from dataclasses import dataclass

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from threestudio.utils.free_lunch import register_free_upblock2d_in, register_free_crossattn_upblock2d_in


@threestudio.register("stable-diffusion-ssd-guidance")
class StableDiffusionDCGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        ip2p_name_or_path: str = "stabilityai/stable-diffusion-2-1-base" 

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False

        guidance_scale: float = 7.5
        enhance_scale: float = 5.5
        image_scale: float = 2.0

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 1000 # 20
        max_iteration: int = 1500

        use_dds: bool = True
        use_ssd: bool = False

        # FreeU
        freeu_b1: float=1.1
        freeu_b2: float=1.1
        freeu_s1: float=0.9
        freeu_s2: float=0.2


    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.ip2p_name_or_path, **pipe_kwargs
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )  # ddim是稳健的，ddpm有些case获取更好结果
        # self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.iteration = 0
        self.max_iteration = self.cfg.max_iteration

        # FreeU
        b1 = self.cfg.freeu_b1
        b2 = self.cfg.freeu_b2
        s1 = self.cfg.freeu_s1
        s2 = self.cfg.freeu_s2

        register_free_upblock2d_in(self.unet, b1, b2, s1, s2)
        register_free_crossattn_upblock2d_in(self.unet, b1, b2, s1, s2)

        threestudio.info(f"Loaded Stable Diffusion!")
        print(self.min_step)

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            threestudio.debug("Start editing...")
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat(
                        [latent_model_input, image_cond_latents], dim=1
                    )

                    noise_pred = self.forward_unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )

                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                )

                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            threestudio.debug("Editing finished.")
        return latents

    def compute_grad_dds(
        self,
        tgt_latents: Float[Tensor, "B 4 DH DW"],
        src_latents: Float[Tensor, "B 4 DH DW"],
        tgt_text_embeddings: Float[Tensor, "BB 77 768"],
        src_text_embeddings: Float[Tensor, "BB 77 768"],
        null_text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
        t_normalized: Int[Tensor, "B"] = None,
    ):
        noise = torch.randn_like(tgt_latents)  # TODO: use torch generator
        # noise = torch.empty_like(tgt_latents).normal_(generator=torch.Generator(device="cuda").manual_seed(t.item()))
        
        with torch.no_grad():
            latents_noisy_tgt = self.scheduler.add_noise(tgt_latents, noise, t)
            latents_noisy_src = self.scheduler.add_noise(src_latents, noise, t)

            latent_model_input = torch.cat([latents_noisy_src, latents_noisy_tgt, latents_noisy_tgt, latents_noisy_tgt], dim=0)

            text_embeddings = torch.cat([ null_text_embeddings, src_text_embeddings, null_text_embeddings,tgt_text_embeddings], dim=0) # null, src, null, tgt

            noise_pred = self.forward_unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )

            noise_pred_src_null, noise_pred_tgt_source, noise_pred_tgt_null, noise_pred_tgt_target = noise_pred.chunk(4)

            grad_ours = (
                self.cfg.image_scale * (noise_pred_tgt_source - noise_pred_src_null) * 0.5 +  # preserving source content
                self.cfg.guidance_scale * (noise_pred_tgt_target - noise_pred_tgt_source) * 0.5 +  # target prompt
                self.cfg.enhance_scale * (noise_pred_tgt_target - noise_pred_tgt_null)  # enhancing target prompt
            )

            # noise_pred_branch1 = noise_pred_tgt_source - noise_pred_src_null # 图片稳定变换
            # noise_pred_branch2 = 7.5 * (noise_pred_tgt_target - noise_pred_tgt_source)  # 编辑变化
            # noise_pred_branch3 = 5.5 * (noise_pred_tgt_target - noise_pred_tgt_null)  # 风格化
            # grad_ours = noise_pred_branch1 * 2.0 + noise_pred_branch2  + noise_pred_branch3

            noise_pred_branch4 = (latents_noisy_tgt - latents_noisy_src)  # img保持

            # grad = (grad_ours * (t_normalized ** (1/math.e)) * 0.75 + 0.075 * noise_pred_branch4 * math.exp(t_normalized))  # weight from DreamCatalyst

            grad = (grad_ours * 1.0 + 0.075 * noise_pred_branch4 * math.exp(t_normalized) * 1.0 ) # most cases

            # other options
            # grad = (grad_ours * 1.0 + 0.075 * noise_pred_branch4 * (t_normalized) * 1.0 )
            # grad = grad_ours

        # grad = grad_ours  

        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        target_prompt_utils: PromptProcessorOutput,
        source_prompt_utils: PromptProcessorOutput,
        # TODO: DDS
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        target_latents: Float[Tensor, "B 4 DH DW"]
        source_latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        target_latents = self.encode_images(rgb_BCHW_HW8)

        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )

        source_latents = self.encode_images(cond_rgb_BCHW_HW8)
        cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        temp = torch.zeros(1).to(rgb.device)
        target_text_embeddings = target_prompt_utils.get_text_embeddings(temp, temp, temp, False)
        # target_text_embeddings = torch.cat(
        #     [target_text_embeddings, target_text_embeddings[-1:]], dim=0
        # )  # [positive, negative, negative]

        source_text_embeddings = source_prompt_utils.get_text_embeddings(temp, temp, temp, False)
        # source_text_embeddings = torch.cat(
        #     [source_text_embeddings, source_text_embeddings[-1:]], dim=0
        # )  # [positive, negative, negative]
        null_text_embeddings = target_text_embeddings[-1:]
        target_text_embeddings = target_text_embeddings[:1]
        source_text_embeddings = source_text_embeddings[:1]


        if self.cfg.use_ssd:
            timesteps = reversed(self.scheduler.timesteps)

            self.min_step = 1 if self.cfg.min_step_percent <= 0 else int(len(timesteps) * self.cfg.min_step_percent)
            max_step = (
                len(timesteps) if self.cfg.max_step_percent >= 1 else int(len(timesteps) * self.cfg.max_step_percent)
            )
            self.max_step = max(max_step, self.min_step + 1)

            timestep_index = torch.full((batch_size,), (self.max_step - self.min_step) * ((self.max_iteration - self.iteration) / self.max_iteration) + self.min_step, dtype=torch.long, device="cpu")

            # t = torch.randint(
            #     self.min_step,
            #     self.max_step + 1,
            #     [batch_size],
            #     dtype=torch.long,
            #     device=self.device,
            # )

            t = timesteps[timestep_index].to(self.device)
            t_noralized = timestep_index[0].item() / len(timesteps)
        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
    
        self.iteration += 1

        if self.cfg.use_dds:
            grad = self.compute_grad_dds(
                target_latents, 
                source_latents,
                target_text_embeddings, 
                source_text_embeddings,
                null_text_embeddings,
                t,
                t_noralized if self.cfg.use_ssd else None
            )
            grad = torch.nan_to_num(grad)
            
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (target_latents - grad).detach()
            loss_dds = 0.5 * F.mse_loss(target_latents, target, reduction="sum") / batch_size
            return {
                "loss_dds": loss_dds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }

        else:
            edit_latents = self.edit_latents(target_text_embeddings, target_latents, cond_latents, t)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if not self.cfg.use_ssd:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/instructpix2pix.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )
    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (
            guidance_out["edit_images"][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .clip(0, 1)
            .numpy()
            * 255
        )
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    import os

    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
