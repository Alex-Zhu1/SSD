# from clip_similarity import ClipSimilarity # type: ignore
from clip_metrics import ClipSimilarity
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import os
import torch
from pathlib import Path
from pytorch_lightning import seed_everything

seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_file_names(directory, extensions):
    """Fetch all file names with specific extensions from the given directory."""
    return [file for file in os.listdir(directory) if file.lower().endswith(tuple(extensions)) and os.path.isfile(os.path.join(directory, file))]

def read_images(images_dir, extensions=("png", "jpg", "jpeg")):
    """Read images from the given directory and return them as a list of tensors."""
    file_names = get_file_names(images_dir, extensions)
    image_paths = [os.path.join(images_dir, file_name) for file_name in file_names]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    
    # 使用索引随机选择 20 张图像
    selected_indices = np.random.choice(len(images), 20, replace=False)
    images = [images[i] for i in selected_indices]
    
    # Changing the array shape from [h,w,c] to [1,c,w,h]
    images = [torch.Tensor(np.array(image).T[None, :, :, :]) for image in images]
    return images


def main():
    # Read and parse arguments
    parser = ArgumentParser()
    parser.add_argument("--original-dir", required=True, type=str)
    parser.add_argument("--edited-dir", required=True, type=str)
    parser.add_argument("--original-caption", required=True, type=str)
    parser.add_argument("--edited-caption", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    original_dir = Path(args.original_dir)
    edited_dir = Path(args.edited_dir)
    original_caption = args.original_caption  # source prompt
    edited_caption = args.edited_caption  # instruction prompt
    # Load original and edited views as tensors
    original_views = read_images(original_dir)  # nerf_ render views
    edited_views = read_images(edited_dir)  # edited views
    clip_similarity = ClipSimilarity()
    sim_dirs = []
    clip_edit = []
    sim_images = []
    sim_00 = []
    # calculate CLIP Direction Similarity for each original/edited image pair
    for i in range(len(original_views)):
        sim_0, sim_1, sim_direction, sim_image = clip_similarity(
            original_views[i], edited_views[i], original_caption, edited_caption
        )
        # print(float(sim_direction))
        clip_edit.append(float(sim_direction))
        sim_dirs.append(float(sim_1))
        sim_images.append(float(sim_image))
        sim_00.append(float(sim_0))
    # Print mean directional similarity
    print("edited-caption", edited_caption)
    print("clip sim", np.mean(sim_dirs))
    print("edir sim", np.mean(clip_edit))
    print("clip img", np.mean(sim_images))
    print("clip 00", np.mean(sim_00))

if __name__=="__main__":
    main()