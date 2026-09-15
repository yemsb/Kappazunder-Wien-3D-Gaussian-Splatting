import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from moge.model.v3 import MoGeModel
from moge.utils.vis import colorize_normal
import torch
from tqdm import tqdm


# def depth_to_rgb_turbo(depth_map, skip_zero=True):
#     """
#     Convert depth map to RGB using Spirula's turbo colormap

#     Args:
#         depth_map: 2D numpy array of depth values (float32, in meters)
#         skip_zero: If True, zero/negative values are set to black

#     Returns:
#         3D numpy array of RGB values (uint8, shape HxWx3)
#     """
#     # Make a copy to avoid modifying original
#     depth = depth_map.copy()

#     # Handle invalid values
#     valid_mask = np.isfinite(depth)
#     if skip_zero:
#         valid_mask = valid_mask & (depth > 0)

#     # Initialize output as black
#     rgb = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

#     if not np.any(valid_mask):
#         return rgb

#     # Find min and max of valid values
#     valid_depths = depth[valid_mask]
#     min_depth = np.min(valid_depths)
#     max_depth = np.max(valid_depths)
#     span = max_depth - min_depth

#     if span < 1e-6:  # Avoid division by zero
#         span = 1.0

#     # Normalize to [0, 1]
#     normalized = np.zeros_like(depth)
#     normalized[valid_mask] = (depth[valid_mask] - min_depth) / span
#     normalized = np.clip(normalized, 0, 1)

#     # Apply turbo colormap (coefficients from Spirula's code)
#     # Turbo colormap coefficients for R, G, B channels
#     turbo_coeffs = np.array([
#         [0.14637796, 2.94711014, -10.15061040, -90.46877071,
#          550.44382083, -1061.31232675, 874.85901369, -266.03287948],
#         [0.08594198, 2.06520532, 9.64581379, -55.09180383,
#          149.40813531, -245.26823636, 202.98596195, -63.82163439],
#         [0.23431523, 6.33792818, 9.26380342, -203.76964931,
#          604.07329733, -766.82678386, 447.99287080, -97.24397473]
#     ], dtype=np.float32)

#     # Apply polynomial for each channel using Horner's method
#     for c in range(3):
#         # Start with highest order coefficient
#         channel_vals = np.full_like(normalized, turbo_coeffs[c, 7], dtype=np.float32)

#         # Apply remaining coefficients
#         for k in range(6, -1, -1):
#             channel_vals = channel_vals * normalized + turbo_coeffs[c, k]

#         # Scale to [0, 255] and clip
#         channel_vals = np.clip(channel_vals, 0, 1) * 255
#         rgb[:, :, c] = np.where(valid_mask, channel_vals.astype(np.uint8), 0)

#     return rgb


def save_depth(depth, filepath):
    """
    Save depth map as grayscale PNG (like Spirula's DepthPng.cpp)

    Args:
        depth_map: 2D numpy array of depth values (float32, in meters)
        filepath: Output file path
    """
    depth_map = depth.copy()
    # initialize output image black RGB image
    depth_map /= np.max(np.nan_to_num(depth_map, nan=0.0, posinf=-1.0))  # Normalize to [0, 1]
    # Set invalid to 0 (black)
    depth_map[~np.isfinite(depth_map)] = 0.0
    output_image = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    output_image[:, :, 0] = (np.round(depth_map * 255)).astype(np.uint8)
    output_image[:, :, 1] = (np.round(depth_map * 255)).astype(np.uint8)
    output_image[:, :, 2] = (np.round(depth_map * 255)).astype(np.uint8)

    cv2.imwrite(filepath, output_image)


def save_normal(normal, filepath):
    """
    Save normal map as RGB PNG (like Spirula's NormalPng.cpp)

    Args:
        normal_map: 3D numpy array of normal values (float32, in range [-1, 1])
        filepath: Output file path
    """
    normal_map = normal.copy()
    # Invert G and B channels
    normal_map[:, :, 1] = 255 - normal_map[:, :, 1]
    normal_map[:, :, 2] = 255 - normal_map[:, :, 2]

    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    cv2.imwrite(filepath, normal_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get depth and normal data from MoGe3 output.")
    parser.add_argument("--images-path", type=str, required=True, help="Path to the images directory.")
    args = parser.parse_args()
    images_path = Path(args.images_path)

    if not images_path.exists() or not images_path.is_dir():
        print(f"Error: The provided images path '{images_path}' does not exist or is not a directory.")
        exit(1)

    os.makedirs(images_path.parent / "depths", exist_ok=True)
    os.makedirs(images_path.parent / "normals", exist_ok=True)
    os.makedirs(images_path.parent / "masks", exist_ok=True)

    # Remove existing files in the depths and normals directories
    for output_path in [images_path.parent / "depths", images_path.parent / "normals"]:
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Instead of the below code (using CLI), we use the moge python library directly to get depth and normal maps.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        print("Using device: CPU")

    model = MoGeModel.from_pretrained("../MoGe3_pipeline/moge-3-vitl/model.pt").to(device)

    for image_file in tqdm(sorted(images_path.glob("*.jpg"))):
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1064, 1064))
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model.infer(image_tensor, fov_x=90, resolution_level=6)

        depth = output["depth"].squeeze().cpu().numpy()
        normal = output["normal"].squeeze().cpu().numpy()
        mask = depth != np.inf  # Create a mask for valid depth values

        normal_colorized = colorize_normal(normal)

        frame_number = image_file.stem.split("_")[1]
        save_depth(depth, images_path.parent / "depths" / f"frame_{frame_number}.png")
        save_normal(normal_colorized, images_path.parent / "normals" / f"frame_{frame_number}.png")
        cv2.imwrite(images_path.parent / "masks" / f"frame_{frame_number}.png", (mask).astype(np.uint8) * 255)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Get depth and normal data from MoGe3 output.")
#     parser.add_argument("--images-path", type=str, required=True, help="Path to the images directory.")
#     args = parser.parse_args()
#     images_path = Path(args.images_path)

#     if not images_path.exists() or not images_path.is_dir():
#         print(f"Error: The provided images path '{images_path}' does not exist or is not a directory.")
#         exit(1)

#     os.makedirs(images_path.parent / "depths", exist_ok=True)
#     os.makedirs(images_path.parent / "normals", exist_ok=True)

#     # Remove existing files in the depths and normals directories
#     for output_path in [images_path.parent / "depths", images_path.parent / "normals"]:
#         for file in os.listdir(output_path):
#             file_path = os.path.join(output_path, file)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)

#     os.makedirs(images_path.parent / "output", exist_ok=True)
#     system_command = f"moge infer -i {images_path} --version v3 --pretrained ../MoGe3_pipeline/model.pt --refine_steps 3 -o {images_path.parent / 'output'} --maps --resize 1064 --fov_x 90"
#     # system_command = f"moge infer -i {images_path} --version v2 --pretrained ../MoGe3_pipeline/moge-2-vitl-normal/model.pt -o {images_path.parent / 'output'} --maps --resize 1064 --fov_x 90"
#     print(f"Running command: {system_command}")
#     os.system(system_command)

#     # Get all depth and normal maps in output using search pattern and move them to the respective directories with proper names
#     # All depths have the pattern "frame_*/depth_vis.png" and all normales have the pattern "frame_*/normal.png"
#     depth_files = list(Path(images_path.parent / "output").rglob("frame_*/depth_vis.png"))
#     normal_files = list(Path(images_path.parent / "output").rglob("frame_*/normal.png"))

#     # Move and rename depth files
#     for depth_file in depth_files:
#         frame_number = depth_file.parent.name.split("_")[1]
#         new_depth_file_name = f"frame_{frame_number}.png"
#         new_depth_file_path = images_path.parent / "depths" / new_depth_file_name
#         # Assuming the output is in turbo cmap with inverted scale, we need to re-invert it
#         depth_image = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
#         if depth_image is not None:
#             # map turbo cmap back to depth values (0-1), invert it (1 - value), and save again as turbo cmap
#             pass
#         depth_file.rename(new_depth_file_path)

#     # Move and rename normal files
#     for normal_file in normal_files:
#         frame_number = normal_file.parent.name.split("_")[1]
#         new_normal_file_name = f"frame_{frame_number}.png"
#         new_normal_file_path = images_path.parent / "normals" / new_normal_file_name
#         # Instead of moving, read the normal map and invert 0 and 1 channels
#         normal_image = cv2.imread(str(normal_file), cv2.IMREAD_UNCHANGED)
#         # Invert 0 and 1 channels
#         if normal_image is not None:
#             channels = normal_image[:, :, [0, 1]]
#             normal_image[:, :, [0, 1]] = np.where(channels == 0, 0, 255 - channels)
#             cv2.imwrite(str(new_normal_file_path), normal_image)

#     # Remove the output directory after moving the files
#     for file in os.listdir(images_path.parent / "output"):
#         file_path = os.path.join(images_path.parent / "output", file)
#         if os.path.isdir(file_path):
#             for sub_file in os.listdir(file_path):
#                 sub_file_path = os.path.join(file_path, sub_file)
#                 if os.path.isfile(sub_file_path):
#                     os.remove(sub_file_path)
#             os.rmdir(file_path)
#         elif os.path.isfile(file_path):
#             os.remove(file_path)
#     os.rmdir(images_path.parent / "output")