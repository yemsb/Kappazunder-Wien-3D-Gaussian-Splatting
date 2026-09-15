from utils import load_and_verify_config, load_reduction_polygon
import argparse
from pathlib import Path
from build_colmap_selection import load_image_meta, load_interior_orientation, load_fov_per_sensor, select_images
from rotation_conversion import sensor_role_from_pitch
from tqdm import tqdm


def generate_gif(config, reduction_polygon):
	"""Creates a GIF from front-facing images in the AoI, if provided."""

	output_dir = config["dataset_output_dir"]
	for los_dir in Path(config["data_path"]).glob("Los*"):
		image_meta_path = los_dir / "Bild-Meta" / "image_meta.txt"
		interior_orientation_path = los_dir / "Bild-Meta" / "interior_orientation.txt"
		raw_images_root = los_dir / "Bild-Rohdaten"

		image_meta_gdf = load_image_meta(image_meta_path)
		fov_by_sensor = load_fov_per_sensor(interior_orientation_path)

		if reduction_polygon is None:
			selected_gdf = image_meta_gdf.copy()
		else:
			image_meta_gdf = select_images(image_meta_gdf, reduction_polygon, fov_by_sensor, config.get("max_frustum_distance"), raw_images_root)
			selected_gdf = image_meta_gdf[image_meta_gdf["selected"]].copy()

		if len(selected_gdf) == 0:
			print("No images selected -- check ROI polygon / MAX_FRUSTUM_DIST.")
			return selected_gdf

		# Choose only front-facing images for the GIF
		sensor_ids = selected_gdf["sensor_id"]
		front_facing_sensors = [i for i, _ in enumerate(sensor_ids) if sensor_role_from_pitch(sensor_ids.iloc[i]) == "front"]
		selected_gdf = selected_gdf.iloc[front_facing_sensors]

		# Go through the df and create a list of paths
		image_paths = []
		for _, row in selected_gdf.iterrows():
			trajectory_str = "Trajektorie_" + str(row["trajectory_id"])
			sensor_str = "Sensor_" + str(row["sensor_id"])
			image_path = raw_images_root / trajectory_str / sensor_str / row["image_name"]
			if image_path.exists():
				image_paths.append(image_path)

		max_image_limit = 60
		# Chose max_image_limit by taking every Nth image from the list, where N is the total number of images divided by max_image_limit (make sure to keep under the limit)
		if len(image_paths) > max_image_limit:
			step = len(image_paths) // max_image_limit
			image_paths = image_paths[::step]

		# Create a GIF from the images
		print("Opening images and creating GIF...")
		try:
			import cv2
			import imageio.v2 as imageio

			images = []
			for image_path in tqdm(image_paths, desc="Resizing images"):
				image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
				if image is None:
					continue
				height, width = image.shape[:2]
				new_width = round(width * 360 / height)
				image = cv2.resize(image, (new_width, 360), interpolation=cv2.INTER_AREA)
				images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
			if not images:
				print("No readable images found.")
				return
			print(f"Saving GIF to {output_dir / 'front_facing.gif'}...")
			imageio.mimsave(output_dir / "front_facing.gif", images, duration=1 / 12, loop=0)
		except ImportError as e:
			from PIL import Image

			print("Warning: cv2 or imageio not found, falling back to PIL for GIF creation (slower).")

			images = [
				image.resize(
					(round(image.width * 360 / image.height), 360),
					Image.Resampling.LANCZOS,
				)
				for image in tqdm((Image.open(p) for p in image_paths), total=len(image_paths), desc="Resizing images")
			]
			print(f"Saving GIF to {output_dir / 'front_facing.gif'}...")
			images[0].save(output_dir / "front_facing.gif", save_all=True, append_images=images[1:], fps=12, loop=0)


def parse():
	parser = argparse.ArgumentParser(description="Generate a GIF from front-facing images in the AoI.")
	parser.add_argument("--config", type=str, help="Path to the config.yaml file.")
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse()
	config_dir, config = load_and_verify_config(args)

	reduction_polygon = load_reduction_polygon(config_dir, config)

	generate_gif(config, reduction_polygon)