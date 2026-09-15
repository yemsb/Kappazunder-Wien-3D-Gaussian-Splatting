from pathlib import Path
import pandas as pd
from shapely.geometry import Polygon
import yaml


def load_AoI(aoi_path):
    """Load the area-of-interest polygon from a CSV file (EPSG:31256)."""
    if not Path(aoi_path).exists():
        raise FileNotFoundError(f"Area-of-interest CSV not found: {aoi_path}")
    
    df = pd.read_csv(aoi_path, sep="\t", header=None, names=["x_m", "y_m"])
    coords = df.to_numpy()
    return Polygon(coords)


# The user provides a config.yaml file with the above parameters.
def load_config(config_path):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required keys
    required_keys = [
        "data_path",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    return config


def load_reduction_polygon(config_dir, config):
    # Look for the AoI polygon CSV if:
    # 1. The config contains no use_AoI key -> skip silently if no CSV found
    # 2. The config contains use_AoI: True -> raise error if no CSV found
    # Skip otherwise (use_AoI: False)
    reduction_polygon = None
    if not "use_AoI" in config:
        aoi_csv_path = config_dir / "AoI.csv"
        if aoi_csv_path.exists():
            reduction_polygon = load_AoI(aoi_csv_path)
    elif config.get("use_AoI", True):
        aoi_csv_path = config_dir / "AoI.csv"
        if aoi_csv_path.exists():
            reduction_polygon = load_AoI(aoi_csv_path)
        else:
            raise FileNotFoundError(f"Area-of-interest CSV not found: {aoi_csv_path}")
    return reduction_polygon


def load_and_verify_config(args):
    if not args.config:
        print("Usage: python create_gif.py --config <path_to_config.yaml>")
        exit(1)
    config_path = args.config # Path to config file
    config_dir = Path(config_path).parent # Path to directory containing config file
    config = load_config(config_path)
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # If directory doesn't exist, create dataset output directory at ./datasets/<config_dir_name>/
    dataset_output_dir = Path("./datasets") / config_dir.name
    config["dataset_output_dir"] = dataset_output_dir
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded config from {config_path}:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    return config_dir,config