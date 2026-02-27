from pathlib import Path
import yaml

def load_paths():

    # RIGHT
    config_file = Path(__file__).resolve().parents[1] / "config" / "paths.yaml"

    with open(config_file, "r") as f:
        return yaml.safe_load(f)
