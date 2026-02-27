import yaml, datetime
from pathlib import Path

def write_step_config(out_dir: Path, step_name: str, **knobs):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "step": step_name,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "knobs": knobs
    }
    with open(out_dir / f"{step_name}.config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
