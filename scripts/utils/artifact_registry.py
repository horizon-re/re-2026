import yaml
from pathlib import Path
import uuid
import datetime

CONFIG_FILE = Path(__file__).resolve().parents[2] / "config" / "artifacts.yaml"

def load_artifacts():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {"runs": []}

def save_artifacts(data):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(data, f)

def get_or_create_run():
    artifacts = load_artifacts()
    if not artifacts["runs"]:
        run_id = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        run_uuid = str(uuid.uuid4())
        run = {"run_id": run_id, "uuid": run_uuid, "steps": {}}
        artifacts["runs"].append(run)
        save_artifacts(artifacts)
        return run
    return artifacts["runs"][-1]


def register_artifact(step, name, path):
    artifacts = load_artifacts()
    run = artifacts["runs"][-1]
    run["steps"].setdefault(step, {})[name] = path
    save_artifacts(artifacts)