import json
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data
