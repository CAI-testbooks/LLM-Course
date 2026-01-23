# src/utils.py
import json
import yaml
from typing import Any


def save_json(data: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(data: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)


def load_yaml(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
