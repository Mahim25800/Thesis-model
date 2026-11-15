"""Apply the promoted multi-entity v3 model to compatible cached feature rows."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.multi_entity import MultiEntityCachedClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/multi_entity_v3/model.pt")
    parser.add_argument("--row", type=int, default=None, help="One zero-based row; omit for the complete cache")
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if args.row is None:
        features, confidences = cache["features"], cache["confidences"]
    else:
        features, confidences = cache["features"][args.row], cache["confidences"][args.row]
    result = MultiEntityCachedClassifier(args.model).predict_features(features, confidences)
    print(json.dumps({key: value.tolist() for key, value in result.items()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
