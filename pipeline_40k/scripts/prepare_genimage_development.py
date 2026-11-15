"""Selectively extract four non-Synthbuster GenImage domains for development."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GENERATOR_ALIASES = {
    "adm": ("adm",),
    "biggan": ("biggan", "big_gan"),
    "vqdm": ("vqdm", "vq-diffusion", "vqdiffusion"),
    "wukong": ("wukong", "wu_kong"),
    "glide": ("glide",),
    "midjourney": ("midjourney",),
    "sdv4": ("sdv4", "stable_diffusion_v_1_4"),
    "sdv5": ("sdv5", "stable_diffusion_v_1_5"),
}
CLASS_ALIASES = {"fake": ("ai", "fake", "1_fake"), "real": ("nature", "real", "0_real")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_member(name: str) -> tuple[str, str] | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    parts = [part.lower().replace(" ", "_") for part in path.parts]
    generator = next((key for key, aliases in GENERATOR_ALIASES.items()
                      if any(alias in part for alias in aliases for part in parts)), None)
    image_class = next((key for key, aliases in CLASS_ALIASES.items()
                        if any(part == alias for alias in aliases for part in parts)), None)
    return None if generator is None or image_class is None else (generator, image_class)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/external_genimage_dev/genimage_test.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/external_genimage_dev/selected")
    parser.add_argument("--limit-per-class", type=int, default=2000)
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument(
        "--generators", nargs="+", default=["adm", "biggan", "vqdm", "wukong"],
        choices=sorted(GENERATOR_ALIASES),
    )
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output_dir.resolve()
    if not archive.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Archive and output must remain inside pipeline_40k")
    if args.limit_per_class <= 0:
        raise ValueError("limit-per-class must be positive")
    if args.selection_offset < 0:
        raise ValueError("selection-offset must be non-negative")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Development extraction directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        selected: dict[tuple[str, str], list[zipfile.ZipInfo]] = {
            (generator, image_class): []
            for generator in args.generators
            for image_class in CLASS_ALIASES
        }
        for member in source.infolist():
            category = classify_member(member.filename)
            if category is not None and category in selected:
                selected[category].append(member)
        missing = [f"{generator}/{image_class}" for (generator, image_class), members in selected.items() if not members]
        if missing:
            examples = [member.filename for member in source.infolist()[:30]]
            raise ValueError(f"Archive layout did not expose {missing}; first members: {examples}")
        manifest = []
        for (generator, image_class), members in sorted(selected.items()):
            members = sorted(members, key=lambda value: value.filename)[
                args.selection_offset:args.selection_offset + args.limit_per_class
            ]
            if len(members) != args.limit_per_class:
                raise ValueError(
                    f"{generator}/{image_class} has too few members for offset "
                    f"{args.selection_offset} and limit {args.limit_per_class}"
                )
            destination_dir = output / generator / image_class
            destination_dir.mkdir(parents=True, exist_ok=True)
            for index, member in enumerate(members):
                suffix = PurePosixPath(member.filename).suffix.lower()
                stable_name = hashlib.sha1(member.filename.encode("utf-8")).hexdigest() + suffix
                destination = destination_dir / stable_name
                with source.open(member) as reader, destination.open("wb") as writer:
                    while block := reader.read(1024 * 1024):
                        writer.write(block)
                manifest.append({
                    "generator": generator, "class": image_class,
                    "source_member": member.filename,
                    "relative_path": str(destination.relative_to(ROOT)).replace("\\", "/"),
                    "compressed_bytes": member.compress_size, "uncompressed_bytes": member.file_size,
                    "selection_position": index,
                })
                if (index + 1) % 250 == 0:
                    print(f"Extracted {generator}/{image_class}: {index + 1}/{len(members)}", flush=True)
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest), encoding="utf-8"
    )
    provenance = {
        "source": "GenImage official validation split via jzousz/GenImage mirror",
        "official_project": "https://github.com/GenImage-Dataset/GenImage",
        "license": "CC BY-NC-SA 4.0 plus GenImage dataset terms; academic/non-commercial use",
        "archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
        "archive_sha256": sha256_file(archive),
        "generators": list(args.generators), "limit_per_class": args.limit_per_class,
        "selection_offset": args.selection_offset,
        "samples": len(manifest),
        "protocol": ({
            "role": "confirmatory external test; frozen before feature extraction",
            "training_domains": ["diffusiondb_coco", "adm", "biggan", "vqdm"],
            "development_generator": "wukong",
            "restriction": "never used for fitting, calibration, checkpoint selection, or ensemble weighting",
        } if set(args.generators) == {"glide", "midjourney", "sdv4", "sdv5"} else {
            "role": "external development",
            "source_domains": ["adm", "biggan", "vqdm"],
            "held_out_development_generator": "wukong",
            "final_external_benchmark": "Synthbuster and separately frozen GenImage domains",
        }),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
