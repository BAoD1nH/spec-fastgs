#!/usr/bin/env python3
"""Download and prepare the datasets used by spec-fastgs.

The archives are downloaded from Google Drive, extracted safely, and merged
into these canonical locations:

    datasets/anisotropic-synthetic/
    datasets/mipnerf360/

The command is safe to run repeatedly: existing files are kept by default.
Use --overwrite only when an existing prepared dataset must be replaced.
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


DATASETS = {
    "anisotropic-synthetic": {
        "url": "https://drive.google.com/file/d/1fVmfIl0DyjK2CUozsFnr7r57PaQFsucr/view?usp=sharing",
        "target": "anisotropic-synthetic",
        "aliases": {
            "anisotropicsynthetic",
            "anisotropicsynthesis",
            "anisotropicsyntheticdataset",
        },
    },
    "mipnerf360": {
        "url": "https://drive.google.com/file/d/1D0C0sW4nPjkq3UoJqjzj9VizTf8wo8Gj/view?usp=sharing",
        "target": "mipnerf360",
        "aliases": {"mipnerf360", "mipnerf360dataset"},
    },
}


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download and prepare spec-fastgs datasets from Google Drive."
    )
    parser.add_argument(
        "--dataset",
        choices=("all",) + tuple(DATASETS),
        default="all",
        help="Dataset to prepare (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "datasets",
        help="Destination directory (default: <repo>/datasets).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / ".cache" / "datasets",
        help="Archive cache (default: <repo>/.cache/datasets).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files. Without this flag, existing files are kept.",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Download archives again even when a cached copy exists.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives after successful extraction.",
    )
    return parser.parse_args()


def normalized_name(name):
    return "".join(character.lower() for character in name if character.isalnum())


def require_gdown():
    try:
        import gdown
    except ImportError:
        print(
            "ERROR: Missing dependency 'gdown'. Install it with:\n"
            "  python -m pip install gdown",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return gdown


def download_archive(name, config, cache_dir, redownload):
    gdown = require_gdown()
    archive_path = cache_dir / (name + ".download")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and archive_path.stat().st_size > 0 and not redownload:
        print("[cache] Using {}".format(archive_path))
        return archive_path

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if partial_path.exists():
        partial_path.unlink()

    print("[download] {}".format(name))
    result = gdown.download(
        url=config["url"],
        output=str(partial_path),
        quiet=False,
        fuzzy=True,
        resume=True,
    )
    if not result or not partial_path.exists() or partial_path.stat().st_size == 0:
        raise RuntimeError("Google Drive download failed for {}".format(name))

    os.replace(str(partial_path), str(archive_path))
    return archive_path


def safe_destination(root, member_name):
    root = root.resolve()
    destination = (root / member_name).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        raise RuntimeError("Unsafe archive path: {}".format(member_name))


def extract_archive(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(str(archive_path)):
        with zipfile.ZipFile(str(archive_path)) as archive:
            for member in archive.infolist():
                safe_destination(destination, member.filename)
            archive.extractall(str(destination))
        return

    if tarfile.is_tarfile(str(archive_path)):
        with tarfile.open(str(archive_path), "r:*") as archive:
            for member in archive.getmembers():
                safe_destination(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(
                        "Archive links are not allowed: {}".format(member.name)
                    )
            archive.extractall(str(destination))
        return

    raise RuntimeError(
        "Unsupported archive downloaded for {} (expected ZIP or TAR).".format(
            archive_path
        )
    )


def visible_children(path):
    return [
        child
        for child in path.iterdir()
        if child.name not in {"__MACOSX", ".DS_Store"}
    ]


def locate_payload(extracted_root, aliases):
    aliases = set(aliases)
    candidates = [extracted_root]
    candidates.extend(
        path
        for path in extracted_root.rglob("*")
        if path.is_dir() and len(path.relative_to(extracted_root).parts) <= 3
    )
    for candidate in candidates:
        if normalized_name(candidate.name) in aliases:
            return candidate

    payload = extracted_root
    while True:
        children = visible_children(payload)
        if len(children) == 1 and children[0].is_dir():
            payload = children[0]
        else:
            return payload


def merge_tree(source, destination, overwrite):
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for source_path in source.rglob("*"):
        relative_path = source_path.relative_to(source)
        destination_path = destination / relative_path

        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        if source_path.is_symlink():
            raise RuntimeError("Dataset contains an unsupported symlink: {}".format(source_path))

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(str(source_path), str(destination_path))
        copied += 1

    return copied, skipped


def validate_dataset(name, target):
    children = visible_children(target) if target.exists() else []
    scene_directories = [child for child in children if child.is_dir()]
    if not scene_directories:
        raise RuntimeError("No scene directories found in {}".format(target))

    scene_names = ", ".join(sorted(path.name for path in scene_directories))
    print("[ready] {}: {}".format(name, target))
    print("        scenes: {}".format(scene_names))


def prepare_one(name, config, args):
    archive_path = download_archive(name, config, args.cache_dir, args.redownload)
    target = args.output_dir / config["target"]

    with tempfile.TemporaryDirectory(prefix="spec-fastgs-dataset-") as temp_dir:
        extracted_root = Path(temp_dir)
        print("[extract] {}".format(archive_path))
        extract_archive(archive_path, extracted_root)
        payload = locate_payload(extracted_root, config["aliases"])
        copied, skipped = merge_tree(payload, target, args.overwrite)

    validate_dataset(name, target)
    print("        copied: {}, existing kept: {}".format(copied, skipped))

    if not args.keep_archives and archive_path.exists():
        archive_path.unlink()


def main():
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    try:
        for name, config in selected.items():
            prepare_one(name, config, args)
    except (OSError, RuntimeError, zipfile.BadZipFile, tarfile.TarError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1

    print("\nDataset preparation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
