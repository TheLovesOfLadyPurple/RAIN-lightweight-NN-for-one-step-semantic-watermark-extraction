import argparse
import sys
from pathlib import Path
from typing import Iterable
from PIL import Image, UnidentifiedImageError

# Deletes images larger than a threshold under the pd12m directory.

def find_image_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.suffix.lower() in exts and path.is_file():
            yield path


def delete_large_images(root: Path, threshold: int, dry_run: bool = False) -> None:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    deleted = 0
    skipped = 0
    errors = 0

    for img_path in find_image_files(root, exts):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except (UnidentifiedImageError, OSError) as exc:
            errors += 1
            print(f"Error reading {img_path}: {exc}")
            continue

        if w * h > threshold:
            deleted += 1
            action = "Would delete" if dry_run else "Deleting"
            print(f"{action} {img_path} ({w}x{h})")
            if not dry_run:
                try:
                    img_path.unlink()
                except OSError as exc:
                    errors += 1
                    print(f"Failed to delete {img_path}: {exc}")
        else:
            skipped += 1

    print(f"Done. Deleted: {deleted}, kept: {skipped}, errors: {errors}.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Delete pd12m images larger than a threshold.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("pd12m"),
        help="Path to pd12m root directory (default: ./pd12m)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1048517,
        help="Maximum allowed width or height before deletion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without removing them.",
    )

    args = parser.parse_args(argv)
    if not args.root.exists() or not args.root.is_dir():
        print(f"Root directory {args.root} does not exist or is not a directory.")
        return 1

    delete_large_images(args.root, args.threshold, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
