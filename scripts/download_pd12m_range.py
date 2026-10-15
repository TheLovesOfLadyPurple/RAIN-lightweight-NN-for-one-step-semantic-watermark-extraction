"""Download a slice of the pd12m-full dataset from Hugging Face.

This pulls selected tar shards from the dataset repo Spawning/pd12m-full.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pd12m-full tar shards")
    parser.add_argument(
        "--repo-id",
        default="Spawning/pd12m-full",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=173,
        help="First shard number (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=182,
        help="Last shard number (inclusive)",
    )
    parser.add_argument(
        "--output",
        default="./pd12m_tars",
        help="Directory to store downloaded tar files",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent downloads",
    )
    return parser.parse_args()


def download_one(repo_id: str, shard: int, out_dir: Path, token: str | None) -> tuple[int, str]:
    filename = f"{shard:05d}.tar"
    target_path = out_dir / filename
    if target_path.exists():
        return shard, "skipped (exists)"

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        return shard, "downloaded"
    except Exception as exc:  # network or auth failure
        return shard, f"failed: {exc}"


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional login step if a token is provided and the repo is gated.
    if token:
        login(token=token)

    shards = list(range(args.start, args.end + 1))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_one, args.repo_id, shard, out_dir, token) for shard in shards]
        for fut in as_completed(futures):
            shard, status = fut.result()
            print(f"{shard:05d}.tar -> {status}")


if __name__ == "__main__":
    main()