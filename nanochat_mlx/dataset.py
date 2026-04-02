"""
Dataset download and iteration utilities.
Downloads ClimbMix-400B parquet shards from HuggingFace.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat_mlx.common import get_base_dir

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")


def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """Return full paths to all parquet files in data_dir."""
    data_dir = DATA_DIR if data_dir is None else data_dir

    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print(f"WARNING: Could not find {data_dir}")
            print("Run: python -m nanochat_mlx.dataset -n 8")
        legacy = os.path.join(base_dir, "base_data")
        if os.path.exists(legacy):
            data_dir = legacy
        else:
            return []

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return [os.path.join(data_dir, f) for f in parquet_files]


def parquets_iter_batched(split, start=0, step=1):
    """Iterate through dataset, yielding batches of text from row groups."""
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


def download_single_file(index):
    """Download a single parquet shard with retry/backoff."""
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1)
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # always download val shard

    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for s in results if s)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
