import os
import tarfile
import urllib.request

import numpy as np
from sklearn.datasets import load_files
from tqdm import tqdm


IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE_NAME = "aclImdb_v1.tar.gz"


def _safe_extract_tar(tar, path):
    abs_path = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(path, member.name))
        if not member_path.startswith(abs_path + os.sep) and member_path != abs_path:
            raise RuntimeError("Unsafe path detected in tar archive.")
    tar.extractall(path)


def download_imdb_data(data_root="data_cache"):
    """
    Downloads and extracts the IMDB dataset into data_root.
    Returns the path to the extracted aclImdb directory.
    """
    os.makedirs(data_root, exist_ok=True)
    archive_path = os.path.join(data_root, IMDB_ARCHIVE_NAME)
    imdb_path = os.path.join(data_root, "aclImdb")

    if not os.path.exists(archive_path):
        tqdm.write(f"Downloading IMDB dataset to {archive_path}...")
        urllib.request.urlretrieve(IMDB_URL, archive_path)
    else:
        tqdm.write(f"IMDB archive already exists at {archive_path}, skipping download.")

    if not os.path.exists(imdb_path):
        tqdm.write(f"Extracting IMDB dataset to {data_root}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_tar(tar, data_root)

    return imdb_path


def load_imdb_data(download_if_missing=True, data_root="data_cache"):
    """
    Loads the full IMDB dataset from aclImdb or data_root/aclImdb.
    Returns (data, target), where data is a list of strings and target is a list/array of labels.
    """
    # Check both potential locations
    potential_paths = [
        "aclImdb",
        os.path.join(data_root, "aclImdb"),
    ]
    imdb_path = None
    for p in potential_paths:
        if os.path.exists(p):
            imdb_path = p
            break

    if imdb_path is None and download_if_missing:
        try:
            download_imdb_data(data_root=data_root)
        except Exception as exc:
            raise FileNotFoundError(
                f"IMDB dataset not found in {potential_paths} and download failed."
            ) from exc
        for p in potential_paths:
            if os.path.exists(p):
                imdb_path = p
                break

    if imdb_path is None:
        raise FileNotFoundError(
            f"IMDB dataset not found in {potential_paths}. Please download it from "
            f"{IMDB_URL} and extract it."
        )

    tqdm.write("Loading IMDB Train Data...")
    train_data = load_files(
        os.path.join(imdb_path, "train"), categories=["pos", "neg"], encoding="utf-8"
    )

    tqdm.write("Loading IMDB Test Data...")
    test_data = load_files(
        os.path.join(imdb_path, "test"), categories=["pos", "neg"], encoding="utf-8"
    )

    # Combine Train and Test
    all_data = train_data.data + test_data.data
    all_target = np.concatenate([train_data.target, test_data.target])

    return all_data, all_target
