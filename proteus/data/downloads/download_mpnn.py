
import argparse
import tarfile
from pathlib import Path

import requests
from cloudpathlib import S3Path
from tqdm import tqdm

from proteus.utils.s3_utils import upload_dir_to_s3

DATASET = "pdb_2021aug02"
mpnn_url = lambda dataset: f"https://files.ipd.uw.edu/pub/training_sets/{dataset}.tar.gz"

def download_mpnn(data_dir: Path, s3_path: S3Path | None = None, debug: bool = False) -> None:
    """download and extract the ProteinMPNN training dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = DATASET if not debug else DATASET + "_sample"
    url = mpnn_url(dataset)

    dataset_archive = dataset + ".tar.gz"
    tar_path = data_dir / dataset_archive
    extracted_path = data_dir / dataset

    # stream download the tarball with progress
    if tar_path.exists() or extracted_path.exists():
        print(f"skipping download, {tar_path} already downloaded/extracted")
    else:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(tar_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="downloading") as pbar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    pbar.update(len(chunk))

    # extract and clean up
    if extracted_path.exists():
        print(f"skipping extraction, {extracted_path} already exists")
    else:
        print(f"extracting to {data_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir, filter="data")

    if tar_path.exists():
        tar_path.unlink()
    print(f"finished downloading {dataset} to {data_dir}")

    # upload to S3 if path provided
    if s3_path:
        upload_dir_to_s3(data_dir / dataset, s3_path)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download proteus training datasets")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ubuntu/proteus/data"),
        help="directory to download data to",
    )
    parser.add_argument(
        "--s3_path",
        type=S3Path,
        default=S3Path("s3://pmpnn-data-bucket"),
        help="optional S3 URI to upload extracted data (e.g. s3://pmpnn-data-bucket/)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="test this on the smaller dataset",
    )
    args = parser.parse_args()
    download_mpnn(args.data_dir, s3_path=args.s3_path, debug=args.debug)
