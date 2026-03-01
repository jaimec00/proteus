
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from cloudpathlib import S3Path
from tqdm import tqdm


REGION = "us-east-1"


def upload_dir_to_s3(local_path: Path, s3_path: S3Path, max_workers: int = 32) -> None:
	"""upload a local directory to S3 with parallel file and per-file multipart uploads"""
	bucket = s3_path.bucket
	prefix = s3_path.key

	transfer_config = TransferConfig(
		max_concurrency=20,
		multipart_threshold=8 * 1024 * 1024,
		multipart_chunksize=8 * 1024 * 1024,
	)

	files = [f for f in local_path.rglob("*") if f.is_file()]
	total = len(files)

	session = boto3.Session(region_name=REGION)
	client = session.client("s3")
	pbar = tqdm(total=total, desc=f"uploading to {s3_path}", unit="file")

	def upload_file(file_path: Path) -> None:
		key = f"{prefix}/{file_path.relative_to(local_path.parent)}" if prefix else str(file_path.relative_to(local_path.parent))
		client.upload_file(str(file_path), bucket, key, Config=transfer_config)
		pbar.update(1)

	# upload files in parallel and collect any failures
	failures = []
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = {executor.submit(upload_file, f): f for f in files}
		for future in as_completed(futures):
			try:
				future.result()
			except Exception as e:
				failures.append((futures[future], e))

	pbar.close()
	if failures:
		for path, err in failures:
			print(f"  failed: {path} — {err}")
		raise RuntimeError(f"{len(failures)}/{total} uploads failed")

	print("s3 upload complete.")
