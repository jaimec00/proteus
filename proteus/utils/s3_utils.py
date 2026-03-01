
import asyncio
from pathlib import Path

import aioboto3
from aiobotocore.config import AioConfig
from boto3.s3.transfer import TransferConfig
from cloudpathlib import S3Path
from tqdm import tqdm
import subprocess
import json

REGION = "us-east-1"

def get_session(aio: bool=False):
	'''
	aws and boto3 resolve credentials differently, use aws cli to do it
	'''
	result = subprocess.run(
		["aws", "configure", "export-credentials", "--format", "process"],
		capture_output=True, text=True, check=True,
	)
	creds = json.loads(result.stdout)
	aws_lib = aioboto3 if aio else boto3
	return aws_lib.Session(
		aws_access_key_id=creds["AccessKeyId"],
		aws_secret_access_key=creds["SecretAccessKey"],
		aws_session_token=creds["SessionToken"],
	)

async def _upload_dir_to_s3(local_path: Path, s3_path: S3Path, max_concurrency: int = 32) -> None:
	"""upload a local directory to S3 with async parallel uploads"""
	bucket = s3_path.bucket
	prefix = s3_path.key

	files = [f for f in local_path.rglob("*") if f.is_file()]
	total = len(files)

	session = get_session(aio=True)
	aio_config = AioConfig(max_pool_connections=max_concurrency)
	transfer_config = TransferConfig(
		max_concurrency=20,
		multipart_threshold=8 * 1024 * 1024,
		multipart_chunksize=8 * 1024 * 1024,
	)
	semaphore = asyncio.Semaphore(max_concurrency)
	failures = []

	pbar = tqdm(total=total, desc=f"uploading to {s3_path}", unit="file")

	async with session.client("s3", region_name=REGION, config=aio_config) as client:

		async def upload_file(file_path: Path) -> None:
			key = f"{prefix}/{file_path.relative_to(local_path.parent)}" if prefix else str(file_path.relative_to(local_path.parent))
			async with semaphore:
				await client.upload_file(str(file_path), bucket, key, Config=transfer_config)
			pbar.update(1)

		tasks = [asyncio.create_task(upload_file(f)) for f in files]
		results = await asyncio.gather(*tasks, return_exceptions=True)

		for file_path, result in zip(files, results):
			if isinstance(result, Exception):
				failures.append((file_path, result))

	pbar.close()
	if failures:
		for path, err in failures:
			print(f"  failed: {path} — {err}")
		raise RuntimeError(f"{len(failures)}/{total} uploads failed")

	print("s3 upload complete.")


async def upload_bytes_to_s3(data: bytes, s3_path: S3Path, client) -> None:
	"""upload a bytes object to S3"""
	await client.put_object(Bucket=s3_path.bucket, Key=s3_path.key, Body=data)


def upload_dir_to_s3(local_path: Path, s3_path: S3Path, max_concurrency: int = 32) -> None:
	"""sync wrapper for async S3 directory upload"""
	asyncio.run(_upload_dir_to_s3(local_path, s3_path, max_concurrency))
