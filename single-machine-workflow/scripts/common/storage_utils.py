"""MinIO S3 upload/download helpers."""

import os

import boto3
from botocore.client import Config


def get_s3_client():
    """Create a boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(client, bucket_name):
    """Create an S3 bucket if it doesn't exist."""
    try:
        client.head_bucket(Bucket=bucket_name)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=bucket_name)
        print(f"Created bucket: {bucket_name}")


def upload_directory(local_dir, bucket, s3_prefix, client=None):
    """Upload a local directory recursively to S3."""
    if client is None:
        client = get_s3_client()
    ensure_bucket(client, bucket)
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path) if s3_prefix else relative_path
            client.upload_file(local_path, bucket, s3_key)
            print(f"Uploaded {local_path} -> s3://{bucket}/{s3_key}")


def download_directory(bucket, s3_prefix, local_dir, client=None):
    """Download files from S3 prefix to a local directory."""
    if client is None:
        client = get_s3_client()
    os.makedirs(local_dir, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = os.path.relpath(key, s3_prefix)
            local_path = os.path.join(local_dir, relative)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            client.download_file(bucket, key, local_path)
            print(f"Downloaded s3://{bucket}/{key} -> {local_path}")
