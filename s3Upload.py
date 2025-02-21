import os
import boto3

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

s3_client = boto3.client("s3",
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def upload_file_to_s3(file_path: str, s3_key: str):
    s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    s3_url = f"https://{S3_BUCKET_NAME}.s3.us-east-1.amazonaws.com/{s3_key}"
    return s3_url

