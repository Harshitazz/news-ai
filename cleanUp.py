from s3Upload import s3_client
import os
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio


S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

def cleanup_old_files():
    """Deletes files older than 24 hours"""
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)

    for obj in response.get("Contents", []):
        if (datetime.datetime.now(datetime.timezone.utc) - obj["LastModified"]).total_seconds() > 86400:
            s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=obj["Key"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_loop())  # ✅ Run cleanup loop in background
    yield
    task.cancel()  # ✅ Cancel cleanup task on shutdown

async def cleanup_loop():
    while True:
        cleanup_old_files()
        await asyncio.sleep(86400)  # Run every 24 hours
