"""Upload LoRA adapter from Tinker checkpoint to Kaggle.

Usage:
    uv run modal run --detach upload_adapter.py

Prerequisites:
    1. KAGGLE_API_TOKEN and TINKER_API_KEY in env.json
    2. Trained adapter checkpoint in Tinker

The script:
    1. Downloads the adapter archive from Tinker to a Modal volume
    2. Creates the Kaggle model instance if it doesn't exist
    3. Uploads the adapter as a new version
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import modal
from requests.exceptions import HTTPError

kaggle_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "kaggle>=1.6.0",
    "tinker>=0.5.1",
    "pydantic>=2.0,<3.0",
)

adapter_vol = modal.Volume.from_name("adapter-weights", create_if_missing=True)

app = modal.App("upload-adapter-to-kaggle")

ADAPTER_DIR = "/adapter/weights"
DEFAULT_INSTANCE = "huikang/nemotron-adapter/Transformers/default"


def _print_files(directory: str) -> list[str]:
    """Print files in a directory with sizes."""
    files = os.listdir(directory)
    for fname in sorted(files):
        size = os.path.getsize(os.path.join(directory, fname))
        print(
            f"  {fname}: {size / 1e9:.2f} GB"
            if size > 1e9
            else f"  {fname}: {size / 1e6:.2f} MB"
        )
    return files


@app.function(
    image=kaggle_image,
    volumes={"/adapter": adapter_vol},
    timeout=3 * 60 * 60,
)
def download_adapter(tinker_model: str, tinker_env: dict[str, str]):
    """Download adapter weights from Tinker to Modal volume."""
    import re
    import tarfile
    import urllib.request

    os.environ.update(tinker_env)
    import tinker

    print(f"Downloading adapter from {tinker_model}...")
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Download archive from Tinker
    model_id = re.search(r"tinker://([a-f0-9-]+)", tinker_model)
    model_id_str = model_id.group(1) if model_id else "unknown"
    print(f"Model ID: {model_id_str}")

    sc = tinker.ServiceClient()
    url = (
        sc.create_rest_client()
        .get_checkpoint_archive_url_from_tinker_path(tinker_model)
        .result()
        .url
    )
    print("Archive URL obtained, downloading...")

    tar_path = f"/tmp/adapter_{model_id_str}.tar"
    urllib.request.urlretrieve(url, tar_path)
    print(f"Downloaded archive: {os.path.getsize(tar_path) / 1e6:.1f} MB")

    # Extract
    with tarfile.open(tar_path) as tar:
        tar.extractall(ADAPTER_DIR)
    os.remove(tar_path)

    # Find adapter files (may be in a subdirectory)
    config_path = None
    weights_path = None
    for root, _dirs, files in os.walk(ADAPTER_DIR):
        for f in files:
            if f == "adapter_config.json":
                config_path = os.path.join(root, f)
            elif f == "adapter_model.safetensors":
                weights_path = os.path.join(root, f)

    if config_path and weights_path:
        # Move to top-level if in subdirectory
        for path in [config_path, weights_path]:
            dest = os.path.join(ADAPTER_DIR, os.path.basename(path))
            if path != dest:
                shutil.move(path, dest)

    print("Extracted adapter files:")
    _print_files(ADAPTER_DIR)
    adapter_vol.commit()


@app.function(
    image=kaggle_image,
    volumes={"/adapter": adapter_vol},
    timeout=3 * 60 * 60,
)
def upload_to_kaggle(kaggle_api_token: str):
    """Upload adapter from Modal volume to Kaggle."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
        f.write(kaggle_api_token)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated")

    if not os.path.exists(ADAPTER_DIR):
        raise ValueError(f"Adapter directory not found: {ADAPTER_DIR}")

    files = _print_files(ADAPTER_DIR)
    print(f"Found {len(files)} files")

    parts = DEFAULT_INSTANCE.split("/")
    owner, model_slug, framework, instance_slug = (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
    )

    def instance_exists() -> bool:
        try:
            api.model_instance_get(DEFAULT_INSTANCE)
            return True
        except HTTPError:
            return False

    if not instance_exists():
        print(f"\nInstance {DEFAULT_INSTANCE} does not exist, creating...")

        upload_dir = tempfile.mkdtemp()
        for fname in files:
            shutil.copy(os.path.join(ADAPTER_DIR, fname), upload_dir)

        metadata = {
            "ownerSlug": owner,
            "modelSlug": model_slug,
            "instanceSlug": instance_slug,
            "framework": framework,
            "licenseName": "Apache 2.0",
            "overview": "Nemotron-3-Nano-30B LoRA adapter",
        }
        metadata_path = os.path.join(upload_dir, "model-instance-metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        print(f"Created metadata: {metadata}")

        api.model_instance_create(upload_dir, dir_mode="skip")
        print("Instance created")
    else:
        print(f"\nInstance {DEFAULT_INSTANCE} already exists")

    print(f"\nUploading new version to {DEFAULT_INSTANCE}...")
    api.model_instance_version_create(DEFAULT_INSTANCE, ADAPTER_DIR, dir_mode="skip")
    print("Version created")

    print("\nUpload complete!")
    return "Success"


def _find_latest_adapter() -> str:
    """Find the latest sampler_weights/final checkpoint via tinker CLI."""
    result = subprocess.run(
        ["uv", "run", "tinker", "-f", "json", "checkpoint", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    candidates = [
        c for c in data["checkpoints"] if c["checkpoint_id"] == "sampler_weights/final"
    ]
    if not candidates:
        raise ValueError("No sampler_weights/final checkpoint found")
    latest = max(candidates, key=lambda c: c["time"])
    return latest["tinker_path"]


@app.local_entrypoint()
def main():
    """Download the latest adapter from Tinker and upload to Kaggle."""
    with open("env.json") as f:
        env = json.load(f)

    tinker_model = _find_latest_adapter()
    print(f"Tinker model: {tinker_model}")
    print(f"Kaggle instance: {DEFAULT_INSTANCE}")

    kaggle_api_token = env["KAGGLE_API_TOKEN"]
    tinker_env = {"TINKER_API_KEY": env["TINKER_API_KEY"]}

    download_adapter.remote(tinker_model, tinker_env)

    result = upload_to_kaggle.remote(kaggle_api_token)
    print(result)
