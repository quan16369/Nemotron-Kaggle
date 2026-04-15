# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:04.826298Z","iopub.execute_input":"2026-03-22T22:56:04.826468Z","iopub.status.idle":"2026-03-22T22:56:04.828414Z","shell.execute_reply.started":"2026-03-22T22:56:04.826457Z","shell.execute_reply":"2026-03-22T22:56:04.828191Z"},"jupyter":{"outputs_hidden":false}}
ADAPTER_PATH = "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"
TEST_GENERATION = True

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:04.832199Z","iopub.execute_input":"2026-03-22T22:56:04.832397Z","iopub.status.idle":"2026-03-22T22:56:06.070916Z","shell.execute_reply.started":"2026-03-22T22:56:04.832389Z","shell.execute_reply":"2026-03-22T22:56:06.070624Z"},"jupyter":{"outputs_hidden":false}}
import shutil

shutil.copytree(
    ADAPTER_PATH,
    "/kaggle/working/",
    dirs_exist_ok=True,
)

shutil.copytree(
    "/kaggle/input/notebooks/huikang/nvidia-nemotron-all-linear",
    "/kaggle/working/reference",
    dirs_exist_ok=True,
)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:06.071520Z","iopub.execute_input":"2026-03-22T22:56:06.071649Z","iopub.status.idle":"2026-03-22T22:56:17.030130Z","shell.execute_reply.started":"2026-03-22T22:56:06.071640Z","shell.execute_reply":"2026-03-22T22:56:17.029851Z"},"jupyter":{"outputs_hidden":false}}
import zipfile

with zipfile.ZipFile("reference/submission.zip", "r") as zip_ref:
    zip_ref.extractall("reference")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Compare configs

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.030559Z","iopub.execute_input":"2026-03-22T22:56:17.030649Z","iopub.status.idle":"2026-03-22T22:56:17.032733Z","shell.execute_reply.started":"2026-03-22T22:56:17.030639Z","shell.execute_reply":"2026-03-22T22:56:17.032552Z"},"jupyter":{"outputs_hidden":false}}
import json

with open("reference/adapter_config.json") as f:
    reference_adapter_config = json.load(f)

print(reference_adapter_config)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.033217Z","iopub.execute_input":"2026-03-22T22:56:17.033328Z","iopub.status.idle":"2026-03-22T22:56:17.045241Z","shell.execute_reply.started":"2026-03-22T22:56:17.033321Z","shell.execute_reply":"2026-03-22T22:56:17.045073Z"},"jupyter":{"outputs_hidden":false}}
import json

with open("adapter_config.json") as f:
    trained_adapter_config = json.load(f)

print(trained_adapter_config)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.045499Z","iopub.execute_input":"2026-03-22T22:56:17.045572Z","iopub.status.idle":"2026-03-22T22:56:17.056314Z","shell.execute_reply.started":"2026-03-22T22:56:17.045565Z","shell.execute_reply":"2026-03-22T22:56:17.056141Z"},"jupyter":{"outputs_hidden":false}}
for k, reference_value in reference_adapter_config.items():
    if k in trained_adapter_config and reference_value != trained_adapter_config[k]:
        print(k)
        print(reference_value)
        print(trained_adapter_config[k])
        print()

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.056570Z","iopub.execute_input":"2026-03-22T22:56:17.056704Z","iopub.status.idle":"2026-03-22T22:56:17.067327Z","shell.execute_reply.started":"2026-03-22T22:56:17.056696Z","shell.execute_reply":"2026-03-22T22:56:17.067168Z"},"jupyter":{"outputs_hidden":false}}
for k, reference_value in reference_adapter_config.items():
    if k not in trained_adapter_config:
        print(k)
        print(reference_value)
        print()

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.067571Z","iopub.execute_input":"2026-03-22T22:56:17.067674Z","iopub.status.idle":"2026-03-22T22:56:17.078115Z","shell.execute_reply.started":"2026-03-22T22:56:17.067666Z","shell.execute_reply":"2026-03-22T22:56:17.077962Z"},"jupyter":{"outputs_hidden":false}}
for k, v in trained_adapter_config.items():
    if k not in reference_adapter_config:
        print(k)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Align configs

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.078371Z","iopub.execute_input":"2026-03-22T22:56:17.078442Z","iopub.status.idle":"2026-03-22T22:56:17.089962Z","shell.execute_reply.started":"2026-03-22T22:56:17.078434Z","shell.execute_reply":"2026-03-22T22:56:17.089801Z"},"jupyter":{"outputs_hidden":false}}
trained_adapter_config["target_modules"] = [
    "k_proj",
    "o_proj",
    "in_proj",
    "q_proj",
    "up_proj",
    "v_proj",
    "down_proj",
    "out_proj",
    "lm_head",
]

with open("adapter_config.json", "w") as f:
    f.write(json.dumps(trained_adapter_config))

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.090220Z","iopub.execute_input":"2026-03-22T22:56:17.090291Z","iopub.status.idle":"2026-03-22T22:56:17.099817Z","shell.execute_reply.started":"2026-03-22T22:56:17.090284Z","shell.execute_reply":"2026-03-22T22:56:17.099653Z"},"jupyter":{"outputs_hidden":false}}
with open("adapter_config.json") as f:
    print(f.read())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Compare adapters

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.100526Z","iopub.execute_input":"2026-03-22T22:56:17.100639Z","iopub.status.idle":"2026-03-22T22:56:17.109932Z","shell.execute_reply.started":"2026-03-22T22:56:17.100631Z","shell.execute_reply":"2026-03-22T22:56:17.109778Z"},"jupyter":{"outputs_hidden":false}}
def trained_adapter_key_rename(key_name: str) -> str:
    key_name = key_name.replace("base_model.model.model", "base_model.model.backbone")
    return key_name

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.110180Z","iopub.execute_input":"2026-03-22T22:56:17.110263Z","iopub.status.idle":"2026-03-22T22:56:17.120723Z","shell.execute_reply.started":"2026-03-22T22:56:17.110256Z","shell.execute_reply":"2026-03-22T22:56:17.120550Z"},"jupyter":{"outputs_hidden":false}}
from safetensors import safe_open

trained_adapter_keys = set()
with safe_open("adapter_model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor_slice = f.get_slice(key)
        trained_adapter_keys.add(
            (key, tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
        )

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.120988Z","iopub.execute_input":"2026-03-22T22:56:17.121146Z","iopub.status.idle":"2026-03-22T22:56:17.153123Z","shell.execute_reply.started":"2026-03-22T22:56:17.121138Z","shell.execute_reply":"2026-03-22T22:56:17.152883Z"},"jupyter":{"outputs_hidden":false}}
from safetensors import safe_open

reference_adapter_keys = set()
with safe_open(
    "reference/adapter_model.safetensors", framework="pt", device="cpu"
) as f:
    for key in f.keys():
        tensor_slice = f.get_slice(key)
        reference_adapter_keys.add(
            (key, tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
        )

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.153421Z","iopub.execute_input":"2026-03-22T22:56:17.153563Z","iopub.status.idle":"2026-03-22T22:56:17.183747Z","shell.execute_reply.started":"2026-03-22T22:56:17.153554Z","shell.execute_reply":"2026-03-22T22:56:17.183547Z"},"jupyter":{"outputs_hidden":false}}
from safetensors import safe_open
import glob

model_keys = set()
for model_safetensors in glob.glob(
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1/*.safetensors"
):
    with safe_open(model_safetensors, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor_slice = f.get_slice(key)
            model_keys.add(
                (key, tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
            )

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.184043Z","iopub.execute_input":"2026-03-22T22:56:17.184122Z","iopub.status.idle":"2026-03-22T22:56:17.185392Z","shell.execute_reply.started":"2026-03-22T22:56:17.184114Z","shell.execute_reply":"2026-03-22T22:56:17.185234Z"},"jupyter":{"outputs_hidden":false}}
# set([".".join(x.split(".")[3:]) for x in trained_adapter_keys]) - set([".".join(x.split(".")[3:]) for x in reference_adapter_keys])

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.185636Z","iopub.execute_input":"2026-03-22T22:56:17.185749Z","iopub.status.idle":"2026-03-22T22:56:17.197461Z","shell.execute_reply.started":"2026-03-22T22:56:17.185741Z","shell.execute_reply":"2026-03-22T22:56:17.197315Z"},"jupyter":{"outputs_hidden":false}}
sorted(model_keys)[-20:]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.197723Z","iopub.execute_input":"2026-03-22T22:56:17.197797Z","iopub.status.idle":"2026-03-22T22:56:17.211408Z","shell.execute_reply.started":"2026-03-22T22:56:17.197790Z","shell.execute_reply":"2026-03-22T22:56:17.211251Z"},"jupyter":{"outputs_hidden":false}}
sorted(reference_adapter_keys)[-20:]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.211672Z","iopub.execute_input":"2026-03-22T22:56:17.211756Z","iopub.status.idle":"2026-03-22T22:56:17.221430Z","shell.execute_reply.started":"2026-03-22T22:56:17.211749Z","shell.execute_reply":"2026-03-22T22:56:17.221284Z"},"jupyter":{"outputs_hidden":false}}
sorted(trained_adapter_keys)[-20:]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.221690Z","iopub.execute_input":"2026-03-22T22:56:17.221766Z","iopub.status.idle":"2026-03-22T22:56:17.230935Z","shell.execute_reply.started":"2026-03-22T22:56:17.221759Z","shell.execute_reply":"2026-03-22T22:56:17.230793Z"},"jupyter":{"outputs_hidden":false}}
len(trained_adapter_keys), len(reference_adapter_keys)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.231182Z","iopub.execute_input":"2026-03-22T22:56:17.231283Z","iopub.status.idle":"2026-03-22T22:56:17.240212Z","shell.execute_reply.started":"2026-03-22T22:56:17.231276Z","shell.execute_reply":"2026-03-22T22:56:17.240067Z"},"jupyter":{"outputs_hidden":false}}
(
    len(trained_adapter_keys - reference_adapter_keys),
    len(reference_adapter_keys - trained_adapter_keys),
)

# %% [markdown] {"execution":{"iopub.status.busy":"2026-03-22T21:22:45.522635Z","iopub.execute_input":"2026-03-22T21:22:45.522940Z","iopub.status.idle":"2026-03-22T21:22:45.525375Z","shell.execute_reply.started":"2026-03-22T21:22:45.522904Z","shell.execute_reply":"2026-03-22T21:22:45.525025Z"},"jupyter":{"outputs_hidden":false}}
# # Update adapter

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:17.240467Z","iopub.execute_input":"2026-03-22T22:56:17.240537Z","iopub.status.idle":"2026-03-22T22:56:19.996770Z","shell.execute_reply.started":"2026-03-22T22:56:17.240530Z","shell.execute_reply":"2026-03-22T22:56:19.996411Z"},"jupyter":{"outputs_hidden":false}}
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# --- Load all trained adapter tensors ---
adapter_tensors = {}
with safe_open("adapter_model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        adapter_tensors[key] = f.get_tensor(key)

# --- Collect adapter base names (without .lora_A/.lora_B.weight suffix) ---
base_names = set()
for key in adapter_tensors:
    base = re.sub(r"\.lora_[AB]\.weight$", "", key)
    base_names.add(base)

# --- Identify Mamba layers needing gate_proj+x_proj → in_proj ---
mamba_merge_layers = {}  # layer_path -> {"gate_proj": base, "x_proj": base}
for base in base_names:
    for proj in ("gate_proj", "x_proj"):
        if f".{proj}" in base:
            layer_path = base.rsplit(f".{proj}", 1)[0]
            mamba_merge_layers.setdefault(layer_path, {})[proj] = base
mamba_merge_bases = set()
for projs in mamba_merge_layers.values():
    mamba_merge_bases.update(projs.values())

# --- Build model_key_shapes for in_proj dimension lookup ---
model_key_shapes = {k: s for k, s, _ in model_keys}

# --- Build output tensors ---
tensors = {}

for base in sorted(base_names):
    lora_A = adapter_tensors[f"{base}.lora_A.weight"]
    lora_B = adapter_tensors[f"{base}.lora_B.weight"]
    renamed = trained_adapter_key_rename(base)

    # Skip empty w3 experts
    if ".experts.w3" in base and lora_A.numel() == 0:
        continue

    # # Skip lm_head (not in reference adapter)
    # if ".lm_head" in base:
    #     continue

    # Skip gate_proj/x_proj — handled in Mamba merge pass below
    if base in mamba_merge_bases:
        continue

    # --- Expert unfusing: w1 → per-expert up_proj, w2 → per-expert down_proj ---
    if ".experts.w1" in base or ".experts.w2" in base:
        # Broadcast shared dimension (one of A/B has shape[0]==1)
        # Use expand + contiguous to avoid shared memory in safetensors
        if lora_A.shape[0] == 1:
            lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
        elif lora_B.shape[0] == 1:
            lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()

        num_experts = lora_A.shape[0]
        proj_name = "up_proj" if ".w1" in base else "down_proj"

        for i in range(num_experts):
            exp_renamed = re.sub(
                r"\.experts\.w[12]",
                f".experts.{i}.{proj_name}",
                renamed,
            )
            tensors[f"{exp_renamed}.lora_A.weight"] = lora_A[i].contiguous()
            tensors[f"{exp_renamed}.lora_B.weight"] = lora_B[i].contiguous()
        continue

    # --- Direct rename for everything else ---
    tensors[f"{renamed}.lora_A.weight"] = lora_A
    tensors[f"{renamed}.lora_B.weight"] = lora_B

# --- Mamba: gate_proj + x_proj → in_proj via SVD ---
for layer_path, projs in sorted(mamba_merge_layers.items()):
    renamed_layer = trained_adapter_key_rename(layer_path)
    in_proj_base = f"{renamed_layer}.in_proj"

    model_in_proj_key = (
        renamed_layer.replace("base_model.model.", "") + ".in_proj.weight"
    )
    in_proj_dim = model_key_shapes[model_in_proj_key][0]

    gate_A = adapter_tensors[f"{projs['gate_proj']}.lora_A.weight"].float()
    gate_B = adapter_tensors[f"{projs['gate_proj']}.lora_B.weight"].float()
    x_A = adapter_tensors[f"{projs['x_proj']}.lora_A.weight"].float()
    x_B = adapter_tensors[f"{projs['x_proj']}.lora_B.weight"].float()
    rank = gate_A.shape[0]

    # Build combined rank-64 representation, then SVD to best rank-32
    A_cat = torch.cat([gate_A, x_A], dim=0)  # (64, in_dim)
    B_block = torch.zeros(in_proj_dim, 2 * rank)
    B_block[: gate_B.shape[0], :rank] = gate_B
    B_block[gate_B.shape[0] : gate_B.shape[0] + x_B.shape[0], rank:] = x_B

    Q_B, R_B = torch.linalg.qr(B_block)
    Q_A, R_A = torch.linalg.qr(A_cat.T)
    core = R_B @ R_A.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)

    k = rank
    new_B = (Q_B @ U[:, :k]) * S[:k].unsqueeze(0)
    new_A = Vh[:k, :] @ Q_A.T

    kept = S[:k].sum().item()
    total = S.sum().item()
    print(
        f"{layer_path}: SVD kept {kept:.2f}/{total:.2f} "
        f"({kept / total * 100:.1f}%) of singular value mass"
    )

    tensors[f"{in_proj_base}.lora_A.weight"] = new_A
    tensors[f"{in_proj_base}.lora_B.weight"] = new_B

print(
    f"\nConverted {len(adapter_tensors)} trained tensors → {len(tensors)} output tensors"
)
save_file(tensors, "adapter_model.safetensors")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:58.488923Z","iopub.execute_input":"2026-03-22T22:56:58.489252Z","iopub.status.idle":"2026-03-22T22:56:58.507967Z","shell.execute_reply.started":"2026-03-22T22:56:58.489240Z","shell.execute_reply":"2026-03-22T22:56:58.507712Z"},"jupyter":{"outputs_hidden":false}}
from safetensors import safe_open

updated_adapter_keys = set()
with safe_open("adapter_model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor_slice = f.get_slice(key)
        updated_adapter_keys.add(
            (key, tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
        )

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:57:07.861325Z","iopub.execute_input":"2026-03-22T22:57:07.861440Z","iopub.status.idle":"2026-03-22T22:57:07.867292Z","shell.execute_reply.started":"2026-03-22T22:57:07.861430Z","shell.execute_reply":"2026-03-22T22:57:07.867083Z"},"jupyter":{"outputs_hidden":false}}
sorted(updated_adapter_keys)[-20:]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:57:21.449718Z","iopub.execute_input":"2026-03-22T22:57:21.450014Z","iopub.status.idle":"2026-03-22T22:57:21.452306Z","shell.execute_reply.started":"2026-03-22T22:57:21.450003Z","shell.execute_reply":"2026-03-22T22:57:21.452084Z"},"jupyter":{"outputs_hidden":false}}
len(updated_adapter_keys), len(reference_adapter_keys)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:57:31.614467Z","iopub.execute_input":"2026-03-22T22:57:31.614626Z","iopub.status.idle":"2026-03-22T22:57:31.621269Z","shell.execute_reply.started":"2026-03-22T22:57:31.614615Z","shell.execute_reply":"2026-03-22T22:57:31.621046Z"},"jupyter":{"outputs_hidden":false}}
(
    len(updated_adapter_keys - reference_adapter_keys),
    len(reference_adapter_keys - updated_adapter_keys),
)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:58:07.673673Z","iopub.execute_input":"2026-03-22T22:58:07.673813Z","iopub.status.idle":"2026-03-22T22:58:07.678272Z","shell.execute_reply.started":"2026-03-22T22:58:07.673804Z","shell.execute_reply":"2026-03-22T22:58:07.678044Z"},"jupyter":{"outputs_hidden":false}}
sorted(reference_adapter_keys - updated_adapter_keys)[:20]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:58:07.804864Z","iopub.execute_input":"2026-03-22T22:58:07.804963Z","iopub.status.idle":"2026-03-22T22:58:07.808118Z","shell.execute_reply.started":"2026-03-22T22:58:07.804955Z","shell.execute_reply":"2026-03-22T22:58:07.807914Z"},"jupyter":{"outputs_hidden":false}}
sorted(updated_adapter_keys - reference_adapter_keys)[:20]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Load model

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:19.997310Z","iopub.execute_input":"2026-03-22T22:56:19.997401Z","iopub.status.idle":"2026-03-22T22:56:50.406346Z","shell.execute_reply.started":"2026-03-22T22:56:19.997391Z","shell.execute_reply":"2026-03-22T22:56:50.405798Z"},"jupyter":{"outputs_hidden":false}}
"""Metric for NVIDIA (129716)."""

import subprocess
import sys

# Set up environment
commands = [
    "uv pip uninstall torch torchvision torchaudio",
    "tar -cf - -C /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script . | tar -xf - -C /tmp",
    "chmod +x /tmp/triton/backends/nvidia/bin/ptxas",
    "chmod +x /tmp/triton/backends/nvidia/bin/ptxas-blackwell",
]
if TEST_GENERATION:
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
sys.path.insert(0, "/tmp")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.406662Z","iopub.status.idle":"2026-03-22T22:56:50.406787Z","shell.execute_reply.started":"2026-03-22T22:56:50.406734Z","shell.execute_reply":"2026-03-22T22:56:50.406741Z"},"jupyter":{"outputs_hidden":false}}
import glob
import math
import multiprocessing
import os
import re
import time
from pathlib import Path

import kagglehub
import pandas as pd
from tqdm import tqdm

# Configuration
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)
DATA_PATH = (
    ""  # Path(kagglehub.dataset_download('metric/nvidia-nemotron-rerun-data-129716'))
)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.407317Z","iopub.status.idle":"2026-03-22T22:56:50.407445Z","shell.execute_reply.started":"2026-03-22T22:56:50.407402Z","shell.execute_reply":"2026-03-22T22:56:50.407408Z"},"jupyter":{"outputs_hidden":false}}
class ParticipantVisibleError(Exception):
    pass


def cache_model(
    path: str | Path,
    exts: tuple[str, ...] = (".bin", ".pt", ".safetensors"),
    num_workers: int | None = None,
    chunk_mb: int = 256,
) -> int:
    """Pre-read model weight files into the OS page cache to speed up later loads.

    Args:
        path        : Directory containing model files, or a single file path.
        exts        : File extensions treated as model weight files.
        num_workers : Number of threads (default = min(CPU cores, 8)).
        chunk_mb    : Size of each read chunk in MB.

    Returns:
        Total bytes read (int).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath: Path) -> tuple[Path, int]:
        """Sequentially read an entire file in chunks."""
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        try:
            with open(fpath, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    total += len(data)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
        return fpath, total

    path = Path(path)
    # Collect files to read
    files: list[Path] = []
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file() and str(p).endswith(exts)]
        files.sort()
    else:
        files = [path] if path.exists() else []

    if not files:
        print(f"No model files found to cache at: {path}")
        return 0

    # Decide number of worker threads
    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    t0 = time.time()
    total_bytes = 0
    # Read files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {fpath.name}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed if elapsed > 0 else 0
    print(f"[cache_model] total read ≈ {gb:.2f} GB")
    print(f"[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s")
    return total_bytes


def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return "NOT_FOUND"

    # Search for boxed answer
    # Match all instances of \boxed{...} or unclosed \boxed{ at the end
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()


def generate_standard_submission(submission_dir: str):
    """Processes an extracted submission archive to produce a standard submission file."""
    # Locate the LoRA files within the extracted directory
    possible_extraction_dirs = {
        "/kaggle/tmp",
        "/kaggle/working",
        submission_dir,
    }
    adapter_configs = []
    for search_dir in possible_extraction_dirs:
        if os.path.exists(search_dir):
            adapter_configs.extend(
                glob.glob(
                    os.path.join(search_dir, "**/adapter_config.json"), recursive=True
                )
            )
    if not adapter_configs:
        raise ParticipantVisibleError(
            "No adapter_config.json found in submission. Found:\n\n"
            f"{submission_dir} {os.listdir(submission_dir)}\n\n"
            f"/kaggle/tmp {os.listdir('/kaggle/tmp')}\n\n"
            f"/kaggle/input/competition_evaluation {os.listdir('/kaggle/input/competition_evaluation')}"
        )

    lora_path = os.path.dirname(adapter_configs[0])

    # Load test data
    test_df = pd.read_csv(DATA_PATH / "test.csv", index_col=None)

    row_id_col = str(test_df.columns.to_list()[0])
    predictions = []
    for item in test_df.itertuples(index=False):
        predictions.append(
            {
                row_id_col: getattr(item, row_id_col),
                "prediction": lora_path,
            }
        )

    submission_df = pd.DataFrame(predictions)

    # Write the standard submission file to the current working directory
    submission_df.to_csv("submission.csv", index=False)


def generate_predictions(
    test_df: pd.DataFrame,
    lora_path: str,
    row_id_col: str,
    max_lora_rank: int,
    max_tokens: int,
    top_p: float,
    temperature: float,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    debug: bool = False,
) -> pd.DataFrame:
    """Load the model and generate predictions for the provided test data.

    Args:
        debug: If True, writes a CSV file with raw model outputs and extracted predictions.
    """
    # Cache Model
    cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TRITON_PTXAS_PATH"] = "/tmp/triton/backends/nvidia/bin/ptxas"

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Initialize vLLM Offline inference Engine
    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in test_df.itertuples(index=False):
        user_content = (
            item.prompt
            + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        )
        # Format using the tokenizer's chat template directly
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            # Fallback if chat template fails
            prompt = user_content
        prompts.append(prompt)

    # Generate predictions using continuous batching
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, lora_path),
    )

    predictions = []
    debug_records = []
    for item, output in zip(test_df.itertuples(index=False), outputs):
        raw_text = output.outputs[0].text
        extracted_answer = extract_final_answer(raw_text)

        row_id_val = getattr(item, row_id_col)

        predictions.append(
            {
                row_id_col: row_id_val,
                "prediction": extracted_answer,
            }
        )

        if debug:
            debug_records.append(
                {
                    row_id_col: row_id_val,
                    "raw_output": raw_text,
                    "extracted_prediction": extracted_answer,
                }
            )

    # Write debug CSV if requested
    if debug and debug_records:
        debug_df = pd.DataFrame(debug_records)
        debug_df.to_csv("debug_predictions.csv", index=False)
        print("Debug data saved to debug_predictions.csv")

    return pd.DataFrame(predictions)


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    max_lora_rank: int = 32,
    max_tokens: int = 3584,
    top_p: float = 1.0,
    temperature: float = 1.0,
    max_num_seqs: int = 128,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    debug: bool = False,
) -> float:
    r"""Evaluate the generated predictions against the ground truth.

    Submissions are evaluated based on their **Accuracy** in solving the provided
    tasks. The NVIDIA Nemotron-3-Nano-30B model is loaded with the participant's
    submitted LoRA adapter (which must include an `adapter_config.json`) using
    the vLLM inference engine. For each test case, the model is prompted to
    generate a response and instructed to place its final answer within a `\boxed{}`
    LaTeX command. The metric extracts the final answer from the generated text,
    prioritizing content within the boxed format while falling back to other
    heuristic patterns or the first numeric value found. A prediction is graded as
    correct if it matches the ground truth either exactly as a string or within a
    relative numerical tolerance of $10^{-2}$. The final score is the proportion of
    correctly answered questions.

    Args:
        solution: DataFrame containing the ground truth answers. Must include the
            row_id_column_name and an 'answer' column.
        submission: DataFrame containing the predicted answers. Must include the
            row_id_column_name and a 'prediction' column.
        row_id_column_name: The name of the ID column used to join solution and
            submission.
        max_lora_rank: Maximum rank for LoRA adapters.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top-p sampling parameter.
        temperature: Temperature sampling parameter.
        max_num_seqs: Maximum number of sequences to process concurrently.
        gpu_memory_utilization: Fraction of GPU memory to allocate for the vLLM execution.
        max_model_len: Maximum context length (input + output tokens).
        debug: If True, writes raw outputs and extracted predictions to a CSV file.

    Returns:
        The accuracy score (fraction of matches) as a float.
    """
    lora_path = submission["prediction"].iloc[0]

    # Load test data and filter it to only include rows present in the solution
    test_df = pd.read_csv(DATA_PATH / "test.csv", index_col=None)
    row_id_col = str(test_df.columns.to_list()[0])
    test_df = test_df[test_df[row_id_col].isin(solution[row_id_column_name])]

    submission = generate_predictions(
        test_df=test_df,
        lora_path=lora_path,
        row_id_col=row_id_column_name,
        max_lora_rank=max_lora_rank,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        debug=debug,
    )

    dataset = solution.merge(submission, on=row_id_column_name)
    num_correct = 0

    # Verify the predictions
    for item in dataset.itertuples(index=False):
        ground_truth = item.answer
        extracted_answer = item.prediction

        match = verify(str(ground_truth), str(extracted_answer))
        if match:
            num_correct += 1

    accuracy = num_correct / len(solution)
    return float(accuracy)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.407743Z","iopub.status.idle":"2026-03-22T22:56:50.407820Z","shell.execute_reply.started":"2026-03-22T22:56:50.407781Z","shell.execute_reply":"2026-03-22T22:56:50.407786Z"},"jupyter":{"outputs_hidden":false}}
# Cache Model
if TEST_GENERATION:
    cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Init vLLM

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.408145Z","iopub.status.idle":"2026-03-22T22:56:50.408219Z","shell.execute_reply.started":"2026-03-22T22:56:50.408181Z","shell.execute_reply":"2026-03-22T22:56:50.408186Z"},"jupyter":{"outputs_hidden":false}}
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_PTXAS_PATH"] = "/tmp/triton/backends/nvidia/bin/ptxas"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.408474Z","iopub.status.idle":"2026-03-22T22:56:50.408547Z","shell.execute_reply.started":"2026-03-22T22:56:50.408510Z","shell.execute_reply":"2026-03-22T22:56:50.408515Z"},"jupyter":{"outputs_hidden":false}}
# www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/evaluation
max_model_len = 8192
max_lora_rank = 32
max_tokens = 7680
top_p = 1.0
temperature = 0.0
max_num_seqs = 64
gpu_memory_utilization = 0.85
max_model_len = 8192

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.408857Z","iopub.status.idle":"2026-03-22T22:56:50.409050Z","shell.execute_reply.started":"2026-03-22T22:56:50.408894Z","shell.execute_reply":"2026-03-22T22:56:50.408898Z"},"jupyter":{"outputs_hidden":false}}
# Initialize vLLM Offline inference Engine

if TEST_GENERATION:
    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=1,
    )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Test generation

# %% [code] {"execution":{"iopub.status.busy":"2026-04-08T00:46:19.493989Z","iopub.execute_input":"2026-04-08T00:46:19.494221Z","iopub.status.idle":"2026-04-08T00:46:21.102942Z","shell.execute_reply.started":"2026-04-08T00:46:19.494196Z","shell.execute_reply":"2026-04-08T00:46:21.101877Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
df = pd.read_csv("/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv")

problem_set = {
    # bit_manipulation
    "3e953bd6", "ea67a727", "aea08eb0", "4ccf511a", "719b314e", "c71f26d6", "f3cdeb29", "87007054",
    "0a50c4a8", "8b4c71ba", "af358750", "0b26be60", "b80795b4", "cb11a232", "19f4b3d6", "52a9d5e4",
    "6a933284", "05b5ffe1", "a63f9c85", "a6f7139f", "f1c1ed58", "6689ee95", "7ed809c8", "17df2ad5",
    "9b2b3698", "6fae6379", "64b20c19", "9460a623", "6d196fe8", "e495b8e7", "7973c1b7", "5bb8c8a5",
    "93ef4c81", "562cfc29", "491e3793", "a8ea0e29", "16e73f0e", "02a66bcb", "cb5ddb11", "59c78e51",
    "2d74e088", "ef37f544", "5f29ae58", "9d09171e", "f8c6ea2c", "bca238ee", "5e0b85b0", "032fc96f",
    "6eb0d262", "7877dd7f", "2630aaf8", "d87caad1", "bb326096", "50a0c4b6", "751d48a2", "826a32cd",
    "dbabfbdd", "a169fa86", "511bb76a", "8fa7ea3a", "4f80d363", "55fbd1ad", "41a702a8", "6dbd9643",
    "28e3bbd3", "e6d3a97b", "779bb1f3", "a4ea5f61", "aaa62881", "77819d4e", "c9ed8124", "5daa92e3",
    "c30a782a", "855b5480", "5c50c07f", "0b16458a", "eb359ea5", "6b11c05f", "3d6df80f", "e2b02e13",
    "004ef7c7", "0f2dec86", "341a468c", "0dd5caf4", "5fead1a1", "7c538bb0", "7c980689", "8a382428",
    "dd900b68", "07434d56", "ea6859d7", "5d489e95", "bd214050", "da7288cb", "00fdc0be", "e3c672be",
    "965ea054", "7ff6d6c3", "eb12e80d", "a316aadc", "52d72862", "7b107eec", "a0847120", "66f9fe57",
    "663fd5e9", "06248efa", "ad7edc85", "b634898d", "847d8897", "f557284b", "f459b7b4", "c39705cd",
    "cd17f040", "736ae137", "abb5d597", "4b628577", "6f91481e", "72ffb208", "0d7aacfc", "bef8b5ec",
    "f3fc7209", "f5d74f50", "8bf273f5", "9972f3f1", "c4a6d52b", "cfb1b2c1", "8a04fbe3", "74ada555",
    "107badb0", "baeb4a11", "5356c59d", "f7aeb894", "f43f6142", "8c06fb6f", "e5ae7b47", "fd7920d3",
    "b2bdef43", "563d22ab", "b398201b", "3c9b8e0e", "37c94738", "c0e04922", "1f69a85f", "dde0558e",
    "c7be75f8", "d6e6f8b5", "8edddcc4", "b3a9b053", "1496dfeb", "4b52b575", "d2b4560f", "4ee88170",
    "2774d7a4", "e893d523", "9077fa09", "25705f72", "e24c8980", "a1c20432", "abba85ac", "4c23d822",
    "21312779", "4059e1a5", "aa2e26f1", "54d2b3b0", "36e96ab0", "d408af38", "12897b38", "4b4f1779",
    "698e737a", "fbbe43a3", "68213a4c", "54ca9d57", "26c83e22", "7e577681", "42e10376", "d5071155",
    "74e525f0", "04183bf9", "83a70d48", "22e28f23", "a37158d6", "37a00064", "1fd65d86", "33910360",
    "24b60af3", "c29cdaed", "7a79ac09", "e2f184b4", "8abcff0f", "ecf2a23b", "1f6c2fd9", "e79c8c5a",
    "33c48893", "c58451f9", "c90fa3a6", "0dcacfab", "fa900ac8", "5ec86a30", "dc593799", "8e7952cc",
    "54b895c1", "8982c48b", "0536483a", "4ada9150", "400c9250", "4723911c", "ec192379", "d6f4e854",
    "c0788894", "02324ba1", "c33ad31a", "691608b9", "3a1f8cf0", "836b85e8", "a33019c4", "6bef21ca",
    "88ca8c4a", "8d2290b0", "8171f73c", "b74a49c4", "30ba0cf4", "db56bba9", "b30f8610", "1dba764b",
    "0a195a74", "3b92cc78", "100e280a", "e0b993bd", "9d4f1368", "6b01e0cf", "14917af8", "c2d8ef67",
    "ccfd3fe3", "7987a44b", "799c6822", "1ed83cae", "99bda3b2", "009a74b6", "a5749dc0", "8c743940",
    "d8e070fa", "cb31384b", "978c688b", "6b8246a0", "4166067c", "d40b3aa3", "8cf926ca", "b2fb65cf",
    "410a5cbd", "16122fba", "f64eb796", "ee4f1423", "89906f16", "f4bb4d9c", "18544cb0", "07ceed47",
    "fc4a85ad", "632444af", "7f73016f", "7b4ac17a", "bc4d68f5", "073942b5", "8b12ff37", "f2f257ed",
    "fd13b7fd", "d25d3b5f", "679af1e1", "ba61b815", "d5220697", "76fbfa25", "7b412ac0", "5d0db0d2",
    "5f66eb60", "18c797f1", "f370fab8", "b595b6e8", "4145ec70", "236034b4", "28ffe70d", "d1c3ed77",
    "fecfb467", "840050ec", "dd9ee40c", "37664f79", "a7cbf6fd", "77f17a8f", "573eaca1", "e7f1ac8e",
    "5756de38", "512dd86d", "0b404f15", "a30a5e37", "2f7f58de", "c8730ea5", "3564baf1", "8785d0c3",
    "851a22cb", "de91def4", "b9734394", "5cc4cf10", "a1d2bb0c", "98e1a8b4", "da4c92a3", "8e5d6fe6",
    "98c4eb34", "3ec3f1b4", "fa67da07", "55bc6738", "59bee375", "bdb93228", "a4770481", "9f2e45e7",
    "739451ab", "0f8fe647", "b558c74d", "638224db", "4ba4a7ec", "a6082142", "b76fd053", "a4f4fb0c",
    "2f51362d", "117b4e53", "eed2f3ae", "6a333ed6", "0c671b98", "d9bdf754", "28bde184", "5bd26372",
    "bdd63604", "d0ea0b5f", "a88c4f08", "2441e2b4", "8d20d0ae", "6818b555", "bcf0d51d", "710cf035",
    "b287ee74", "021ed764", "a0ffbfd2", "186ade13", "23410e94", "7b6f55dc", "8a057351", "b5261a95",
    "362e9e42", "9e2b594d", "413086bc", "6890870d", "f8fc43d2", "302dc36e", "e5bb9b26", "2460c01a",
    "412186a6", "7283eb09", "57b03b2b", "cf9bebe9", "10d29630", "b16455a2", "2b50eb88", "d6187157",
    "52a052a0", "f84c009a", "e912794a", "3302f383", "5cedf608", "106aeb25", "3400e0d5", "b9f42afb",
    "dae576ce", "1c671acc", "3dc918eb", "bb429016", "3f4f7b9b", "97f45beb", "455b6b61", "574c9e17",
    "02021540", "78db4aa3", "bc520eb0", "94c25c56", "4b925449", "29e458d6", "2c6cb766", "2dcb7ec6",
    "14a72508", "cb7c2230", "9c5c6401", "dc99f250", "3cfc5f9c", "3296c5b5", "3abe72f4", "cf1cca51",
    "405262aa", "a91414ee", "bcdf9198", "c6a1ba04", "3b7dab4c", "294557bb", "3e000b40", "0c1a09ce",
    # cipher
    "c30e03b9", "556ee87d", "47366481", "d8590d73", "ec466776", "85e7ec53", "550bb9ef", "100c650a",
    "4a940571", "a6620888", "4e1a4d0e", "e0819bcd", "c31c1019", "03deaa1c", "0cb16796", "9ca96a88",
    "f32745f1", "ed7780b9", "55c7342b", "38023e84", "26fe52a0", "1419a453", "b9222db1", "f42d5e08",
    "d1611217", "ddaa5f84", "4925c815", "b2748dc4", "b1ec44a1", "482d0ff2", "aa6f0d54", "6258d182",
    "f83118d7", "39470bbb", "dfad71ea", "597dc93a", "55d86f7e", "a6e3134e", "2792186b", "47d45654",
    "02102281", "50a6d9d8", "8d10c393", "f63c1191", "515ebada", "8e905ad6", "e903bc06", "e1536141",
    "a3c3be4d", "04c1e45b", "5f8be7e2", "61abed03", "6079c08f", "7f7340d6", "248cb850", "50c066b0",
    "5f0da536", "867b8d6c", "b19f4815", "01c5cb71", "5e2d5262", "a3efb940", "62885797", "183e90da",
    "b82f6bc4", "31610ce5", "aa6641fb", "9ca2d781", "f6afb6b5", "22d94427", "32c6db19", "08bc3a02",
    "c3b430b1", "16642d10", "f05e77f3", "df81fd9b", "3661796c", "c3d00ce8", "62c13a9b", "d9e00612",
    # cryptarithm_deduce
    "1d10ccaf", "a4ee9fa6", "08f6216d", "34b4cf96", "97d6db7a", "64d775e5", "d6c03e21", "1c7a0091",
    "b13d511a", "a08fbb68", "8e56a42e", "065abaf6", "d72d8c1c", "85c5b2a2", "e977d0b7", "0dcfd566",
    "dea42835", "02a04b59", "eac0e775", "6897f05e", "d0b1e41a", "b1b10e83", "21b90d9f", "39a1f5e9",
    "bc83b0a1", "0133bcec", "f8bbabab", "ed61a9d6", "02b8d816", "7d8f22b1", "23b0eb54",
    # cryptarithm_guess
    "deed3497", "c7844441", "25ee72c3", "9dfe5ac9", "e38c423b", "0da1841f", "258b796b", "2e9973b7",
    "0fcf912a", "07b440f0", "55f4fa64",
    # equation_numeric_deduce
    "836d6c4a", "9e5d97de", "ed561f79", "b26d8818", "852d16cb", "1019ee55", "26b716ed", "93481650",
    "4a8eafe3", "25f2f2cd", "8bc6a26c", "1e248995", "d8735cc4", "118889da", "93f3ae6b", "3a8a4ebc",
    "e9b69bf8", "fe393711", "92471ca4", "1d6c587d", "ff8409cd", "d8b3310a", "4c19c1cb", "324f918b",
    "912d2b79", "45582be0", "936b3ae5", "31eb8247", "c7420a23", "74b9b0ec", "a04ecffd", "79f29eb5",
    "79acd75c", "17c98340", "339149c9", "aa7f06f4", "6e60b0c5", "e518256e", "4011d4e5", "7a962e17",
    "f6db706e", "937c06c5", "87a902eb", "9d68ef62", "df309eeb", "d64a6578", "84030b0b", "56343b77",
    "c0f32a1e", "97dee6aa", "c170f7d4", "4bc6e6b5", "9b458fbc", "9ae3b78e", "42553fde", "fee66e16",
    "a19a75ba", "5787c3d0", "35672155", "8dea05d7", "34c563c5", "f333b67f", "2fb20366", "49bc1b7b",
    "07cbed38", "e7cf0394", "41babefd", "5fe8d710", "4c57a53f", "3e42c4f9", "4d1ae327", "6ab04968",
    "9a5b6b28", "5e67b1a1", "9dd8adaa", "1ee54412", "118f8c86", "c11777c0", "bc42e664", "3efc46c1",
    "5d44a0b2", "4c6cf9d9", "10d7bd1b", "7ef4d5d6", "f0a2d457", "67f1bc8a", "eb68289b", "1b6366af",
    "cd94ec17", "26168be6", "c541eb82", "7e2e8a95", "6b769a9e", "3687b4bb", "e9a84dd6", "8d652f91",
    "44e69b7e", "6b393b81", "cf517124", "13ad7bce", "1861c08f", "c748b04f", "d78ce112", "2beb5851",
    "49578b02", "04322d27", "debff779", "891942ba", "51dafb5b", "3ee36e0a", "8011cb24", "9f2fae58",
    "3b2e0cc3", "d46df3b0", "f28681ad", "912c6ea5", "d67f28ad", "67de8e10", "9a9f6025", "dc7a5e7b",
    # equation_numeric_guess
    "8df3daad", "662fd21c", "9fa9ecdc", "9c91b226", "8ae8e12a", "be877da5", "69fe4b0d", "ef1b13ac",
    "b7ad0671", "c763054a", "6e6401d7", "5d89a09c", "260f20c1", "e7b87b82", "e9e6b620", "83f2724b",
    "32a31236", "4e840a1a", "7c0c5227", "dc178d1c", "66a0856f",
    # gravity
    "cd8b5299", "649c2246", "b8b99140", "49a13c0b", "50ea5164", "6735003f", "6cdcf6d9", "a34bd133",
    "62380ed2", "af78b80e", "e100c796", "5957f4e5", "0d2f174c", "5c69e4c6", "00ed1836", "d61f4e24",
    "0a039aa6", "16b45644", "a9a49dfe", "fb0df6ae", "109a6016", "81385151", "f73349d4", "8c36a40e",
    "177d60f2", "404f84df", "c14582f7", "a8ce8537", "1520fb1d", "6cec77c9", "f731876f", "cb57aeca",
    "4cd4cbb7", "112bd01f", "360b6f4e", "7ee39525", "d71bb1fc", "a3c3d077", "c8594a19", "27d3c9e1",
    "d61269a7", "1dde1e65", "e560f574", "89b232e6", "fc9b063b", "47e63612", "de78c53f", "3672cece",
    "d87a13ce", "cbf13263", "5fe169fc", "c83ef461", "0e897e03", "10b19ba1", "ccddd1be", "5116e8ec",
    "c5bea6cd", "47a42bf5", "fd38d6f0", "6d1e50f7", "f63f77f8", "a043e4da", "c7e1ae7d", "14fd5550",
    "e406f8e6", "8e6bc995", "2cbb9290", "6d93a375", "780d3ceb", "60db6fdd", "89f164d6", "2bf73df4",
    "d080f2a8", "e44b5be4", "8bbb6458", "645fe504", "e62041e4", "1c37e8fe", "3dbbbe28", "ae6c8cf6",
    # numeral
    "bb9c9d6f", "f71d0599", "a23251e6", "b2ee2241", "0e3d5b3d", "79eace73", "7826240a", "fe44dfa9",
    "6489821f", "85c6761c", "b8496b92", "e97073be", "8c281ee9", "658e7335", "63ab2c50", "5c919729",
    "1f30afb9", "5b05cc20", "a327e1d2", "8c444d7c", "0dd90079", "e4240858", "05992f55", "ac465f14",
    "6d4e39cb", "ad121190", "aa24c0a9", "6708d238", "c539bcca", "18570e64", "a9cc9eee", "794361a4",
    "be8d4d84", "91697374", "eb83a44c", "838981e4", "94ad1872", "93f6583a", "f9d9b3fd", "85edf718",
    "d9516e43", "086cce4f", "853dcb9c", "861f02de", "e0545ff2", "1af4f53a", "d8053ec2", "6d0aa254",
    "0122d53a", "904295f6", "d777480b", "2ed54b3f", "1e2de753", "883e87f4", "a9a5e10f", "11697763",
    "1fefd874", "20d62620", "8dec3179", "5f86e4da", "d8e57ee5", "faa24059", "b993173b", "40509ae1",
    "446fa429", "fc800721", "9f775c18", "2345be90", "d46a570b", "c92650e9", "37cd3797", "312e2579",
    "eeb33e03", "57fa8b99", "3aeee6cd", "8e10f699", "c6fe6285", "360ea139", "78f78c8e",
    # unit_conversion
    "fbda33d1", "dbc542da", "e9461e93", "b3ab21d2", "e063d66b", "51994791", "cb7475e5", "4ce156ee",
    "874fb96a", "12009c93", "e3ff3a78", "5e75615e", "f2d67c62", "522c0009", "fcb9068b", "10b1ffc8",
    "533ed10e", "0fd8b43e", "9e919776", "31092823", "fe6fb7b4", "c3c3e734", "f9215fe9", "b36aa89c",
    "082c1a06", "7aa5f509", "31810496", "84e91011", "45a0ccde", "97d7a79a", "c0c0699a", "420ea7ee",
    "c102a112", "29b79c2b", "ba469498", "34514d28", "23ec50f8", "c78a5deb", "5add849f", "50cd5357",
    "e8c4a0bf", "05fcec41", "ddafe911", "21e46a71", "1e99e3e2", "ed5b5c60", "014c7478", "5776269d",
    "d4bea0cd", "90ae7158", "99e18140", "c7f785b6", "36391105", "5666e340", "d035c02e", "68368ba9",
    "aa6f0af1", "a4f9e327", "137bb1f4", "ea8977e0", "9b10b67b", "8860525a", "4176d785", "7c1685b2",
    "161ca0ab", "b12377a9", "2425d9ff", "d97da22d", "2cb5b118", "835d7556", "a966930a", "414c2259",
    "b61e875a", "7418fc5e", "58f02dad", "1bf3f0f1", "a575e7b7", "37519088", "3e26fa1b", "8affee55",
}

# change the variable name to run the shorter version
# backup_problem_set
# problem_set
backup_problem_set = {
    # bit_manipulation
    "836b85e8", "b20b39bf", "af358750", "3f9bd1e7", "0528d502", "9992bbd0", "812131f1", "84e3f9f7",  # min_lp: -20.0965 .. -18.6948
    "114a41e3", "585f2ff6", "ea6859d7", "2fa48efe", "5f76ba09", "6a186446", "5d0db0d2", "75898981",  # min_lp: -18.6263 .. -17.7624
    "c7a37cda", "989dde0a", "d623e937", "100e280a", "f6a95641", "302dc36e", "3c9b8e0e", "b6e4a36d",  # min_lp: -17.6909 .. -17.3764
    "bd214050", "d50683b4", "f7346f0c", "5dd3345c", "dfc4839c", "0e7a6920", "31a4c9ef", "f8fc43d2",  # min_lp: -17.3617 .. -16.6266
    "19f4b3d6", "093de4ea", "9bd65991", "c6fa3e3f", "b8e0c853", "cb3317fe", "4ada9150", "c90fa3a6",  # min_lp: -16.5006 .. -15.4493
    # cipher
    "cf821623", "31c72d27", "73cd9008", "3975d230", "4db54201", "c1ffb3ac", "2a6c343e", "0e46fd1d",  # min_lp: -19.4908 .. -18.5764
    "49b244e3", "7b8e4432", "3019f44e", "4f8f23d6", "1fcbbb93", "987a223b", "84d10c70", "0dad87bf",  # min_lp: -18.5705 .. -18.2567
    # cryptarithm_deduce
    "1c7a0091", "ed61a9d6", "02b8d816", "d6c03e21", "2f6531cb", "a4ee9fa6", "02a04b59", "81b6d789",  # min_lp: -16.8100 .. -8.6917
    "bc83b0a1", "b1b10e83", "dea42835", "2c017f70", "b13d511a", "6897f05e", "64d775e5", "24e1f1d5",  # min_lp: -8.1260 .. -3.0486
    # cryptarithm_guess
    "c7844441", "0fcf912a", "07b440f0", "25ee72c3", "9dfe5ac9", "deed3497", "258b796b", "0da1841f",  # min_lp: -12.5001 .. -0.1081
    "e38c423b", "2e9973b7", "55f4fa64",  # min_lp: -0.0486 .. -0.0202
    # equation_numeric_deduce
    "91488dc9", "7e2e8a95", "35a89469", "a04ecffd", "27cec7a9", "d6a2e332", "04322d27", "8bc6a26c",  # min_lp: -21.2890 .. -17.3401
    "5c743e8a", "30763ac0", "91b42a45", "fecad63c", "c857a727", "b69391b8", "45df54db", "e5956ffa",  # min_lp: -17.0230 .. -15.1098
    # equation_numeric_guess
    "662fd21c", "e9e6b620", "8ae8e12a", "dc178d1c", "9c91b226", "c763054a", "66a0856f", "ef1b13ac",  # min_lp: -19.6251 .. -2.2399
    "8df3daad", "e7b87b82", "5d89a09c", "be877da5", "7c0c5227", "69fe4b0d", "b7ad0671", "4e840a1a",  # min_lp: -2.1279 .. -0.1560
    # gravity
    "d0dd2df7", "74a50b2c", "f9f20a7a", "ef3c7703", "85bc954c", "8de7d8bc", "22bb13b8", "87eb7ce0",  # min_lp: -18.8125 .. -16.7516
    "8a24aef9", "3c53c8af", "a6f1b553", "7b47f88d", "c0f6e1b8", "44dbe7d3", "827a6b1b", "6f3a0625",  # min_lp: -16.6875 .. -15.1253
    # numeral
    "3b4ebafd", "ad6ff612", "5a6ed2bf", "0adca57b", "d79d0cfd", "e6b04620", "f47276a4", "1e2de753",  # min_lp: -18.7504 .. -12.9401
    "0122d53a", "685bb0b1", "797ae611", "588a4ce8", "f19ffbf1", "8c281ee9", "972ef18a", "5092f0e0",  # min_lp: -12.4973 .. -11.1420
    # unit_conversion
    "082c1a06", "3dcaf042", "87342969", "8e1cff16", "d566ff0e", "598af975", "51a22965", "d3d82844",  # min_lp: -22.2500 .. -20.1875
    "e6157d05", "cd1280b0", "bbb61c3a", "740e0460", "be2416ec", "63ec749f", "26e6819a", "99948ad9",  # min_lp: -19.7500 .. -17.4077
}
# df = df[df.id.isin(problem_set)].copy()

# %% [code] {"jupyter":{"outputs_hidden":false}}
problem_texts = list(df["prompt"])

if TEST_GENERATION:
    tokenizer = llm.get_tokenizer()
    prompts = []
    for problem_text in problem_texts:
        # Format using the tokenizer's chat template directly
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.409717Z","iopub.status.idle":"2026-03-22T22:56:50.409790Z","shell.execute_reply.started":"2026-03-22T22:56:50.409753Z","shell.execute_reply":"2026-03-22T22:56:50.409758Z"},"jupyter":{"outputs_hidden":false}}
# Generate predictions using continuous batching (without adapter, for baseline comparison)
if TEST_GENERATION:
    # outputs = llm.generate(
    #     prompts,
    #     sampling_params=sampling_params,
    # )
    pass

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.410006Z","iopub.status.idle":"2026-03-22T22:56:50.410073Z","shell.execute_reply.started":"2026-03-22T22:56:50.410038Z","shell.execute_reply":"2026-03-22T22:56:50.410043Z"},"jupyter":{"outputs_hidden":false}}
print(os.listdir("."))

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.410308Z","iopub.status.idle":"2026-03-22T22:56:50.410377Z","shell.execute_reply.started":"2026-03-22T22:56:50.410342Z","shell.execute_reply":"2026-03-22T22:56:50.410347Z"},"jupyter":{"outputs_hidden":false}}
possible_extraction_dirs = {
    "/kaggle/tmp",
    "/kaggle/working",
    # submission_dir,
}
adapter_configs = []
for search_dir in possible_extraction_dirs:
    if os.path.exists(search_dir):
        adapter_configs.extend(
            glob.glob(
                os.path.join(search_dir, "**/adapter_config.json"), recursive=True
            )
        )

lora_path = os.path.dirname(adapter_configs[0])

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.410584Z","iopub.status.idle":"2026-03-22T22:56:50.410653Z","shell.execute_reply.started":"2026-03-22T22:56:50.410617Z","shell.execute_reply":"2026-03-22T22:56:50.410622Z"},"jupyter":{"outputs_hidden":false}}
print(os.listdir("/kaggle/working"))

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.410934Z","iopub.status.idle":"2026-03-22T22:56:50.411005Z","shell.execute_reply.started":"2026-03-22T22:56:50.410969Z","shell.execute_reply":"2026-03-22T22:56:50.410973Z"},"jupyter":{"outputs_hidden":false}}
with open("adapter_config.json") as f:
    print(f.read())

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.411249Z","iopub.status.idle":"2026-03-22T22:56:50.411315Z","shell.execute_reply.started":"2026-03-22T22:56:50.411281Z","shell.execute_reply":"2026-03-22T22:56:50.411285Z"},"jupyter":{"outputs_hidden":false}}
print(lora_path)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-22T22:56:50.411535Z","iopub.status.idle":"2026-03-22T22:56:50.411601Z","shell.execute_reply.started":"2026-03-22T22:56:50.411566Z","shell.execute_reply":"2026-03-22T22:56:50.411571Z"},"jupyter":{"outputs_hidden":false}}
# Generate predictions using continuous batching (with adapter)
if TEST_GENERATION:
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, lora_path),
    )
    df["output"] = [output.outputs[0].text for output in outputs]
    df["predicted"] = df["output"].apply(extract_final_answer)
    df["correct"] = df.apply(lambda row: verify(str(row["answer"]), str(row["predicted"])), axis=1)
    df["minlogprob"] = [
        min(lp.logprob for token_dict in output.outputs[0].logprobs for lp in token_dict.values())
        if output.outputs[0].logprobs else None
        for output in outputs
    ]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Produce submission

# %% [code] {"jupyter":{"outputs_hidden":false}}
import zipfile as _zf

print(os.listdir("."))
shutil.rmtree("reference", ignore_errors=True)
with _zf.ZipFile("submission.zip", "w", _zf.ZIP_DEFLATED) as zf:
    for file in os.listdir("."):
        if file.startswith(".") or file == "submission.zip" or not os.path.isfile(file):
            continue
        zf.write(file)
        os.remove(file)

# %% [code] {"jupyter":{"outputs_hidden":false}}
if TEST_GENERATION:
    df.to_csv("predictions.csv", index=False)
    if len(df[~df["correct"]]) > 0:
        df[~df["correct"]].to_csv("predictions_wrong.csv", index=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-22T22:56:50.412155Z","iopub.status.idle":"2026-03-22T22:56:50.412227Z","shell.execute_reply.started":"2026-03-22T22:56:50.412190Z","shell.execute_reply":"2026-03-22T22:56:50.412195Z"}}
print(os.listdir("."))