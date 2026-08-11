# Ray Megatron Kubernetes Setup

This repository contains a small Kubernetes/Ray setup for pretraining a Qwen-style model with NVIDIA NeMo/Megatron-Bridge. It includes cluster setup scripts, KubeRay manifests, Ray training jobs, and local Hugging Face-style Qwen configuration/tokenizer files.

## Repository Layout

```text
.
├── setup.sh
├── clenup.sh
├── kai-queue.yaml
└── training/
    ├── launch_single_node.sh
    ├── pretrain.py
    ├── pretrain_qwen.py
    ├── qwen_pretrain.py
    ├── ray-cluster.yaml
    ├── ray-job-submission.yaml
    └── qwen/
        ├── config.json
        ├── tokenizer.json
        └── tokenizer_config.json
```

## What This Project Does

The project is intended to run Qwen pretraining workloads on a GPU-enabled Kubernetes environment using:

- Minikube for local Kubernetes.
- NVIDIA GPU Operator and Network Operator for GPU support.
- NVIDIA KAI Scheduler for GPU queue scheduling.
- KubeRay for running Ray clusters and Ray jobs on Kubernetes.
- NVIDIA NeMo container image `nvcr.io/nvidia/nemo:26.04`.
- Megatron-Bridge for converting Hugging Face model configuration into a Megatron-compatible training setup.

The training configuration targets an 8-GPU worker setup, with H100/NCCL-related environment tuning in the Ray/Kubernetes files.

## Top-Level Files

### `setup.sh`

Sets up the local Kubernetes AI stack:

1. Installs Minikube if it is missing.
2. Starts Minikube with Docker, GPU passthrough, persistent storage, and default storage addons.
3. Installs Helm if it is missing.
4. Installs the NVIDIA GPU Operator.
5. Installs the NVIDIA Network Operator.
6. Installs NVIDIA KAI Scheduler.
7. Installs the KubeRay operator with KAI scheduler integration.
8. Installs `kube-prometheus-stack` for monitoring.
9. Installs a local Weights & Biases server.

Note: the W&B Helm command currently has a likely shell syntax issue:

```bash
--set license=$LICENCE
--wait
```

The `--set license=$LICENCE` line probably needs a trailing `\` before `--wait`.

### `clenup.sh`

Deletes Ray resources from the Kubernetes cluster:

```bash
kubectl delete rayjob ubuntu-demo-kuberay-job-rayjob
kubectl delete raycluster ubuntu-demo-kuberay-cluster-raycluster
```

Note: the filename is spelled `clenup.sh`. The resource names in this script do not match the `qwen-pretrain-*` resource names used by the current manifests.

### `kai-queue.yaml`

Defines a KAI Scheduler queue named `qwen-pretrain-queue`.

The queue has unlimited quota and limits for CPU, GPU, and memory:

- `quota: -1`
- `limit: -1`
- `overQuotaWeight: 1`

## Training Files

### `training/pretrain.py`

Runs Qwen pretraining directly through Megatron-Bridge.

Key behavior:

- Uses `Qwen/Qwen3.5-4B` as the Hugging Face model ID.
- Converts the Hugging Face config into a Megatron provider with `AutoBridge`.
- Does not load pretrained weights.
- Sets sequence length and max position embeddings to `4096`.
- Uses tensor parallel size `1`.
- Uses pipeline parallel size `1`.
- Uses micro batch size `1` and global batch size `8`.
- Runs for `10000` training iterations.
- Saves checkpoints to `./checkpoints_qwen`.
- Writes TensorBoard logs to `./tensorboard_qwen`.
- Uses `bf16_mixed` precision.
- Enables the distributed optimizer.
- Sets NCCL-related memory environment variables before importing PyTorch.

This file is launched by `training/launch_single_node.sh`.

### `training/launch_single_node.sh`

Runs `training/pretrain.py` with `torchrun`.

Defaults:

- `NNODES=1`
- `NODE_RANK=0`
- `MASTER_ADDR=localhost`
- `MASTER_PORT=6000`
- `NPROC_PER_NODE=8`

The script can be adapted for multi-node runs by exporting `NNODES`, `NODE_RANK`, and `MASTER_ADDR` before launching it.

### `training/pretrain_qwen.py`

Runs the Qwen pretraining configuration inside Ray Train.

Key behavior:

- Uses `ray.train.torch.TorchTrainer`.
- Uses `ScalingConfig(num_workers=8, use_gpu=True)`.
- Uses local model files at `/workspace/qwen`.
- Initializes Ray with `ray.init(dashboard_host="0.0.0.0")`.
- Applies NCCL memory optimization variables inside each Ray worker.

This script is suitable for Ray-managed distributed training where Ray starts locally or connects according to the surrounding environment.

### `training/qwen_pretrain.py`

Another Ray Train entrypoint for Qwen pretraining, tuned for the KubeRay setup.

Key behavior:

- Uses local model files at `/workspace/qwen`.
- Connects to Ray using `address="ray://localhost:10001"`.
- Uses `ScalingConfig(num_workers=8, use_gpu=True)`.
- Enables FP8 hybrid format on the model.
- Enables FlashAttention.
- Sets H100/NCCL environment variables through Ray `runtime_env`.
- Uses CPU and NCCL tuning variables such as `OMP_NUM_THREADS`, `CUDA_DEVICE_MAX_CONNECTIONS`, `NCCL_NVLS_ENABLE`, and `NCCL_BUFFSIZE`.

This is the script referenced by `training/ray-job-submission.yaml`.

## Kubernetes Manifests

### `training/ray-cluster.yaml`

Defines a standalone KubeRay `RayCluster` named `qwen-pretrain-cluster`.

Cluster shape:

- Ray version: `2.9.0`
- Namespace: `default`
- KAI queue label: `qwen-pretrain-queue`
- Head pod:
  - Image: `nvcr.io/nvidia/nemo:26.04`
  - CPU: `8`
  - Memory: `32Gi`
  - GPU count: `0`
  - Ports: dashboard `8265`, Ray client `10001`, GCS `6379`
- Worker group:
  - Group name: `h100-gpu-workers`
  - Replicas: `1`
  - Max replicas: `8`
  - GPU limit/request: `8`
  - CPU limit/request: `96`
  - Memory limit/request: `800Gi`
  - Shared memory volume: `256Gi`

Both head and worker pods mount host storage from `/mnt/data/training` to `/workspace`.

### `training/ray-job-submission.yaml`

Defines a KubeRay `RayJob` named `qwen-pretrain-job`.

Key behavior:

- Uses `HTTPMode` submission.
- Runs this entrypoint:

```bash
python /workspace/qwen_pretrain.py
```

- Defines an embedded Ray cluster spec similar to `ray-cluster.yaml`.
- Uses the NVIDIA NeMo image `nvcr.io/nvidia/nemo:26.04`.
- Mounts `/mnt/data/training` as `/workspace`.
- Requests one 8-GPU worker pod with 96 CPUs and 800Gi memory.
- Sets NCCL, PyTorch, and CPU-threading environment variables for H100 training.

## Qwen Model Files

### `training/qwen/config.json`

Hugging Face-style model configuration for a Qwen 3.5 architecture.

Notable values:

- Architecture: `Qwen3_5ForConditionalGeneration`
- Model type: `qwen3_5`
- Text hidden size: `2560`
- Text layers: `32`
- Attention heads: `16`
- Vocabulary size: `248320`
- Max position embeddings: `262144`
- Text dtype: `bfloat16`
- Includes vision configuration and image/video token IDs.

The training scripts override runtime sequence length to `4096`.

### `training/qwen/tokenizer.json`

Tokenizer model data.

Notable values:

- Tokenizer format version: `1.0`
- Model type: BPE
- Vocabulary entries: `248044`
- Merge rules: `247587`
- Added tokens: `26`
- Decoder type: ByteLevel

This is a large generated tokenizer file and usually should not be hand-edited.

### `training/qwen/tokenizer_config.json`

Tokenizer runtime configuration.

Notable values:

- Tokenizer class: `Qwen2Tokenizer`
- Model max length: `262144`
- EOS token: `<|im_end|>`
- Pad token: `<|endoftext|>`
- Includes chat template logic for text, image, video, tool-call, and tool-response formatting.
- Defines extra special tokens for audio, image, video, and vision markers.

## Typical Usage

From the repository root:

```bash
./setup.sh
kubectl apply -f kai-queue.yaml
kubectl apply -f training/ray-job-submission.yaml
```

For a direct single-node `torchrun` launch from the `training` directory:

```bash
cd training
./launch_single_node.sh
```

## Notes

- The Kubernetes manifests assume GPU-capable nodes and NVIDIA container support.
- The manifests mount `/mnt/data/training` into containers as `/workspace`; ensure the training files and `qwen/` directory are available there.
- The training scripts assume Megatron-Bridge, Ray, PyTorch, and NVIDIA/NeMo dependencies are available in the runtime image.
- Generated files such as `.DS_Store` and `.git/` internals are not part of the runtime workflow.
