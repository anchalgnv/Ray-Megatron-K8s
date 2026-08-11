#!/bin/bash
set -e

echo "========================================================"
echo " Starting Kubernetes AI Cluster Setup with Minikube"
echo "========================================================"

# 1. Check if minikube is installed, IF NOT install it for ubuntu
if ! command -v minikube &> /dev/null; then
    echo "[Step 1] Minikube not found. Installing for Ubuntu..."
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    sudo install minikube-linux-amd64 /usr/local/bin/minikube
    rm minikube-linux-amd64
    echo "[Step 1] Minikube installed successfully."
else
    echo "[Step 1] Minikube is already installed."
fi

# 2. Start Minikube with GPU access and persistent storage enabled
echo "[Step 2] Starting Minikube with GPU and Persistent Storage..."
minikube start  --mount-string $(pwd):/mnt/data --driver=docker --gpus=all --addons=default-storageclass,storage-provisioner

# Pre-requisite: Check if Helm is installed, install if not
if ! command -v helm &> /dev/null; then
    echo "[Pre-requisite] Helm not found. Installing..."
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    bash get_helm.sh
    rm get_helm.sh
fi

# 3. Install NVIDIA GPU and Network Operator using Helm
echo "[Step 3] Adding NVIDIA Helm repository..."
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

echo "[Step 3] Installing NVIDIA GPU Operator..."
helm upgrade -i gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --wait

echo "[Step 3] Installing NVIDIA Network Operator..."
helm upgrade -i network-operator nvidia/network-operator \
  --namespace network-operator \
  --create-namespace \
  --wait

# 4. Install KubeRay Operator with NVIDIA KAI scheduler enabled
echo "[Step 4] Installing NVIDIA KAI Scheduler..."
helm upgrade -i kai-scheduler oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  --namespace kai-system \
  --create-namespace \
  --set "global.gpuSharing=true" \
  --version v0.15.3 \
  --wait

echo "[Step 4] Installing KubeRay Operator with KAI Scheduler integration..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

helm upgrade -i kuberay-operator kuberay/kuberay-operator \
  --namespace kuberay-system \
  --create-namespace \
  --set batchScheduler.name=kai-scheduler \
  --wait

# 5. Setup kube-prometheus operator to monitor Ray cluster
echo "[Step 5] Installing kube-prometheus-stack for Ray monitoring..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade -i prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --wait

# 6. Setup W&B local server in the cluster
echo "[Step 6] Installing Weights & Biases (W&B) local server..."
helm repo add wandb https://wandb.github.io/helm-charts
helm repo update

helm upgrade -i wandb wandb/wandb \
  --namespace wandb \
  --create-namespace \
  --set global.host="localhost" \
  --version=0.3.7 \
  --set license=$LICENCE
  --wait

echo "========================================================"
echo " Setup complete! The AI Stack is successfully deployed."
echo "========================================================"
