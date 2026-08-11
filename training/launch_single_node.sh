#!/bin/bash
# run_pretrain.sh

# =====================================================================
# Distributed Environment Variables
# These defaults run the script on a single node (localhost).
# To scale to multiple nodes, export these variables externally:
#   export NNODES=4
#   export NODE_RANK=0  # (0, 1, 2, 3...)
#   export MASTER_ADDR="10.0.0.1" # The IP of the master node
# =====================================================================

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-6000}

# Number of GPUs per node
NPROC_PER_NODE=8

echo "====================================================="
echo "Starting Megatron-Bridge Pretraining..."
echo "Nodes: $NNODES | Rank: $NODE_RANK | Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node: $NPROC_PER_NODE"
echo "====================================================="

# Execute the python script utilizing standard PyTorch distributed launch (torchrun).
# No Slurm is required, simply run this script directly on your instance(s).
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain.py
