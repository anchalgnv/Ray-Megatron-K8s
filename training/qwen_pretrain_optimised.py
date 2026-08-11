import os

# Prevent the Ray driver process from hogging VRAM on GPU 0.
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step

def qwen3_5_4b_pretrain_config():
    cfg = _pretrain_common()
    hf_model_id = "/workspace/qwen" 
    
    # 1. Model Configuration
    cfg.model = AutoBridge.from_hf_pretrained(hf_model_id).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096
    cfg.model.max_position_embeddings = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    # Enable Transformer Engine FP8 under the hood
    cfg.model.fp8_format = 'hybrid'
    cfg.model.fp8_margin = 0
    # Force FlashAttention
    cfg.model.use_flash_attn = True

    
    # 2. Tokenizer Configuration
    cfg.tokenizer.tokenizer_model = hf_model_id
    cfg.tokenizer.trust_remote_code = True
    
    # 3. Dataset Configuration
    cfg.dataset.blend = None
    # [FIX 1]: Reverted to 0 to prevent Docker/Kubernetes Shared Memory (IPC) thrashing.
    cfg.dataset.num_workers = 0  
    cfg.dataset.seq_length = 4096
    
    # 4. Training Loop Configuration
    cfg.train.micro_batch_size = 1  
    cfg.train.global_batch_size = 8  
    cfg.train.train_iters = 10000
    cfg.train.eval_iters = 50
    cfg.train.eval_interval = 500
    cfg.train.save_interval = 1000
    
    cfg.train.tensorboard_dir = "./tensorboard_qwen"
    cfg.train.save_dir = "./checkpoints_qwen"
    
    # Precision
    cfg.mixed_precision = "bf16_mixed"

    # Distributed Optimizer (ZeRO-1)
    if not hasattr(cfg, 'optim'):
        from megatron.bridge.training.config import OptimizerConfig
        cfg.optim = OptimizerConfig()
    cfg.optim.use_distributed_optimizer = True
    #cfg.optim.overlap_grad_reduce = True
    #cfg.optim.delay_grad_reduce = True

    return cfg

def train_loop_per_worker():
    config = qwen3_5_4b_pretrain_config()
    pretrain(config, forward_step)

if __name__ == "__main__":
    # [FIX 2]: Inject NCCL and H100 flags at the Ray container boot level.
    # This guarantees NCCL sees the flags BEFORE PyTorch initializes.
    runtime_env = {
        "env_vars": {
            # H100 NVLink & GPUDirect optimizations
            "NCCL_NVLS_ENABLE": "1",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_IB_DISABLE": "0",
            # Megatron Async Stream Requirements
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            # CPU limits to prevent worker thrashing
            "OMP_NUM_THREADS": "4",
            "NCCL_BUFFSIZE":"8388608"
        }
    }
    
    # Initialize the Ray Context with the strictly defined environment
    ray.init(dashboard_host="0.0.0.0", runtime_env=runtime_env, address="ray://localhost:10001")
    
    scaling_config = ScalingConfig(
        num_workers=8,
        use_gpu=True,
    )
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
    )
    
    result = trainer.fit()
    print("Training finished successfully! Metrics:", result.metrics)
