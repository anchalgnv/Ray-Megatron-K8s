import os
import torch
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step

def qwen3_5_4b_pretrain_config():
    """
    Returns a pre-training config for a Qwen 4B architecture.
    Uses Megatron-Bridge to convert HuggingFace configs dynamically into Megatron format.
    """
    cfg = _pretrain_common()
    hf_model_id = "/workspace/qwen" 
    
    # 1. Model Configuration
    cfg.model = AutoBridge.from_hf_pretrained(hf_model_id).to_megatron_provider(load_weights=False)
    
    # Sequence length override to prevent initial mismatch & OOM
    cfg.model.seq_length = 4096
    cfg.model.max_position_embeddings = 4096
    
    # 2. Parallelism Settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    
    # 3. Tokenizer Configuration
    cfg.tokenizer.tokenizer_model = hf_model_id
    cfg.tokenizer.trust_remote_code = True
    
    # 4. Dataset Configuration
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 0  
    cfg.dataset.seq_length = 4096
    
    # 5. Training Loop Configuration
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

    return cfg

def train_loop_per_worker():
    """
    The function executed by each Ray worker process. 
    Ray Train automatically populates the PyTorch Distributed environment vars.
    """
    # Force the NCCL memory optimizations into the localized worker processes
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    
    config = qwen3_5_4b_pretrain_config()
    pretrain(config, forward_step)

if __name__ == "__main__":
    # Initialize the Ray Context. 
    # If run in KubeRay (RayJob), this connects to the Ray cluster.
    # If run locally in Docker, this automatically starts a local Ray instance.
    ray.init(dashboard_host="0.0.0.0")
    
    # Define scaling config for the TorchTrainer
    # This automatically provisions 8 tasks requesting 1 GPU each on the cluster.
    scaling_config = ScalingConfig(
        num_workers=8,
        use_gpu=True,
    )
    
    # Wrap the Megatron execution inside Ray's TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
    )
    
    # Launch the distributed training
    result = trainer.fit()
    print("Training finished successfully! Metrics:", result.metrics)
