"""
GRPO Skeleton: Colocated Synchronous Training Loop (Simplified)
--------------------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation using TransformerLM
 - perform policy updates using GRPO algorithm
 - implement keyword inclusion reward function

This version combines Generator and Learner into a single actor for simplified
synchronous training without replay buffer, training directly on each trajectory.
"""

import argparse
import asyncio
import ray
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.adamw import AdamW
from cse599o_basics.train_utils import decode, load_checkpoint
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    # gradient_clipping
)


# ===================== Basic setup =====================

G = 4  # group size (number of responses per prompt)
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
THETA = 10000
CHECKPOINT_PATH = "/homes/iws/samarjit/workspace/cse599o/cse599o-assignment1-basics/checkpoints/ckpt_2780189.pt"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: torch.Tensor,  # shape: [G]
        values: Optional[torch.Tensor] = None,  # shape: [G]
    ):
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.values = values


# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the TransformerLM model
        self.model = TransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            rope_theta=THETA
        ).to(self.device)
        optim_args = {
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-08
        }
        self.optim = AdamW(self.model.parameters(), **optim_args)
        self.tokenizer = tiktoken.get_encoding("gpt2")

        if os.path.exists(CHECKPOINT_PATH):
            load_checkpoint(CHECKPOINT_PATH, self.model, self.optim)
        

    def generate_trajectories(self, prompts: List[str]) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        TODO: Implement this method
        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        trajectories = []
        for prompt in prompts:
            responses = []
            log_probs_list = []
            for g in range(G):
                response, log_probs = decode(self.model, self.tokenizer, self.optim, 
                      prompt=prompt)
                responses.append(response)
                log_probs_list.append(torch.tensor(log_probs, device=self.device))
            trajectories.append(Trajectory(
                prompts=[prompt]*G,
                responses=responses,
                rewards=torch.zeros(G, device=self.device),  # Placeholder
                log_probs=torch.cat(log_probs_list, dim=0)
            ))
        return trajectories

class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the same TransformerLM model as Generator
        self.model = TransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            rope_theta=THETA
        ).to(self.device)
        optim_args = {
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-08
        }
        self.optim = AdamW(self.model.parameters(), **optim_args)
        self.tokenizer = tiktoken.get_encoding("gpt2")

        if os.path.exists(CHECKPOINT_PATH):
            load_checkpoint(CHECKPOINT_PATH, self.model, self.optim)

    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        # TODO: Implement GRPO advantage computation
        # This should implement the group-relative advantage computation
        # that's central to GRPO algorithm
        advantages = []
        for traj in trajectories:
            # keyword is the last element of the prompt
            keyword = traj.prompts[0].split()[-1]
            advantage, _, _ = compute_group_normalized_reward(
                reward_fn=self._keyword_inclusion_reward_fn,
                rollout_responses=traj.responses,
                repeated_ground_truths=[keyword]*G,
                group_size=G,
                advantage_eps=1e-8,
                normalized_by_std=False
            )
            advantages.append(advantage)
        return torch.cat(advantages, dim=0).to(self.device)
    
    def update_policy(self, trajectories: List[Trajectory]) -> float:
        """Perform one policy update step."""
        # TODO: Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value

        advantages = self.compute_advantages(trajectories)
        # policy_log_probs = torch.cat([traj.log_probs for traj in trajectories], dim=0)
        # we expect that for a trajectory, log_probs is a tensor of shape (G, seq_len)
        # we want to concatenate each trajectory so that we have (num_trajectories * G, seq_len)
        # first verify shape of log_probs
        print(f"Log probs shape per trajectory: {trajectories[0].log_probs.shape}", flush=True)
        policy_log_probs = torch.cat([traj.log_probs.unsqueeze(0) for traj in trajectories], dim=0).view(-1, trajectories[0].log_probs.shape[-1])
        old_log_probs = policy_log_probs.detach()
        self.optim.zero_grad()
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=torch.ones_like(policy_log_probs, device=self.device),
            gradient_accumulation_steps=1,
            loss_type="grpo_clip",
            advantages=advantages.unsqueeze(1),
            old_log_probs=old_log_probs,
            cliprange=0.2
        )

        self.optim.step()
        
        return loss.item()

    def _keyword_inclusion_reward_fn(self, response: str, keywords: List[str]) -> dict:
        """Compute reward based on keyword inclusion in the response."""
        for keyword in keywords:
            if keyword not in response:
                return {"reward": 0.0}
        return {"reward": 1.0}

# ===================== Combined Actor =====================

@ray.remote(num_gpus=1)
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self):
        Generator.__init__(self)
        Learner.__init__(self)
        self.step_count = 0
    
    def training_step(self, prompts: List[str]) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        # Generate trajectories for the batch of prompts
        trajectories = self.generate_trajectories(prompts)
        
        # Update policy using GRPO
        loss = self.update_policy(trajectories)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories),
            'avg_reward': float(torch.cat([traj.rewards for traj in trajectories]).mean()) if trajectories else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }


# ===================== Training loop =====================

def run_training(num_steps: int = 10, num_workers: int = 1):
    """Run colocated GRPO training with text generation."""
    
    # Create workers  
    workers = [ColocatedWorker.remote() for _ in range(num_workers)]
    
    # TODO: Define training prompts
    prompts = []
    with open("cse599o_alignment/prompts/keywords.txt", "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(f"Generate a story that includes {line.strip()}")
    
    for step in range(num_steps):
        # just get 4 prompts, assume 1 worker for simplicity
        num_prompts = 4
        batch_prompts = prompts[step * num_prompts: (step + 1) * num_prompts]
        if num_workers == 1:
            worker = workers[0]
            result = ray.get(worker.training_step.remote(batch_prompts))
            print(f"Step {result['step']}: Loss={result['loss']:.4f}, "
                  f"Num Trajectories={result['num_trajectories']}, "
                  f"Avg Reward={result['avg_reward']:.4f}")
        
    # Get final statistics
    final_stats = ray.get([worker.get_statistics.remote() for worker in workers])
    for i, stats in enumerate(final_stats):
        print(f"Worker {i} - Steps: {stats['step_count']}, "
              f"Model Params: {stats['model_parameters']}")


def run_once(num_steps: int = 10, num_workers: int = 1):
    """Entry point for training."""
    run_training(num_steps, num_workers)


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of colocated workers")
    args = parser.parse_args()
    
    ray.init(ignore_reinit_error=True, _temp_dir="/local1/samarjit",
             runtime_env={
                "excludes": [
                    ".git/**",                           # git metadata and objects
                    ".venv/**",                          # virtual environment
                    "submission_*/**",                   # submission folders (6.9GB)
                    "checkpoint/**",                     # checkpoint folder (731MB)
                    "tests/fixtures/**",                 # test fixtures (large model files)
                    "wandb/**",                          # wandb logs
                    "*.nsys-rep",                        # profiling files
                    "*.pt", "*.pth", "*.safetensors",   # model weight files
                    "*.tar", "*.zip", "*.gz",           # archives
                    "__pycache__/**",                   # Python cache
                    "*.egg-info/**"                     # package info
                ]})
    
    try:
        run_once(num_steps=args.steps, num_workers=args.workers)
    finally:
        ray.shutdown()
