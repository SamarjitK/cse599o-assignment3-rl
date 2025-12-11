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
from cse599o_basics.model_utils import softmax
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

N_GRPO_STEPS = 2
LEARNING_RATE = 5e-4
SAMPLING_TEMP = 0.8
SAMPLING_MAX_TOKENS = 60
ADVANTAGE_EPS = 1e-8
LOSS_TYPE = "grpo_clip"
USE_STD_NORM = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Helpers =====================

def create_model_optim(device) -> tuple[TransformerLM, AdamW]:
    """Create TransformerLM model and AdamW optimizer."""
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=THETA
    ).to(device)
    optim_args = {
        "lr": LEARNING_RATE,
        "weight_decay": 0.01,
        "betas": [
            0.9,
            0.999
        ],
        "eps": 1e-08
    }
    optim = AdamW(model.parameters(), **optim_args)

    if os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(CHECKPOINT_PATH, model, optim)

    return model, optim

def regenerate_probs(prompt: str, response: str, 
                     model: TransformerLM, tokenizer: tiktoken.Encoding, device) -> torch.Tensor:
        prompt_tokens = tokenizer.encode(prompt, allowed_special="all")
        resp_tokens = tokenizer.encode(response, allowed_special="all")
        input_tokens = prompt_tokens + resp_tokens
        input_tensor = torch.tensor(input_tokens, device=device).unsqueeze(0)  # (1, seq_len)

        # still running into issues even with the seq_len - 1. So lets assert and check that that isn't the issue
        assert input_tensor[:, :-1].shape[1] <= SAMPLING_MAX_TOKENS, f"input_tensor too long: {input_tensor[:, :-1].shape[1]} >= {SAMPLING_MAX_TOKENS}. Prompt len: {len(prompt_tokens)}, Response len: {len(resp_tokens)}\
            for ref: prompt='{prompt}', response='{response}'"

        outputs = model(input_tensor[:, :-1])  # (1, seq_len-1, vocab_size)
        probs = softmax(outputs, dim=-1)  # (1, seq_len-1, vocab_size)

        target_tokens = input_tensor[:, 1:].squeeze(0)  # (seq_len-1,)
        selected = probs[0].gather(1, target_tokens.unsqueeze(1))
        return torch.log(selected.squeeze(1))  # (seq_len-1,)

# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: List[torch.Tensor],  # shape: [G]
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
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.gen_model, self.gen_optim = create_model_optim(self.device)

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
                response, log_probs = decode(
                    self.gen_model, self.tokenizer, self.gen_optim, 
                    max_tokens=SAMPLING_MAX_TOKENS, temperature=SAMPLING_TEMP,
                    prompt=prompt)
                responses.append(response)
                log_probs_list.append(torch.tensor(log_probs, device=self.device))
            trajectories.append(Trajectory(
                prompts=[prompt]*G,
                responses=responses,
                rewards=torch.zeros(G, device=self.device),  # Placeholder
                log_probs=log_probs_list
            ))
        return trajectories

class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the same TransformerLM model as Generator
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.learn_model, self.learn_optim = create_model_optim(self.device)

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
                advantage_eps=ADVANTAGE_EPS,
                normalized_by_std=USE_STD_NORM,
            )
            advantages.append(advantage)
        return torch.cat(advantages, dim=0).to(self.device)
    
    def update_policy(self, trajectories: List[Trajectory]) -> tuple[float, torch.Tensor]:
        """Perform one policy update step."""
        # TODO: Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value

        advantages = self.compute_advantages(trajectories).unsqueeze(-1)  # shape: (batch_size, 1)
        batch_size = len(trajectories) * G
        old_log_probs = torch.zeros(batch_size, SAMPLING_MAX_TOKENS, device=self.device)
        mask = torch.zeros(batch_size, SAMPLING_MAX_TOKENS, device=self.device)
        for i, traj in enumerate(trajectories):
            start_i = i * G
            prompt_len = len(self.tokenizer.encode(traj.prompts[0]))
            for g in range(G):
                log_probs = traj.log_probs[g]
                response_len = log_probs.shape[0]
                old_log_probs[start_i + g, (prompt_len - 1):(prompt_len + response_len - 1)] = log_probs
                mask[start_i + g, (prompt_len - 1):(prompt_len + response_len - 1)] = 1.0
        
        old_log_probs = old_log_probs.detach()

        self.learn_model.train()
        policy_log_probs = torch.zeros_like(old_log_probs).to(self.device)
        for i, traj in enumerate(trajectories):
            start_i = i * G
            for g in range(G):
                prompt = traj.prompts[g]
                response = traj.responses[g]
                selected_probs = regenerate_probs(
                    prompt, response, self.learn_model, self.tokenizer, self.device)
                policy_log_probs[start_i + g, :selected_probs.shape[0]] = selected_probs
    
        self.learn_optim.zero_grad()
        # create deep copy of policy_log_probs to return later
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=mask,
            gradient_accumulation_steps=1,
            loss_type=LOSS_TYPE,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=0.2
        )

        self.learn_optim.step()
        
        return loss.item(), mask.detach()

    def _keyword_inclusion_reward_fn(self, response: str, keyword: str) -> dict:
        """Compute reward based on keyword inclusion in the response."""
        # for keyword in keywords:
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
        self.ref_model, self.ref_optim = create_model_optim(self.device)
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def training_step(self, prompts: List[str]) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        # Generate trajectories for the batch of prompts

        # track time taken for generation
        gen_start = time.time()
        print(f"Generating trajectories for {len(prompts)} prompts...", flush=True)
        trajectories_list = []
        for _ in range(N_GRPO_STEPS):
            trajectories_list.append(self.generate_trajectories(prompts))
        print(f"Generated trajectories in {time.time() - gen_start:.2f} seconds.", flush=True)
        
        # Update policy using GRPO
        update_start = time.time()
        print("Updating policy...", flush=True)
        for i in range(N_GRPO_STEPS):
            loss, mask = self.update_policy(trajectories_list[i])
        print(f"Policy updated in {time.time() - update_start:.2f} seconds.", flush=True)
        last_traj = trajectories_list[-1]

        kl_start = time.time()
        print("Computing KL divergences...", flush=True)
        with torch.no_grad():
            old_lp = torch.zeros_like(mask).to(self.device)
            ref_lp = torch.zeros_like(mask).to(self.device)
            cur_lp = torch.zeros_like(mask).to(self.device)
            for i, traj in enumerate(last_traj):
                start_i = i * G
                for g in range(G):
                    prompt = traj.prompts[g]
                    response = traj.responses[g]
                    ref_probs = regenerate_probs(prompt, response, 
                                             self.ref_model, self.tokenizer, self.device)
                    ref_lp[start_i + g, :ref_probs.shape[0]] = ref_probs
                    cur_probs = regenerate_probs(prompt, response, 
                                             self.learn_model, self.tokenizer, self.device)
                    cur_lp[start_i + g, :cur_probs.shape[0]] = cur_probs
                    old_probs = regenerate_probs(prompt, response, 
                                             self.gen_model, self.tokenizer, self.device)
                    old_lp[start_i + g, :old_probs.shape[0]] = old_probs

            kl_old = ((cur_lp - old_lp) * mask).sum() / (G * len(last_traj))
            kl_ref = ((cur_lp - ref_lp) * mask).sum() / (G * len(last_traj))
        print(f"KL divergences computed in {time.time() - kl_start:.2f} seconds.", flush=True)
        print(f"KL(Current || Old): {kl_old.item():.6f}, KL(Current || Ref): {kl_ref.item():.6f}", flush=True)

        # Weight synchronization to generator
        sync_start = time.time()
        print("Synchronizing weights to generator...", flush=True)
        self.gen_model.load_state_dict(self.learn_model.state_dict())
        print(f"Weights synchronized in {time.time() - sync_start:.2f} seconds.", flush=True)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories_list[-1]),
            'avg_reward': float(torch.cat([traj.rewards for traj in last_traj]).mean()) if last_traj else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.gen_model.parameters()) if hasattr(self, 'model') else 0
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
        num_prompts = 4
        batch_prompts = prompts[step * num_prompts: (step + 1) * num_prompts]
        for worker in workers:
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
            #  _system_config={
            #         "enable_metrics_collection": False,
            #  },
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
