"""
GRPO Skeleton: Minimal Asynchronous Training Loop
------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with text generation using TransformerLM (Generator)
 - compute rewards for text responses (Scorer)
 - perform policy updates using GRPO algorithm (Learner)
 - synchronize model weights between Generator and Learner
"""

import asyncio
import argparse
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np

from cse599o_basics.model import TransformerLM
from cse599o_basics.optimizer import AdamW
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    gradient_clipping
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
CHECKPOINT_PATH = ""


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""
    def __init__(
        self,
        version: int,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: torch.Tensor,  # shape: [G]
        values: Optional[torch.Tensor] = None,  # shape: [G]
    ):
        self.version = version
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.values = values


# ===================== Actors =====================

@ray.remote
class TrajectoryQueue:
    """Buffer between Generator and Scorer."""
    def __init__(self):
        self.q = asyncio.Queue()

    def put(self, traj: Trajectory):
        # TODO: implement trajectory queuing
        pass

    def get(self):
        # TODO: implement trajectory retrieval with timeout
        return None


@ray.remote
class ReplayBuffer:
    """Stores scored trajectories for the Learner."""
    def __init__(self):
        self.data = []

    def put(self, traj: Trajectory):
        # TODO: store completed trajectories here
        pass

    def sample(self, k: int):
        # TODO: sample k trajectories for training
        return []


@ray.remote
class Scorer:
    """Assigns rewards to generated text responses."""
    def __init__(self, traj_q, replay_buf):
        self.traj_q = traj_q
        self.replay_buf = replay_buf
        self.running = False

    def run(self):
        """Continuously fetch trajectories, assign rewards, and store them."""
        self.running = True
        while self.running:
            # TODO: Get trajectories from queue, compute rewards, store in replay buffer
            # This should implement a reward function that evaluates text quality
            # e.g., keyword inclusion, safety, helpfulness, etc.
            pass

    def stop(self):
        self.running = False


@ray.remote
class Learner:
    """Learns policy updates from the replay buffer using TransformerLM."""
    def __init__(self, replay_buf):
        self.device = get_device()
        # TODO: Initialize the TransformerLM model
        # self.model = TransformerLM(
        #     vocab_size=VOCAB_SIZE,
        #     context_length=CONTEXT_LENGTH,
        #     num_layers=NUM_LAYERS,
        #     d_model=D_MODEL,
        #     num_heads=NUM_HEADS,
        #     d_ff=D_FF,
        #     theta=THETA
        # ).to(self.device)
        # self.model.load_checkpoint(CHECKPOINT_PATH)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.version = 0
        self.replay_buf = replay_buf

    def step(self):
        """One GRPO/PPO-style update step."""
        # TODO: sample from replay buffer, compute advantages, update model
        # This should implement GRPO policy gradient updates for text generation
        loss = torch.tensor(0.0, device=self.device)
        self.version += 1
        return float(loss.item())

    def get_weights(self):
        # TODO: Return model weights for synchronization with Generator
        return {}  # {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_version(self):
        return self.version


@ray.remote
class Generator:
    """Generates text responses using TransformerLM policy."""
    def __init__(self, traj_q):
        self.device = get_device()
        # TODO: Initialize the TransformerLM model
        # self.model = TransformerLM(
        #     vocab_size=VOCAB_SIZE,
        #     context_length=CONTEXT_LENGTH,
        #     num_layers=NUM_LAYERS,
        #     d_model=D_MODEL,
        #     num_heads=NUM_HEADS,
        #     d_ff=D_FF,
        #     theta=THETA
        # ).to(self.device)
        # self.model.load_checkpoint(CHECKPOINT_PATH)
        # self.tokenizer = tiktoken.get_encoding("gpt2")
        self.traj_q = traj_q
        self.version = 0

    def generate(self, prompts: List[str]):
        """Generate text responses and send to Scorer."""
        # TODO: Generate G responses for each prompt using TransformerLM
        # - Tokenize prompts
        # - Generate text responses using self.model
        # - Calculate log probabilities for generated tokens
        # - Create Trajectory objects and send to trajectory queue
        pass

    def update(self, weights: Dict, version: int):
        """Load updated learner weights."""
        # TODO: Update model weights from learner
        # sd = self.model.state_dict()
        # for n, w in weights.items():
        #     sd[n] = w.to(self.device)
        # self.model.load_state_dict(sd)
        self.version = version


# ===================== Training loop =====================

def run_training(num_steps: int = 3):
    """Run disaggregated GRPO training with text generation."""
    traj_q = TrajectoryQueue.remote()
    replay_buf = ReplayBuffer.remote()
    learner = Learner.remote(replay_buf)
    scorer = Scorer.remote(traj_q, replay_buf)
    generator = Generator.remote(traj_q)

    # TODO: Driver code for the training loop
    pass


def run_once(num_steps: int = 3):
    """Entry point for training."""
    run_training(num_steps)



# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    run_once(num_steps=args.steps)
