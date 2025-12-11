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

BATCH_SIZE = 4
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

def _keyword_inclusion_reward_fn(response: str, keyword: str) -> dict:
    """Compute reward based on keyword inclusion in the response."""
    # for keyword in keywords:
    if keyword not in response:
        return {"reward": 0.0}
    return {"reward": 1.0}


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""
    def __init__(
        self,
        version: int,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: List[torch.Tensor],  # shape: [G]
        values: Optional[torch.Tensor] = None,  # shape: [G]
    ):
        self.version = version
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.values = values

    def move_to_device(self, device):
        """Move trajectory tensors to specified device."""
        if self.rewards is not None:
            self.rewards = self.rewards.to(device)
        self.log_probs = [lp.to(device) for lp in self.log_probs]
        return self


# ===================== Actors =====================

@ray.remote
class TrajectoryQueue:
    """Buffer between Generator and Scorer."""
    def __init__(self):
        self.q = asyncio.Queue()

    async def put(self, traj: Trajectory):
        # TODO: implement trajectory queuing
        await self.q.put(traj)

    async def get(self):
        # TODO: implement trajectory retrieval with timeout
        begin = time.time()
        while self.q.qsize() == 0:
            await asyncio.sleep(0.1)
            if time.time() - begin > 5.0:
                return None
        traj = await self.q.get()
        return traj


@ray.remote
class ReplayBuffer:
    """Stores scored trajectories for the Learner."""
    def __init__(self):
        self.data = []

    def put(self, traj: Trajectory):
        # TODO: store completed trajectories here
        self.data.append(traj)

    def sample(self, k: int):
        # TODO: sample k trajectories for training
        if k > len(self.data):
            return None
        indices = np.random.choice(len(self.data), size=k, replace=False)
        return [self.data[i] for i in indices]


@ray.remote
class Scorer:
    """Assigns rewards to generated text responses."""
    def __init__(self, traj_q, replay_buf):
        self.traj_q = traj_q
        self.replay_buf = replay_buf
        self.running = False
        self.device = get_device()

    async def run(self):
        """Continuously fetch trajectories, assign rewards, and store them."""
        self.running = True
        while self.running:
            # TODO: Get trajectories from queue, compute rewards, store in replay buffer
            # This should implement a reward function that evaluates text quality
            # e.g., keyword inclusion, safety, helpfulness, etc.
            # traj: Trajectory = await self.traj_q.get()
            # if traj is None:
            #     await asyncio.sleep(0.1)
            #     continue
            traj: Trajectory = await self.traj_q.get.remote()
            if traj is None:
                await asyncio.sleep(0.1)
                continue
            traj = traj.move_to_device(self.device)
            keyword = traj.prompts[0].split()[-1]  # Example: use last word of prompt as keyword
            advantage, _, _ = compute_group_normalized_reward(
                reward_fn=_keyword_inclusion_reward_fn,
                rollout_responses=traj.responses,
                repeated_ground_truths=[keyword]*G,
                group_size=G,
                advantage_eps=ADVANTAGE_EPS,
                normalized_by_std=USE_STD_NORM
            )
            traj.rewards = advantage#.to(traj.rewards.device)
            self.replay_buf.put.remote(traj.move_to_device("cpu"))
            print(f"scored a trajectory for version {traj.version}", flush=True)

    def stop(self):
        self.running = False


@ray.remote(num_gpus=1)
class Learner:
    """Learns policy updates from the replay buffer using TransformerLM."""
    def __init__(self, replay_buf):
        self.device = get_device()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.learn_model, self.learn_optim = create_model_optim(self.device)
        self.version = 0
        self.replay_buf = replay_buf

    def step(self):
        """One GRPO/PPO-style update step."""
        # TODO: sample from replay buffer, compute advantages, update model
        # This should implement GRPO policy gradient updates for text generation
        num_prompts = BATCH_SIZE

        for _ in range(N_GRPO_STEPS):
            trajectories = ray.get(self.replay_buf.sample.remote(num_prompts))
            if trajectories is None:
                return None
            old_log_probs = torch.zeros((num_prompts * G, SAMPLING_MAX_TOKENS), device=self.device)
            mask = torch.zeros((num_prompts * G, SAMPLING_MAX_TOKENS), device=self.device)
            for i, traj in enumerate(trajectories):
                traj = traj.move_to_device(self.device)
                start_i = i * G
                prompt_len = len(self.tokenizer.encode(traj.prompts[0], allowed_special="all"))
                for g in range(G):
                    log_probs = traj.log_probs[g]
                    resp_len = log_probs.shape[0]
                    old_log_probs[start_i + g, (prompt_len - 1):(prompt_len + resp_len - 1)] = log_probs
                    mask[start_i + g, (prompt_len - 1):(prompt_len + resp_len - 1)] = 1.0

            old_log_probs = old_log_probs.detach()

            self.learn_model.train()
            policy_log_probs = torch.zeros_like(old_log_probs).to(self.device)
            advantages = torch.zeros((num_prompts * G, 1), device=self.device)
            for i, traj in enumerate(trajectories):
                advantages[i*G:(i+1)*G, :] = traj.rewards.unsqueeze(1)
                start_i = i * G
                for g in range(G):
                    prompt = traj.prompts[g]
                    response = traj.responses[g]
                    selected_probs = regenerate_probs(
                        prompt, response, self.learn_model, self.tokenizer, self.device)
                    policy_log_probs[start_i + g, :selected_probs.shape[0]] = selected_probs

            self.learn_optim.zero_grad()
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=mask,
                gradient_accumulation_steps=1,
                loss_type=LOSS_TYPE,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=0.2
            )

        print(f"Learner completed step for version {self.version} with loss {loss.item()}", flush=True)

        # loss = torch.tensor(0.0, device=self.device)
        self.version += 1
        return float(loss.item())

    def get_weights(self):
        # TODO: Return model weights for synchronization with Generator
        return {k: v.cpu() for k, v in self.learn_model.state_dict().items()}

    def get_version(self):
        return self.version


@ray.remote(num_gpus=1)
class Generator:
    """Generates text responses using TransformerLM policy."""
    def __init__(self, traj_q):
        self.device = get_device()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.gen_model, self.gen_optim = create_model_optim(self.device)
        self.traj_q = traj_q
        self.version = 0

    def get_version(self):
        return self.version

    def generate(self, prompts: List[str]):
        """Generate text responses and send to Scorer."""
        # TODO: Generate G responses for each prompt using TransformerLM
        # - Tokenize prompts
        # - Generate text responses using self.model
        # - Calculate log probabilities for generated tokens
        # - Create Trajectory objects and send to trajectory queue
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
            traj = Trajectory(version=self.version,
                prompts=[prompt]*G,
                responses=responses,
                rewards=torch.zeros(G, device=self.device),  # Placeholder
                log_probs=log_probs_list
            ).move_to_device("cpu")
            self.traj_q.put.remote(traj)
        print(f"Generated {len(prompts)*G} responses for version {self.version}", flush=True)

    def update(self, weights: Dict, version: int):
        """Load updated learner weights."""
        sd = self.gen_model.state_dict()
        for n, w in weights.items():
            sd[n] = w.to(self.device)
        self.gen_model.load_state_dict(sd)
        self.version = version
        return version


# ===================== Training loop =====================

# Implement disaggregated off-policy GRPO where rollout generation and policy opti
# mization are run in parallel on different GPUs. Rollout generation will execute on an older version of
#  the model. We’ll limit staleness of rollouts so that generation is at most one version behind training.
#  Use Ray to implement separate Generator and Learner actors, each using 1 GPU (2 GPUs total). You
#  should be able to directly reuse the Generator and Learner classes implemented in the colocated setup.
#  The changes needed will be in the driver loop (run\_once). The driver loop should call Generator
#  and Learner methods in order so that it implements the “one-version-behind” constraint: ensure the
#  generator is never more than one policy version behind the learner.

def run_training(num_steps: int = 3):
    """Run disaggregated GRPO training with text generation."""
    traj_q = TrajectoryQueue.remote()
    replay_buf = ReplayBuffer.remote()
    learner = Learner.remote(replay_buf)
    scorer = Scorer.remote(traj_q, replay_buf)
    generator = Generator.remote(traj_q)

    # TODO: Driver code for the training loop
    scorer.run.remote()

    # training prompts
    prompts = []
    with open("cse599o_alignment/prompts/keywords.txt", "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(f"Generate a story that includes {line.strip()}")

    for step in range(num_steps):
        print(f"=== Step {step} ===", flush=True)

        # 1. Generator generates rollouts using current policy
        gen_version = ray.get(generator.get_version.remote())
        print(f"Generator version: {gen_version}")
        batch_prompts = prompts[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
        ray.get(generator.generate.remote(batch_prompts))

        print("Waiting for learner step...", flush=True)

        # 2. Learner performs a training step
        loss = ray.get(learner.step.remote())
        learn_version = ray.get(learner.get_version.remote())
        print(f"Learner version: {learn_version}, Loss: {loss}", flush=True)

        # 3. Update Generator with new weights if needed
        if learn_version > gen_version:
            weights = ray.get(learner.get_weights.remote())
            gen_version = ray.get(generator.update.remote(weights, learn_version))
            print(f"Updated generator to version {gen_version}", flush=True)
        else:
            print(f"prefilled", flush=True)
            step -= 1  # repeat this step if no update

def run_once(num_steps: int = 3):
    """Entry point for training."""
    run_training(num_steps)



# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, _temp_dir="/local1/samarjit")
    run_once(num_steps=args.steps)
