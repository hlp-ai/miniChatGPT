from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from loss import KPairwiseLoss, CrossEntropyLoss, ValueLoss, PolicyLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import statistics
from gpt import GPTRewardModel, GPTCritic, GPTActor
from tqdm import tqdm
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from typing import Union
from torchinfo import summary
from configs import TrainingConfig
from tokenizer import TiktokenTokenizer


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')


@dataclass
class Experience:
    completion: torch.Tensor
    actor_log_probs: torch.Tensor
    attention_mask: torch.Tensor
    kl_penalized_reward: torch.Tensor
    advantage: torch.Tensor
    num_actions: int
    estimated_kl: torch.Tensor
    values: torch.Tensor
    action_mask: torch.Tensor


class PPOTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, actor: GPTActor, critic: GPTCritic,
                 reward_model: GPTRewardModel, sft_model: GPTActor,
                 train_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"ppo_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = "cuda"
        self.max_new_tokens = 128

        self.orig_actor = actor
        self.orig_critic = critic
        self.orig_sft_model = sft_model
        self.orig_reward_model = reward_model

        self.actor = self.orig_actor  # torch.compile(self.orig_actor)
        self.critic = self.orig_critic  # torch.compile(self.orig_critic)
        self.sft_model = self.orig_sft_model  # torch.compile(self.orig_sft_model)
        self.reward_model = self.orig_reward_model  # torch.compile(self.orig_reward_model)
        # Separate actor loss from critic loss to save optimizer memory
        self.actor_criterion = PolicyLoss()
        self.critic_criterion = ValueLoss()

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=12,
                                           prefetch_factor=4,
                                           pin_memory=True)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=cfg.actor_lr,
                                          betas=(self.cfg.adam_beta1,
                                                 self.cfg.adam_beta1))
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=cfg.critic_lr,
                                           betas=(self.cfg.adam_beta1,
                                                  self.cfg.adam_beta1))

        self.writer = SummaryWriter(f'./runs/{self.run_name}/logs',
                                    max_queue=50)
        self.total_epochs = cfg.total_epochs
        self.debug = False
        self.save_freq = 50  # 500
        self.dtype = torch.float16
        self.tokenizer = TiktokenTokenizer("gpt2")
        self.finetune_method = cfg.finetune_method

        hp = {
            "max_new_tokens": self.max_new_tokens,
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "dtype": str(self.dtype),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)
        print("Initialized PPO Trainer")

    def save_states(self, step, is_last=False):
        file_name = f'{self.run_name}_actor_final.pt' if is_last else f'{self.run_name}_actor_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.orig_actor.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.actor_optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')
        file_name = f'{self.run_name}_critic_final.pt' if is_last else f'{self.run_name}_critic_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict': self.orig_critic.state_dict(),
                'optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, f'./runs/{self.run_name}/{file_name}')

    def kl_penalized_reward(
        self,
        reward: torch.Tensor,
        log_prob_rl: torch.Tensor,
        log_prob_sft: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> Union[torch.Tensor, torch.Tensor]:
        # log(π_RL(y|x) / π_SFL(y|x)) = log(π_RL(y|x)) - log(π_SFL(y|x))
        ratio = log_prob_rl - log_prob_sft
        # k3 in http://joschu.net/blog/kl-approx.html
        estimated_kl = (torch.exp(ratio) - 1) - ratio
        if action_mask:
            estimated_kl = estimated_kl * action_mask
            estimated_kl.sum(dim=1) / action_mask.sum(dim=1)
        estimated_kl = estimated_kl.mean(
            dim=1, keepdim=True)  # estimated_kl -> (B, 1)
        return reward - self.cfg.kl_beta * estimated_kl, estimated_kl

    @torch.no_grad()
    def make_experience(self, idx, input_masks, input_lengths):
        self.reward_model.eval()
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()

        # TODO: Batch generate
        completion, attention_mask, num_actions, action_mask = self.actor.batch_generate(
            idx, input_masks, input_lengths, self.max_new_tokens, temperature=1.0, top_k=50)

        if self.debug:
            print(" --- Make Experience --- ")
            print("completion", completion.shape)
            print("input_masks", input_masks.shape)
            print("num_actions", num_actions)
            print("action_mask", action_mask.shape)
            print("idx", idx.shape)
            print("input_masks", input_masks.shape)

        actor_log_probs = self.actor.forward_actor(
            completion,
            attention_mask,  # (B, num_actions)
            num_actions)
        sft_log_probs = self.sft_model.forward_actor(
            completion, attention_mask, num_actions)  # (B, num_actions)
        values = self.critic.forward_critic(completion,
                                            attention_mask, num_actions).view(-1, 1)  # (B, 1)
        reward = self.reward_model(completion,
                                   attention_mask)  # (B, 1)

        if self.debug:
            print("actor_log_probs", actor_log_probs.shape)
            print("sft_log_probs", sft_log_probs.shape)
            print("values", values.shape)
            print("reward", reward.shape)

        kl_penalized_reward, estimated_kl = self.kl_penalized_reward(
            reward, actor_log_probs, sft_log_probs)
        advantage = kl_penalized_reward - values

        if self.debug:
            print("kl_penalized_reward", kl_penalized_reward)
            print("advantage", advantage.shape)

        return Experience(
            completion, actor_log_probs, attention_mask, kl_penalized_reward, advantage, num_actions, estimated_kl,
            values, action_mask)

    def fit(self):
        scaler = GradScaler(enabled=self.dtype != torch.float32)
        for epoch in range(self.total_epochs):
            for step, (prompt, input_masks, input_lengths) in enumerate(pbar := tqdm(self.train_dataloader)):
                prompt, input_masks, input_lengths = prompt.to(self.device), input_masks.to(
                    self.device), input_lengths.to(self.device)
                if self.debug:
                    print("prompt", prompt.shape)
                max_input_length = torch.max(input_lengths)
                prompt = prompt[:, :max_input_length]
                if self.debug:
                    print("input_lengths", input_lengths)
                    print("prompt after", prompt.shape)

                total_steps = step + epoch * len(self.train_dataloader)

                with torch.autocast(device_type=self.device,
                                    dtype=self.dtype,
                                    enabled=self.dtype != torch.float32):
                    experience = self.make_experience(
                        prompt, input_masks, input_lengths)

                    self.actor.train()
                    curr_actor_log_probs = self.actor.forward_actor(
                        experience.completion, experience.attention_mask, experience.num_actions)

                    if self.debug:
                        print("curr_actor_log_probs",
                              curr_actor_log_probs.shape)
                        print("actor_log_probs", experience.actor_log_probs.shape)

                    actor_loss = self.actor_criterion(curr_actor_log_probs,
                                                      experience.actor_log_probs,
                                                      experience.advantage,
                                                      experience.action_mask)
                    scaler.scale(actor_loss).backward()
                    scaler.step(self.actor_optimizer)
                    # actor_loss.backward()
                    # self.actor_optimizer.step()
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_lossf = actor_loss.item()

                    self.critic.train()
                    new_values = self.critic.forward_critic(
                        experience.completion, experience.attention_mask, experience.num_actions).view(-1, 1)

                    if self.debug:
                        print("new_value", new_values.shape)
                        print("reward", experience.kl_penalized_reward.shape)

                    critic_loss = self.critic_criterion(new_values, experience.kl_penalized_reward, experience.values,
                                                        experience.action_mask)

                    scaler.scale(critic_loss).backward()
                    scaler.step(self.critic_optimizer)
                    # critic_loss.backward()
                    # self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_lossf = critic_loss.item()

                    scaler.update()

                self.writer.add_scalar('KL', experience.estimated_kl.mean(), total_steps)
                self.writer.add_scalar('mean_advantage', experience.advantage.mean(),
                                       total_steps)
                self.writer.add_scalar('mean_reward', experience.kl_penalized_reward.mean(),
                                       total_steps)
                self.writer.add_scalar('mean_value', new_values.mean(),
                                       total_steps)
                self.writer.add_scalar('Loss/actor/step', actor_lossf,
                                       total_steps)
                self.writer.add_scalar('Loss/critic/step', critic_lossf,
                                       total_steps)
                pbar.set_description(
                    f"actor loss {round(actor_lossf, 3)}, critic loss {round(critic_lossf, 3)}"
                )

                if total_steps == 50 or total_steps == 500 or (
                    total_steps != 0
                    and total_steps % self.save_freq == 0):
                    self.save_states(total_steps)

                if self.debug:
                    return

        self.save_states(None, True)


class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.max_steps = cfg.max_steps
        self.eval_freq = 1
        self.save_freq = 100  # 20000
        self.train_dataloader = iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.model = model
        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        self.finetune_method = cfg.finetune_method

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        summary(self.model, input_data=torch.ones(1, 1024).long())

        # opt_model = torch.compile(self.model)
        opt_model = self.model
        opt_model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        opt_model.train()
        step = 0

        t0 = time.time()
        while step < self.max_steps:
            x, y = next(self.train_dataloader)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                y_hat = opt_model(x)  # (B, 1)
                loss = self.criterion(y_hat, y)  # (B, 1)

            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(opt_model.parameters(),
                                               self.grad_clip)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            lossf = loss.item()

            iter_time = time.time() - t0
            t0 = time.time()
            print(
                f"step {step}, batch loss {round(lossf, 3)}, {round(1.0 / iter_time, 2)} iters/s"
            )
            writer.add_scalar('Loss/train/step', lossf, step)

            if step != 0 and step % self.save_freq == 0:
                self.save_states(step)

            step += 1

        self.save_states(step, True)


class RewardModelTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.run_name = f"rm_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.total_epochs = cfg.total_epochs
        self.eval_freq = 1
        self.save_freq = 100  # 30000
        self.model = model
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=cfg.batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        self.model = model
        self.criterion = KPairwiseLoss()
        self.finetune_method = cfg.finetune_method
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        summary(self.model, input_data=torch.ones(1, 1024).long())

        opt_model = self.model  # torch.compile(self.model)
        opt_model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        for epoch in range(self.total_epochs):
            opt_model.train()
            for step, (completions, attention_masks) in enumerate(
                pbar := tqdm(self.train_dataloader)):
                total_steps = step + epoch * len(self.train_dataloader)
                completions = completions.to(self.device)
                attention_masks = attention_masks.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    # TODO: Support K completions instead of only 2
                    # TODO: Support gradient accumulation
                    positive_scores = opt_model(
                        completions[:, 0, :],
                        attention_masks[:, 0, :])  # (B, 1)
                    negative_scores = opt_model(
                        completions[:, 1, :],
                        attention_masks[:, 1, :])  # (B, 1)
                    loss = self.criterion(
                        torch.cat((positive_scores, negative_scores),
                                  dim=-1))  # (B, 2)

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(opt_model.parameters(),
                                                   self.grad_clip)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf, total_steps)
                pbar.set_description(f"batch loss {round(lossf, 3)}")

                if total_steps != 0 and total_steps % self.save_freq == 0:
                    self.save_states(total_steps)

            if epoch % self.eval_freq == 0:
                opt_model.eval()
                with torch.no_grad():
                    tp = 0
                    total = 0
                    losses = []
                    for step, (completions, attention_masks) in enumerate(
                        self.test_dataloader):
                        completions = completions.to(self.device)
                        attention_masks = attention_masks.to(self.device)

                        positive_scores = opt_model(
                            completions[:, 0, :],
                            attention_masks[:, 0, :])  # (B, 1)
                        negative_scores = opt_model(
                            completions[:, 1, :],
                            attention_masks[:, 1, :])  # (B, 1)
                        loss = self.criterion(
                            torch.cat((positive_scores, negative_scores),
                                      dim=-1))  # (B, 2)
                        lossf = loss.item()
                        losses.append(lossf)
                        writer.add_scalar(
                            'Loss/test/step', lossf,
                            step + epoch * len(self.test_dataloader))
                        tp += torch.count_nonzero(
                            positive_scores > negative_scores)
                        total += positive_scores.shape[0]

                    acc = tp / total
                    epoch_loss = statistics.mean(losses)

                writer.add_scalar('Loss/test/epoch', epoch_loss, epoch)
                writer.add_scalar('Acc/test/epoch', acc, epoch)
                print(f'Epoch: {epoch + 1}, Test Loss: {lossf}, Acc: {acc}')

        self.save_states(total_steps, True)
