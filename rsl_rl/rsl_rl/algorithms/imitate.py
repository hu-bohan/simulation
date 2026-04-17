import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.storage.imitate_storage import Imitate_Storage

class Imitate:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 device='cpu',
                 clip_param=1
                 ):

        self.device = device
        self.learning_rate = learning_rate
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, act_dim, num_obs, step_length, device):
        self.storage = Imitate_Storage(num_envs, act_dim, num_obs, step_length, device)

    def act(self, obs):
        actions = self.actor_critic.act_inference(obs)
        return actions

    def update(self):
        mean_loss = 0
        num_updates = 0
        generator=self.storage.feed_forward_generator(self.num_mini_batches, self.num_learning_epochs)

        for sample_teacher_act, sample_student_obs in generator:
            sample_student_act = self.act(sample_student_obs)
            loss = (sample_student_act - sample_teacher_act).square().mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_loss += loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss /= num_updates

        return {
            "mean_loss":mean_loss,
        }