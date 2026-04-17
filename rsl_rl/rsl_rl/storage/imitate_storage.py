import torch
import numpy as np

class Imitate_Storage:

    def __init__(self, num_envs, act_dim, obs_dim, step_length, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.num_envs = num_envs
        self.step_length = step_length
        self.teacher_act = torch.zeros(num_envs, step_length, act_dim).to(device)
        # self.student_act = torch.zeros(num_envs, step_length, act_dim).to(device)
        # self.teacher_obs = torch.zeros(num_envs, step_length, obs_dim).to(device)
        self.student_obs = torch.zeros(num_envs, step_length, obs_dim).to(device)
        self.device = device

        self.step = 0
        self.num_samples = 0
    
    def insert(self, idx, teacher_act, student_act, teacher_obs, student_obs):
        """Add new states to memory."""
        # print(teacher_act.shape)
        # print(self.teacher_act.shape)
        self.teacher_act[:,idx,:] = teacher_act
        # self.student_act[:,idx,:] = student_act
        # self.teacher_obs[:,idx,:] = teacher_obs
        self.student_obs[:,idx,:] = student_obs

    def clear(self):
        self.teacher_act = None
        # self.student_act = None
        # self.teacher_obs = None
        self.student_obs = None
        torch.cuda.empty_cache()


    def feed_forward_generator(self, num_mini_batches, num_epochs):
        batch_size = self.num_envs * self.step_length
        mini_batch_size = batch_size // num_mini_batches #2048x32/4
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
        teacher_act = self.teacher_act.reshape(-1, self.teacher_act.shape[-1]) 
        student_obs = self.student_obs.reshape(-1, self.student_obs.shape[-1]) 
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]
                sample_teacher_act = teacher_act[batch_idx,:]
                sample_student_obs = student_obs[batch_idx,:]
                yield sample_teacher_act,sample_student_obs
