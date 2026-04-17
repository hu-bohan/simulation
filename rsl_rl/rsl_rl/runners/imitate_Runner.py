import time
import os
from collections import deque
import statistics
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import *
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.env import VecEnv
from legged_gym.utils.helpers import fix_model_structure

class imitate_Runner:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.teacher_input_dim = self.cfg["teacher_input_dim"]
        self.student_input_dim = self.env.num_obs

        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        self.teacher_actor_critic: ActorCritic = actor_critic_class(num_actor_obs=self.teacher_input_dim,
                                                        num_critic_obs=self.teacher_input_dim,
                                                        num_actions=self.env.num_actions,
                                                        actor_hidden_dims=self.cfg["teacher_actor_hidden_dims"],
                                                        critic_hidden_dims=self.cfg["teacher_critic_hidden_dims"],
                                                        activation='elu',
                                                        output_activation='tanh',
                                                        init_noise_std=1.0,
                                                        fixed_std=False,).to(self.device)
                                                        # **self.policy_cfg).to(self.device)
        student_actor_critic: ActorCritic = actor_critic_class(num_actor_obs=self.student_input_dim,
                                                        num_critic_obs=self.student_input_dim,
                                                        num_actions=self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        self.alg = Imitate(student_actor_critic, device=self.device,**self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.use_teacher_act = self.cfg["use_teacher_act"]

        # init storage and model
        self.alg.init_storage(env.num_envs, env.num_actions, self.student_input_dim, self.num_steps_per_env, self.device)
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.tol_iter = self.cfg["max_iterations"]

        self.env.reset()
    
    def learn(self,num_learning_iterations, init_at_random_ep_len=True):
        #加载教师网络
        path = self.cfg["teacher_model"]
        loaded_dict = torch.load(path)
        self.teacher_actor_critic.load_state_dict(loaded_dict['model_state_dict'])

        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        teacher_obs = self.env.get_teacher_obs()
        student_obs = self.env.get_student_obs()
        teacher_obs, student_obs = teacher_obs.to(self.device), student_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        for it in range(num_learning_iterations):
            start = time.time()
            # Rollout
            for i in range(self.num_steps_per_env):
                #collection step 收集数据
                # with torch.inference_mode():
                teacher_act = (self.teacher_actor_critic.act_inference(teacher_obs.detach())).detach()
                student_act = self.alg.act(student_obs.detach()).detach()
                self.alg.storage.insert(i,teacher_act,student_act,teacher_obs.detach(),student_obs.detach())
                act = student_act
                if self.use_teacher_act:
                    act = teacher_act
                _, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids = self.env.step(act)
                if self.env.episode_length_buf[0] == 0:
                    for i in range(8):
                        # 这里需要注意用学生还是老师的act
                        teacher_act = (self.teacher_actor_critic.act_inference(teacher_obs.detach())).detach()
                        student_act = self.alg.act(student_obs.detach()).detach()
                        act = student_act
                        if self.use_teacher_act:
                            act = teacher_act
                        _, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids = self.env.step(act)
                        teacher_obs = self.env.get_teacher_obs()

                teacher_obs = self.env.get_teacher_obs()
                student_obs = self.env.get_student_obs()
                teacher_obs, student_obs = teacher_obs.to(self.device), student_obs.to(self.device)

                stop = time.time()
                collection_time = stop - start
                # Learning step
                start = stop
            
            #train
            learning_result_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(learning_result_dict,it)
            if it % self.save_interval == 0:
                self.save(it)
    
    def log(self,learning_result_dict,it):
        print(f"iteration:{it}, Loss:{learning_result_dict['mean_loss']}")
        self.writer.add_scalar('Loss/mean_loss', learning_result_dict['mean_loss'], it)

    def load(self,path):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        scripted_model = torch.jit.script(self.alg.actor_critic)
        scripted_model.save('logs/roll_robot_r_imitate/student_history_jit.pt')

    def save(self,it):
        path=os.path.join(self.log_dir, 'model_{}.pt'.format(it))
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': iter,
            }, path)
        
    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference