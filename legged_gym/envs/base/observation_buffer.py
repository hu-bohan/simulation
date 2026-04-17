import torch

class ObservationBuffer:
    def __init__(self, num_envs, num_obs, include_history_steps, device):

        self.num_envs = num_envs
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps
        self.device = device

        self.num_obs_total = num_obs * include_history_steps

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs_total, device=self.device, dtype=torch.float)

    def reset(self, reset_idxs, new_obs):
        #把第一帧的数据填充到历史
        if len(reset_idxs)==0: return
        self.obs_buf[reset_idxs] = new_obs.repeat(1, self.include_history_steps)

    def insert(self, new_obs):
        # Shift observations back.
        self.obs_buf[:, : self.num_obs * (self.include_history_steps - 1)] = self.obs_buf[:,self.num_obs : self.num_obs * self.include_history_steps]

        # Add new observation.
        # num_obs * -1 新的值在最后
        self.obs_buf[:, -self.num_obs:] = new_obs

    def get_obs_vec(self, obs_ids):
        """Gets history of observations indexed by obs_ids.
        
        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                include_history_steps - 1 is the oldest observation.
        """
        #旧的值在左,新的值在右
        obs = []
        for obs_id in reversed(sorted(obs_ids)): #obs_id大->小
            slice_idx = self.include_history_steps - obs_id - 1 # slice_id小->大
            obs.append(self.obs_buf[:, slice_idx * self.num_obs : (slice_idx + 1) * self.num_obs]) #小先append
        return torch.cat(obs, dim=-1)

