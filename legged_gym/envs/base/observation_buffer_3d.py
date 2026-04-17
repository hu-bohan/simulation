import torch

class ObservationBuffer3D:
    def __init__(self, num_envs, num_obs, include_history_steps, device):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps
        self.device = device

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs, self.include_history_steps), 
                                   device=self.device, dtype=torch.float)
        """
        使用三维数组来存储观测值(num_envs, num_obs, history_steps) \\
        0=旧 -> end=新
        """

    def reset(self, reset_ids, new_obs):
        "将新的观测值填充到所有的历史buf中"
        if len(reset_ids) == 0: 
            return
        if len(reset_ids) != new_obs.shape[0]:
            raise Exception("new_obs环境数量不等于reset_ids")
        
        self.obs_buf[reset_ids,:,:] = new_obs.unsqueeze(-1).repeat(1, 1, self.include_history_steps)

    def insert(self, new_obs):
        # 将观测值向前移动一步
        self.obs_buf[:, :, :-1] = self.obs_buf[:, :, 1:] # [:-1]不包括-1

        # 将新的观测值插入到最后
        self.obs_buf[:, :, -1] = new_obs

    def get_obs_vec(self, obs_ids):
        """Gets history of observations indexed by obs_ids.
        
        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                include_history_steps - 1 is the oldest observation.
        不要用这个
        """
        # 旧的值在左, 新的值在右
        obs = []
        for obs_id in reversed(sorted(obs_ids)):  # obs_id 大 -> 小
            slice_idx = self.include_history_steps - obs_id - 1  # slice_id 小 -> 大
            obs.append(self.obs_buf[:, :, slice_idx])  # 小先 append
        return torch.cat(obs, dim=-1)
    
    def get_latest_obs(self, time_length):
        """返回最近的n个obs\\
        shape=(env, obs, history)
        """
        return self.obs_buf[:,:,self.include_history_steps-time_length:]
    
    def get_obs_by_t(self, time_before:torch.tensor):
        """
        返回距离最新obs往前t处的obs\\
        t=0,返回end=h-1(最新)
        t=h-1,返回0(最旧)
        """
        if type(time_before) is not torch.tensor:
            # 单个数值
            time_before=torch.ones(self.num_envs,device=self.device,dtype=torch.int,requires_grad=False)*time_before
        
        if torch.any(time_before>self.include_history_steps-1) or torch.any(time_before<0):
            raise Exception("out of range")
        
        if len(time_before)!=self.num_envs:
            raise Exception("length error")
        
        env_indices=torch.arange(self.num_envs,dtype=torch.int32,device=self.device)
        buf_index=self.include_history_steps-time_before-1
        return self.obs_buf[env_indices,:,buf_index]


from legged_gym.envs.base.observation_buffer import ObservationBuffer
if __name__ == "__main__":
    num_envs = 2
    num_obs = 3
    include_history_steps = 4
    device = torch.device("cpu")

    buffer_original = ObservationBuffer(num_envs, num_obs, include_history_steps, device)
    buffer_modified = ObservationBuffer3D(num_envs, num_obs, include_history_steps, device)

    # 测试 reset 操作
    reset_idxs = [0, 1]  # 重置所有环境
    new_obs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)  # 新的观测值
    buffer_original.reset(reset_idxs, new_obs)
    buffer_modified.reset(reset_idxs, new_obs)

    # 检查 reset 后的输出是否一致
    obs_ids = [0, 1, 2, 3]  # 获取所有历史步骤
    original_output = buffer_original.get_obs_vec(obs_ids)
    modified_output = buffer_modified.get_obs_vec(obs_ids)
    assert torch.allclose(
        original_output, modified_output
    ), f"Reset 操作后输出不一致: \n{original_output}\n{modified_output}"

    # 测试 insert 操作
    new_obs = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device=device)  # 新的观测值
    buffer_original.insert(new_obs)
    buffer_modified.insert(new_obs)

    # 检查 insert 后的输出是否一致
    original_output = buffer_original.get_obs_vec(obs_ids)
    modified_output = buffer_modified.get_obs_vec(obs_ids)
    assert torch.allclose(
        original_output, modified_output
    ), f"Insert 操作后输出不一致: \n{original_output}\n{modified_output}"

    print("所有测试通过！")

    print(buffer_modified.get_latest_obs(1))
    print(buffer_modified.get_obs_by_t(0))
    print(buffer_modified.get_obs_by_t(torch.tensor([0,1])))
    