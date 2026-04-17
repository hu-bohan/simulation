# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple
import math

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = get_quat_yaw(quat)
    return quat_apply(quat_yaw, vec)

def get_quat_yaw(quat):
    "获得yaw旋转分量"
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_yaw

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

@torch.jit.script
def torch_rand_float_1d(lower, upper, shape, device):
    # type: (float, float, int, str) -> Tensor
    return (upper - lower) * torch.rand(shape, device=device) + lower

def vector_apply_yaw(tensor, yaw_angle):
    """
    tensor: 形状为(num, 3)的张量
    yaw_angle: 绕z轴的旋转角度rad(num)
    返回原向量在坐标系内旋转后的坐标
    """
    cos_theta = torch.cos(yaw_angle)
    sin_theta = torch.sin(yaw_angle)

    x1=tensor[:,0]
    y1=tensor[:,1]
    z1=tensor[:,2]
    x2=x1*(cos_theta)+y1*(-sin_theta)
    y2=x1*(sin_theta)+y1*(cos_theta)
    z2=z1

    return torch.stack([x2,y2,z2],dim=1)

def fit_plane(px:torch.Tensor, py:torch.Tensor, pz:torch.Tensor):
    """
    拟合平面方程 ax + by + z = c \\
    返回方程系数[a,b,c], 归一化的法向量(默认朝下)
    """
    device = px.device
    # 求解方程 ax + by - c = -z
    # A=[x,y,-1], x=[a,b,c], b=[-z]
    A = torch.stack([px, py, -torch.ones_like(px)], dim=-1)
    coefficients = torch.linalg.lstsq(A, -pz).solution # 求矩阵方程Ax=b的最小二乘解

    # 提取平面方程参数
    a=coefficients[:,0]
    b=coefficients[:,1]
    c=coefficients[:,2]

    # 法向量[a,b,1]
    nz=1*torch.ones(coefficients.shape[0],device=device)
    normal_vector = torch.stack([a, b, nz],dim=1)
    normal_vector_n = normal_vector/ torch.norm(normal_vector, dim=-1, keepdim=True)

    return [a,b,c],-normal_vector_n

def linear_interp(low,high,x):
    # x=[0,1]
    return low+(high-low)*x

def linear_interp_v2(y_low,y_high,x_low,x_high,x):
    r=(x-x_low)/(x_high-x_low)
    return linear_interp(y_low,y_high,r)

def exp_interp(low,high,x):
    base=10
    h = math.log(high,base)
    l = math.log(low,base)
    m = linear_interp(l,h,x)
    return math.pow(base,m)

def exp_interp_v2(y_low,y_high,x_low,x_high,x):
    r=(x-x_low)/(x_high-x_low)
    return exp_interp(y_low,y_high,r)