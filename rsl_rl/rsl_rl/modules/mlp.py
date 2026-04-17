import torch
import torch.nn as nn

class mlp(nn.Module): 
    def __init__(self,
                 mlp_input_dim,
                 mlp_hidden_dims,
                 mlp_output_dim,
                 activation = 'elu',
                 output_activation = 'elu',
                 has_output_activation = True, # 最后一层全连接有没有激活函数
                 init_last_weight = False, # 初始化最后一层权重 让初始输出接近0
                 **kwargs):
        if kwargs:
            print("mlp.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        super().__init__()

        if type(activation) is str:
            activation = get_activation(activation)
        if type(output_activation) is str:
            output_activation = get_activation(output_activation)

        mlp_layers = []

        # 第一层
        mlp_layers.append(nn.Linear(mlp_input_dim, mlp_hidden_dims[0]))
        mlp_layers.append(activation)

        for l in range(len(mlp_hidden_dims)):
            if l == len(mlp_hidden_dims) - 1:
                # 最后一层
                mlp_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_output_dim))
                if has_output_activation:
                    mlp_layers.append(output_activation) #最后一层的激活函数特殊处理
            else:
                # 中间层
                mlp_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_hidden_dims[l + 1]))
                mlp_layers.append(activation)

        if init_last_weight:
            # 初始化最后一层的参数为mean=0,std=0.1
            final_layer = mlp_layers[-2]  # 获取最后一层全连接层
            nn.init.normal_(final_layer.weight, mean=0, std=0.1)
            nn.init.constant_(final_layer.bias, 0)

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, obs):
        return self.mlp(obs)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None