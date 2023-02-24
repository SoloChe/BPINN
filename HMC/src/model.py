import torch
import torch.nn as nn
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, layers, activation):
        super().__init__()

        n_hidden = len(layers) 
        self.activation = activation
        layer_list = []

        for i in range(n_hidden-2):

            linear = nn.Linear(layers[i], layers[i+1])

            nn.init.xavier_normal_(linear.weight.data, gain=1.0)
            nn.init.zeros_(linear.bias.data)

            layer_list.append(
                ('layer_%d' % i, linear)
            )
            layer_list.append(
                ('activation_%d' % i, self.activation())
            )
        
        linear = torch.nn.Linear(layers[n_hidden-2], layers[n_hidden-1])
        nn.init.xavier_normal_(linear.weight.data, gain=1.0)
        nn.init.zeros_(linear.bias.data)

        layer_list.append(('layer_%d' % (n_hidden-2), linear))
        
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

class Res_block(nn.Module):
    def __init__(self, activation, input_dim, output_dim) -> None:
        super().__init__()

        self.activation = activation
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.layer2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
    
        res = x
        out = self.layer1(x)
        out = self.activation(out)
        # 
        out = self.layer2(out)
        out = self.activation(out + res)
        return out

class MLP_res(nn.Module):
    def __init__(self, activation, layers) -> None:
        super().__init__()

        self.activation = activation
        layer_list = []
        n_layer = len(layers)

        input_layer = nn.Linear(layers[0], layers[1])
        layer_list.append(input_layer)
        layer_list.append(self.activation)

        for i in range(1, n_layer-2):
            res_block = Res_block(activation,layers[i], layers[i+1])
            layer_list.append(res_block)

        last_layer = nn.Linear(layers[n_layer - 2], layers[n_layer - 1])
        layer_list.append(last_layer)

        self.model = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.model(x)
        return x



