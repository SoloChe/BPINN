import torch
import torch.nn as nn

import numpy as np
from utils import log_gaussian_loss, gaussian, get_kl_Gaussian_divergence

class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior

        scale = (2 / self.input_dim) ** 0.5
        rho_init = np.log(np.exp((2 / self.input_dim) ** 0.5) - 1)

        self.weight_mus = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.05, 0.05))
        self.weight_rhos = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -2))

        self.bias_mus = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.05, 0.05))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-3, -2))

    def forward(self, x, sample=True):

        if sample:
            # sample gaussian noise for each weight and each bias
            weight_epsilons = self.weight_mus.data.new(self.weight_mus.size()).normal_()
            bias_epsilons = self.bias_mus.data.new(self.bias_mus.size()).normal_()

            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            bias_stds = torch.log(1 + torch.exp(self.bias_rhos))

            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons * weight_stds
            bias_sample = self.bias_mus + bias_epsilons * bias_stds
            output = torch.mm(x, weight_sample) + bias_sample

            # computing the KL loss term
            KL_loss_weight = get_kl_Gaussian_divergence(self.prior.mu, self.prior.sigma**2, self.weight_mus, weight_stds**2)
            KL_loss_bias = get_kl_Gaussian_divergence(self.prior.mu, self.prior.sigma**2, self.bias_mus, bias_stds**2)
            KL_loss = KL_loss_weight + KL_loss_bias

            return output, KL_loss
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, KL_loss

class BBP_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units):
        super(BBP_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim - 1

        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, no_units, gaussian(0, 1))
        self.layer2 = BayesLinear_Normalq(no_units, no_units, gaussian(0, 1))
        self.layer3 = BayesLinear_Normalq(no_units, no_units, gaussian(0, 1))
        self.layer4 = BayesLinear_Normalq(no_units, no_units, gaussian(0, 1))
        self.layer5 = BayesLinear_Normalq(no_units, no_units, gaussian(0, 1))
        self.layer6 = BayesLinear_Normalq(no_units, no_units, gaussian(0, 1))
        self.layer7 = BayesLinear_Normalq(no_units, output_dim, gaussian(0, 1))

        # activation to be used between hidden layers
        self.activation = nn.Tanh()

    def forward(self, x):

        KL_loss_total = 0
       
        x, KL_loss = self.layer1(x)
        x = self.activation(x)
        KL_loss_total += KL_loss

        x, KL_loss = self.layer2(x)
        x = self.activation(x)
        KL_loss_total += KL_loss

        x, KL_loss = self.layer3(x)
        x = self.activation(x)
        KL_loss_total += KL_loss

        x, KL_loss = self.layer4(x)
        x = self.activation(x)
        KL_loss_total += KL_loss

        x, KL_loss = self.layer5(x)
        x = self.activation(x)
        KL_loss_total += KL_loss

        x, KL_loss = self.layer6(x)
        x = self.activation(x)
        KL_loss_total += KL_loss

        x, KL_loss = self.layer7(x)
        KL_loss_total += KL_loss

        return x, KL_loss_total
