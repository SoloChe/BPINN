import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import log_gaussian_loss, gaussian, get_kl_Gaussian_divergence, get_kl_divergence
import priors

class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior

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
            varpost_weight = gaussian(self.weight_mus, weight_stds)
            varpost_bias = gaussian(self.bias_mus, bias_stds)
            KL_loss_weight = get_kl_divergence(weight_sample, self.prior, varpost_weight)
            KL_loss_bias = get_kl_divergence(bias_sample, self.prior, varpost_bias)
            # KL_loss_weight = get_kl_Gaussian_divergence(self.prior.mu, self.prior.sigma**2, self.weight_mus, weight_stds**2)
            # KL_loss_bias = get_kl_Gaussian_divergence(self.prior.mu, self.prior.sigma**2, self.bias_mus, bias_stds**2)
            KL_loss = KL_loss_weight + KL_loss_bias

            return output, KL_loss
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output

class BayesLinear_Normalq_local(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq_local, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.05, 0.05))
        self.W_p = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -2))
        self.b_mu = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.05, 0.05))
        self.b_p = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-3, -2))

    def forward(self, x, sample=True):
        #         print(self.training)

        if sample:
            # calculate std
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            act_W_mu = torch.mm(x, self.W_mu)  # self.W_mu + std_w * eps_Ws
            act_W_std = torch.sqrt(torch.mm(x.pow(2), std_w.pow(2)))

            eps_W = self.W_mu.data.new(act_W_std.size()).normal_(mean = 0, std = 1)
            eps_b = self.b_mu.data.new(std_b.size()).normal_(mean = 0, std = 1)

            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
            act_b_out = self.b_mu + std_b * eps_b

            output = act_W_out + act_b_out

            ## todo
            KL_loss_weight = get_kl_Gaussian_divergence(self.prior.mu, self.prior.sigma**2, self.W_mu, std_w**2)
            KL_loss_bias = get_kl_Gaussian_divergence(self.prior.mu, self.prior.sigma**2, self.b_mu, std_b**2)
            KL_loss = KL_loss_weight + KL_loss_bias
            return output, KL_loss
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output

class BBP_Model(nn.Module):
    def __init__(self, layers, prior, local = False):
        super(BBP_Model, self).__init__()

        n_layer = len(layers)
        self.output_dim = layers[-1] - 1
        self.layer_list = []
        

        self.activation = nn.Tanh()
        for i in range(n_layer - 2):
            if local:
                b_layer = BayesLinear_Normalq_local(layers[i], layers[i+1], prior)
            else:
                b_layer = BayesLinear_Normalq(layers[i], layers[i+1], prior)
            self.layer_list.append(b_layer)
            
        # last layer
        if local:
            self.last_layer = BayesLinear_Normalq_local(layers[n_layer - 2], layers[n_layer - 1], gaussian(0, 1))
        else:
            self.last_layer = BayesLinear_Normalq(layers[n_layer - 2], layers[n_layer - 1], gaussian(0, 1))

        self.layer_list_torch = nn.ModuleList(self.layer_list)
     
    def forward(self, x):
        KL_loss_total = 0
        for layer in self.layer_list_torch:
            x, KL_loss = layer(x)
            x = self.activation(x)
            KL_loss_total += KL_loss

        x, KL_loss = self.last_layer(x)
        KL_loss_total += KL_loss
        return x, KL_loss_total

class BBP_Model_PINN:
    def __init__(self, xt_lb, xt_ub, u_lb, u_ub,
                 layers, loss_func, opt, local,
                 learn_rate, batch_size, n_batches,
                 prior, device):

        self.device = device
        self.xt_lb = torch.from_numpy(xt_lb).float().to(self.device)
        self.xt_ub = torch.from_numpy(xt_ub).float().to(self.device)
        self.u_lb = torch.from_numpy(u_lb).float().to(self.device)
        self.u_ub = torch.from_numpy(u_ub).float().to(self.device)


        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.n_batches = n_batches

        self.prior = prior

        self.network = BBP_Model(layers, self.prior, local)
        self.loss_func = loss_func
        
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr = self.learn_rate)
        
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr = 1e-3, 
        #                                     steps_per_epoch = no_batches, epochs = num_epochs)
        
        self.initial_para()
        self.network = self.network.to(self.device)
        self.optimizer = opt(self.network.parameters(), lr = self.learn_rate)
    

    def initial_para(self):
        raise NotImplementedError('initialize parameters in PDE/ODE first.')     
    def net_F(self):
        raise NotImplementedError('You need to define physical law.')
    def fit(self):
        raise NotImplementedError('You need to define fitting.')
    
    def net_U(self, x, t):
        xt = torch.cat((x,t), dim=1)
        xt = 2*(xt-self.xt_lb)/(self.xt_ub-self.xt_lb) - 1
        out, KL_loss = self.network(xt)

        u = out[:, 0:1]
        log_noise_u = out[:, 1:2]
        return u, log_noise_u, KL_loss

    def predict(self, xt, n_sample, best_net):
        xt = torch.tensor(xt, requires_grad = True).float().to(self.device)
        xt = 2*(xt-self.xt_lb)/(self.xt_ub-self.xt_lb) - 1

        self.network.eval()
        samples = []
        noises = []
        for i in range(n_sample):
            out_pred, _ = best_net(xt)
            u_pred = out_pred[:,0:1]
            noise_u = out_pred[:,1:2].exp()

            u_pred = u_pred*(self.u_ub-self.u_lb) + self.u_lb # reverse scaling
            noise_u = noise_u*(self.u_ub-self.u_lb)

            samples.append(u_pred.detach().cpu().numpy())
            noises.append(noise_u.detach().cpu().numpy())
        return np.array(samples), np.array(noises)


