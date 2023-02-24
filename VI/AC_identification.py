import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer



import scipy.io
from src.utils import log_gaussian_loss, get_kl_Gaussian_divergence, gaussian, get_kl_divergence
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: {}'.format(device))

from src.model import BBP_Model_PINN

class BBP_Model_PINN_AC(BBP_Model_PINN):
    def __init__(self, xt_lb, xt_ub, u_lb, u_ub, normal,
                    layers, loss_func, opt, local, res, activation,
                    learn_rate, batch_size, n_batches, 
                    prior, numerical, identification, device):
        super().__init__(xt_lb, xt_ub, u_lb, u_ub, normal,
                            layers, loss_func, opt, local, res, activation,
                            learn_rate, batch_size, n_batches, 
                            prior, numerical, identification, device)

    def initial_para(self):
        
        self.lambda1_mus = nn.Parameter(torch.Tensor(1).uniform_(0, 2))
        self.lambda1_rhos = nn.Parameter(torch.Tensor(1).uniform_(-3, 2))
        self.lambda2_mus = nn.Parameter(torch.Tensor(1).uniform_(0, 0.05))
        self.lambda2_rhos = nn.Parameter(torch.Tensor(1).uniform_(-3, -2))
        self.lambda3_mus = nn.Parameter(torch.Tensor(1).uniform_(0, 0.05))
        self.lambda3_rhos = nn.Parameter(torch.Tensor(1).uniform_(-3, -2))
        self.alpha = nn.Parameter(torch.Tensor(1).uniform_(0, 0.5))
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(0, 0.5))

        self.network.register_parameter('lambda1_mu', self.lambda1_mus)
        self.network.register_parameter('lambda2_mu', self.lambda2_mus)
        self.network.register_parameter('lambda3_mu', self.lambda3_mus)
        self.network.register_parameter('lambda1_rho', self.lambda1_rhos)
        self.network.register_parameter('lambda2_rho', self.lambda2_rhos)
        self.network.register_parameter('lambda3_rho', self.lambda3_rhos)
        self.network.register_parameter('alpha', self.alpha)
        self.network.register_parameter('beta', self.beta)

        self.prior_lambda1 = self.prior
        self.prior_lambda2 = self.prior
        self.prior_lambda3 = self.prior

  

    def net_F(self, x, t, u, lambda1_sample, lambda2_sample, lambda3_sample):
        lambda_1 = torch.exp(lambda1_sample)        
        lambda_2 = torch.exp(lambda2_sample)
        lambda_3 = torch.exp(lambda3_sample)

        # u, _, _ = self.net_U(x, t)
        u = u*(self.u_ub-self.u_lb) + self.u_lb # reverse scaling

        u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                                    retain_graph=True,
                                    create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                                    retain_graph=True,
                                    create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    retain_graph=True,
                                    create_graph=True)[0]

        F = u_t - lambda_1*u_xx + lambda_2*u**3 - lambda_3*u
        # 0.0001 5 5
        return F

    def net_F_inference(self, x, t, u):
       
        # u, _, _ = self.net_U(x, t)
        u = u*(self.u_ub-self.u_lb) + self.u_lb # reverse scaling

        u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                                    retain_graph=True,
                                    create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                                    retain_graph=True,
                                    create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    retain_graph=True,
                                    create_graph=True)[0]
        F = u_t - 0.0001*u_xx + 5*u**3 - 5*u

        return F

    def fit(self, X, t, U, n_samples):
        self.network.train()

        # X = torch.tensor(self.X, requires_grad=True).float().to(device)
        # t = torch.tensor(self.t, requires_grad=True).float().to(device)
        U = (U-self.u_lb)/(self.u_ub-self.u_lb) # scaling
        # U = (U-self.u_mean)/self.u_std # scaling

        # reset gradient and total loss
        self.optimizer.zero_grad()

      
        fit_loss_F_total = 0
        fit_loss_U_total = 0
        KL_loss_total = 0
        for _ in range(n_samples):
            
            if self.identification:
                lambda1_epsilons = self.lambda1_mus.data.new(self.lambda1_mus.size()).normal_()
                lambda1_stds = torch.log(1 + torch.exp(self.lambda1_rhos))
                lambda2_epsilons = self.lambda2_mus.data.new(self.lambda2_mus.size()).normal_()
                lambda2_stds = torch.log(1 + torch.exp(self.lambda2_rhos))
                lambda3_epsilons = self.lambda3_mus.data.new(self.lambda3_mus.size()).normal_()
                lambda3_stds = torch.log(1 + torch.exp(self.lambda3_rhos))

                lambda1_sample = self.lambda1_mus + lambda1_epsilons * lambda1_stds
                lambda2_sample = self.lambda2_mus + lambda2_epsilons * lambda2_stds
                lambda3_sample = self.lambda3_mus + lambda3_epsilons * lambda3_stds

               
                KL_loss_lambda1 = get_kl_Gaussian_divergence(self.prior_lambda1.mu, self.prior_lambda1.sigma**2, self.lambda1_mus, lambda1_stds**2)
                KL_loss_lambda2 = get_kl_Gaussian_divergence(self.prior_lambda2.mu, self.prior_lambda2.sigma**2, self.lambda2_mus, lambda2_stds**2)
                KL_loss_lambda3 = get_kl_Gaussian_divergence(self.prior_lambda3.mu, self.prior_lambda3.sigma**2, self.lambda3_mus, lambda3_stds**2)
                
                u_pred, log_noise_u, KL_loss_para = self.net_U(X, t)
                KL_loss_total += (KL_loss_para + KL_loss_lambda1 + KL_loss_lambda2 + KL_loss_lambda3)
                f_pred = self.net_F(X, t, u_pred, lambda1_sample, lambda2_sample, lambda3_sample)
            else:
                u_pred, log_noise_u, KL_loss_para = self.net_U(X, t)
                f_pred = self.net_F_inference(X, t, u_pred)
                KL_loss_total += KL_loss_para

            

            # calculate fit loss based on mean and standard deviation of output
            fit_loss_U_total += self.loss_func(u_pred, U, log_noise_u.exp(), self.network.output_dim)
            fit_loss_F_total += self.loss_func(f_pred, torch.zeros_like(f_pred), (self.alpha.exp()+1)*torch.ones_like(f_pred), self.network.output_dim)

        # KL_loss_total = KL_loss_para 
        # minibatches and KL reweighting
        self.coef = self.alpha.exp()+1
        self.coef2 = self.beta.exp()
        KL_loss_total = KL_loss_total/self.n_batches/n_samples
        total_loss = (KL_loss_total + fit_loss_U_total + fit_loss_F_total) / (n_samples*X.shape[0])
        
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return fit_loss_U_total/n_samples, fit_loss_F_total/n_samples, KL_loss_total, total_loss


if __name__ == '__main__':

    data = scipy.io.loadmat('../Data/AC.mat')

    t = data['tt'].flatten()[:,None] # 201 x 1
    x = data['x'].flatten()[:,None] # 512 x 1 
    Exact_ = np.real(data['uu']).T # 201 x 512

    noise = 0.1
    Exact = Exact_ + noise*np.std(Exact_)*np.random.randn(201, 512)

    X, T = np.meshgrid(x,t) # 201 x 512
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # 102912 x 2
    u_star = Exact.flatten()[:,None]  # 102912 x 1

    N_u_test = 10000
    idx_test = np.random.choice(X_star.shape[0], N_u_test, replace = False)
    X_test = X_star[idx_test,:]
    u_test = u_star[idx_test,:]

    # Domain bounds of x, t
    xt_lb = X_star.min(0)
    xt_ub = X_star.max(0)

    # training data
    N_u = 1000
    idx = np.random.choice(X_star.shape[0], N_u, replace = False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]

    u_min = u_train.min(0)
    u_max = u_train.max(0)

    X = torch.tensor(X_u_train[:,0:1], requires_grad = True, device = device).float()
    t = torch.tensor(X_u_train[:,1:2], requires_grad = True, device = device).float()
    U = torch.tensor(u_train, requires_grad = True, device = device).float()

    #%% model 
    local = True
    identification = True
    numerical = False

    learn_rate = 2e-3
    opt = torch.optim.AdamW
    loss_func = log_gaussian_loss
    
    

    # prior = spike_slab_2GMM(mu1 = 0, mu2 = 0, sigma1 = 0.1, sigma2 = 0.0005, pi = 0.75)
    prior = gaussian(0, 1)

    num_epochs = 40000 

    n_batches = 1
    batch_size = len(X_u_train)

    res = True
    normal = True
    n_hidden = 50
    activation = nn.Tanh()

    
    if res:
        layers = [2, n_hidden, n_hidden, n_hidden, 2] # res
    else:
        layers = [2, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, 2]

    pinn_model = BBP_Model_PINN_AC(xt_lb, xt_ub, u_min, u_max, normal,
                                        layers, loss_func, opt, local, res, activation,
                                        learn_rate, batch_size, n_batches,
                                        prior, numerical, identification, device)
    #%%

    n_fit = 10
    comment = f'AC n_sample = {N_u} n_fit = {n_fit} res = {res}'
    writer = SummaryWriter(comment = comment)

    fit_loss_U_train = np.zeros(num_epochs)
    fit_loss_F_train = np.zeros(num_epochs)
    KL_loss_train = np.zeros(num_epochs)
    loss = np.zeros(num_epochs)


    for i in range(num_epochs):

        EU, EF, KL_loss, total_loss = pinn_model.fit(X, t, U, n_samples = n_fit)
        
        fit_loss_U_train[i] = EU.item()
        fit_loss_F_train[i] = EF.item()
        KL_loss_train[i] = KL_loss.item()
        loss[i] = total_loss.item()

        writer.add_scalar("loss/total_loss", loss[i], i)
        writer.add_scalar("loss/U_loss", fit_loss_U_train[i], i)
        writer.add_scalar("loss/F_loss", fit_loss_F_train[i], i)
        writer.add_scalar("loss/KL_loss", KL_loss_train[i], i)
        

        # if i % 2000 == 0:
        #     F_test = net.sample_F(X_u_test_25)
        #     fig, axs = plt.subplots(2, 2, figsize=(20, 8))
        #     axs[0,0].hist(F_test[:,0])
        #     axs[0,1].hist(F_test[:,100])
        #     axs[1,0].hist(F_test[:,150])
        #     axs[1,1].hist(F_test[:,255])
        #     plt.savefig('./plots/epoch{}.tiff'.format(i))


        if i % 10 == 0 or i == num_epochs - 1:

            print("Epoch: {:5d}/{:5d}, total loss = {:.3f}, Fit loss U = {:.3f}, Fit loss F = {:.3f}, KL loss = {:.3f}".format(i + 1, num_epochs, 
                loss[i], fit_loss_U_train[i], fit_loss_F_train[i], KL_loss_train[i]))

            if i % 100 == 0 or i == num_epochs - 1:
                samples_star, _ = pinn_model.predict(X_test, 50, pinn_model.network)
                u_pred_star = samples_star.mean(axis = 0)
                error_star = np.linalg.norm(u_test-u_pred_star, 2)/np.linalg.norm(u_test, 2)

                samples_train, _ = pinn_model.predict(X_u_train, 50, pinn_model.network)
                u_pred_train = samples_train.mean(axis=0)
                
                error_train = np.linalg.norm(u_train-u_pred_train, 2)/np.linalg.norm(u_train, 2)


                writer.add_scalars("loss/train_test", {'train':error_train, 'test':error_star}, i)
                print("Epoch: {:5d}/{:5d}, error_test = {:.5f}, error_train = {:.5f}".format(i+1, num_epochs, error_star, error_train))

            if identification:
                lambda1_mus = np.exp(pinn_model.lambda1_mus.item())
                lambda1_stds = torch.log(1 + torch.exp(pinn_model.lambda1_rhos)).item()
                
                lambda2_mus = np.exp(pinn_model.lambda2_mus.item())
                lambda2_stds = torch.log(1 + torch.exp(pinn_model.lambda2_rhos)).item()

                lambda3_mus = np.exp(pinn_model.lambda3_mus.item())
                lambda3_stds = torch.log(1 + torch.exp(pinn_model.lambda3_rhos)).item()
                print("Epoch: {:5d}/{:5d}, lambda1_mu = {:.5f}, lambda2_mu = {:.3f}, lambda3_mu = {:.3f}, lambda1_std = {:.3f}, lambda2_std = {:.3f}, lambda3_std = {:.3f}".format(i + 1, num_epochs,
                                                                                                                                                lambda1_mus, lambda2_mus, lambda3_mus,
                                                                                                                                                lambda1_stds, lambda2_stds, lambda3_stds))
            
            print("Epoch: {:5d}/{:5d}, alpha = {:.5f}, beta = {:.5f}".format(i+1, num_epochs, pinn_model.coef.item(), \
                                                                            pinn_model.coef2.item()))
            print()

    writer.close()
    #%%

    x = data['x'].flatten()[:,None]
    X_u_test_25 = np.hstack([x, 0.25*np.ones_like((x))]); u_test_25 = Exact[50]; u_mean_25 = Exact_[50]
    X_u_test_50 = np.hstack([x, 0.50*np.ones_like((x))]); u_test_50 = Exact[100]; u_mean_50 = Exact_[100]
    X_u_test_75 = np.hstack([x, 0.75*np.ones_like((x))]); u_test_75 = Exact[150]; u_mean_75 = Exact_[150]


    def get_res(X):
        samples, noises = pinn_model.predict(X, 100, pinn_model.network)
        u_pred = samples.mean(axis = 0)

        aleatoric = (noises**2).mean(axis = 0)**0.5
        epistemic = samples.var(axis = 0)**0.5
        total_unc = (aleatoric**2 + epistemic**2)**0.5
        return u_pred.ravel(), aleatoric.ravel(), epistemic.ravel(), total_unc.ravel()


    x = x.ravel()
    u_pred_25, ale_25, epi_25, total_unc_25 = get_res(X_u_test_25)
    u_pred_50, ale_50, epi_50, total_unc_50 = get_res(X_u_test_50)
    u_pred_75, ale_75, epi_75, total_unc_75 = get_res(X_u_test_75)

    u_pred, ale, epi, total_unc = get_res(X_star)

    #%% plot
    import matplotlib
    import matplotlib.gridspec as gridspec
    matplotlib.rc('xtick', labelsize=12) 
    matplotlib.rc('ytick', labelsize=12) 
    matplotlib.rc('font', size=18)


    plt.figure(figsize = (20, 5))
    gs0 = gridspec.GridSpec(1, 3)
    ax = plt.subplot(gs0[0,0])
    # ax.scatter(x, u_test_25, s = 10, marker = 'x', color = 'black', alpha = 0.5, label = 'Exact')
    ax.plot(x, u_mean_25, 'b-', linewidth = 2, label = 'Prediction')
    ax.plot(x, u_pred_25, 'r--', linewidth = 2, label = 'Prediction')
    ax.fill_between(x, u_pred_25-2*total_unc_25, u_pred_25+2*total_unc_25, color = 'g', alpha = 0.5, label = 'Epistemic + Aleatoric')
    ax.set_xlim([-1.2, 1.2])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$')

    ax = plt.subplot(gs0[0,1])
    # ax.scatter(x, u_test_50, s = 10, marker = 'x', color = 'black', alpha = 0.5, label = 'Exact')
    ax.plot(x, u_mean_50, 'b-', linewidth = 2, label = 'Prediction')
    ax.plot(x, u_pred_50, 'r--', linewidth = 2, label = 'Prediction')
    ax.fill_between(x, u_pred_50-2*total_unc_50, u_pred_50+2*total_unc_50, color = 'g', alpha = 0.5, label = 'Epistemic + Aleatoric')
    ax.set_xlim([-1.2, 1.2])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.5$')

    ax = plt.subplot(gs0[0,2])
    # ax.scatter(x, u_test_75, s = 10, marker = 'x', color = 'black', alpha = 0.5, label = 'Exact')
    ax.plot(x, u_mean_75, 'b-', linewidth = 2, label = 'Prediction')
    ax.plot(x, u_pred_75, 'r--', linewidth = 2, label = 'Prediction')
    ax.fill_between(x, u_pred_75-2*total_unc_75, u_pred_75+2*total_unc_75, color = 'g', alpha = 0.5, label = 'Epistemic + Aleatoric')
    ax.set_xlim([-1.2, 1.2])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.75$')
    
    # ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    # plt.savefig('./plots/prediction_AC.tiff')


    
    plt.figure(figsize=(8,3))
    h = plt.imshow(np.abs(u_pred.reshape((201, 512)).T - Exact_.T), interpolation='nearest', cmap='rainbow', 
                    extent=[0, 1, -1, 1], 
                    origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.plot(X_u_train[:,1], X_u_train[:,0], 'kx', markersize = 2)
    # plt.savefig('./plots/prediction_AC_data.tiff')

    
    
# %% save model
model_path = './model_save'
file_name = f'/AC_{N_u}_01.pth'
pinn_model.save_model(i, model_path + file_name)

# %%
