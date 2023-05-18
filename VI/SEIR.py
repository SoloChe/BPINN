import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
import argparse



import scipy.io
from src.utils import log_gaussian_loss, get_kl_Gaussian_divergence, gaussian, get_kl_divergence
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: {}'.format(device))

from src.model import BBP_Model_PINN
from src.data_simu_COVID import model



def state_init(n_x, n_y): 
    
    S_xy_0 = np.abs(np.random.normal(150, 30, size=(n_x, n_y)).astype(int))
    N = np.sum(S_xy_0)
    
    print(f'number of people = {N}')

    # Infected
    I_xy_0 = np.zeros((n_x, n_y))
    I_ind = []
    for _ in range(5):
        ia, ib = np.random.randint(0,n_x-1), np.random.randint(0,n_y-1)
        I_ind.append((ia, ib))
        I_xy_0[ia, ib] +=  int(S_xy_0[ia, ib]/5)
        S_xy_0[ia, ib] -= int(S_xy_0[ia, ib]/5)

    n_infected = np.sum(I_xy_0)
    print(f'number of infected people = {n_infected}')


    # Recoverd
    R_xy_0 = np.zeros((n_x, n_y)) 
    E_xy_0 = np.zeros((n_x, n_y)) 

    u_0 = np.concatenate((np.expand_dims(S_xy_0,axis=(0)), 
                            np.expand_dims(E_xy_0,axis=(0)), 
                            np.expand_dims(I_xy_0,axis=(0)),
                            np.expand_dims(R_xy_0,axis=(0)),
                            ), axis=0)/N

    assert u_0.shape == (4, n_x, n_y)
    return u_0, N

def get_simu_data(u_0, h, dt, n_x, n_y, n_timestep, para_simu, N):
   
    m = model(n_x, n_y, n_timestep, para_simu, N)
    u_t = m.simu(u_0, h, dt)

    print(u_t.shape)

    X_simu = np.concatenate( [np.sum( u_t[i], axis=(1,2))[None,...] for i in range(n_timestep)], axis=0)
    return u_t, X_simu
    
    
class BBP_Model_PINN_SEIR(BBP_Model_PINN):
    def __init__(self, xt_lb, xt_ub, u_lb, u_ub, normal,
                    layers, loss_func, opt, local, res, activation,
                    learn_rate, batch_size, n_batches, 
                    prior, numerical, identification, device):
        super().__init__(xt_lb, xt_ub, u_lb, u_ub, normal,
                            layers, loss_func, opt, local, res, activation,
                            learn_rate, batch_size, n_batches, 
                            prior, numerical, identification, device)

    def initial_para(self):

        self.para_mus = nn.ParameterDict({
                                    # 'nS_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 1)),
                                    # 'nE_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 1)),
                                    # 'nI_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 1)),
                                    # 'nR_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 1)),
                                    'beta_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 0.1)), 
                                    'a_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 1)), 
                                    'gamma_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 1)),
                                    'd_mu':nn.Parameter(torch.Tensor(1).uniform_(0, 1))
                                    })

        self.para_rhos = nn.ParameterDict({
                                    # 'nS_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2)),
                                    # 'nE_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2)),
                                    # 'nI_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2)),
                                    # 'nR_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2)),
                                    'beta_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2)), 
                                    'a_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2)), 
                                    'gamma_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2)),
                                    'd_rho':nn.Parameter(torch.Tensor(1).uniform_(-3, 2))
                                    })
        
        for (key_mu, value_mu), (key_rho, value_rho)  in zip(self.para_mus.items(), self.para_rhos.items()):
            self.network.register_parameter(key_mu, value_mu)
            self.network.register_parameter(key_rho, value_rho)

        self.alpha = nn.Parameter(torch.Tensor(1).uniform_(0, 1))
        self.network.register_parameter('alpha', self.alpha)

        self.pde_para_prior = [gaussian(0, 1), gaussian(0, 1), gaussian(0, 1), gaussian(0, 1)]

    

    def net_U(self, x, y, t):
        xt = torch.cat((x, y, t), dim=1)
        xt = 2*(xt-self.xt_lb) / (self.xt_ub-self.xt_lb) - 1
        out, KL_loss = self.network(xt)
        u = out[:, 0:4]
        log_noise_u = out[:, 4:]
        return u, log_noise_u, KL_loss

    def net_F(self, x, y, t, para_samples, N):

        u, _, _ = self.net_U(x, y, t)
        if self.normal:
            u = u*(self.u_ub-self.u_lb) + self.u_lb # reverse scaling

        U_t = []; U_xx = []; U_yy = []
        
        for i in range(4):
            u_t = torch.autograd.grad(u[:,i:i+1], t, 
                                        torch.ones_like(u[:,i:i+1]),
                                        retain_graph=True,
                                        create_graph=True)[0]
            
            u_x = torch.autograd.grad(u[:,i:i+1], x, 
                                        torch.ones_like(u[:,i:i+1]), 
                                        retain_graph=True,
                                        create_graph=True)[0]

            u_y = torch.autograd.grad(u[:,i:i+1], y, 
                                        torch.ones_like(u[:,i:i+1]), 
                                        retain_graph=True,
                                        create_graph=True)[0]

            u_xx = torch.autograd.grad(u_x, x, 
                                        torch.ones_like(u[:,i:i+1]),
                                        retain_graph=True,
                                        create_graph=True)[0]
           
            u_yy = torch.autograd.grad(u_y, y, 
                                        torch.ones_like(u[:,i:i+1]),
                                        retain_graph=True,
                                        create_graph=True)[0]
            U_t.append(u_t)
            U_xx.append(u_xx)
            U_yy.append(u_yy)

        S = u[:,0:1]; E = u[:,1:2]; I = u[:,2:3]; 

        # nS, nE, nI, nR = para_samples['nS_mu'].exp(), para_samples['nE_mu'].exp(), para_samples['nI_mu'].exp(), para_samples['nR_mu'].exp()
        nS, nE, nI, nR = 0.1, 0.1, 0.1, 0.1
        beta, a, gamma, d = para_samples['beta_mu'].exp(), para_samples['a_mu'].exp(), para_samples['gamma_mu'].exp(), para_samples['d_mu'].exp()
        # beta, a, gamma, d = 0.005, 0.2, 0.1, 0.1
        N_S = nS*(U_xx[0] + U_yy[0]) - beta*S*I*N
        N_E = nE*(U_xx[1] + U_yy[1]) + beta*S*I*N - a*E
        N_I = nI*(U_xx[2] + U_yy[2]) + a*E - gamma*I - d*I
        N_R = nR*(U_xx[3] + U_yy[3]) + gamma*I 

        F_S = U_t[0] - N_S; 
        F_E = U_t[1] - N_E; 
        F_I = U_t[2] - N_I; 
        F_R = U_t[3] - N_R
        return torch.cat((F_S, F_E, F_I, F_R), dim=1)



    def fit(self, data_loader, n_samples, N):
        self.network.train()
        
        for XYT, U in data_loader:
            fit_loss_F_total = 0
            fit_loss_U_total = 0
            KL_loss_total = 0
           
            X, Y, t, U = XYT[:,0:1].to(self.device), XYT[:,1:2].to(self.device), XYT[:,2:3].to(self.device), U.to(self.device)
            if self.normal:
                U = (U-self.u_lb)/(self.u_ub-self.u_lb) # scaling
            # reset gradient and total loss
            self.optimizer.zero_grad()

            for _ in range(n_samples):

                if self.identification:
                    para_samples = {}

                    for (name, value_mu), (_, value_rho), prior  in zip(self.para_mus.items(), self.para_rhos.items(), self.pde_para_prior):
                        ep = value_mu.data.new(value_mu.size()).normal_()
                        std = torch.log(1 + torch.exp(value_rho))
                        sample = value_mu  
                        + ep * std
                        para_samples[name] = sample

                        KL_loss_total += get_kl_Gaussian_divergence(prior.mu, prior.sigma**2, value_mu, std**2)
                     
                    u_pred, log_noise_u, KL_loss_model_para = self.net_U(X, Y, t)
                    f_pred = self.net_F(X, Y, t, para_samples, N)
                    KL_loss_total += KL_loss_model_para
               
                fit_loss_U_total += self.loss_func(u_pred, U, log_noise_u.exp())
                # print(fit_loss_U_total) 
                fit_loss_F_total += self.loss_func(f_pred, torch.zeros_like(f_pred), (self.alpha.exp()+1)*torch.ones_like(f_pred))

            # KL_loss_total = KL_loss_para 
            # minibatches and KL reweighting
            KL_loss_total = KL_loss_total/self.n_batches/n_samples
            total_loss = (KL_loss_total + fit_loss_U_total + fit_loss_F_total) / (n_samples*4*X.shape[0])
            
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return fit_loss_U_total/n_samples, fit_loss_F_total/n_samples, KL_loss_total, total_loss

    def predict(self, xyt, n_sample, best_net):
        xyt = torch.tensor(xyt, requires_grad = True).float().to(self.device)
        xyt = 2*(xyt-self.xt_lb)/(self.xt_ub-self.xt_lb) - 1

        self.network.eval()
        samples = [] 
        noises = []
        for _ in range(n_sample):
            out_pred, _ = best_net(xyt)
            u_pred = out_pred[:,0:4]
            noise_u = out_pred[:,4:].exp()

            if self.normal:
                u_pred = u_pred*(self.u_ub-self.u_lb) + self.u_lb # reverse scaling
                noise_u = noise_u*(self.u_ub-self.u_lb)

            samples.append(u_pred.detach().cpu().numpy()[...,None])
            noises.append(noise_u.detach().cpu().numpy()[...,None])
        return np.concatenate(samples, axis=2), np.concatenate(noises, axis=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train PINN on SEIR Model')
    parser.add_argument('--n_x', default=10, type=int, help='size x')
    parser.add_argument('--n_y', default=10, type=int, help='size y')
    parser.add_argument('--h', default=0.5, type=float, help='dx and dy')
    parser.add_argument('--dt', default=0.5, type=float, help='dt')
    parser.add_argument('--n_timestep', default=200, type=int, help='n_dt')
    parser.add_argument('--epochs', default=50000, type=int, help='epochs')
    parser.add_argument('--warm_up', default=50, type=int, help='warm-up epochs')
    parser.add_argument('--resume', default=False, type=bool, help='resume or not')
    parser.add_argument('--save', default=True, type=bool, help='save or not')
    args = parser.parse_args('') # if .ipynb
    # args = parser.parse_args()
    model_path = './model_save'
    n_x = args.n_x
    n_y = args.n_y
    u_0, N = state_init(n_x, n_y) 

    h = args.h
    dt = args.dt
    n_timestep = args.n_timestep

    para_simu = {'nS':0.1, 'nE':0.1, 'nI':0.1, 'nR':0.1,  
                    'beta':0.005, 'a':0.2, 'gamma':0.1, 'd':0.1}
    
    if not args.resume:

        u_t, X_simu = get_simu_data(u_0, h, dt, n_x, n_y, n_timestep, para_simu, N)
        # np.save('../Data/simu_data_seir', u_t)
        plt.figure()    
        plt.plot(X_simu[:,0], 'r-', linewidth = 2)
        plt.plot(X_simu[:,1], 'g-', linewidth = 2)
        plt.plot(X_simu[:,2], 'b-', linewidth = 2)
        plt.plot(X_simu[:,3], 'k-', linewidth = 2)
        plt.legend(['S', 'E', 'I', 'R'])
        # plt.savefig('./pic/simu_data.tiff')
    else:
        u_t = np.load('../Data/simu_data_seir')
        X_simu = np.concatenate( [np.sum( u_t[i], axis=(1,2))[None,...] for i in range(n_timestep)], axis=0)

    #%%
   
    x = np.array([h*i for i in range(n_x)], dtype=float)
    y = np.array([h*i for i in range(n_y)], dtype=float)
    t = np.array([dt*i for i in range(n_timestep)], dtype=float)

    Y, X = np.meshgrid(y,x) 
    X, Y = X.reshape((-1,1)), Y.reshape((-1,1))


    data_loc = np.hstack([X, Y])
    data_loc = np.tile(data_loc, (n_timestep,1))
    data_t = np.vstack( [np.reshape([t[i]]*n_x*n_y, (-1,1)) for i in range(n_timestep)])
    data = np.hstack((data_loc, data_t))
    u = np.vstack( [np.hstack([u_t[j,i,:,:].reshape((-1,1)) for i in range(4)]) for j in range(n_timestep)] )

    n_train = 4000
    n_test = 5000
    idx = np.array(range(n_x*n_y*n_timestep))
    train_idx = np.random.choice(n_x*n_y*n_timestep, n_train, replace=False)
    test_idx = np.setdiff1d(idx, train_idx)[:n_test]

    data_train = data[train_idx, ...]
    data_test = data[test_idx, ...]
    u_train = u[train_idx, ...]
    u_test = u[test_idx, ...]

    u_lb = u_train.min(0)
    u_ub = u_train.max(0)
    xyt_lb = data_train.min(0) 
    xyt_ub = data_train.max(0)  
   
    batch_size = 2000
    tensor_xyt = torch.tensor(data[train_idx, ...], requires_grad=True).float()
    tensor_u = torch.tensor(u[train_idx,...]).float()

    dataset = TensorDataset(tensor_xyt, tensor_u) 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    

    #%% model 
    local = True
    identification = True
    numerical = False

    learn_rate = 2e-3
    opt = torch.optim.AdamW
    loss_func = log_gaussian_loss
    
    

    # prior = spike_slab_2GMM(mu1 = 0, mu2 = 0, sigma1 = 0.1, sigma2 = 0.0005, pi = 0.75)
    prior = gaussian(0, 1)

    num_epochs = 100000

    n_batches = int(n_train / batch_size)

    res = True 
    normal = True
    n_hidden = 50
    activation = nn.Tanh()

    
    if res:
        layers = [3, n_hidden, n_hidden, n_hidden, 8] # res
    else:
        layers = [3, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, 8]

    pinn_model = BBP_Model_PINN_SEIR(xyt_lb, xyt_ub, u_lb, u_ub, normal,
                                    layers, loss_func, opt, local, res, activation,
                                    learn_rate, batch_size, n_batches,
                                    prior, numerical, identification, device)
    #%%

    n_fit = 10
    comment = f'SEIR n_sample = {n_train} n_fit = {n_fit} res = {res}'
    writer = SummaryWriter(comment = comment)

    fit_loss_U_train = np.zeros(num_epochs)
    fit_loss_F_train = np.zeros(num_epochs)
    KL_loss_train = np.zeros(num_epochs)
    loss = np.zeros(num_epochs)


    for i in range(num_epochs):

        EU, EF, KL_loss, total_loss = pinn_model.fit(data_loader, n_samples = n_fit, N = N)
        
        fit_loss_U_train[i] = EU.item()
        fit_loss_F_train[i] = EF.item()
        KL_loss_train[i] = KL_loss.item()
        loss[i] = total_loss.item()

        writer.add_scalar("loss/total_loss", loss[i], i)
        writer.add_scalar("loss/U_loss", fit_loss_U_train[i], i)
        writer.add_scalar("loss/F_loss", fit_loss_F_train[i], i)
        writer.add_scalar("loss/KL_loss", KL_loss_train[i], i)
        

        if i % 10 == 0 or i == num_epochs - 1:

            print("Epoch: {:5d}/{:5d}, total loss = {:.3f}, Fit loss U = {:.3f}, Fit loss F = {:.3f}, KL loss = {:.3f}, alpha = {:.5f}".format(i + 1, num_epochs, 
                loss[i], fit_loss_U_train[i], fit_loss_F_train[i], KL_loss_train[i], pinn_model.alpha.exp().item()+1))
            
            if i % 100 == 0 or i == num_epochs - 1:
                samples_star, _ = pinn_model.predict(data_test, 50, pinn_model.network)
                u_pred_star = samples_star.mean(axis = 2)
                error_star = np.linalg.norm(u_test-u_pred_star, 2)/np.linalg.norm(u_test, 2)

                samples_train, _ = pinn_model.predict(data_train, 50, pinn_model.network)
                u_pred_train = samples_train.mean(axis = 2)
                
                error_train = np.linalg.norm(u_train-u_pred_train, 2)/np.linalg.norm(u_train, 2)


                writer.add_scalars("loss/train_test", {'train':error_train, 'test':error_star}, i)
                print("Epoch: {:5d}/{:5d}, error_test = {:.5f}, error_train = {:.5f}".format(i+1, num_epochs, error_star, error_train))

            if i % 5000 == 0:
                
                file_name = f'/SEIR_{n_train}.pth'
                pinn_model.save_model(i, model_path + file_name)

            if identification:
                # printing = (i+1, num_epochs, 
                #             pinn_model.para_mus['nS_mu'].exp().item(), pinn_model.para_mus['nE_mu'].exp().item(),
                #             pinn_model.para_mus['nI_mu'].exp().item(), pinn_model.para_mus['nR_mu'].exp().item(),
                #             pinn_model.para_mus['beta_mu'].exp().item(), pinn_model.para_mus['a_mu'].exp().item(),
                #             pinn_model.para_mus['gamma_mu'].exp().item(), pinn_model.para_mus['d_mu'].exp().item())
                # print("Epoch: {:5d}/{:5d}, S = {:.3f}, E = {:.3f}, I = {:.3f}, R = {:.3f}, beta = {:.3f}, a = {:.3f}, gamma = {:.3f}, d = {:.3f}".format(*printing))

                # std_ = lambda rho: torch.log(1 + torch.exp(rho))
                # printing = (i+1, num_epochs, 
                #             std_(pinn_model.para_rhos['nS_rho']).item(), std_(pinn_model.para_rhos['nE_rho']).item(),
                #             std_(pinn_model.para_rhos['nI_rho']).item(), std_(pinn_model.para_rhos['nR_rho']).item(),
                #             std_(pinn_model.para_rhos['beta_rho']).item(), std_(pinn_model.para_rhos['a_rho']).item(),
                #             std_(pinn_model.para_rhos['gamma_rho']).item(), std_(pinn_model.para_rhos['d_rho']).item())
                # print("Epoch: {:5d}/{:5d}, Ss = {:.3f}, Es = {:.3f}, Is = {:.3f}, Rs = {:.3f}, betas = {:.3f}, as = {:.3f}, gammas = {:.3f}, ds = {:.3f}".format(*printing))

                printing = (i+1, num_epochs, 
                            pinn_model.para_mus['beta_mu'].exp().item(), pinn_model.para_mus['a_mu'].exp().item(),
                            pinn_model.para_mus['gamma_mu'].exp().item(), pinn_model.para_mus['d_mu'].exp().item())
                print("Epoch: {:5d}/{:5d}, beta = {:.3f}, a = {:.3f}, gamma = {:.3f}, d = {:.3f}".format(*printing))

                std_ = lambda rho: torch.log(1 + torch.exp(rho))
                printing = (i+1, num_epochs, 
                            std_(pinn_model.para_rhos['beta_rho']).item(), std_(pinn_model.para_rhos['a_rho']).item(),
                            std_(pinn_model.para_rhos['gamma_rho']).item(), std_(pinn_model.para_rhos['d_rho']).item())
                print("Epoch: {:5d}/{:5d}, betas = {:.3f}, as = {:.3f}, gammas = {:.3f}, ds = {:.3f}".format(*printing))
            
            
            print()

    writer.close()
    
    

# %%
def plot(data):
    pred, _ = pinn_model.predict(data, 50, pinn_model.network)
    pred = pred.mean(axis = 2)
    
    temp = int(n_x*n_y)
    XT_pred = np.zeros((n_timestep, temp, 4)) 

    for i in range(n_timestep): 
        XT_pred[i] = pred[temp*i:temp*(i+1),:]
        

    X_simu_pred = np.sum(XT_pred, axis=1) 

    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    axs[0,0].plot(X_simu[:,0], 'b-', linewidth = 2)
    axs[0,0].plot(X_simu_pred[:,0], 'k--', linewidth = 2)
    axs[0,0].legend(['Exact', 'Predict'])
    axs[0,0].set_title('S')

    axs[0,1].plot(X_simu[:,1], 'b-', linewidth = 2)
    axs[0,1].plot(X_simu_pred[:,1], 'k--', linewidth = 2)
    axs[0,1].set_title('E')

    axs[1,0].plot(X_simu[:,2], 'b-', linewidth = 2)
    axs[1,0].plot(X_simu_pred[:,2], 'k--', linewidth = 2)
    axs[1,0].set_title('I')

    axs[1,1].plot(X_simu[:,3], 'b-', linewidth = 2)
    axs[1,1].plot(X_simu_pred[:,3], 'k--', linewidth = 2)
    axs[1,1].set_title('R') 

    plot(data)

     

# %%