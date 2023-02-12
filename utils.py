import torch
import numpy as np

def log_gaussian_loss(output, target, sigma, no_dim): # negative

    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    
    return -(log_coeff + exponent).sum()

# def log_gaussian_loss(output, target, sigma, no_dim, sum_reduce=True):
#     exponent = -0.5*(target - output)**2/sigma**2
#     log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    
#     if sum_reduce:
#         return -(log_coeff + exponent).sum()
#     else:
#         return -(log_coeff + exponent)


def get_kl_divergence(weights, prior, varpost):
    prior_loglike = prior.loglike(weights)

    varpost_loglike = varpost.loglike(weights)

    # varpost_lik = varpost_loglik.exp()

    return (varpost_loglike - prior_loglike).sum() # weight is sampled directly from q(w|theta)
    # return (varpost_lik*(varpost_loglik - prior_loglik)).sum() # weight is sampled from the range of possible w

def get_kl_Gaussian_divergence(prior_mus, prior_cov, varpost_mus, varpost_cov):
    # KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()

    KL_loss = 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * varpost_cov.numel()
    KL_loss += 0.5 * (varpost_cov / prior_cov).sum()
    KL_loss +=  0.5 * ((varpost_mus - prior_mus) ** 2 / prior_cov).sum()
    return KL_loss
class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglike(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        return (exponent + log_coeff).sum()



if __name__ == '__main__':

    # test
    loss = log_gaussian_loss(1, 1, torch.tensor([0.1]), 1)
    print(loss)

