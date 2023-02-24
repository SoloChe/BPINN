import numpy as np

class model:
    def __init__(self, n_x, n_y, n_timestep, para, N):
        super().__init__()
        self.para = para
        self.n_x = n_x
        self.n_y = n_y
        self.n_timestep = n_timestep
        self.N = N

       
    def simu(self, x0, h, dt):
        X = np.zeros((self.n_timestep, 4, self.n_x, self.n_y))
        X[0] = x0
        for i in range(1, self.n_timestep):
            x = self.Solver(x0, h, dt)
            x0 = x
            X[i] = x
        
        return X

    def U_xx_yy(self, D, X, h):
        g_xx_yy = np.zeros((self.n_x, self.n_y))
        for i in range(self.n_x):
            for j in range(self.n_y):
                
                # 
                if i == 0 and j == 0:
                    g_xx_yy[i,j] = (X[i+1,j] + X[i,j+1] - 2*X[i,j]) / h**2
                elif i == 0 and j == self.n_y-1:
                    g_xx_yy[i,j] = (X[i+1,j] + X[i,j-1] - 2*X[i,j]) / h**2
                elif i == self.n_x-1 and j == 0:
                    g_xx_yy[i,j] = (X[i-1,j] + X[i,j+1] - 2*X[i,j]) / h**2
                elif i == self.n_x-1 and j == self.n_y-1:
                    g_xx_yy[i,j] = (X[i-1,j] + X[i,j-1] - 2*X[i,j]) / h**2

                # 
                elif i == 0 and j != 0 and j != self.n_y-1: # up
                    g_xx_yy[i,j] = (X[i+1,j] + X[i,j-1] + X[i,j+1] - 3*X[i,j]) / h**2
                
                elif j == 0 and i != 0 and i != self.n_x-1: # left
                    g_xx_yy[i,j] = (X[i+1,j] + X[i-1,j] + X[i,j+1] - 3*X[i,j]) / h**2

                elif i == self.n_x-1 and j != 0 and j != self.n_y-1: # down
                    g_xx_yy[i,j] = (X[i,j-1] + X[i-1,j] + X[i,j+1] - 3*X[i,j]) / h**2
                
                elif j == self.n_y-1 and i != 0 and i != self.n_x-1: # right
                    g_xx_yy[i,j] = (X[i+1,j] + X[i-1,j] + X[i,j-1] - 3*X[i,j]) / h**2

                else:
                    g_xx_yy[i,j] = (X[i+1,j] + X[i-1,j] + X[i,j+1] + X[i,j-1] - 4*X[i,j]) / h**2

        return D*g_xx_yy
                                
  
    def Solver(self, X, h, dt):
        
        # X: n_states x n_x x n_y
        # initialize solution matrix

        # dynamics 
        S0 = X[0]; E0 = X[1]; I0 = X[2]; R0 = X[3]; 

        # N = S0 + E0 + I0 + R0 
        # N += 1e-6

        S1 = S0 + dt*self.U_xx_yy(self.para['nS'], S0, h) \
            - dt*(self.para['beta']*S0*I0) 
            

        E1 = E0 + dt*self.U_xx_yy(self.para['nE'], E0, h) \
            + dt*(self.para['beta']*S0*I0 - self.para['a']*E0)
                   

        I1 = I0 + dt*self.U_xx_yy(self.para['nI'], I0, h) \
           + dt*(self.para['a']*E0 - self.para['gamma']*I0 - self.para['d']*I0)
                                       
        R1 = R0 + dt*self.U_xx_yy(self.para['nR'], R0, h)  \
            + dt*(self.para['gamma']*I0) 
            
        X1 = np.concatenate([S1[None,:], E1[None,:], I1[None,:], R1[None,:]], axis=0)
                
        return X1


        