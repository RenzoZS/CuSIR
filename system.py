import cupy as cp
import matplotlib.pyplot as plt


forces_lamb = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 lamb, int16 L' ,
    'float64 fS, float64 fI',
    '''
    int x = i % L;
    int y = (int) i/L;

    fS = -S[i]*I[i] + S[(x+1)%L + L*y] + S[(x-1+L)%L+L*y] + S[x + L*((y+1)%L)] + S[x + L*((y-1+L)%L)] - 4*S[i];
    fI = S[i]*I[i] - lamb*I[i] + I[(x+1)%L + L*y] + I[(x-1+L)%L+L*y] + I[x + L*((y+1)%L)] + I[x + L*((y-1+L)%L)] - 4*I[i];
    ''',
    'forces_lamb')

forces_R = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 R, int16 L',
    'float64 fS, float64 fI',
    '''
    int x = i % L;
    int y = (int) i/L;

    fS = -R*S[i]*I[i];
    fI = R*S[i]*I[i] - I[i] + (I[(x+1)%L + L*y] + I[(x-1+L)%L+L*y] + I[x + L*((y+1)%L)] + I[x + L*((y-1+L)%L)] - 4*I[i]);
    
    ''',
    'forces_R')

forces_beta = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 beta, float64 gamma, float64 D, int16 L',
    'float64 fS, float64 fI',
    '''
    int x = i % L;
    int y = (int) i/L;

    fS = -beta*S[i]*I[i];
    fI = beta*S[i]*I[i] - gamma*I[i] + D*(I[(x+1)%L + L*y] + I[(x-1+L)%L+L*y] + I[x + L*((y+1)%L)] + I[x + L*((y-1+L)%L)] - 4*I[i]);
    ''',
    'forces_beta')

forces_RH = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, raw float64 H, float64 vx, float64 vy, float64 R, int16 L',
    'float64 fS, float64 fI',
    '''
    int x = i % L;
    int y = (int) i/L;

    fS = -R*S[i]*I[i] ;

    float reaction = R*S[i]*I[i] - I[i];
    float diffusion = I[(x+1)%L + L*y] + I[(x-1+L)%L+L*y] + I[x + L*((y+1)%L)] + I[x + L*((y-1+L)%L)] - 4*I[i];
    float convective_height = -((H[(x+1)%L + L*y] - H[(x-1+L)%L+L*y])*(I[(x+1)%L + L*y] - I[(x-1+L)%L+L*y]) + (H[x + L*((y+1)%L)] - H[x + L*((y-1+L)%L)])*(I[x + L*((y+1)%L)] - I[x + L*((y-1+L)%L)]))/4;
    float convective_wind = (vx[i]*(I[(x+1)%L + L*y] - I[(x-1+L)%L+L*y]) + vy[i]*(I[x + L*((y+1)%L)] - I[x + L*((y-1+L)%L)]))/2;
    fI =  reaction + diffusion + 34.5*convective_height + convective_wind;
    ''',
    'forces_R_H')

forces = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, raw float64 H, float64 beta, float64 gamma, float64 D, float64 vx, float64 vy, int16 L',
    'float64 fS, float64 fI',
    '''
    int x = i % L;
    int y = (int) i / L;

    fS = - beta * S[i] * I[i];

    float reaction = beta * S[i] * I[i] - gamma * I[i];
    float diffusion = D * (I[(x+1)%L + L*y] + I[(x-1+L)%L+L*y] + I[x + L*((y+1)%L)] + I[x + L*((y-1+L)%L)] - 4*I[i]);
    float convective_height = -((H[(x+1)%L + L*y] - H[(x-1+L)%L+L*y])*(I[(x+1)%L + L*y] - I[(x-1+L)%L+L*y]) + (H[x + L*((y+1)%L)] - H[x + L*((y-1+L)%L)])*(I[x + L*((y+1)%L)] - I[x + L*((y-1+L)%L)]))/4;
    float convective_wind = -(vx*(I[(x+1)%L + L*y] - I[(x-1+L)%L+L*y]) + vy*(I[x + L*((y+1)%L)] - I[x + L*((y-1+L)%L)]))/2;

    fI = reaction + diffusion + convective_height + convective_wind;
    ''',
    'forces')

forces_0 = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 beta, float64 gamma, float64 D, float64 vx, float64 vy, int16 L',
    'float64 fS, float64 fI',
    '''
    int x = i % L;
    int y = (int) i / L;

    fS = - beta * S[i] * I[i];

    float reaction = beta * S[i] * I[i] - gamma * I[i];
    float diffusion = D * (I[(x+1)%L + L*y] + I[(x-1+L)%L+L*y] + I[x + L*((y+1)%L)] + I[x + L*((y-1+L)%L)] - 4*I[i]);
    float convective_wind = -(vx*(I[(x+1)%L + L*y] - I[(x-1+L)%L+L*y]) + vy*(I[x + L*((y+1)%L)] - I[x + L*((y-1+L)%L)]))/2;

    fI = reaction + diffusion + convective_wind;
    ''',
    'forces')

smooth = cp.ElementwiseKernel(
    'int16 L, raw float64 X', 'float64 Y',
    '''

    int x = i % L;
    int y = (int) i/L;
    
    Y = (X[(x+1)%L + L*y] + X[(x-1+L)%L+L*y] + X[x + L*((y+1)%L)] + X[x + L*((y-1+L)%L)] + X[i])/5;

    ''','smooth')

class System:

    def __init__(self,N=1024):
        self.N = N
        self.S = cp.ones((self.N,self.N))
        self.I = cp.zeros((self.N,self.N))
        self.fS = cp.zeros((N,N))
        self.fI = cp.zeros((N,N))
        self.beta = 0
        self.gamma = 0
        self.D = 0
        self.R = 0
        self.H = 0
        self.vx = 0 
        self.vy = 0
        self.t = 0
        self.t_it = 0
        self.dt = .1

    def reset(self):
        self.S = cp.ones((self.N,self.N))
        self.I = cp.zeros((self.N,self.N))
        self.t = 0
        self.t_it = 0
    

    def set_initial_conditions(self,S0,I0):
        self.S = S0
        self.I = I0
    
    def set_plane_initial_conditions(self,x0 = 1):
        self.I[:,x0] = 1
        self.S[:,x0] = 0
    
    def set_point_initial_conditions(self,x0,y0):
        self.I[x0,y0] = 1
        self.S[x0,y0] = 0
    
    def set_dic_beta(self, beta0 = 1., p = 0):
        self.beta = cp.random.choice([0,beta0],size=(self.N,self.N),p=[p,1-p])

    def set_smooth_beta(self,beta0 = 1., p = 0 , n = 1):
        self.beta = cp.random.choice([0,beta0],size=(self.N,self.N),p=[p,1-p])
        for _ in range(n):
            beta_aux = cp.copy(self.beta)
            smooth(self.N,beta_aux,self.beta)

    def set_dic_smooth_beta(self,beta0 = 1., p = 0 , n = 1):
        self.beta = cp.random.choice([0,beta0],size=(self.N,self.N),p=[p,1-p])
        for _ in range(n):
            beta_aux = cp.copy(self.beta)
            smooth(self.N,beta_aux,self.beta)
        
        beta_mean = cp.mean(self.beta)
        self.beta[self.beta>beta_mean] = beta0
        self.beta[self.beta<beta_mean] = 0

    def set_dic_R(self,R0=2.,p = 0):
        self.R = cp.random.choice([0,R0],size=(self.N,self.N),p = [p,1-p])

    def set_smooth_R(self,R0=2.,p=0,n=1):
        self.R = cp.random.choice([0,R0],size=(self.N,self.N),p = [p,1-p])
        for _ in range(n):
            R_aux = cp.copy(self.R)
            smooth(self.N,R_aux,self.R)
    
    def set_dic_smooth_R(self,R0=2.,p=0,n=1):
        self.R = cp.random.choice([0,R0],size=(self.N,self.N),p = [p,1-p])
        
        for _ in range(n):
            R_aux = cp.copy(self.R)
            smooth(self.N,R_aux,self.R)

        R_mean = cp.mean(self.R)
        self.R[self.R>R_mean] = R0
        self.R[self.R<R_mean] = 0
    
    def set_hyper_R(self,R0=2.,p=0,K=1):
        self.R = cp.random.choice([-R0,R0],size=(self.N,self.N),p = [p,1-p])
        # circle mask
        x = cp.arange(self.N)
        y = cp.arange(self.N)
        cx = self.N//2
        cy = self.N//2
        mask = ((x[cp.newaxis,:]-cx)**2 + (y[:,cp.newaxis]-cy)**2 <= K**2)*((x[cp.newaxis,:]-cx)**2 + (y[:,cp.newaxis]-cy)**2 > 0)

        while True:
            Rm = self.R.mean()
            R_fft = cp.fft.fft2(self.R)
            R_fft = cp.fft.fftshift(R_fft)
            R_fft[mask] = 0
            R_fft = cp.fft.ifftshift(R_fft) 
            R_aux = cp.fft.ifft2(R_fft).real
            R_aux[R_aux>Rm] = R0
            R_aux[R_aux<Rm] = -R0
            if (self.R == R_aux).all():
                break
            self.R = R_aux
    
    def set_egg_H(self,k=1,h=1):
        Y,X = cp.mgrid[0:self.N:1,0:self.N:1]
        self.H = h*(cp.cos(k*2*cp.pi/(self.N-1)*X) + cp.cos(k*2*cp.pi/(self.N-1)*Y))/2
    
    def fft_R(self):
        return cp.fft.fftshift(cp.fft.fft2(self.R))

    def fft_beta(self):
        return cp.fft.fftshift(cp.fft.fft2(self.beta))
    
    def update(self):
        forces(self.S,self.I,self.H,self.beta,self.gamma,self.D,self.vx,self.vy,self.N,self.fS,self.fI)
        self.S += self.dt*self.fS
        self.I += self.dt*self.fI
        self.t += self.dt
        self.t_it += 1
    
    def update_0(self):
        forces_0(self.S,self.I,self.beta,self.gamma,self.D,self.vx,self.vy,self.N,self.fS,self.fI)
        self.S += self.dt*self.fS
        self.I += self.dt*self.fI
        self.t += self.dt
        self.t_it += 1
    
    def update_beta(self):
        forces_beta(self.S,self.I,self.beta,self.gamma,self.D,self.N,self.fS,self.fI)
        self.S += self.dt*self.fS
        self.I += self.dt*self.fI
        self.t += self.dt
        self.t_it += 1
    
    def update_RH(self):
        forces_RH(self.S,self.I,self.H,self.R,self.N,self.fS,self.fI)
        self.S += self.dt*self.fS
        self.I += self.dt*self.fI
        self.t += self.dt
        self.t_it += 1
    
    def get_S1(self):
        return cp.asnumpy(self.S[:,int(self.N*.2):int(self.u_cm())-int(self.N*.2)].mean())
    
    def get_p(self):
        return cp.asnumpy(cp.count_nonzero(self.beta)/self.N)

    def rigid_x(self):
        self.S[:,0] = self.S[:,-1] = self.I[:,0] = self.I[:,-1] = 0
    
    def rigid_y(self):
        self.S[0,:] = self.S[-1,:] = self.I[0,:] = self.I[-1,:] = 0
    
    def u(self):
        return cp.argmax(self.I,axis=1)
    
    def u_cm(self):
        return cp.asnumpy(cp.mean(self.u()))
    
    def u_sigma(self):
        return cp.asnumpy(cp.std(self.u()))
    
    def u_fft(self):
        return cp.square(cp.abs(cp.fft.fft(self.u())))
    
    def f_I(self):
        return cp.roll(cp.mean(self.I,axis=0),int(-self.u_cm())+self.N//2)
    
    def ff_I(self):
        return cp.mean(self.I,axis=0)
    
    def I_max(self):
        return cp.asnumpy(cp.mean(cp.max(self.I,axis=1)))
    


        
    
    