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

forces_R0 = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 R0, int16 L' ,
    'float64 fS, float64 fI',
    '''
    int x = i % L;
    int y = (int) i/L;

    fS = -R0*S[i]*I[i] + S[(x+1)%L + L*y] + S[(x-1+L)%L+L*y] + S[x + L*((y+1)%L)] + S[x + L*((y-1+L)%L)] - 4*S[i];
    fI = R0*S[i]*I[i] - I[i] + I[(x+1)%L + L*y] + I[(x-1+L)%L+L*y] + I[x + L*((y+1)%L)] + I[x + L*((y-1+L)%L)] - 4*I[i];
    ''',
    'forces_R0')

class System:

    def __init__(self,N=1024):
        self.N = N
        self.S = []
        self.I = []
        self.fS = cp.zeros((N,N))
        self.fI = cp.zeros((N,N))
        self.R0 = []
        self.t = 0
        self.t_it = 0
    
    def set_initial_conditions(self,S0,I0):
        self.S = S0
        self.I = I0
    
    def set_plane_initial_conditions(self,x0 = 1):
        self.S = cp.ones((self.N,self.N))
        self.I = cp.zeros((self.N,self.N))
        self.fS = cp.zeros((self.N,self.N))
        self.fI = cp.zeros((self.N,self.N))
        self.I[:,x0] = 1
        self.S[:,x0] = 0
        self.t = 0
        self.t_it = 0
    
    def set_dic_R0(self,R_0=5.,p=0):
        self.R0 = cp.random.choice([0,R_0],size=(self.N,self.N),p = [p,1-p])
    
    def update(self,tstep=.1):
        forces_R0(self.S,self.I,self.R0,self.N,self.fS,self.fI)
        self.S += tstep*self.fS
        self.I += tstep*self.fI
        self.t += tstep
        self.t_it += 1

    def rigid_x(self):
        self.S[:,0] = self.S[:,-1] = self.I[:,0] = self.I[:,-1] = 0
    
    def rigid_y(self):
        self.S[0,:] = self.S[-1,:] = self.I[0,:] = self.I[-1,:] = 0
    
    def u(self):
        return cp.argmax(self.I,axis=1)
    
    def u_cm(self):
        return cp.mean(self.u())
    
    def u_sigma(self):
        return cp.std(self.u())
    
    def u_fft(self):
        return cp.square(cp.abs(cp.fft.fft(self.u())))
    
    def f_I(self):
        return cp.roll(cp.mean(self.I,axis=0),int(-self.u_cm())+self.N//2)
    
    def mean_Ymax(self):
        return cp.mean(cp.max(self.I,axis=1))
    
    def plot_u(self):
        plt.plot(cp.asnumpy(self.u()))

    def plot_I(self):
        plt.imshow(cp.asnumpy(self.I))
        plt.axis('off')
        plt.show()
    
    def plot_S(self):
        plt.matshow(cp.asnumpy(self.S))
    
    