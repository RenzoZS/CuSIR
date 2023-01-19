import cupy as cp
from datetime import date

#cuda kernels

forces_0 = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, raw float64 H, float64 beta, float64 gamma, float64 DI,float64 DS, float64 vx, float64 vy, uint32 Lx, uint32 Ly',
    'float64 fS, float64 fI',
    '''
    int x = i % Lx;
    int y = (int) i / Lx;

    float diffusionS = DS * (S[(x+1)%Lx + y*Lx] + S[(x-1+Lx)%Lx + y*Lx] + S[x + ((y+1)%Ly)*Lx] + S[x + ((y-1+Ly)%Ly)*Lx] - 4*S[i]);
    fS = - beta * S[i] * I[i] +diffusionS;

    float reaction = beta * S[i] * I[i] - gamma * I[i];
    float diffusionI = DI * (I[(x+1)%Lx + Lx*y] + I[(x-1+Lx)%Lx+Lx*y] + I[x + Lx*((y+1)%Ly)] + I[x + Lx*((y-1+Ly)%Ly)] - 4*I[i]);
    float convective_height = -((H[(x+1)%Lx + Lx*y] - H[(x-1+Lx)%Lx+Lx*y])*(I[(x+1)%Lx + Lx*y] - I[(x-1+Lx)%Lx+Lx*y]) + (H[x + Lx*((y+1)%Ly)] - H[x + Lx*((y-1+Ly)%Ly)])*(I[x + Lx*((y+1)%Ly)] - I[x + Lx*((y-1+Ly)%Ly)]))/4;
    float convective_wind = -(vx*(I[(x+1)%Lx + Lx*y] - I[(x-1+Lx)%Lx+Lx*y]) + vy*(I[x + Lx*((y+1)%Ly)] - I[x + Lx*((y-1+Ly)%Ly)]))/2;

    fI = reaction + diffusionI + convective_height + convective_wind;
    ''',
    'forces')


forces_1 = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 beta, float64 gamma, float64 DI,float64 DS, float64 alpha, uint32 Lx, uint32 Ly',
    'float64 fS, float64 fI',
    '''
    int x = i % Lx;
    int y = (int) i/Lx;

    fS = -beta*S[i]*powf(I[i],alpha) + DS* (S[(x+1)%Lx + y*Lx] + S[(x-1+Lx)%Lx + y*Lx] + S[x + ((y+1)%Ly)*Lx] + S[x + ((y-1+Ly)%Ly)*Lx] - 4*S[i]);
    fI = beta*S[i]*powf(I[i],alpha) - gamma*I[i] + DI*(I[(x+1)%Lx + Lx*y] + I[(x-1+Lx)%Lx + Lx*y] + I[x + Lx*((y+1)%Ly)] + I[x + Lx*((y-1+Ly)%Ly)] - 4*I[i]);
    ''',
    'forces_beta')

forces_2 = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 beta, float64 gamma, float64 DI, float64 DS, float64 vx, float64 vy, uint32 Lx, uint32 Ly',
    'float64 fS, float64 fI',
    '''
    int x = i % Lx;
    int y = (int) i / Lx;

    fS = - beta * S[i] * I[i] + DS * (S[(x+1)%Lx + y*Lx] + S[(x-1+Lx)%Lx + y*Lx] + S[x + ((y+1)%Ly)*Lx] + S[x + ((y-1+Ly)%Ly)*Lx] - 4*S[i]);

    float reaction = beta * S[i] * I[i] - gamma * I[i];
    float diffusion = DI * (I[(x+1)%Lx + Lx*y] + I[(x-1+Lx)%Lx+Lx*y] + I[x + Lx*((y+1)%Ly)] + I[x + Lx*((y-1+Ly)%Ly)] - 4*I[i]);
    float convective_wind = -(vx*(I[(x+1)%Lx + Lx*y] - I[(x-1+Lx)%Lx+Lx*y]) + vy*(I[x + Lx*((y+1)%Ly)] - I[x + Lx*((y-1+Ly)%Ly)]))/2;

    fI = reaction + diffusion + convective_wind;
    ''',
    'forces')

smooth = cp.ElementwiseKernel(
    'uint32 Lx, uint32 Ly, raw float64 X', 'float64 Y',
    '''
    int x = i % Lx;
    int y = (int) i/Lx;
    
    Y = (X[(x+1)%Lx + Lx*y] + X[(x-1+Lx)%Lx+Lx*y] + X[x + Lx*((y+1)%Ly)] + X[x + Lx*((y-1+Ly)%Ly)] + X[i])/5;

    ''','smooth')

euler = cp.ElementwiseKernel(
    'float64 fS, float64 fI, float64 dt','float64 S, float64 I',
    '''
    S = S + dt*fS;
    I = I + dt*fI;
    ''',
    'euler')


class system:
    '''
    Class to solve the SIR model on a 2D grid using the finite difference method and the Euler method. 

    Parameters

    ----------

    Lx : int
        Number of grid points in the x direction
    
    Ly : int
        Number of grid points in the y direction

    beta : float or array
        Infection rate

    gamma : float or array

    DI : float
        Diffusion coefficient for the infected population
    
    DS : float
        Diffusion coefficient for the susceptible population

    alpha : float
        Power law exponent for the infection rate
    
    H : array
        Height field
    
    vx : float or array
        Wind velocity in the x direction
    
    vy : float or array
        Wind velocity in the y direction
    
    dt : float
        Time step

    '''




    def __init__(self,Lx=1024,Ly=1024):

        '''
        Initialize the system on a 2D grid.

        Parameters

        ----------

        Lx : int
            Number of grid points in the x direction

        Ly : int
            Number of grid points in the y direction

        '''
        
        self.Lx = Lx
        self.Ly = Ly
        self.S = cp.ones((Ly,Lx))
        self.I = cp.zeros((Ly,Lx))
        self.fS = cp.zeros((Ly,Lx))
        self.fI = cp.zeros((Ly,Lx))
        self.beta = 0
        self.gamma = 0
        self.alpha = 1
        self.DI = 0
        self.DS = 0
        self.H = 0
        self.vx = 0 
        self.vy = 0
        self.t = 0
        self.t_it = 0
        self.dt = .01
        self.date = str(date.today())

    def __del__(self):
        del self.S
        del self.I
        del self.fS
        del self.fI
        del self.beta
        return None

    def reset(self):
        '''
        Reset the system to the initial conditions (S=1, I=0) and set the time to 0. Does not reset the parameters.

        '''

        self.S = cp.ones((self.Ly,self.Lx))
        self.I = cp.zeros((self.Ly,self.Lx))
        self.fS = cp.zeros((self.Ly,self.Lx))
        self.fI = cp.zeros((self.Ly,self.Lx))
        self.t = 0
        self.t_it = 0
        self.date = str(date.today())
    
    def set_initial_conditions(self,S0,I0):
        '''
        Set the initial conditions for the susceptible and infected populations.

        Parameters

        ----------

        S0 : array
            Initial condition for the susceptible population

        I0 : array
            Initial condition for the infected population
        '''

        self.S = S0
        self.I = I0
    
    def set_plane_initial_conditions(self,x0 = 1, I0 = 1):
        '''
        Set the initial conditions for the susceptible and infected populations to a plane wave.

        Parameters

        ----------

        x0 : int
            Position of the plane wave in the x direction
        
        I0 : float
            Amplitude of the plane wave
        '''

        self.I[:,x0] = I0
        self.S[:,x0] = 1 - I0

    def set_tilded_initial_conditions(self,m = 1, I0 = 1):
        '''
        Set the initial conditions for the susceptible and infected populations to a tilted plane wave.

        Parameters

        ----------

        m : float
            Ly/(m*Lx) is the slope of the plane wave
        
        I0 : float
            Amplitude of the plane wave
        '''
        

        if m == 0:
            self.set_plane_initial_conditions(I0 = I0)
        elif m>0:
            n = int(cp.ceil(m*(self.Lx-2)))    
            dn = 1/m
            j = 0
            for i in range(1,self.Ly-1):
                self.I[i,j] = I0 
                self.S[i,:j+1] = 1 - I0 
                if i >= (j+1)*dn:
                    j += 1
            self.I[0,0] = I0
            self.S[:,0] = 1 - I0
            self.I[-1,n-1] = I0
            self.S[-1,:n] = 1 - I0
        else:
            n = int(cp.ceil(-m*(self.Lx-2)))
            dn = -1/m
            j = 0
            for i in range(1,self.Ly):
                self.I[-i,j] = I0 
                self.S[-i,:j+1] = 1 - I0 
                if i-1 >= (j+1)*dn:
                    j += 1
            self.I[-1,0] = I0
            self.S[:,0] = 1 - I0
            self.I[0,n-1] = I0
            self.S[0,:n] = 1 - I0
    
    def set_point_initial_conditions(self,x0,y0, I0 = 1):
        '''
        Set the initial conditions for the susceptible and infected populations to a point.

        Parameters

        ----------

        x0 : int
            Position of the point in the x direction
        
        y0 : int
            Position of the point in the y direction

        I0 : float
            Amplitude of the point
        '''

        self.I[x0,y0] = I0
        self.S[x0,y0] = 1 - I0
    
    def set_dic_beta(self, beta0 = 1., p = 0):
        '''
            Set the infection rate to a dicotomic distribution.

            Parameters

            ----------

            beta0 : float
                Non zero value of the infection rate
            
            p : float
                Fraction of the grid where the infection rate is zero
        '''

        self.beta = cp.random.rand(self.Ly,self.Lx)
        self.beta[self.beta>p] = 1
        self.beta[self.beta<p] = 0
        self.beta = self.beta*beta0

    def set_smooth_beta(self,beta0 = 1., p = 0 , n = 1):
        '''
            Set the infection rate to a smooth distribution. Takes the averages value to first neighbours over a dicotomic distribution.

            Parameters

            ----------

            beta0 : float
                Non zero value of the infection rate
            
            p : float
                Fraction of the grid where the infection rate is zero

            n : int
                Number of iterations of the smoothing algorithm
        '''

        self.set_dic_beta(beta0,p)
        for _ in range(n):
            beta_aux = cp.copy(self.beta)
            smooth(self.Lx,self.Ly,beta_aux,self.beta)

    def set_dic_smooth_beta(self,beta0 = 1., p = 0 , n = 1):
        '''
            Set the infection rate to a smooth distribution and then dicotomize it.

            Parameters

            ----------

            beta0 : float
                Non zero value of the infection rate
            
            p : float
                Fraction of the grid where the infection rate is zero

            n : int
                Number of iterations of the smoothing algorithm
        '''



        self.set_dic_beta(beta0,p)
        for _ in range(n):
            beta_aux = cp.copy(self.beta)
            smooth(self.Lx,self.Ly,beta_aux,self.beta)
        
        beta_mean = cp.mean(self.beta)
        self.beta[self.beta>beta_mean] = beta0
        self.beta[self.beta<beta_mean] = 0
    
    def set_cahn_hilliard(self,beta_mean = 1.,e0 = 2/3):

        '''
            Set the infection rate to a Cahn-Hilliard distribution.

            Parameters

            ----------

            beta_mean : float
                Mean value of the infection rate
            
            e0 : float
                Cahn-Hilliard parameter
        '''
        D = .01
        gamma = .5

        laplacian = cp.ElementwiseKernel(
            'raw float32 X, uint32 Lx, uint32 Ly','float32 Y',
            '''
            int x = i % Lx;
            int y = (int) i/Lx;
            Y = X[(x+1)%Lx + Lx*y] + X[(x-1+Lx)%Lx+Lx*y] + X[x + Lx*((y+1)%Ly)] + X[x + Lx*((y-1+Ly)%Ly)] - 4*X[i];
            ''',
            'laplacian')

        A = .0001
        c = cp.random.choice([-A,A],size=(self.Ly,self.Lx)).astype('float32')

        t_max = 10000
        a = cp.zeros_like(c)
        b = cp.zeros_like(c)
        d = cp.zeros_like(c)

        for _ in range(t_max):
            laplacian(4*e0*c*(c**2-1),self.Lx,self.Ly,a)
            laplacian(c,self.Lx,self.Ly,b)
            laplacian(b,self.Lx,self.Ly,d)
            c = c + D*(a-gamma*d)
        c = (c - c.min())
        c = c/c.max()
        c = c/c.mean()*beta_mean

        self.beta = c.astype('float64')
    
    def set_egg_H(self,k=1,h=1):
        '''
            Set the height field to an egg container shape.

            Parameters

            ----------

            k : float
                Wave number for the egg container

            h : float
                Height of the egg container
        '''

        Y,X = cp.mgrid[0:self.Ly:1,0:self.Lx:1]
        self.H = h*(cp.cos(k*2*cp.pi/(self.Lx-1)*X) + cp.cos(k*2*cp.pi/(self.Ly-1)*Y))/2

    def fft_beta(self):
        '''
            Compute the Fourier transform of the infection rate distribution.
        '''

        return cp.fft.fftshift(cp.fft.fft2(self.beta))
    
    def update(self):
        '''
            Update the susceptible and infected populations according to the SIR model.
        '''


        if isinstance(self.H,cp.ndarray):
            forces_0(self.S,self.I,self.H,self.beta,self.gamma,self.DI,self.DS,self.vx,self.vy,self.Lx,self.Ly,self.fS,self.fI)
        elif self.vx != 0 or self.vy != 0:
            forces_2(self.S,self.I,self.beta,self.gamma,self.DI,self.DS,self.vx,self.vy,self.Lx,self.Ly,self.fS,self.fI)
        else:
            forces_1(self.S,self.I,self.beta,self.gamma,self.DI,self.DS,self.alpha,self.Lx,self.Ly,self.fS,self.fI)
        self.fS[:,-1] = 0
        self.fI[:,-1] = 0
        #self.fS[:,0] = 0
        #self.fI[:,0] = 0
        euler(self.fS,self.fI,self.dt,self.S,self.I) 
        self.t += self.dt
        self.t_it += 1

    def update2(self):
        '''
            Update the susceptible and infected populations according to the SIR model looking only a region around the displacement field.
        '''

        x1 = int(self.u_cm() - 1024)
        x2 = int(self.u_cm() + 1024)
        if x1 < 0:
            x1 = 0
        if x2 > self.Lx - 1:
            x2 = self.Lx - 1
        
        forces_1(self.S[:,x1:x2],self.I[:,x1:x2],self.beta[:,x1:x2],self.gamma,self.DI,self.DS,self.alpha,x2-x1,self.Ly,self.fS[:,x1:x2],self.fI[:,x1:x2])
        self.fS[:,x1] = self.fS[:,x2-1] = self.fI[:,x1] = self.fI[:,x2-1] = 0
        euler(self.fS[:,x1:x2],self.fI[:,x1:x2],self.dt,self.S[:,x1:x2],self.I[:,x1:x2])
        self.t += self.dt
        self.t_it += 1


    def solve(self,it_max):
        '''
            Solve the SIR model for a given number of iterations.

            Parameters

            ----------

            it_max : int
                Number of iterations
        '''

        while self.t_it < it_max:
            self.update()
            self.rigid_x()
    
    def get_S1(self):

        '''
            Get the susceptible population remaining after the front has passed.
        '''
        return cp.asnumpy(self.S[:,int(self.Lx*.2):int(self.u_cm())-int(self.Lx*.2)].mean())
    
    def get_p(self):
            
        '''
            Get the fraction of places with zero transmission rate.
        '''

        return cp.asnumpy(cp.count_nonzero(self.beta)/(self.Lx*self.Ly))

    def rigid_x(self):
        '''
            Apply rigid boundary conditions in the x direction.
        '''
        self.S[:,-1] =  0
        self.I[:,-1] = 0
        #self.S[:,0] = 0
        #self.I[:,0] = 0

    def rigid_y(self):
        '''
            Apply rigid boundary conditions in the y direction.
        '''
        self.S[-1,:] = 0
        self.I[-1,:] = 0
    
    
    def tilded_y(self,m = 0):
        '''
            Apply periodic tilded boundary conditions in the y direction. Must be used with set_tilted_initial_conditions.

            Parameters

            ----------

            m : float
                Tilding parameter. Slope is Ly/(m*Lx)
        '''

        if m > 0:
            n = int(cp.ceil(m*(self.Lx-2)))
            self.S[0,:-(n-1)] = self.S[-2,n-1:] 
            self.I[0,:-(n-1)] = self.I[-2,n-1:]
            self.S[-1,n-1:] = self.S[1,:-(n-1)]
            self.I[-1,n-1:] = self.I[1,:-(n-1)]
            self.I[-1,:(n-1)] = 0
            self.I[0,-(n-1):] = 0
            self.S[-1,:(n-1)] = 0
            self.S[0,-(n-1):] = 0
            self.rigid_x()        
        elif m < 0:
            n = int(cp.ceil(-m*(self.Lx-2)))
            self.S[0,n-1:] = self.S[-2,:-(n-1)]
            self.I[0,n-1:] = self.I[-2,:-(n-1)]
            self.S[-1,:-(n-1)] = self.S[1,n-1:]
            self.I[-1,:-(n-1)] = self.I[1,n-1:]
            self.I[-1,-(n-1):] = 0
            self.S[-1,-(n-1):] = 0
            self.I[0,:(n-1)] = 0
            self.S[0,:(n-1)] = 0        
            self.rigid_x()
        else:
            self.rigid_x()    


    def u(self):
        '''
            Compute the displacement field.

            Returns

            -------

            u : array
                Displacement field
        '''
        return cp.argmax(self.I,axis=1)
    
    def u1(self):
        '''
            Compute the displacement field. Second kind.

            Returns

            -------

            u : array
                Displacement field. Second kind.
        '''
        return cp.dot(self.I,cp.arange(self.Lx))/(cp.sum(self.I,axis=1))
    
    def u2(self):
        '''
            Compute <u^2>.

            Returns

            -------

            u : array
                <u^2>
        '''
        return cp.dot(self.I,cp.arange(self.Lx)**2)/(cp.sum(self.I,axis=1))
    
    def u_cm(self):
        '''
            Compute the center of mass of the infected population.

            Returns

            -------

            u : array
                Center of mass of the infected population.
        '''
        return cp.mean(self.u())

    def u_cm1(self):
        '''
            Compute the center of mass of the infected population. Second kind.

            Returns

            -------

            u : array
                Center of mass of the infected population. Second kind.
        '''
        return cp.mean(self.u1())
    
    def u_sigma(self):
        '''
            Compute the standard deviation of the infected population.

            Returns

            -------

            u_sigma : array
                Standard deviation of the infected population.
        '''
        return cp.std(self.u())
    
    def u_sigma1(self):
        '''
            Compute the standard deviation of the infected population. Second kind.

            Returns

            -------

            u_sigma1 : array
                Standard deviation of the infected population. Second kind.
        '''
        return cp.std(self.u1())
    
    def width(self):
        '''
            Compute the width of the infected population.

            Returns

            -------

            width : array
                Width of the infected population.
        '''
        return cp.sqrt(self.u2()-self.u1()**2).mean()
    
    def u_fft(self):
        '''
            Compute the Fourier transform of the displacement field.

            Returns

            -------

            u_fft : array
                Fourier transform of the displacement field.
        '''

        return cp.square(cp.abs(cp.fft.rfft(self.u())))

    def u_fft1(self):
        '''
            Compute the Fourier transform of the displacement field. Second kind.

            Returns

            -------

            u_fft1 : array
                Fourier transform of the displacement field. Second kind.
        '''
        return cp.square(cp.abs(cp.fft.rfft(self.u1())))
    
    def f(self):
        '''
            Compute the infection front's profile.

            Returns

            -------

            f : array
                Infection front's profile.
        '''
        return cp.mean(self.I,axis=0)
        
    
    def f_std(self):
        '''
            Compute the standard deviation of the infection front's profile.

            Returns

            -------

            f_std : array
                Standard deviation of the infection front's profile.
        '''
        return cp.std(self.I,axis=0)
    
    def g(self):
        '''
            Compute the susceptible population's profile.

            Returns

            -------

            g : array
                Susceptible population's profile.
        '''
        return cp.mean(self.S,axis=0)


    def g_std(self):
        '''
            Compute the standard deviation of the susceptible population's profile.

            Returns

            -------

            g_std : array
                Standard deviation of the susceptible population's profile.
        '''
        return cp.std(self.S,axis=0)
    
    def kpz(self):

        dI = cp.gradient(self.I,axis=1,edge_order=2)
        dI2 = cp.gradient(dI,axis=1,edge_order=2)

        return dI2*dI
    
    def I_max(self):
        '''
            Compute the maximum values of the infected population in the x direction and take the mean over the y direction. 

            Returns

            -------

            I_max : array
                Maximum value of the infected population.
        '''
        return cp.mean(cp.max(self.I,axis=1))


        
    
    
