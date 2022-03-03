import numpy as np
import cupy as cp
import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt


forces = cp.ElementwiseKernel(
    'raw float64 X, raw float64 Y, raw float64 params,raw float64 beta,raw float64 gamma, int16 L' ,
    'float64 fX,float64 fY',
    '''
    double N = params[0]; double nu = params[1]; double mu = params[2]; double Dx = params[3]; double Dy = params[4];
    
    int x = i % L;
    int y = (int) i/L;

    fX =  nu - beta[i]*X[i]*Y[i]/N - mu*X[i] + Dx*(X[(x+1)%L + L*y] + X[(x-1+L)%L+L*y] + X[x + L*((y+1)%L)] + X[x + L*((y-1+L)%L)] - 4*X[i]);
    fY = beta[i]*X[i]*Y[i]/N - (gamma[i]+mu)*Y[i] + Dy*(Y[(x+1)%L + L*y] + Y[(x-1+L)%L+L*y] + Y[x + L*((y+1)%L)] + Y[x + L*((y-1+L)%L)] - 4*Y[i]);
    ''',
    'forces')

smoth = cp.ElementwiseKernel(
    'int16 L, raw float64 X', 'float64 Y',
    '''
    int x = i % L;
    int y = (int) i/L;

    Y = (X[(x+1)%L + L*y] + X[(x-1+L)%L+L*y] + X[x + L*((y+1)%L)] + X[x + L*((y-1+L)%L)] + X[i])/5;
    ''','smoth')

def smothing(X,n):
  for i in range(n):
    smoth(cp.asnumpy(cp.shape(X))[1],X,X) 
  return X

def dic_smothing(X,n):
  for i in range(n):
    smoth(cp.asnumpy(cp.shape(X))[1],X,X)
  X_mean = cp.mean(X)
  X[X > X_mean] = 1
  X[X < X_mean] = 0
  return X

def scan(p_vec,n_vec,flag='',flag2='',gamma0=.2,beta0=1.):
  for j in n_vec:
    for p in tqdm(p_vec):
      X,Y,pos,max,width,fft,q,f,beta,time = solver(gamma0=gamma0,beta0=beta0,p_beta = p, smoth_steps=j,beta_type=flag)
      cp.save('u_cm_'+ str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,pos)
      cp.save('Y_max_' + str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,max)
      cp.save('width_' + str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,width)
      cp.save('fft_' + str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,fft)
      cp.save('q_' + str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,q)
      cp.save('time_' + str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,time)
      cp.save('f_' + str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,f)
      cp.save('beta_'+str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,beta)
      cp.save('X_'+str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,X)
      cp.save('Y_'+str(j)+ '_' + str(p) + '_' + flag+ '_' + flag2,Y)  
    np.save('p_vec'+str(j)+'_'+flag + '_' + flag2,p_vec)

def makeI_gif(Lx=1024,Ly=1024,N=1,nu=0.,mu=0.,Dx=0,Dy=1,beta0=1.,gamma0=.2,p_beta=0,p_gamma=0,tstep=.1,smoth_steps=0,beta_type='',t_max = 50000,flag=''):
  
  def line_infection(x):
    X[:,x] = cp.zeros_like(X[:,x])
    Y[:,x] = cp.ones_like(Y[:,x])
  def u():
    return cp.argmax(Y,axis=1)
  def u_cm():
    return cp.mean(u())
  def dic_matrix(x0,x1,p,Lx,Ly):
    return cp.random.choice([x0,x1],size=(Ly,Lx),p = [p,1-p])
  def rigid_x():
    X[:,0] = X[:,-1] = Y[:,0] = Y[:,-1] = 0
  
  fig,ax = plt.subplots(figsize=(12,8))
  plt.xticks([])
  plt.yticks([])


  beta = dic_matrix(0,beta0,p_beta,Lx,Ly)
  if beta_type =='dic_smoth':
    dic_smothing(beta,smoth_steps)
  if beta_type =='smoth':
    smothing(beta,smoth_steps)
  gamma = dic_matrix(0,gamma0,p_gamma,Lx,Ly)
  params = cp.array([N,nu,mu,Dx,Dy])

  # Condición inicial
  X = cp.ones((Ly,Lx))
  Y = cp.zeros((Ly,Lx))
  line_infection(1)
  Y0 = Y
  
  pos = cp.array([u_cm()])
  #Y_max = cp.array([mean_Ymax()])
  #width = cp.array([u_sigma()]) 
  fX = cp.zeros((Ly,Lx))
  fY = cp.zeros((Ly,Lx))
  #fft = cp.zeros_like(u_fft())
  #fI = cp.zeros_like(f_I())
  t = 0
  tsteps = 0
  im = ax.matshow(cp.asnumpy(Y),cmap='hot')
  txt = plt.text(0.5, 1.01,0, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
  ims = [[im,txt]]
  
  #meansteps = 0

  while (pos[tsteps] < Lx*.8 and tsteps < t_max):
    
    forces(X,Y,params,beta,gamma,Lx,fX,fY)
    X = X + tstep*fX
    Y = Y + tstep*fY
    rigid_x()

    pos = cp.append(pos,u_cm())
    tsteps += 1
    t += tstep 
    im = ax.matshow(cp.asnumpy(Y),cmap='hot')
    ax.axis('off')
    txt = plt.text(0.5, 1.01,'t = ' + str(t), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
    ims.append([im,txt])

  anim = animation.ArtistAnimation(fig,ims,blit=True,interval=20,repeat_delay=1000)
  return anim

def solver(Lx=1024,Ly=1024,N=1,nu=0.,mu=0.,Dx=0,Dy=1,beta0=1.,gamma0=.2,p_beta=0,p_gamma=0,tstep=.1,smoth_steps=0,beta_type='',t_max = 50000):

  def line_infection(x):
    X[:,x] = cp.zeros_like(X[:,x])
    Y[:,x] = cp.ones_like(Y[:,x])

  def rigid_x():
    X[:,0] = X[:,-1] = Y[:,0] = Y[:,-1] = 0

  def rigid_y():
    X[0,:] = X[-1,:] = Y[0,:] = Y[-1,:] = 0
  
  def point_infection(x0,y0):
    X[y0,x0] = 0
    Y[y0,x0] = 1 
  
  def u():
    return cp.argmax(Y,axis=1)

  def u_cm():
    return cp.mean(u())
  
  def u_sigma():
    return np.std(u())

  def u_fft():
    return cp.square(cp.abs(cp.fft.fft(u())))
  
  def f_I():
    return cp.roll(np.mean(Y,axis=0),int(-u_cm())+Lx//2)
  
  def dic_matrix(x0,x1,p,Lx,Ly):
    return cp.random.choice([x0,x1],size=(Ly,Lx),p = [p,1-p])
  
  def mean_Ymax():
    return cp.mean(cp.max(Y,axis=1))

  # Heterogeneidad
  beta = dic_matrix(0,beta0,p_beta,Lx,Ly)
  if beta_type =='dic_smoth':
    dic_smothing(beta,smoth_steps)
  if beta_type =='smoth':
    smothing(beta,smoth_steps)
  gamma = dic_matrix(0,gamma0,p_gamma,Lx,Ly)
  params = cp.array([N,nu,mu,Dx,Dy])

  # Condición inicial
  X = cp.ones((Ly,Lx))
  Y = cp.zeros((Ly,Lx))
  line_infection(1)
  
  pos = cp.array([u_cm()])
  Y_max = cp.array([mean_Ymax()])
  width = cp.array([u_sigma()]) 
  fX = cp.zeros((Ly,Lx))
  fY = cp.zeros((Ly,Lx))
  fft = cp.zeros_like(u_fft())
  fI = cp.zeros_like(f_I())

  tsteps = 0
  meansteps = 0

  while (pos[tsteps] < Lx*.8 and tsteps < t_max):
    
    forces(X,Y,params,beta,gamma,Lx,fX,fY)
    X = X + tstep*fX
    Y = Y + tstep*fY 
 
    #Bordes rígidos
    rigid_x()
    #rigid_y()

    pos = cp.append(pos,u_cm())
    Y_max = cp.append(Y_max,mean_Ymax())
    width = cp.append(width,u_sigma())
    if pos[tsteps]>100:
      fft += u_fft()
      fI += f_I()
      meansteps += 1
    
    tsteps += 1
  
  fft = fft/meansteps
  q = cp.fft.fftfreq(Ly)
  fI = fI/meansteps
  time = cp.arange(len(pos))*tstep
  

  return  X,Y,pos,Y_max,width,fft[:len(fft)//2],q[:len(fft)//2],fI,beta,time