import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.collections import LineCollection



def dic_matrix(x0,x1,p,Lx,Ly):
  return np.random.choice([x0,x1],size=(Ly,Lx),p = [p,1-p])

def f_I_front(x,beta=1.,gamma=.2,D=1.,S0=1):
  return np.exp(-c_0(beta,gamma,D)/(2*D)*x)

def f_I_back(x,S1=0,beta=1,gamma=.2,D=1,S0=1):
  return np.exp((-c_0(beta,gamma,D,S0)/(2*D)+np.sqrt(beta/D*(S0-S1)))*x)

def u(Y):
  return np.argmax(Y,axis=1)

def c_0(beta=1,gamma=.2,D=1.,S0=1.):
  return 2*np.sqrt(D*beta*(S0-gamma/beta))

def wave_form(Y):
  return np.mean(Y,axis=0)

def lineal(x,a,b):
  return a*x + b

def sqrt_root(x,a,b):
  return a*np.sqrt(1-x/b)

def root(x,a,b,c):
  return a*(1-x/b)**c

def velocity(t,x):
    fit, cov = curve_fit(lineal,t,x)
    return fit[0]

def plot_scan(p_vec,n,x,y,flag='',flag2='',cbar=False):
  fig, ax = plt.subplots(figsize=(12,8))
  xs = [np.load(x + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  ys = [np.load(y + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  line_segments = LineCollection([np.column_stack([x, y]) for x,y in zip(xs,ys)],cmap='copper')
  line_segments.set_array(p_vec)
  ax.add_collection(line_segments)
  if cbar:
    axcb = fig.colorbar(line_segments)
    axcb.set_label('$p$',fontsize=16)
  ax.autoscale()
  return fig,ax

def plot_beta_scan(p_vec,n,x,y,flag='',flag2='',cbar=False):
  fig, ax = plt.subplots(figsize=(12,8))
  xs = [np.load(x + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  ys = [np.load(y + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  beta_mean = np.array([np.mean(np.load('beta_' + str(n)+'_'+str(p)+'_'+flag+'_'+ flag2+'.npy')) for p in p_vec])
  line_segments = LineCollection([np.column_stack([x, y]) for x,y in zip(xs,ys)],cmap='copper')
  line_segments.set_array(beta_mean)
  ax.add_collection(line_segments)
  if cbar:
    axcb = fig.colorbar(line_segments)
    axcb.set_label(r'$\overline{\beta}$',fontsize=16,rotation=0,labelpad=10)
  ax.autoscale()

def fp_scan(p_vec,n,flag='',flag2='',cbar=False):
  fig, ax = plt.subplots(figsize=(14,8))
  ys = [np.load('f_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  x = np.arange(len(ys[0]))
  line_segments = LineCollection([np.column_stack([x, y]) for y in ys],cmap='copper')
  line_segments.set_array(p_vec)
  ax.add_collection(line_segments)
  if cbar:
    axcb = fig.colorbar(line_segments)
    axcb.set_label('$p$',fontsize=16)
  ax.autoscale()

def fbeta_scan(p_vec,n,flag='',flag2='',cbar=False):
  fig, ax = plt.subplots(figsize=(14,8))
  ys = [np.load('f_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  x = np.arange(len(ys[0]))
  beta_mean = np.array([np.mean(np.load('beta_' + str(n)+'_'+str(p)+'_'+flag+'_'+ flag2+'.npy')) for p in p_vec])
  line_segments = LineCollection([np.column_stack([x, y]) for y in ys],cmap='copper')
  line_segments.set_array(beta_mean)
  ax.add_collection(line_segments)
  if cbar:
    axcb = fig.colorbar(line_segments)
    axcb.set_label(r'$\overline{\beta}$',fontsize=16)
  ax.autoscale()


def cp_scan(p_vec,n,flag='',flag2=''):
  pc = np.array([])
  c = np.array([])
  c_err = np.array([])
  for p in p_vec:
    time = np.load('time_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy')
    pos = np.load('u_cm_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2 + '.npy')
    w = np.load('width_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy')
    if w[-1]<100 and pos[-1]>400:
      fit,cov = curve_fit(lineal,time[pos>400],pos[pos>400])
      if fit[0] > .01:
        pc = np.append(pc,p)
        c = np.append(c,fit[0])
        c_err = np.append(c_err,cov[0,0]) 
  return pc,c,c_err

def cn_scan(p,n_vec,flag='',flag2=''): 
  c = np.array([])
  c_err = np.array([])
  for j in n_vec:
    time = np.load('time_' + str(j)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy')
    pos = np.load('u_cm_' + str(j)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy')
    c_,cov_ = curve_fit(lineal,time[pos>50],pos[pos>50])
    c = np.append(c,c_)
    c_err = np.append(cov_,cov_[0,0]) 
  return c,c_err

def Yp_scan(p_vec,n,flag='',flag2=''):
    Y = np.array([])
    pc = np.array([])
    for p in p_vec:
      Ymax = np.load('Y_max_'+str(n)+'_'+str(p)+'_'+flag+'_'+flag2+'.npy')
      pos = np.load('u_cm_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2 + '.npy')
      w = np.load('width_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy')
      if w[-1]<50:
        Y = np.append(Y,np.mean(Ymax[pos>400]))
        pc = np.append(pc,p)
    return pc,Y

def fftp_scan(p_vec,n,flag='',flag2=''):
  fig, ax = plt.subplots(figsize=(14,8))
  xs = [np.load('q_' + str(n) + '_' + str(p) + '_' + flag+ '_' + flag2+'.npy') for p in p_vec]
  ys = [np.load('fft_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  
  line_segments = LineCollection([np.column_stack([x, y]) for x,y in zip(xs,ys)],cmap='copper')
  
  line_segments.set_array(p_vec)
  ax.add_collection(line_segments)
  ax.set_xlim(.007,.5)
  plt.loglog()
  axcb = fig.colorbar(line_segments)
  axcb.set_label('$p$',fontsize=16)

def fftbeta_scan(p_vec,n,flag='',flag2=''):
  fig, ax = plt.subplots(figsize=(14,8))
  xs = [np.load('q_' + str(n) + '_' + str(p) + '_' + flag+ '_' + flag2+'.npy') for p in p_vec]
  ys = [np.load('fft_' + str(n)+ '_' + str(p) + '_' +flag+ '_' + flag2+'.npy') for p in p_vec]
  line_segments = LineCollection([np.column_stack([x, y]) for x,y in zip(xs,ys)],cmap='copper')
  beta_mean = np.array([np.mean(np.load('beta_' + str(n)+'_'+str(p)+'_'+flag+'_'+ flag2+'.npy')) for p in p_vec])  
  line_segments.set_array(beta_mean)
  ax.add_collection(line_segments)
  ax.set_xlim(.007,.5)
  plt.loglog()
  axcb = fig.colorbar(line_segments)
  axcb.set_label(r'$\overline{\beta}$',fontsize=16)
