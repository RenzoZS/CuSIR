import cv2
from cusir.system import system
import cupy as cp

def I(action, x0, y0, flags, *userdata):
    global draw

    if x0 >= N or y0 >= N:
        return 0
    if action == cv2.EVENT_LBUTTONDOWN:
        draw = True
        s.I[y0,x0] = 1
        s.S[y0,x0] = 0
    elif action == cv2.EVENT_LBUTTONUP:
        draw = False
    elif draw:
        if action == cv2.EVENT_MOUSEMOVE:
            s.I[y0,x0] = 1
            s.S[y0,x0] = 0

def p_slide(val):
    global p
    p = val/max
    s.set_dic_beta(beta0=beta0,p=p)
    
def vx_slide(val):
    s.vx = -max_vx*(1-val/max) + max_vx*(val/max)

def vy_slide(val):
    s.vy = -(-max_vy*(1-val/max) + max_vy*(val/max))

N = 1024
N1 = N
beta0 = 1

s = system(N,N)
s.gamma = .2
s.alpha = 1
s.DI = 1
s.vx = 0
s.vy = 0
s.dt = .05
s.set_dic_beta(beta0=beta0,p=.5)
s.set_plane_initial_conditions()

img = cp.zeros((N,N,3),cp.uint8)

max = 1000
max_vx = 3
max_vy = 3
draw = False
p = 0

cv2.namedWindow('Controls')
cv2.namedWindow('I',cv2.WINDOW_GUI_EXPANDED)
cv2.createTrackbar('p','Controles',0,max,p_slide )
cv2.createTrackbar('vx','Controles',0,max,vx_slide)
cv2.createTrackbar('vy','Controles',0,max,vy_slide)
text_template = 'p = {:.2f} beta = {:.2f} gamma = {:.2f}  D = {:.2f} vx = {:.2f} vy = {:.2f} Imax = {:.2f}'

while True:
    img[:,:,2] = s.I*255/s.I.max()
    img[:,:,0] = (1-s.I-s.S)*255/(1-s.I-s.S).max()
    img[:,:,1] = s.S*130/s.S.max()
    imgnp = img.get()
    cv2.putText(imgnp,text_template.format(p, s.beta.mean(),s.gamma,s.DI,s.vx,s.vy,s.I.max()),(10,30),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)
    cv2.setMouseCallback("I", I)
    #cv2.imshow('I', (s.I/s.I.max()).get(),)
    cv2.imshow('I', imgnp,)
    
    s.update()
    s.rigid_x()

    if cv2.waitKey(1) == ord('q'):
        s.reset()
        break

    if cv2.waitKey(1) == ord('r'):
        s.reset()
        s.vx = 0
        s.vy = 0

cv2.destroyAllWindows()