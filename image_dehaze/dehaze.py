import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from opts import OPTIONS

from path import Path
import math
from tqdm import tqdm






def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1) 
    imvec = im.reshape(imsz,3) 

    indices = darkvec.argsort() 
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx 
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95 
    im3 = np.empty(im.shape,im.dtype) 

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz) 
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r)) 
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r)) 
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r)) 
    cov_Ip = mean_Ip - mean_I*mean_p 

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r)) 
    var_I   = mean_II - mean_I*mean_I 

    a = cov_Ip/(var_I + eps) 
    b = mean_p - a*mean_I 

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r)) 
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r)) 

    q = mean_a*im + mean_b 
    return q 

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
    gray = np.float64(gray)/255 
    r = 60 
    eps = 0.0001 
    t = Guidedfilter(gray,et,r,eps) 

    return t 

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype) 
    t = cv2.max(t,tx) 

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def mask(img):
    (a,b,c,d) = args.mask_points

    if len(img.shape)==3:
        return img[b:d,a:c,:]
    else:
        return img[b:d,a:c]
def normal(img):
    max = img.max()
    min = img.min()
    return abs(img)/(max-min)


def depth_init():
    c = np.linspace(start=719,stop=0,num=720)
    c = np.expand_dims(c,axis=1)
    depth =[]
    for i in range(1280):
        depth.append(c.copy())

    depth =np.array(depth)
    depth=depth.transpose([1,0,2]).squeeze(axis=2)
    return depth


def main(args):
    show = False

    dir = Path(args.dir)
    dir_I = Path('./dir_I')
    dir_I.mkdir_p()
    dir_J=Path('dir_J')
    dir_J.mkdir_p()

    dir_t = Path('dir_t')
    dir_t.mkdir_p()

    dir_sigma = Path('dir_sigma')
    dir_sigma.mkdir_p()

    dir_dark = Path('dir_dark')
    dir_dark.mkdir_p()
    depth = depth_init()
    depth = mask(depth)+1

    files =dir.files()
    files.sort()
    for idx,item in tqdm(enumerate(files)):
        src = cv2.imread(item)
        src=mask(src)
        I = src.astype('float64') / 255
        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A, 15)
        t = TransmissionRefine(src, te)
        J = Recover(I, t, A, 0.1)
        sigma_d = np.log(t)
        #print(bd)
        sigma = -sigma_d/dark/depth
    #show
        if show:
            plt.imsave(dir_I / 'I_{:03d}'.format(idx), I)
            plt.imsave(dir_t / 't_{:03d}'.format(idx), t)
            plt.imsave(dir_J / 'J_{:03d}'.format(idx), J)
            plt.imsave(dir_sigma / 'sig_{:03d}'.format(idx), sigma)
            plt.imsave(dir_dark / 'dark_{:03d}'.format(idx), dark)

        else:
            np.save(dir_I / '{:03d}.npy'.format(idx), I)
            np.save(dir_t/'{:03d}.npy'.format(idx),t)
            np.save(dir_J/'{:03d}.npy'.format(idx),J)
            np.save(dir_sigma/'{:03d}.npy'.format(idx),sigma)
            np.save(dir_dark/'{:03d}.npy'.format(idx),dark)
def draw():
    x_t = np.loadtxt('it.txt')
    dir =Path('dir_sigma')
    sigs_c=[]
    sigs_mean=[]
    #plt.ylim([0,2])
    for item in dir.files('*.npy'):
        npy = np.load(item)
        w,h = npy.shape
        sigma = npy[int(w/2),int(h/2)]
        sigs_c.append(sigma)

        sigma = npy.min()
        sigs_mean.append(sigma)

   # plt.plot(x_t,sigs_c)
    plt.plot(x_t,sigs_mean,'-o')
    #plt.legend(['sigma'])
    plt.show()

    for item in sigs_mean:
        print(item)
    pass
def draw2():
    x_t = np.loadtxt('it.txt')+1
    dir_I = Path('./dir_I')
    files = dir_I.files('*.npy')
    files.sort()
    baseline = np.load('./002.npy')
    diffs_mean=[]
    diffs_min=[]

    for item in files:
        npy = np.load(item)
        diff = np.abs(npy-baseline).mean()
        diffs_mean.append(diff)
        diff = np.abs(npy-baseline).max()
        diffs_min.append(diff)
    diffs_mean=np.array(diffs_mean)/25
    plt.plot(x_t,diffs_mean,'b-o')
    plt.legend(['sigma'])
    for item in diffs_mean:
        print(item)

    plt.show()
def draw3():
    p = Path('/home/roit/Desktop/sigma.csv')
    df = pd.read_csv(p,index_col=False)
    plt.plot(1.3/df['y'],'b-o')
    plt.ylabel('MOR')
    plt.title('MOR')
    plt.xlabel('frame idx')

    plt.show()
    plt.legend('sigma')
    pass

if __name__ == '__main__':
    import pandas as pd
    args = OPTIONS().args()

    #main(args)
    #draw2()
    draw3()

