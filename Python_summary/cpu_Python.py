import numpy as np
import time       

v0=np.loadtxt('V0.txt')
w0=np.loadtxt('W0.txt')
X=np.loadtxt('nnmf-2429-by-361-face.txt')

def NNMF(X,r,tol,V=v0,W=w0,verbose=1):
    V = v0[:,0:r]
    W = w0[0:r,:]
    #Frobinius norm at previous step 
    B = np.dot(V,W)
    L = np.linalg.norm(X-B)**2 
    iteration = 0
    while 1:
        #update V
        V *= np.dot(X,W.T) 
        V /= np.dot(B,W.T) 
        B = np.dot(V,W) 
        #update W
        W *= np.dot(V.T,X)
        W /= np.dot(V.T,B)
        B = np.dot(V,W)
        Lnew = np.linalg.norm(X-B)**2
        if abs(Lnew-L) <= tol*(L+1):
            break 
        else:
            L = Lnew
            iteration += 1
            if(verbose and iteration%50==0):
                print "At iteration %i, the loss is %.2f" %(iteration, L)
    return V,W,iteration


for i in range(5):
    r = i*10+10
    start = time.time()
    V,W,iteration = NNMF(X,r,1e-4,verbose=0)
    end = time.time()
    print "Elapsed time for Rank-%i model is %.2f seconds" %(r,end-start)