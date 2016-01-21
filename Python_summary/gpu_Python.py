import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import skcuda.linalg as linalg
import time       

v0 = np.loadtxt('V0.txt').astype(np.float32)
w0 = np.loadtxt('W0.txt').astype(np.float32)
X  = np.loadtxt('nnmf-2429-by-361-face.txt').astype(np.float32)

linalg.init()
def NNMF_gpu(X,r,tol,V=v0,W=w0,verbose=1):
    Vr = V[:,0:r].copy()
    Wr = W[0:r,:].copy()
    X_gpu = gpuarray.to_gpu(X)
    V_gpu = gpuarray.to_gpu(Vr) 
    W_gpu = gpuarray.to_gpu(Wr) 
    #Frobinius norm at previous step 
    B_gpu = linalg.dot(V_gpu, W_gpu) 
    L = linalg.norm(X_gpu-B_gpu)**2 
    iteration = 0
    while 1: #update V
        V_gpu *= linalg.dot(X_gpu,linalg.transpose(W_gpu)) 
        V_gpu /= linalg.dot(B_gpu,linalg.transpose(W_gpu)) 
        B_gpu = linalg.dot(V_gpu, W_gpu)
        #update W
        W_gpu *= linalg.dot(linalg.transpose(V_gpu),X_gpu)
        W_gpu /= linalg.dot(linalg.transpose(V_gpu),B_gpu)
        B_gpu = linalg.dot(V_gpu, W_gpu)
        Lnew = linalg.norm(X_gpu-B_gpu)**2
        if abs(Lnew-L) <= tol*(L+1):
            break
        else:
            L = Lnew
            iteration += 1
            if(verbose and iteration%50==0):
                print "At iteration %i, the loss is %.2f" %(iteration, L)
    return V_gpu,W_gpu,iteration


for i in range(5):
    r = i*10+10
    start = time.time()
    V_gpu,W_gpu,iteration = NNMF_gpu(X,r,1e-4,verbose=0)
    end = time.time()
    print "Elapsed time for Rank-%i model is %.2f seconds" %(r,end-start)
