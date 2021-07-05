import numpy as np
import multiprocessing

def etaKernel(rVec,epsilon,dim):
    if dim == 1:
        etaVals = 1/(2*epsilon*np.sqrt(np.pi)) * np.exp(-rVec**2./(4*epsilon**2))
    elif dim == 2:
        etaVals = 4/(epsilon**2*np.pi) * np.exp(-rVec**2./(epsilon**2))
    else:
        etaVals = np.ones(rVec.shape)*(15/np.pi**2)/(np.abs(rVec)**10+np.ones(rVec.shape))
    return etaVals

def periodicBoundaries(strengthMat,h,cutoff,bW,rows,cols):
    strengthBound = strengthMat

    # permute left/right boundary
    sb = np.flip(np.arange(0, bW - 1, 1)).astype(int)
    sm = np.arange(bW+1, 2*bW, 1).astype(int)
    strengthBound[:,cols-sb-1,:] = strengthMat[:,sm - 1,:]

    sb = np.arange(1, bW, 1).astype(int)
    sm = np.flip(np.arange(bW, 2*bW - 1, 1)).astype(int)
    strengthBound[:,sb - 1,:] = strengthMat[:,cols-(sm)-1,:]

    # permute upper/lower boundary
    sb = np.flip(np.arange(0, bW - 1, 1)).astype(int)
    sm = np.arange(bW+1, 2*bW, 1).astype(int)
    strengthBound[rows-sb-1,:,:] = strengthMat[sm-1,:,:]
    sb = np.arange(1, bW, 1).astype(int)
    sm = np.flip(np.arange(bW, 2*bW-1, 1)).astype(int)
    strengthBound[sb-1,:,:] = strengthMat[sm-1,:,:]

    # permute the remaining corners up/down
    sb = np.arange(1, bW, 1).astype(int)
    sm = np.flip(np.arange(bW, 2*bW - 1, 1)).astype(int)
    strengthBound[sb-1,sb-1,:] = strengthMat[rows-sm-1,cols-sm-1,:]
    sb = np.flip(np.arange(0, bW - 1, 1)).astype(int)
    sm = np.arange(bW + 1, 2 * bW, 1).astype(int)
    strengthBound[rows-sb-1,cols-sb-1,:] = strengthMat[sm-1,sm-1,:]

    sb_1 = np.arange(1, bW, 1).astype(int)
    sb_2 = np.flip(np.arange(0, bW - 1, 1)).astype(int)
    sm_1 = np.flip(np.arange(bW, 2*bW - 1, 1)).astype(int)
    sm_2 = np.arange(bW + 1, 2*bW, 1).astype(int)
    strengthBound[sb_1-1,cols-sb_2-1,:] = strengthMat[rows-sm_1-1,sm_2-1,:]
    
    sb_1 = np.arange(1, bW, 1).astype(int)
    sb_2 = np.flip(np.arange(0, bW - 1, 1)).astype(int)
    sm_1 = np.flip(np.arange(bW, 2*bW - 1, 1)).astype(int)
    sm_2 = np.arange(bW + 1, 2*bW, 1).astype(int)
    strengthBound[rows - sb_2 - 1,sb_1-1,:] = strengthMat[sm_2-1,cols - sm_2-1,:]

    return strengthBound

def applyPSE(particleMat,verletList,epsilon,numStren, dim):
    numParticles = len(verletList)
    pseSum = np.zeros((numParticles,numStren))

    for i in range(0,numParticles):
        neighParticleMat = particleMat[verletList[i],:]
        if len(neighParticleMat) > 0:
            neighVecs = neighParticleMat[:,0:dim]-np.tile(particleMat[i,0:dim],(neighParticleMat.shape[0],1))
            particleDists = np.sqrt(np.sum(neighVecs*neighVecs,1))

            for j in range(0, numStren):
                pseSum[i,j] = np.sum((neighParticleMat[:,dim+1+j]-particleMat[i,dim+1+j])*etaKernel(particleDists,epsilon,dim))
        else:
            pseSum[i,0] = 0
        
    return pseSum
