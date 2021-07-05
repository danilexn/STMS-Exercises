import numpy as np

def createCellList(particlePos,lBounds,uBounds,cellSide):

    numParticles, dim = particlePos.shape
    indices=np.zeros((numParticles, 1))

    domainSize = (uBounds-lBounds)*np.ones((dim,1))
    
    sideSize = np.floor(domainSize/cellSide).astype(int)
    numCells = np.where(sideSize < 1, 1, sideSize).flatten()
    indParticlePos = np.zeros((numParticles,dim)).astype(int)

    for i in range(0,dim):
        indParticlePos[:,i]=np.floor((particlePos[:,i]-min(particlePos[:,i]))/(domainSize[i]-min(particlePos[:,i]))*numCells[i])

    if dim == 1:
        indices = indParticlePos

    elif dim == 2:
        for i in range(0, numParticles):
            indices[i,0] = np.ravel_multi_index([indParticlePos[i,0],indParticlePos[i,1]], numCells)

    particleMat = np.hstack((particlePos,indices))
    maxCellNum = int(max(particleMat[:,dim]))
    cellList = np.empty(np.prod(numCells),dtype=object)

    for c in range(0, maxCellNum):
        currInd = np.where(particleMat[:,dim] == c)
        cellList[c] = currInd[:]
        
    return particleMat, cellList, numCells
