import numpy as np

def adjacentCells(cellIndex,numCells):
    dim = numCells.shape[0]
    adjCellIndices = []

    if dim==3:

        I, J, K = np.unravel_index(cellIndex, numCells)

        iIndices= np.arange(max(I-1,0), min(I+2,numCells[0]), 1)
        jIndices= np.arange(max(J-1,0), min(J+2,numCells[1]), 1)
        kIndices= np.arange(max(K-1,0), min(K+2,numCells[2]), 1)

        for i in iIndices:
            for j in jIndices:
                for k in kIndices:
                    if np.any(np.array([i,j,k])-np.array([I,J,K])):
                        adjCellIndices += np.ravel_multi_index((i, j, k),numCells)

    elif dim==2:
        I, J = np.unravel_index(cellIndex, numCells)

        iIndices= np.arange(max(I-1,0), min(I+2,numCells[0]), 1)
        jIndices= np.arange(max(J-1,0), min(J+2,numCells[1]), 1)

        for i in iIndices:
            for j in jIndices:
                if np.any(np.array([i,j])-np.array([I,J])):
                    adjCellIndices.append(np.ravel_multi_index((i, j),numCells))

    return np.array(adjCellIndices)
    
def flatten_array(arr):
    flat = []
    for a in arr:
        if a == None:
            continue
        for e in a:
            flat += e.tolist()
    return np.array(flat)
    
def createVerletList(particleMat,cellList,numCells,cutoff):
    numParticles, dim = particleMat.shape
    dim = dim-1;

    verletList = np.empty(numParticles,dtype=object)

    for n in range(0,numParticles):
        currCellInd = int(particleMat[n,dim])
        adjCellIndices = adjacentCells(currCellInd, numCells)
        allCellIndices = np.append(adjCellIndices, np.array(currCellInd))

        currPartIndices = cellList[allCellIndices]
        currPartIndices = flatten_array(currPartIndices)

        numRows = len(currPartIndices)

        tempMat = np.tile(particleMat[n,0:dim],(numRows,1))
        tempDiff = tempMat - particleMat[currPartIndices,0:dim]
        tempDist = np.sum(tempDiff**2,1)
        temp = currPartIndices[np.where(tempDist <= cutoff**2)]
        verletList[n] = temp[np.where(temp != n)]
        
    return verletList
