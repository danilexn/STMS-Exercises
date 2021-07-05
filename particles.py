import numpy as np

def createParticles(numParticles,dim,lBounds,uBounds,kind):
    particlePos = []

    if kind == 0:
        particlePos = lBounds + (uBounds-lBounds) * np.random.uniform(size = (numParticles,dim))

    elif kind == 1:
        numPerDim = round(numParticles**(1/dim))
        cellWidth = (uBounds-lBounds)/(2*numPerDim)

        if dim==1:
            particlePos = np.linspace(lBounds+cellWidth,uBounds-cellWidth,numPerDim)
        elif dim==2:
            xVec = np.linspace(lBounds+cellWidth,uBounds-cellWidth,numPerDim)
            yVec = np.linspace(lBounds+cellWidth,uBounds-cellWidth,numPerDim)
            X, Y = np.meshgrid(xVec,yVec)
            particlePos = np.vstack((Y.flatten(), X.flatten())).T
            
    return particlePos
