import numpy as np
import scipy.special
from joblib import Parallel, delayed
import multiprocessing

def Analytic_Infinitsite(mutation,sampleSize):
    print(sampleSize)
    Prob = np.zeros([sampleSize-1, 100])
    for n in range(2, sampleSize+1):
        if (n == 2):
            for j in range(100):
                Prob[n-2, j] = ((mutation/(1+mutation))**j*(1/(1+mutation)))
        else:
            for j in range(100):
                for l in range(j):
                    Prob[n-2, j] += Prob[n-3, j-l]*((mutation/(n-1+mutation))**l*((n-1)/(n-1+mutation)))
    return Prob

def CoalescentEventTime(tau,taub,timeTensor,n,N,NPrim,vecSize):
    tTot = np.sum(timeTensor,axis=0)
    mean = 2*N/(n*(n-1))
    timeToNextCoalecentEvent = np.random.exponential(scale=mean, size=vecSize)
    newTotalTime = timeToNextCoalecentEvent+tTot
    logicArrayTau = np.logical_and(newTotalTime > tau, newTotalTime < tau+taub)
    sumArrayTau = np.sum(logicArrayTau)
    bottleneckMean = 2*NPrim/(n*(n-1))
    bottleneckTimeToNextCoalecentEvent = np.random.exponential(scale=bottleneckMean, size=sumArrayTau)
    timeToNextCoalecentEvent[logicArrayTau] = bottleneckTimeToNextCoalecentEvent
    return timeToNextCoalecentEvent

def CoalescentMutationsEvents(scaledMutationProb, n, timeToNextCoalecentEvent, treas,mutationTensor, vecSize,sampleSize,populationSize):
    t = sampleSize-n
    mutationProb = scaledMutationProb/(2*populationSize)
    mean = mutationProb * timeToNextCoalecentEvent
    numMutationsBranch = np.zeros([vecSize,sampleSize])
    for i in range(vecSize):
        numMutationsBranch[i,:] = np.random.poisson(mean[i], size=sampleSize)
    #mutationsInEvent = np.sum(numMutationsBranch[:,:n],axis=1)
    logicIndex = np.zeros([n,vecSize], dtype= np.int16)
    mutationsInEvent=0
    for k in range(vecSize):
        _ , logicIndex[:,k] = np.unique(treas[:,k],return_index=True)
        h = numMutationsBranch[k,logicIndex[:,k]]   # fix so that the right mutations are counted
        mutationsInEvent += np.sum(h, axis=None)
    for k in range(vecSize):
        mutationTensor[:,t,0,k] = numMutationsBranch[k,treas.T[k,:]]

    mutationTensor[:, t, 1, :] = treas
    mutationTensor[0, t, 2, :] = timeToNextCoalecentEvent
    mutationTensor[0, t, 3, :] = mutationsInEvent/vecSize

    for j in range(vecSize):
        temp = (np.random.choice(np.unique(treas[:,j]), size=2,replace=False))
        indexing = np.amax(temp, axis=None)
        treas[treas[:,j]==indexing,j] = np.amin(temp)


    return mutationTensor,treas

def PairwiseCount(mutationTensor,sampleSize,vecSize):
    pairwiseDifferencesAverage = np.zeros([sampleSize,sampleSize,vecSize], dtype=np.float32)
    pairwiseDifferencesVarVec = np.zeros(vecSize, dtype=np.float32)
    for i in range (sampleSize):
        for k in range(vecSize):
            for j in range (sampleSize):
                logicTemp = (mutationTensor[j,i,1,k] != mutationTensor[:,i,1,k])
                pairwiseDifferencesAverage[:,j,k] += (mutationTensor[:,i,0,k]+mutationTensor[j,i,0,k]) * logicTemp
            #pairwiseDifferencesAverage[:,:,k] /= vecSize
            pairwiseDifferencesVarVec[k] = np.sum(pairwiseDifferencesAverage[:,:,k]) / (2)
    pairwiseDifferencesVar = np.var(pairwiseDifferencesVarVec)

    pairwiseDifferencesAverage = np.sum(pairwiseDifferencesAverage)/(vecSize*2)

    mutationVar =np.var( np.sum(mutationTensor[0,:,3,:], axis=1)/(vecSize))
    mutationAverage = np.sum(mutationTensor[0,:,3,:])/(vecSize)


    estimatorTVec = pairwiseDifferencesVarVec / scipy.special.comb(sampleSize, 2)
    estimatorSVec = (np.sum(mutationTensor[0,:,3,:], axis=0)/(vecSize)) / np.sum(1 / np.arange(1, sampleSize - 1))

    estimatorSVar = np.var(estimatorSVec)
    estimatorTVar = np.var(estimatorTVec)

    estimatorT = pairwiseDifferencesAverage / scipy.special.comb(sampleSize, 2)
    estimatorS = mutationAverage / np.sum(1 / np.arange(1, sampleSize - 1))

    dVar =np.var( estimatorTVec - estimatorSVec)
    dAverage = estimatorT - estimatorS
    tajimaD = dAverage/np.sqrt(dVar)

    returnVec = np.zeros([2,4])
    returnVec[0,0] = estimatorS
    returnVec[1,0] = estimatorSVar
    returnVec[0,1] = estimatorT
    returnVec[1,1] = estimatorTVar
    returnVec[0,2] = dAverage
    returnVec[1,2] = dVar
    returnVec[:,3] = tajimaD
    return returnVec


def Coalescent_InfinitSite(mutation = 10,sampleSize = 50, populationSize = 10000000, populationScale = 0.01, tau = 200000, vecSize = 10000):
    N = populationSize
    NPrim = populationSize*populationScale
    taub = NPrim * 3
    n = sampleSize
    timeTensor = np.zeros([sampleSize,vecSize], dtype=np.float32)
    mutationTensor = np.zeros([sampleSize,sampleSize,4,vecSize], dtype=np.float32)
    treas = np.zeros([sampleSize, vecSize], dtype=np.int8)
    treeStart = np.arange(sampleSize,dtype=np.int8)
    # Init treas
    for i in range(vecSize):
        treas[:, i] = treeStart
    mutationTensor[:, 0, 1, :] = treas
    for j in range(sampleSize-1):
        timeToNextCoalecentEvent = CoalescentEventTime(tau, taub, timeTensor, n, N, NPrim, vecSize)
        timeTensor[j,:] = timeToNextCoalecentEvent
        mutationTensor,treas = CoalescentMutationsEvents(mutation, n, timeToNextCoalecentEvent, treas,mutationTensor, vecSize, sampleSize,populationSize)
        n -= 1
    estimatorReturnVec = PairwiseCount(mutationTensor, sampleSize, vecSize)
    timeReturnTensor = timeTensor
    return estimatorReturnVec, timeReturnTensor
Coalescent_InfinitSite(10,50,10000000,1,100000,1000)