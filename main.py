import Process as ps
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import Print as pr

def main():
    scaledMutationRate = 10
    sampleSize = 50
    populationSize = 10000000
    populationScaling = 0.01
    tau = np.arange(populationSize*0.02,populationSize*10.02,populationSize*0.02, dtype=np.float32)
    recursionVec = 10000
    lengthTau = len(tau)
    #estimatorReturnVec = np.zeros([2,4,lengthTau])
    #timeReturnTensor = np.zeros([2,4,lengthTau])

    num_cores = multiprocessing.cpu_count()
    estimatorReturnVec = Parallel(n_jobs=num_cores)(delayed(ps.Coalescent_InfinitSite)
                                            (scaledMutation = scaledMutationRate,
                                             sampleSize = sampleSize,
                                             populationSize = populationSize,
                                             populationScale = populationScaling,
                                             tau = tau[i],
                                             vecSize = recursionVec)
                                            for i in range(lengthTau))
    pr.Plotting(tau = tau, estimatorReturnVec= estimatorReturnVec)
if __name__ == '__main__':
    main()