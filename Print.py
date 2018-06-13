import matplotlib.pyplot as plt
import numpy as np
import Process as pr

def Plotting(tau = 0,sampleSize = 50, estimatorReturnVec = 0, timeReturnTensor = 0):

    #Get analytic Data



    # Splitting the Data
    data1 = np.zeros([2,5,len(tau)])
    for i in range(len(tau)):
        data1[:,:,i] = np.asarray(estimatorReturnVec[i])

    # Estimator ThetaS
    plt.figure(1)
    strFigureTitle = r"$\Theta$S"
    plt.suptitle(strFigureTitle, fontsize=16)
    plt.text(0.5, 0.04, r'$\tau$', ha='center')


    plt.subplot(2, 1, 1)
    plt.plot(tau, data1[0,0,:], '-')
    plt.title(r'Average $\Theta$S')
    plt.ylabel(r'$\Theta$S')

    plt.subplot(2, 1, 2)
    plt.plot(tau, data1[1,0,:], '-')
    plt.title(r'Variance $\Theta$S')
    plt.ylabel(r'Var( $\Theta$S)')

    # Estimator ThetaT
    plt.figure(2)
    strFigureTitle = r"$\Theta$T"
    plt.suptitle(strFigureTitle, fontsize=16)
    plt.text(0.5, 0.04, r'$\tau$', ha='center')

    plt.subplot(2, 1, 1)
    plt.plot(tau, data1[0, 1, :], '-')
    plt.title(r'Average $\Theta$T')
    plt.ylabel(r'$\Theta$T')

    plt.subplot(2, 1, 2)
    plt.plot(tau, data1[1, 1, :], '-')
    plt.title(r'Variance $\Theta$T')
    plt.ylabel(r'Var( $\Theta$T)')

    # Estimator Tajima's d numirator
    plt.figure(3)
    strFigureTitle = "Tajima's d numirator"
    plt.suptitle(strFigureTitle, fontsize=16)
    plt.text(0.5, 0.04, r'$\tau$', ha='center')

    plt.subplot(2, 1, 1)
    plt.plot(tau, data1[0, 2, :], '-')
    plt.title(r'Average $\Theta$T - $\Theta$T')
    plt.ylabel(r'$\Theta$T - $\Theta$S')

    plt.subplot(2, 1, 2)
    plt.plot(tau, data1[1, 2, :], '-')
    plt.title(r'Variance $\Theta$T - $\Theta$T')
    plt.ylabel(r'Var( $\Theta$T - $\Theta$S)')

    # Estimator Tajima's d numirator
    plt.figure(4)
    strFigureTitle = "Tajima's d"
    plt.suptitle(strFigureTitle, fontsize=16)
    plt.text(0.5, 0.04, r'$\tau$', ha='center')

    plt.plot(tau, data1[0, 3, :], '-')
    plt.title(r'Average $\Theta$T - $\Theta$T')
    plt.ylabel("Tajima's d")

    # Time average

    plt.figure(5)
    strFigureTitle = "Time Average"
    plt.suptitle(strFigureTitle, fontsize=16)
    plt.text(0.5, 0.04, r'$\tau$', ha='center')

    plt.plot(tau, data1[0, 4, :], '-')
    plt.title(r'Average $\Theta$T - $\Theta$T')
    plt.ylabel("time Average")
    plt.xlabel(r'$\tau$')
    plt.show()



