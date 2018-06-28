import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.patches import Polygon

def Chi2Calculations():
    returnVal = np.zeros([2,26])
    x = np.flip(np.arange(0,26,1),axis=0)
    y = np.arange(25,51,1)
    chi2num = (x - 50*0.495)**2/(50*0.495)  +(y-50*0.495)**2/(50*0.495)
    returnVal[0,:] = chi2num
    returnVal[1,chi2num<3.841] = 1
    return returnVal

def func(population):
    newPopulationIndexParent1 = np.random.randint(50, size=50)
    newPopulationIndexParent2 = np.random.randint(50, size=50)
    newAllelIndexParent1 = np.random.randint(2, size=50)
    newAllelIndexParent2 = np.random.randint(2, size=50)

    Parent1Allele = np.append(newPopulationIndexParent1, newAllelIndexParent1, axis=0)
    Parent1Allele = Parent1Allele.reshape((2, 50))
    Parent2Allele = np.append(newPopulationIndexParent2, newAllelIndexParent2, axis=0)
    Parent2Allele = Parent2Allele.reshape((2, 50))
    offsprings = np.append(population[Parent1Allele[0, :], Parent1Allele[1, :]],
                           population[Parent2Allele[0, :], Parent2Allele[1, :]])
    offsprings = offsprings.reshape((2, 50))
    counter1 = np.sum(np.logical_xor(offsprings[0, :], offsprings[1, :]))
    return counter1
def main():
    populationSize = 50
    triles  = 1000000
    heterozygotStart = 0.5

    AlleleRate = 0.5

    chi2Val = Chi2Calculations()

    population = np.zeros([50,2], dtype=np.int8)

    population[:25,:] = np.array([1,0])
    population[25:38,:] = np.array([1,1])


    num_cores = multiprocessing.cpu_count()
    count = Parallel(n_jobs=num_cores)(delayed(func)
                                            (population)
                                            for i in range(triles))

    #count = (np.asarray(count)-50*0.495)**2/25
    count = np.asarray(count)/50
    #count = count[count>=0]
    #HistogramCount, bin = np.histogram(count, bins=np.arange(min(count),max(count),0.01))



    fig, ax = plt.subplots()

    n_bins = len(set(count))
    print(n_bins)
    #ax.hist(count,bins=n_bins,density=True,alpha = 0.5, label='Simulated chi2')

    df = 1
    x = np.linspace(chi2.ppf(0.2, df), chi2.ppf(0.99, df), 100)
    ax.plot(x, chi2.pdf(x, df),'r-', lw = 1, alpha = 1, label = 'chi2 pdf')


    plt.ylim(ymin=0)

    a = np.amax(chi2Val[0, chi2Val[1, :] == 1])
    b = 10
    xi = np.linspace(a, b, 100)
    verts1 = [(a, 0)] + list(zip(xi, chi2.pdf(xi, df))) + [(b, 0)]
    poly1 = Polygon(verts1, alpha=1, facecolor='b', edgecolor='0.5', label=r'$\chi^2$ 0.95>')

    a = chi2Val[0, 5]

    xi = np.linspace(a, b, 100)
    verts2 = [(a, 0)] + list(zip(xi, chi2.pdf(xi, df))) + [(b, 0)]
    poly2 = Polygon(verts2, alpha = 0.5, facecolor='r', edgecolor='0.5', label='Heterozygosy < 0.42')
    ax.add_patch(poly2)
    ax.add_patch(poly1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\chi^2$')
    plt.suptitle(r'Significance of $\chi^2$ test', fontsize=16)

    #ax.title(r'Validation of using $\chi^2$ test')
    plt.show()


if __name__  ==  '__main__':
    main()