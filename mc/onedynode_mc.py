import numpy as np
import matplotlib.pyplot as plt
import click
from tqdm import tqdm

@click.command()
@click.option('--nevent',type=int,default=100)
@click.option('--ienergy',type=float,default=100)
@click.option('--poisson',type=float,default=0.1)
@click.option('--qreso',type=float,default=0.2)
@click.option('--preso',type=float,default=0.05)
@click.option('--rebin',type=int,default=1)
@click.option('--probbs',type=float,default=0.01)
@click.option('--linear',is_flag=True,default=False)
def main(nevent,ienergy,poisson,qreso,preso,rebin,probbs,linear):
    data = []
    poisson_pdf = np.random.poisson(lam=poisson,size=nevent)
    for nphotons in tqdm(poisson_pdf,leave=False):
        n_pe = 0
        for j in range(nphotons):
            n_pe += onedynode(ienergy,qreso,probbs) 
        data.append(n_pe/ienergy)
    data_normal = np.array([np.random.normal(loc=n,scale=preso) for n in data])

    hist_max = 1.2 * np.max(data_normal)
    hist_min = 1.2 * np.min(data_normal)
    #plt.hist(data,bins=int((hist_max-hist_min+1)*ienergy/rebin),range=(hist_min,hist_max),histtype='step')
    plt.hist(data_normal,bins=int((hist_max-hist_min+1)*ienergy/rebin),range=(hist_min,hist_max),histtype='step',label='All')
    plt.hist(data_normal[poisson_pdf==1],bins=int((hist_max-hist_min+1)*ienergy/rebin),range=(hist_min,hist_max),histtype='step',label='1PE')
    plt.hist(data_normal[poisson_pdf==2],bins=int((hist_max-hist_min+1)*ienergy/rebin),range=(hist_min,hist_max),histtype='step',label='2PE')
    plt.hist(data_normal[poisson_pdf==3],bins=int((hist_max-hist_min+1)*ienergy/rebin),range=(hist_min,hist_max),histtype='step',label='3PE')
    plt.legend()
    if not linear:
        plt.yscale('log')
    plt.show()

def onedynode(i_energy,qreso,prob_back_scatter):
    mean_energy_loss_per_unit_distance = 1 # eV

    energy = i_energy
    n_se = 0
    while energy > 0:
        prob = np.random.rand()
        if prob < prob_back_scatter:
            energy *= (-1)
            n_se += 1
            break
        energy -= mean_energy_loss_per_unit_distance
        n_se += 1
    return n_se * np.random.normal(loc=1,scale=qreso)

if __name__ == '__main__':
    main()
