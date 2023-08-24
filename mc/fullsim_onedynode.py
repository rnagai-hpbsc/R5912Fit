import numpy as np
import matplotlib.pyplot as plt
import click 
from tqdm import tqdm 
import tables

m_e = 511e3 # eV

def dynode_SEE(
        i_energy, # incident energy 
        eloss = 1, # eV
        prob_bs = 0.0001, # back scatter probability
        qreso = 0
    ):

    E = i_energy
    n_see = 0

    v = np.sqrt(E**2 + 2*E*m_e)/(E+m_e)

    while E > 0:
        p = np.random.rand()
        n_see += 1
        E = E*(-1) if p < prob_bs else (E - eloss) if E > eloss else 0

    if qreso > 0: n_see *= np.random.normal(loc=1,scale=qreso)
    
    return n_see, E

@click.command()
@click.option('--pois',type=float,default=0.1,help='Poisson mean (default=0.1)')
@click.option('--nevt',type=int,default=1000,help='Number of Entries (default=100)')
@click.option('--hv',type=float,default=500,help='Applied High Voltage (default=500V)')
@click.option('--preso',type=float,default=0,help='Pedestal resolution [%] (default=0)')
@click.option('--qreso',type=float,default=0,help='Charge resolution [%] (default=0)')
@click.option('--nph',type=float,default=np.nan,help='Number of Photons instead of Poisson (default=None)')
@click.option('--bs',type=float,default=0.0001,help='back scatter probability (default=0.0001)')
@click.option('--linear',is_flag=True)
def main(pois,nevt,hv,preso,qreso,nph,bs,linear):
    data = {'all':[], 'bs':[]}
    data_gaus = {'all':[], 'bs':[]}
    # cathode-dynode distance 
    length = 10 # cm 
    if np.isnan(nph):
        pois_pdf = np.random.poisson(lam=pois,size=nevt)
    else:
        pois_pdf = np.zeros(nevt) + int(nph)

    for n_ph in tqdm(pois_pdf,leave=False):
        n_pe = 0
        n_pe_bs = np.nan
        for j in range(int(n_ph)):
            n_pe_, outE = dynode_SEE(i_energy=hv,prob_bs=bs,qreso=qreso)
            n_pe += n_pe_
            if outE < 0 : 
                if np.isnan(n_pe_bs): n_pe_bs = 0
                n_pe_bs += n_pe_
        data['all'].append(n_pe)
        data['bs'].append(n_pe_bs)

    for key in data:
        data_gaus[key] = [np.random.normal(loc=n,scale=preso/100) for n in np.array(data[key])/hv]

    h_max = 1.2 * np.max(data_gaus['all'])
    h_min = 1.2 * np.min(data_gaus['all'])
    n_bins = 500#int((h_max-h_min+1)*10)

    y,    x, _ = plt.hist(data_gaus['all'],bins=n_bins,range=(h_min,h_max),histtype='step',label='All')
    y_bs, _, _ = plt.hist(data_gaus['bs'],bins=n_bins,range=(h_min,h_max),histtype='step',label='BS')
    plt.plot([(x[i+1]+x[i])/2 for i in range(len(x)-1)], y-y_bs)
    plt.legend()
    if linear:
        if np.isnan(nph): plt.ylim(0,np.max(y)*pois*(1-qreso)*np.sqrt(2*np.pi)*qreso)
    else:
        plt.yscale('log')
    plt.show()

if __name__=='__main__':
    main()
