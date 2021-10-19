import numpy as np 
from scipy.stats import poisson, norm
from scipy.special import comb
import os 
import matplotlib.pyplot as plt 
import click

@click.command()
@click.option('--scale',type=float,default=1.e7)
@click.option('--pmean',type=float,default=0.5)
@click.option('--gain',type=float,default=1.)
@click.option('--mpe',type=int,default=10)
@click.option('--chres',type=float,default=0.3)
@click.option('--eres',type=float,default=0.05)
def main(scale,pmean,gain,mpe,chres,eres):
    qres = chres * gain

    x = np.linspace(-2,10,480)
    SPEymax = getIdealNphotonDist(gain,mpe,pmean,gain,qres,eres,scale)*1.1

    plt.figure()
    plt.plot(x,getIdealNphotonDist(x,mpe,pmean,gain,qres,eres,scale))
    plt.plot(x,getIdealNphotonDists(x,mpe,pmean,gain,qres,eres,scale)[1],ls=':',color='tab:blue')
    plt.plot(x,getIdealNphotonDists(x,mpe,pmean,gain,qres,eres,scale)[2],ls=':',color='tab:blue')
    plt.plot(x,get1lossR5912Dist(x,mpe,pmean,gain,qres,eres,scale))
    plt.plot(x,get1lossR5912Dists(x,mpe,pmean,gain,qres,eres,scale)[1],ls=':',color='tab:orange')
    plt.plot(x,get1lossR5912Dists(x,mpe,pmean,gain,qres,eres,scale)[2],ls=':',color='tab:orange')
    plt.plot(x,get1lossR5912LossDist(x,mpe,pmean,gain,qres,eres,scale),ls=':',color='tab:red')

    plt.xlim(-.5,3)
    plt.ylim(0,SPEymax)
    #plt.yscale('log')
    #plt.ylim(1e-18,1e-2)
    plt.show()
    return

def getNphotonDist(x,n,pmean,gain,res,elecres,scale):
    results = 0
    for k in range(n+1):
        results += poisson.pmf(k,pmean)*getKphotonGauss(x,k,gain,res,elecres,scale)
    return results

def get1lossR5912LossDist(x,n,pmean,gain,res,elecres,scale):
    results = get1lossR5912Dists(x,n,pmean,gain,res,elecres,scale)
    return results[len(results)-2]

def get1lossR5912Dist(x,n,pmean,gain,res,elecres,scale):
    results = get1lossR5912Dists(x,n,pmean,gain,res,elecres,scale)
    return results[len(results)-1]

def get1lossR5912Dists(x,n,pmean,gain,res,elecres,scale):
    a = np.array([16.8, 4, 5, 3.33, 1.67, 1, 1.2, 1.5, 2.2, 3, 2.4]) # R5912-100 Divider

    # gain is calculated by g = A^{n} * R_1 * ...* R_n * V^{kn} 
    # where A, k are constant, R_n is the divider ratio, and V is the bias voltage. 

    loss1 = 0.05
    k = 0.75
    Dy1 = a[0]**k * (gain*scale / a[:10].prod()**k)**0.1
    Dy2_10 = ((a[0]+a[1])*a[2:10].prod())**k * (gain*scale / a[:10].prod()**k)**0.9
    print (f'{a.prod()}, {Dy1}, {Dy2_10}, {gain*scale}, {Dy1*Dy2_10}')

    results = []
    total = 0
    loss = 0
    for i in range(n+1):
        contrib = 0
        for j in range(i+1):
            dist = poisson.pmf(i,pmean)*(comb(i,j)*((1.-loss1)**j)*(loss1**(i-j))*getKphotonGauss(x,(i and 1),j*gain+(i-j)*Dy2_10/scale,res*(j+(i-j)*Dy2_10/gain/scale),elecres,scale))
            contrib += dist
            if j!=i: 
                loss += dist
        results.append(contrib)
        total += contrib
    results.append(loss)
    results.append(total)
    return results

def get2DynodeDist(x,n,pmean,gain,res,elecres,scale):
    Dy1 = 1.5*gain*scale
    Dy2 = 2./3.*gain*scale 
    loss1 = 0.05
    results = 0
    for k in range(n+1): 
        results += poisson.pmf(k,pmean)*((1.-loss1)*getKphotonGauss(x,k,gain,res,elecres,scale)+loss1*getKphotonGauss(x,k,Dy2,res,elecres,scale))
    return results

def getIdealNphotonDist(x,n,pmean,gain,res,elecres,scale):
    results = getIdealNphotonDists(x,n,pmean,gain,res,elecres,scale)
    return results[len(results)-1]

def getIdealNphotonDists(x,n,pmean,gain,res,elecres,scale):
    results = []
    total = 0
    for k in range(n+1):
        contrib = poisson.pmf(k,pmean)*getKphotonGauss(x,k,gain,res,elecres,scale)
        results.append(contrib)
        total += contrib
    results.append(total)
    return results 

def getKphotonGauss(x,k,gain,res,elecres,scale): 
    return norm.pdf(x*scale,loc=k*gain*scale,scale=np.sqrt(k*res**2+elecres**2)*scale)

if __name__ == "__main__":
    main()
