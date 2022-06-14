import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import poisson,norm
from scipy.special import erf, gamma, erfc
import click

def MpeFunc(x,pmean,gain,qreso,elfluc,npe):
    return poisson.pmf(npe,pmean)*norm.pdf(x,npe*gain,np.sqrt(npe*qreso**2+elfluc**2))

def idealFunc(x,pmean,gain,qreso,elfluc,N,tnum):
    result = x**0-1
    for i in range(tnum):
        result += MpeFunc(x,pmean,gain,qreso,elfluc,i)
    return result*N

def furrydist(x,gain,sigma):
    if (gain>0): 
        result = 1/gain*np.exp(-x/gain)*(1+erf(x*np.sqrt(2)/sigma))
    else:
        result = 0
    return result

ufurry = np.vectorize(furrydist,otypes=[float])

def polyadist(x,gain,sigma,b):
    if (x>0) & (gain>0) & (b>0):
        result = (gain*b)**(-1/b)/gamma(1/b)*x**(1/b-1)*np.exp(-x/gain/b)*erfc((x-gain)/np.sqrt(2)/sigma)
    elif (x>0) & (gain>0) & (b==0):
        result = gain**x/gamma(x+1)*np.exp(-gain)
    else:
        result = 0
    return result

upolya = np.vectorize(polyadist,otypes=[float])

def furryFunc(x,pmean,gain,qreso,elfluc,p1,N,tnum):
    result = x*0
    for i in range(tnum):
        poisson_term = poisson.pmf(i,pmean)
        furry_term = ufurry(x,i*gain,np.sqrt(i*qreso**2+elfluc**2))
        norm_term = norm.pdf(x,i*gain,np.sqrt(i*qreso**2+elfluc**2))
        result += poisson_term * (p1*furry_term + (1-p1)*norm_term)
    return result

def polyaFunc(x,pmean,gain,qreso,elfluc,p1,b,N,tnum):
    result = x*0
    for i in range(tnum):
        poisson_term = poisson.pmf(i,pmean)
        polya_term = upolya(x,i*gain,np.sqrt(i*qreso**2+elfluc**2),b)
        norm_term = norm.pdf(x,i*gain,np.sqrt(i*qreso**2+elfluc**2))
        result += poisson_term * (p1*polya_term + (1-p1)*norm_term)
    return result

pmean = 0.3
gain = 1
qreso = 0.3
elfluc = 0.05
N = 1
NPE = 3
p1 = 0.3
p2 = 0.1
b = 0.5
d = 0.1

xdata = np.linspace(-1,4,500)

@click.group()
def cli():
    pass

@cli.command()
def ideal():
    ydata = idealFunc(xdata,pmean,gain,qreso,elfluc,N,NPE)
    plt.figure(figsize=(5,3.75))
    plt.plot(xdata,ydata,color='blue',lw=2,label='total')
    for i in range(NPE):
        plt.plot(xdata, MpeFunc(xdata,pmean,gain,qreso,elfluc,N,i),ls=':',label=f'{i:d}PE')
    plt.xlabel('Number of photo-electrons')
    plt.ylabel('Probability')
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.97)
    plt.grid()
    plt.legend()
    plt.savefig('plots/whole.pdf')
    plt.ylim(-0.06,0.46)
    plt.savefig('plots/detail.pdf')
    plt.show()

@cli.command()
def furry():
    ydata = furryFunc(xdata,pmean,gain,qreso,elfluc,p1,N,NPE)
    plt.figure(figsize=(5,3.75))
    plt.plot(xdata,ydata,color='blue',lw=2,label='total')
    plt.xlabel('Number of photo-electrons')
    plt.ylabel('Probability')
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.97)
    plt.grid()
    plt.legend()
    #plt.savefig('plots/whole.pdf')
    plt.ylim(-0.06,0.46)
    #plt.savefig('plots/detail.pdf')
    plt.show() 

@cli.command()
def polya():
    ydata = polyaFunc(xdata,pmean,gain,qreso,elfluc,p1,b,N,NPE)
    plt.figure(figsize=(5,3.75))
    plt.plot(xdata,ydata,color='blue',lw=2,label='total')
    plt.xlabel('Number of photo-electrons')
    plt.ylabel('Probability')
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.97)
    plt.grid()
    plt.legend()
    #plt.savefig('plots/whole.pdf')
    plt.ylim(-0.06,0.46)
    #plt.savefig('plots/detail.pdf')
    plt.show() 

if __name__ == '__main__':
    cli()
