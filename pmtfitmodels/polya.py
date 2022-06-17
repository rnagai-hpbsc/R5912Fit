import numpy as np
from scipy.stats import poisson, norm
from scipy.special import erf, gamma, erfc

def polyadist(x,gain,sigma,b):
    return (gain*b)**(-1/b)/gamma(1/b)*x**(1/b-1)*np.exp(-x/gain/b)*erfc((x-gain)/np.sqrt(2)/sigma) if x*gain*b>0 else 0

def upolya(x,gain,sigma,b):
    vec_polya = np.vectorize(polyadist,otypes=[float])
    return vec_polya(x,gain,sigma,b)

def polyaiPE(x,pmean,gain,sigma,daqsigma,p1,b,i):
    totalsigma = np.sqrt(i*sigma**2+daqsigma**2)
    return poisson.pmf(i,pmean)*(p1*upolya(i*gain,totalsigma,b)+(1-p1)*norm.pdf(x,i*gain,totalsigma))

def polyaMPE(x,pmean,gain,sigma,daqsigma,p1,N,tnum):
    result = x*0
    for i in range(tnum):
        result += polyaiPE(x,pmean,gain,sigma,daqsigma,p1,b,i)
    return result * N
