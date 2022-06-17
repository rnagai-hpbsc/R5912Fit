import numpy as np 
from scipy.stats import poisson, norm
from scipy.special import erf

def furrydist(x,gain,sigma):
    return 1/gain*np.exp(-x/gain)*(1+erf(x/np.sqrt(2)/sigma)) if gain>0 else 0

def ufurry(x,gain,sigma):
    vec_furry = np.vectorize(furrydist,otypes=[float])
    return vec_furry(x,gain,sigma)

def furryiPE(x,pmean,gain,sigma,daqsigma,p1,i):
    totalsigma = np.sqrt(i*sigma**2+daqsigma**2)
    return poisson.pmf(i,pmean)*(p1*ufurry(x,i*gain,totalsigma)+(1-p1)*norm.pdf(x,i*gain,totalsigma))

def furryMPE(x,pmean,gain,sigma,daqsigma,p1,N,tnum):
    result = x*0
    for i in range(tnum):
        result += furryiPE(x,pmean,gain,sigma,daqsigma,p1,i)
    return result * N
