import numpy as np 
import matplotlib.pyplot as plt
from math import gamma
from scipy.stats import poisson,norm

def MpeFunc(x,pmean,gain,qreso,elfluc,N,npe):
    return N*poisson.pmf(npe,pmean)*norm.pdf(x,npe*gain,np.sqrt(npe*qreso**2+elfluc**2))

def idealFunc(x,pmean,gain,qreso,elfluc,N,tnum):
    result = x**0-1
    for i in range(tnum):
        result += MpeFunc(x,pmean,gain,qreso,elfluc,N,i)
    return result

pmean = 0.5
gain = 1
qreso = 0.3
elfluc = 0.05
N = 1
NPE = 3

xdata = np.linspace(-1,4,500)
ydata = idealFunc(xdata,pmean,gain,qreso,elfluc,N,NPE)
print(np.sum(ydata)*0.01)

plt.figure(figsize=(5,3.75))
plt.plot(xdata,ydata,color='blue',lw=2,label='total')
for i in range(NPE):
    plt.plot(xdata, MpeFunc(xdata,pmean,gain,qreso,elfluc,N,i),ls=':',label=f'{i:d}PE')
plt.xlabel('Number of photo-electrons')
plt.ylabel('Probability')
plt.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.97)
plt.grid()
plt.legend()
plt.savefig('whole.pdf')
plt.ylim(-0.06,0.46)
plt.savefig('detail.pdf')
plt.show()

