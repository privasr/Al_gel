import numpy as np
import glob
from matplotlib import pyplot as plt
from lmfit import minimize, Parameters, report_fit
import math
# from scipy.optimize import least_squares
# %%
lista = sorted(glob.glob("*.dat"), key=len)


def abrir(archivo):
    return np.loadtxt(archivo, unpack=True)


# q, I = abrir(lista[0])
#%%

def residual(params,q,I):
    G = params['G']
    R = params['R']
    e = params['e']
    s = params['s']
    '''e=d-s'''
    q_1 = (1/R) * math.sqrt(e*(3-s)/2)
    D = G * pow(q_1,e) * np.exp(-pow(q_1,2) * pow(R,2)/(3-s))
    model = (q <= q_1) * (G / pow(q,s)) * np.exp(-pow(q,2) * pow(R,2)/(3-s)) + (q > q_1) * D / pow(q,e+s)  
    return (I - model)

params = Parameters()
params.add('G', value = 1,  vary = True, min=0)
params.add('R', value = 1,  vary = True, min=10e-10)
params.add('e', value = 1,  vary = True, min=0)
params.add("s", value = 0,  vary = True, min = 0, max = 2.99)

#%%
i = 0
while i<len(lista):
    q, I = abrir(lista[i])
    #Filtro los primeros datos
    ar = np.where(q>0.2)
    q_new, I_new = q[ar], I[ar]
    out = minimize(residual, params, args = (q_new,I_new))
    res = out.params.valuesdict()
    print('La serie de datos corresponde al archivo:', lista[i])
    print(report_fit(out))
    yf = I_new - out.residual
    i += 1
    
#%%
plt.figure()
plt.plot(q_new, I_new ,'o', markersize=4, label='datos')
# plt.plot(q_new, yf, 'r-', label='modelo fiteado')
plt.xlabel('q$(nm^{-1})$')
plt.ylabel('Intensity')
plt.legend(loc='best')
plt.tight_layout()
#%%