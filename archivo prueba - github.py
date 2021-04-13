import numpy as np
import glob
from matplotlib import pyplot as plt
from lmfit import minimize, Parameters, report_fit
#%%
lista = sorted(glob.glob("*.dat"), key = len)

def abrir(archivo):
    return np.loadtxt(archivo, unpack = True)
q,I = abrir(lista[0])
#%%
def gp (q,I):
    G = 1
    R = 1
    d = 1
    q_1 = pow((1/R*(3*d/2)),(1/2))
    model = ((q<=q_1) * (G*np.exp((pow(q,2)*pow(R,2)/3))) + (q>q_1) * (G*(np.exp(-d/2)*pow((3*d/2),(d/2))*(1/R))/pow(q,d)))
    return model

def residual(params, gp):
    G = params['G']
    R = params['R']
    d = params['d']
    
    return (I - gp (q,I))

params = Parameters()
params.add('G', value = 1,  vary = True)
params.add('R', value = 1,  vary = True)
params.add('d', value = 1,  vary = True)

#Filtro los primeros datos
ar = np.where(q>0.2)
q_new, I_new = q[ar], I[ar]

out = minimize(residual, params, args = (q_new,I_new))
res = out.params.valuesdict()
print(report_fit(out))
yf = I - out.residual

#%%
plt.plot(q_new,I_new)
# plt.plot(q,yf)
# plt.xlim(0.1,1)
plt.ylim(0.3e6, 4e6)
plt.xlabel('q$(nm^{-1})$')
plt.ylabel('Intensity')
