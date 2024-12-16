import numpy as np
import glob
from matplotlib import pyplot as plt
# from lmfit import minimize, Parameters, report_fit
from scipy.optimize import least_squares
# %%
lista = sorted(glob.glob("*.dat"), key=len)


def abrir(archivo):
    return np.loadtxt(archivo, unpack=True)


q, I = abrir(lista[0])
# %%


def gp(params, q):
    G = params[0]
    R = params[1]
    d = params[2]
    q_1 = pow((1/R*(3*d/2)), (1/2))
    model = np.piecewise(q, [q<=q_1, q>q_1], [lambda q:G*np.exp((pow(q, 2)*pow(R, 2)/3)), lambda q: G*(np.exp(-d/2)*pow((3*d/2), (d/2))*(1/R))/pow(q, d)])
    # model = ((q <= q_1) * (G*np.exp((pow(q, 2)*pow(R, 2)/3))) + (q > q_1)
               # * (G*(np.exp(-d/2)*pow((3*d/2), (d/2))*(1/R))/pow(q, d)))
    return model

param_list = []

def residual(params, q, I):
    y_modelo = gp(params, q)
    # plt.clf()
    # plt.plot(q,I,'o',q,y_modelo,'r-')
    # plt.pause(0.05)
    param_list.append(params)
    return y_modelo - I


# params = Parameters()
# params.add('G', value=1,  vary=True)
# params.add('R', value=1,  vary=True)
# params.add('d', value=1,  vary=True)

# Filtro los primeros datos
ar = np.where(q > 0.2)
q_new, I_new = q[ar], I[ar]

parametros_iniciales = [1,1,1]
res = least_squares(residual, parametros_iniciales, args=(q_new, I_new), verbose=1)
print('Los parámetros G,R y d son:',res.x)
# res = out.params.valuesdict()
# print(report_fit(out))
# yf = gp(res.x, q_new)

# Calculamos la matriz de covarianza "pcov"
def calcular_cov(res,y_datos):
    U, S, V = np.linalg.svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * S[0]
    S = S[S > threshold]
    V = V[:S.size]
    pcov = np.dot(V.T / S**2, V)

    s_sq = 2 * res.cost / (y_datos.size - res.x.size)
    pcov = pcov * s_sq
    return pcov

pcov = calcular_cov(res,I_new)

# De la matriz de covarinza podemos obtener los valores de desviación estándar
# de los parametros hallados
pstd = np.sqrt(np.diag(pcov))

#
y_modelo = gp(res.x, q_new)

# %%
plt.plot(q_new, I_new)
plt.plot(q_new,y_modelo)
# plt.xlim(0.1,1)
# plt.ylim(0.3e6, 4e6)
plt.xlabel('q$(nm^{-1})$')
plt.ylabel('Intensity')
#%%
plt.figure()
# plt.plot(q_new, I_new ,  'o', markersize=4, label='datos')
plt.plot(q_new, y_modelo, 'r-',               label='modelo fiteado')
# plt.xlim(0.1,1)
# plt.ylim(0.3e6, 4e6)
plt.xlabel('q$(nm^{-1})$')
plt.ylabel('Intensity')
plt.legend(loc='best')
plt.tight_layout()
