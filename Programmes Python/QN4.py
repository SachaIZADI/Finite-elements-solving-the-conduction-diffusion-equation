##Question 4 - schéma de Crank-Nicholson - équation de convection

from math import *
import numpy as np
import numpy.linalg as alg
from matplotlib.pyplot import*
from matplotlib import animation


##Initialisation des paramètres
global x0, s
x0 = 20
s = 1
def f(x) :
    return((s*sqrt(2*pi))**(-1)*exp(-(x-x0)**2/(2*s**2)))

global L, T, V, nu, dx, dt
L=50
T=5
V=1
nu=1
dx=0.1
dt=0.025

Nx=floor(L/dx)
Nt=floor(T/dt)

U=np.zeros((Nx,1))
for j in range(Nx):
    U[j]=f(j*dx)
    
X=np.zeros((Nx,1))
for j in range(Nx):
    X[j]=j*dx

legend_plot=[] #tableau servant à mettre une légende sur le graphe


## Définition des matrices A = I_N + K_c et B = I_N - K_c tq U_{n+1} = inv(A)*B*U_{n} = C*U_{n}
Kc=np.zeros((Nx,Nx))
Kc[0][1]=1
Kc[Nx-1][Nx-2]=-1
for j in range(1,Nx-1):
    Kc[j][j-1]=-1
    Kc[j][j+1]=1
    
Kd=np.zeros((Nx,Nx))
Kd[0][0]=1
for j in range(1,Nx):
    Kd[j][j-1]=-1
    Kd[j][j]=1

A = np.eye(Nx)+V*dt/(4*dx)*Kc
B = np.eye(Nx)-V*dt/(4*dx)*Kd
C=np.dot(alg.inv(A),B)

## Mise en oeuvre de la résolution
for i in range(Nt+1):
    U=np.dot(C,U)
    if i%50==0 :
        plot(X,U)
        legend_plot+=["$\ u(t="+str(round(i*dt,2))+",\ x) $"]
        

grid()
title("$\ Equation\ de\ convection\ unidimensionnelle\ -\ Schéma\ de\ Crank-Nicholson$")
xlabel("$\ x $")
ylabel("$\ u(t,x) $")
legend(legend_plot, loc="best")

show()
