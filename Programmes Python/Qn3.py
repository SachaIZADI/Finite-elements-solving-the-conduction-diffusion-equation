##Question 3 - comparaison du schéma explicite décentré amont sous la condition CFL et sans la condition CFL - équation de convection

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
T=16
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

# Autres valeurs du pas de temps pour mettre en évidence la condition CFL
dx1=0.1
dt1=0.3

Nx1=floor(L/dx1)
Nt1=floor(T/dt1)

U1=np.zeros((Nx1,1))
for j in range(Nx1):
    U1[j]=f(j*dx1)
    
X1=np.zeros((Nx1,1))
for j in range(Nx1):
    X1[j]=j*dx1

legend_plot=[] #tableau servant à mettre une légende sur le graphe


## Définition des matrices A =I_N + K_c tq U_{n+1}=A*U_{n}
Kd=np.zeros((Nx,Nx))
Kd[0][0]=1
for j in range(1,Nx):
    Kd[j][j-1]=-1
    Kd[j][j]=1

A = np.eye(Nx)-V*dt/(2*dx)*Kd

Kd1=np.zeros((Nx1,Nx1))
Kd1[0][0]=1
for j in range(1,Nx1):
    Kd1[j][j-1]=-1
    Kd1[j][j]=1

A1 = np.eye(Nx1)-V*dt1/(2*dx1)*Kd1

plot(X,U)
## Mise en oeuvre de la résolution
for i in range(Nt+1):
    U=np.dot(A,U)
plot(X,U)


for i in range(Nt1+1):
    U1=np.dot(A1,U1)
plot(X1,U1)


grid()
title("$\ Equation\ de\ convection\ unidimensionnelle\ -\ Schéma\ explicite\ décentré\ amont$")
xlabel("$\ x $")
ylabel("$\ u(t,x) $")
legend(["$u^0(x)$","$u^n(x)\ sous\ CFL$", "$u^n(x)\ sans\ CFL$"], loc="best")

show()
