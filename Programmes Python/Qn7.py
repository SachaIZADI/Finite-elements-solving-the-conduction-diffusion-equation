##Question 7 - schéma de Crank-Nicholson - équation de convection-diffusion

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


## Définition des matrices B = I_N + K_c - A et C = I_N - K_c + A tq U_{n+1} = inv(B)*C*U_{n} = D*U_{n}
Kc=np.zeros((Nx,Nx))
Kc[0][1]=1
Kc[Nx-1][Nx-2]=-1
Kc[Nx-1][Nx-1]=1
for j in range(1,Nx-1):
    Kc[j][j-1]=-1
    Kc[j][j+1]=1

A=np.zeros((Nx,Nx))
A[0][0]=-2
A[0][1]=1
for j in range(1,Nx-1):
    A[j][j-1]=1
    A[j][j]=-2
    A[j][j+1]=1
A[Nx-1][Nx-1]=-1
A[Nx-1][Nx-2]=1

B = np.eye(Nx)+V*dt/(4*dx)*Kc-nu*dt/2/(2*(dx)**2*dt)*A
C = np.eye(Nx)-V*dt/(4*dx)*Kc+nu*dt/2/(2*(dx)**2*dt)*A
inv_B=alg.inv(B)
D = np.dot(inv_B,C)


## Mise en oeuvre de la résolution
clf
for i in range(Nt+1):
    U=np.dot(D,U)
    if (dt*i-floor(dt*i)==0) :
        plot(X,U)
        legend_plot+=["$\ u(t="+str(round(i*dt,2))+",\ x) $"]
        

grid()
title("$\ Convection-diffusion\ avec\ source$")
xlabel("$\ x $")
ylabel("$\ u(t,x) $")
legend(legend_plot, loc="best")

show()
