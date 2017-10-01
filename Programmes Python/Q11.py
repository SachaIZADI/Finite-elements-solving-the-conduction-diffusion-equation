##Question 11 - résolution de l'équation de convection-diffusion

from math import *
import numpy as np
from matplotlib.pyplot import*
import scipy.sparse as sp
import scipy.sparse.linalg as alg

##Initialisation des paramètres
global x0, y0, s
x0 = 25
y0=25
s = 1
def f(x,y) :
    return((s*sqrt(2*pi))**(-1)*exp(-((x-x0)**2+(y-y0)**2)/(2*s**2)))

global L, T, nu, dx, dt
L=50
T=20
nu=1
dt=0.1
dx=0.5
N=floor(L/dx)
Nt=floor(T/dt)

#u_0=f_{x0,y0,s}
U=np.zeros((N**2,1))
for i in range(N):
    for j in range(N):
        U[i*N+j]=f((i+1)*dx, (j+1)*dx)



## On écrit le système R*U^{n+1} = S*U^n
B=4*sp.eye(N)-sp.eye(N,k=1)-sp.eye(N,k=-1)
diagonals=np.array([-B for k in range(N)])
A=sp.block_diag(diagonals)
A=A+sp.eye(N**2,k=N)+sp.eye(N**2,k=-N)

Kcx=1/(2*dx)*(sp.eye(N**2,k=N)-sp.eye(N**2,k=-N))
Vy=sp.eye(N,k=1)-sp.eye(N,k=-1)
diagonals=np.array([Vy for k in range(N)])
Kcy=1/(2*dx)*sp.block_diag(diagonals)

R=sp.identity(N**2)+dt/2*(Kcx+Kcy)-nu*dt/2*A
S=sp.identity(N**2)-dt/2*(Kcx+Kcy)+nu*dt/2*A

clf()
title("Solution approchée de l'équation de convection-diffusion")

## On résout le système R*U^{n+1} = S*U^n
alpha=0
for t in range (Nt):
    Sprime=sp.csr_matrix(S)
    V=Sprime.dot(U)
    U=alg.spsolve(R,V)

    if t%50==0 :
        alpha+=1
        # Afficher les matrices U(x_i,y_j) de taille N*N (et pas le vecteur de taille N^2)
        U_bis=np.eye(N)
        i=-1
        for k in range(N**2):
            if (k%N==0):
                i+=1
            U_bis[i][k%N]=U[k]
        subplot(220+alpha)
        grid()
        imshow(U_bis,origin='lower')
        colorbar()

show()