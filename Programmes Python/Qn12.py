##Question 12 - dispersion du polluant dans l'océan

from math import *
import numpy as np
from matplotlib.pyplot import*
import scipy.sparse as sp
import scipy.sparse.linalg as alg

##Initialisation des paramètres
global x0, y0, s
x0=0
y0=25
s = 1
def f(x,y) :
    return((s*sqrt(2*pi))**(-1)*exp(-((x-x0)**2+(y-y0)**2)/(2*s**2)))

global L, T, nu, dx, dt
L=50
T=16
nu=1
dt=0.1
dx=0.5

Nx=floor(L/dx)
N=Nx
Nt=floor(T/dt)
#u_0=0
U=np.zeros(Nx**2)


## On écrit le système R*U^{n+1} = S*U^n+F
B=4*sp.eye(Nx)-sp.eye(Nx,k=1)-sp.eye(Nx,k=-1)
diagonals=np.array([-B for k in range(Nx)])
A=sp.block_diag(diagonals)
A=A+sp.eye(Nx**2,k=Nx)+sp.eye(Nx**2,k=-Nx)

Kcx=1/(2*dx)*(sp.eye(Nx**2,k=Nx)-sp.eye(Nx**2,k=-Nx))
Vy=sp.eye(Nx,k=1)-sp.eye(Nx,k=-1)
diagonals=np.array([Vy for k in range(Nx)])
Kcy=1/(2*dx)*sp.block_diag(diagonals)

R=sp.identity(Nx**2)+dt/2*(Kcx+Kcy)-nu*dt/2*A
S=sp.identity(Nx**2)-dt/2*(Kcx+Kcy)+nu*dt/2*A
Sprime=sp.csr_matrix(S)

F=np.zeros(Nx**2)
for i in range(Nx):
    for j in range(Nx):
        F[i*Nx+j]=dt*f((i+1)*dx, (j+1)*dx)

clf()
title("Solution approchée de l'équation de convection-diffusion")

## On résout le système R*U^{n+1} = S*U^n +F
alpha=0
for t in range (Nt):
    V=Sprime.dot(U)
    U=alg.spsolve(R,V+F)
    
    if t%40==0 :
        alpha+=1
        # Afficher les matrices U(x_i,y_j) de taille N*N (et pas le vecteur de taille N^2)
        U_bis=np.eye(Nx)
        i=-1
        for k in range(Nx**2):
            if (k%N==0):
                i+=1
            U_bis[i][k%Nx]=U[k]
        subplot(220+alpha)
        grid()
        imshow(U_bis.T,origin='lower')
        colorbar()
show()