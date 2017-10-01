##Question 9 - Evaluation de l'erreur = f(h)

from math import *
import numpy as np
from matplotlib.pyplot import*
import scipy.sparse as sp
import scipy.sparse.linalg as alg

##Initialisation des paramètres

global L
L=50

DX=[0.1,0.15, 0.2, 0.25, 0.3,0.4,0.5]
#DX=[0.05+j*0.5/1000 for j in range(1001)]
Erreur=[]

def f(x,y) :
    return(sin(2*pi*x/L)*sin(2*pi*y/L))

    
for dx in DX :
    N=floor(L/dx)
    
    b=np.zeros((N**2,1))
    for i in range(N):
        for j in range(N):
            b[i*N+j]=f((i+1)*dx, (j+1)*dx)
    b=dx**2*b
    
    B=4*sp.eye(N)-sp.eye(N,k=1)-sp.eye(N,k=-1)
    diagonals=np.array([-B for k in range(N)])
    A=sp.block_diag(diagonals)
    A=A+sp.eye(N**2,k=N)+sp.eye(N**2,k=-N)
    
    U=alg.spsolve(-A,b) #solution numérique

    F=np.eye(N)
    U_bis=np.eye(N)
    i=-1
    for k in range(len(b)):
        if (k%N==0):
            i+=1
        F[i][k%N]=b[k]
        U_bis[i][k%N]=U[k]

    V=L**2/(8*pi**2*dx**2)*F #solution exacte
    
    # On définit l'erreur comme la différence entre la solution approchée et la solution exacte au point (i_max, i_max) où la solution exacte est maximale (i.e. en L/(4dx)) - on normalise cet écart par rapport à la valeur de la solution exacte en ce point.
    i_max=floor(L/(4*dx))
    Err = abs((U_bis[i_max][i_max]-V[i_max][i_max])/V[i_max][i_max])
    Erreur+=[Err]


clf()
grid()
title("Erreur=$f(\Delta x)$")
xlabel("$\Delta x$")
ylabel("Erreur = $(u(x_i,y_j)-U_{i,j})/u(x_i,y_j)$")
plot(DX, Erreur)
show()