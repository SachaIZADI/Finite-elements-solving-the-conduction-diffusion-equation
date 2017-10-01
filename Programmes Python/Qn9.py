##Question 9 - résolution de l'équation de Laplace avec des matrices creuses

from math import *
import numpy as np
from matplotlib.pyplot import*
import scipy.sparse as sp
import scipy.sparse.linalg as alg

##Initialisation des paramètres

global L, dx
L=50
dx=0.1

def f(x,y) :
    return(sin(2*pi*x/L)*sin(2*pi*y/L))

def u(x,y) :
    return(L**2*f(x,y)/(8*pi**2))
    

N=floor(L/dx)

# second membre
b=np.zeros((N**2,1))
for i in range(N):
    for j in range(N):
        b[i*N+j]=f((i+1)*dx, (j+1)*dx)
b=dx**2*b

# matrice de laplacien
B=4*sp.eye(N)-sp.eye(N,k=1)-sp.eye(N,k=-1)
diagonals=np.array([-B for k in range(N)])
A=sp.block_diag(diagonals)
A=A+sp.eye(N**2,k=N)+sp.eye(N**2,k=-N)

# résolution du système
U=alg.spsolve(-A,b)

## Afficher les matrices U(x_i,y_j) de taille N*N (et pas le vecteur de taille N^2)
F=np.eye(N)
U_bis=np.eye(N)
i=-1
for k in range(len(b)):
    if (k%N==0):
        i+=1
    F[i][k%N]=b[k]
    U_bis[i][k%N]=U[k]

V=L**2/(8*pi**2*dx**2)*F

clf()
grid()
title("Solution approchée de $\Delta u(x,y)=f(x,y)$")
xlabel("x/dx")
ylabel("y/dy")
## Commenter / décommenter l'une des deux lignes qui suivent pour observer la solution exacte V ou la solution approchée U
#imshow(V) 
imshow(U_bis)
colorbar()
show()