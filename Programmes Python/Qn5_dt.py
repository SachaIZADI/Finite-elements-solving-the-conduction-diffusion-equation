##Question 5 - ordre du schéma de Crank-Nicholson vs ordre du schéma explicite centré - équation de convection

##Etapes: 
#* Initialiser les paramètres
#* Calculer la valeur en t=T de u^n(x_0+Vt) et de v^n(x_0+Vt)
#* Calculer u^0(x_0)
#* Faire la différence
#* itérer pour différentes valeurs de dt --> Graphe

#NB : Crank-Nicholson = U //// Explicite centré = V

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

global L, T, V, nu
L=50
T=2
V=1
nu=1



k=0.25 #k=V*dt/dx

DX=np.zeros((20,1))
for j in range(20):
    DX[j]=[0.01 +j*0.01]

Err_CN=np.zeros((20,1))
Err_DE=np.zeros((20,1))

q=0

for dX in DX :
    dT=dX*k
    Nt=floor(T/dT)
    dX=float(dX)
    
    Nx=floor(L/ dX)
    ## Définition des matrices Kc et Kd
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

    A = np.eye(Nx)+V*dT/(2*dX)*Kc
    B = np.eye(Nx)-V*dT/(2*dX)*Kd
    C = np.dot(alg.inv(A),B)
    D = np.eye(Nx)-V*dT/(2*dX)*Kc

    
    U=np.zeros((Nx,1))
    for j in range(Nx):
        U[j]=f(j*dX)
    
    X=np.zeros((Nx,1))
    for j in range(Nx):
        X[j]=j*dX
    
    ## Tracé des courbes |u_n(x=x0+VT)-f(x0)|=F(dt) et de |u_n(x=x0+VT)-f(x0)|=F(dx) pour les deux schémas


    #Crank - Nicholson :
    for i in range(Nt+1):
        U=np.dot(C,U)
    Err_CN[q]=[abs(max(U)-f(x0))]
    
    #Décentré - Explicite :
    for i in range(Nt+1):
        U=np.dot(D,U)
    Err_DE[q]=[abs(max(U)-f(x0))]
    
    q+=1

DT=DX*k
plot(DT,Err_CN)
plot(DT,Err_DE)
grid()
title("$\ Erreur\ :\ |\max_{j}(U_j^n)-f(x_0)|\ =\ F(dt) $")
xlabel("$\ dt $")
ylabel("$\ Err(dt) $")
legend(["$Crank-Nicholson$","$explicite\ décentré$"], loc="best")

show()

