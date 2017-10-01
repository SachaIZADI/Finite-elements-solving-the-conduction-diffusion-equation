## Question 13 - ce programme permet de tracer le champ des vitesses V(x,y)
import numpy as np
from matplotlib.pyplot import*
from math import *

n = 50 # dimensions du domaine
L=50 # dimensions du domaine
X, Y = np.mgrid[0:n:3, 0:n:3]

k,l=1,1

U=np.cos(k*pi*Y/L)*np.sin(l*pi*X/L) #Vx
V=np.sin(k*pi*Y/L)*np.cos(l*pi*X/L) # Vy

Vit=np.sqrt(np.square(U)+np.square(V)) # norme de V

quiver(X, Y, U, V,        # données
           Vit,                   # couleur des flèches en fonction de la norme de V
           cmap=cm.seismic,     # colour map
           headlength=7)        # longueur des flèches

# Affichage
title('Champ des vitesses $\overrightarrow{V}(x,y)$')
colorbar()
show()                

