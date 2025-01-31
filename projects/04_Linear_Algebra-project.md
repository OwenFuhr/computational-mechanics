---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# CompMech04-Linear Algebra Project
## Practical Linear Algebra for Finite Element Analysis

+++

In this project we will perform a linear-elastic finite element analysis (FEA) on a support structure made of 11 beams that are riveted in 7 locations to create a truss as shown in the image below. 

![Mesh image of truss](../images/mesh.png)

+++

The triangular truss shown above can be modeled using a [direct stiffness method [1]](https://en.wikipedia.org/wiki/Direct_stiffness_method), that is detailed in the [extra-FEA_material](./extra-FEA_material.ipynb) notebook. The end result of converting this structure to a FE model. Is that each joint, labeled $n~1-7$, short for _node 1-7_ can move in the x- and y-directions, but causes a force modeled with Hooke's law. Each beam labeled $el~1-11$, short for _element 1-11_, contributes to the stiffness of the structure. We have 14 equations where the sum of the components of forces = 0, represented by the equation

$\mathbf{F-Ku}=\mathbf{0}$

Where, $\mathbf{F}$ are externally applied forces, $\mathbf{u}$ are x- and y- displacements of nodes, and $\mathbf{K}$ is the stiffness matrix given in `fea_arrays.npz` as `K`, shown below

_note: the array shown is 1000x(`K`). You can use units of MPa (N/mm^2), N, and mm. The array `K` is in 1/mm_

$\mathbf{K}=EA*$

$  \left[ \begin{array}{cccccccccccccc}
 4.2 & 1.4 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 1.4 & 2.5 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 8.3 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 4.2 & -1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & -1.4 & 2.5 \\
\end{array}\right]~\frac{1}{m}$

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu
plt.style.use('fivethirtyeight')
```

```{code-cell} ipython3
fea_arrays = np.load('./fea_arrays.npz')
K=fea_arrays['K']
K
```

In this project we are solving the problem, $\mathbf{F}=\mathbf{Ku}$, where $\mathbf{F}$ is measured in Newtons, $\mathbf{K}$ `=E*A*K` is the stiffness in N/mm, `E` is Young's modulus measured in MPa (N/mm^2), and `A` is the cross-sectional area of the beam measured in mm^2. 

There are three constraints on the motion of the joints:

i. node 1 displacement in the x-direction is 0 = `u[0]`

ii. node 1 displacement in the y-direction is 0 = `u[1]`

iii. node 7 displacement in the y-direction is 0 = `u[13]`

We can satisfy these constraints by leaving out the first, second, and last rows and columns from our linear algebra description.

+++

### 1. Calculate the condition of `K` and the condition of `K[2:13,2:13]`. 

a. What error would you expect when you solve for `u` in `K*u = F`? 

b. Why is the condition of `K` so large? __The problem is underconstrained. It describes stiffness of structure, but not the BC's. So, we end up with sumF=0 and -sumF=0__

c. What error would you expect when you solve for `u[2:13]` in `K[2:13,2:13]*u=F[2:13]`

```{code-cell} ipython3
print(np.linalg.cond(K))
print(np.linalg.cond(K[2:13,2:13]))

print('expected error in x=solve(K,b) is {}'.format(10**(16-16)))
print('expected error in x=solve(K[2:13,2:13],b) is {}'.format(10**(2-16)))
```

### 2. Apply a 300-N downward force to the central top node (n 4)

a. Create the LU matrix for K[2:13,2:13]

b. Use cross-sectional area of $0.1~mm^2$ and steel and almuminum moduli, $E=200~GPa~and~E=70~GPa,$ respectively. Solve the forward and backward substitution methods for 

* $\mathbf{Ly}=\mathbf{F}\frac{1}{EA}$

* $\mathbf{Uu}=\mathbf{y}$

_your array `F` is zeros, except for `F[5]=-300`, to create a -300 N load at node 4._

c. Plug in the values for $\mathbf{u}$ into the full equation, $\mathbf{Ku}=\mathbf{F}$, to solve for the reaction forces

d. Create a plot of the undeformed and deformed structure with the displacements and forces plotted as vectors (via `quiver`). Your result for aluminum should match the following result from [extra-FEA_material](./extra-FEA_material.ipynb). _note: The scale factor is applied to displacements $\mathbf{u}$, not forces._

> __Note__: Look at the [extra FEA material](./extra-FEA_material). It
> has example code that you can plug in here to make these plots.
> Including background information and the source code for this plot
> below.


![Deformed structure with loads applied](../images/deformed_truss.png)

```{code-cell} ipython3
#Create force array with F = -300N applied at node 4.
nodes = np.array([[1,0,0],[2,0.5,3**0.5/2],[3,1,0],[4,1.5,3**0.5/2],[5,2,0],[6,2.5,3**0.5/2],[7,3,0]])
nodes[:,1:3]*=l
F=np.zeros(2*len(nodes)-3)
F[5]=-300

#Modify K with BCs
K_BC = K[2:13,2:13]
#Element Properties
#Area in mm^2
A = 0.1
#Steel (MPa)
Est = 200e3
#Aluminum (MPa)
EAl = 70e3

#Part a
#Use Scipy to decompose the matrix into PLU components
P, L, U = lu(K_BC)

#Forward Substitution Solution
FW_Sol = np.linalg.solve(L,F)

uf = np.linalg.solve(U,FW_Sol)

u=np.zeros(2*len(nodes))
u[2:13] = uf
u_store = u #Store u for use in second part
React = K@u
#Compensate for Young's modulus and area
uSt = u/(Est*A)
uAl = u/(EAl*A)

#Adapted from Professor's work
xy={0:'x',1:'y'}

print('The reactions for a steel or aluminum truss are:')
for i, R in enumerate(React):
    print('R_{}{} = {:.1f} N'.format(int(i/2)+1,xy[i%2], R))
    

print('\nThe deformations for the steel truss are:')
for i, u in enumerate(uSt):
    print('u_{}{} = {:.1f} mm'.format(int(i/2)+1,xy[i%2],u))

print('\nThe deformations for the aluminum truss are:')
for i, u in enumerate(uAl):
    print('u_{}{} = {:.1f} mm'.format(int(i/2)+1,xy[i%2],u))
```

```{code-cell} ipython3
#From Professor's work
l=300 # mm
nodes = np.array([[1,0,0],[2,0.5,3**0.5/2],[3,1,0],[4,1.5,3**0.5/2],[5,2,0],[6,2.5,3**0.5/2],[7,3,0]])
nodes[:,1:3]*=l
elems = np.array([[1,1,2],[2,2,3],[3,1,3],[4,2,4],[5,3,4],[6,3,5],[7,4,5],[8,4,6],[9,5,6],[10,5,7],[11,6,7]])
print('node array\n---------------')
print(nodes)
print('element array\n---------------')
print(elems)

ix = 2*np.block([[np.arange(0,5)],[np.arange(1,6)],[np.arange(2,7)],[np.arange(0,5)]])
iy = ix+1

r = np.block([n[1:3] for n in nodes])
r
```

```{code-cell} ipython3
plt.plot(r[ix],r[iy],'-',color='k')
plt.plot(r[ix],r[iy],'o',color='b')
plt.plot(r[0],r[1],'^',color='r',markersize=20)
plt.plot(r[0],r[1],'>',color='k',markersize=20)
plt.plot(r[-2],r[-1],'^',color='r',markersize=20)
# label the nodes
for n in nodes:
    if n[2]>0.8*l: offset=0.1
    else: offset=-l/5
    plt.text(n[1]-l/3,n[2]+offset,'n {}'.format(int(n[0])),color='b')
# label the elements
for e in elems:
    n1=nodes[e[1]-1]
    n2=nodes[e[2]-1]
    x=np.mean([n2[1],n1[1]])
    y=np.mean([n2[2],n1[2]])
    # ----------------->need elem labels<-----------------
    plt.text(x-l/5,y-l/10,'el {}'.format(int(e[0])),color='r')
plt.title('Our truss structure\neach element is 2 nodes and length L\ntriangle markers are constraints')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.axis(l*np.array([-0.5,3.5,-1,1.5]));
```

```{code-cell} ipython3
#Adapted from professor's work
def f(s, u, matlname):
    plt.figure(figsize=(8,5))
    plt.plot(r[ix],r[iy],'-',color=(0,0,0,1))
    plt.plot(r[ix]+u[ix]*s,r[iy]+u[iy]*s,'-',color=(1,0,0,1))
    #plt.quiver(r[ix],r[iy],u[ix],u[iy],color=(0,0,1,1),label='displacements')
    plt.quiver(r[ix],r[iy],React[ix],React[iy],color=(1,0,0,1),label='applied forces')
    plt.quiver(r[ix],r[iy],u[ix],u[iy],color=(0,0,1,1),label='displacements')
    plt.axis(l*np.array([-0.5,3.5,-0.5,2]))
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Deformation of {} Truss Under {:.0f} N'.format(matlname, React[7]))
    plt.legend(bbox_to_anchor=(1,0.5))
    

f(1, uSt, 'Steel')
f(1, uAl, 'Aluminum')
```

### 3. Determine cross-sectional area

a. Using aluminum, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

b. Using steel, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

c. What are the weights of the aluminum and steel trusses with the
chosen cross-sectional areas?

```{code-cell} ipython3
#Solving part a and b iteratively

#Set up range of areas
Amax, step = 100, 0.005
Xsecs = np.arange(0.1,Amax,step)

#Iterate until the total y deformation of aluminum and steel at node 4 becomes less than 0.2mm, graph results
totdef = []

for i,A in enumerate(Xsecs):
    uAl_t = u_store/(EAl*A)
    totdef.append(abs(uAl_t[7]))
    if abs(uAl_t[7]) < 0.2:
        A_Al = A
        print('Aluminum:\nThe minimum cross-sectional area for deformations less than 0.2 mm is {:.2f} mm^2'.format(A))
        plt.figure(figsize=(8,5))
        plt.plot(Xsecs[:len(totdef)],totdef, 'r-')
        plt.plot(Xsecs[len(totdef)], 0.2, 'b*', markersize=20,label='A = {:.2f} $mm^2$'.format(A))
        plt.xlabel('Cross-Sectional Area ($mm^2$)')
        plt.ylabel('Total Deformation ($mm$)')
        plt.title('Aluminum Total Deformation vs. Cross-Sectional Area')
        plt.legend();
        break

totdef = []

for i,A in enumerate(Xsecs):
    uSt_t = u_store/(Est*A)
    totdef.append(abs(uSt_t[7]))
    if abs(uSt_t[7]) < 0.2:
        A_steel = A
        print('\nSteel:\nThe minimum cross-sectional area for deformations less than 0.2 mm is {:.2f} mm^2'.format(A))
        plt.figure(figsize=(8,5))
        plt.plot(Xsecs[:len(totdef)],totdef, 'r-')
        plt.plot(Xsecs[len(totdef)], 0.2, 'b*', markersize=20,label='A = {:.2f} $mm^2$'.format(A))
        plt.xlabel('Cross-Sectional Area ($mm^2$)')
        plt.ylabel('Total Deformation ($mm$)')
        plt.title('Steel Total Deformation vs. Cross-Sectional Area')
        plt.legend();
        break
```

```{code-cell} ipython3
#Finding the weights

#Densities:
rhoSteel = 7870 #kg/m^3 (Average from carbon steel https://amesweb.info/Materials/Density_of_Steel.aspx)
rhoAl = 2700 #kg/m^3 (from https://www.periodic-table.org/Aluminium-density/)

#weight (N) = density * Xsec Area(mm^2) * number of elements * element length (mm)* g (m/s^2)
StWeight = rhoSteel/(1000**3) * A_steel * 11 * 300 * 9.81
AlWeight = rhoAl/(1000**3) * A_Al * 11 * 300 * 9.81

print('''The weight of the steel truss with an element cross-sectional area of {:.2f} mm is {:.2f} N

The weight of the Aluminum truss with an element cross-sectional area of {:.2f} mm is {:.2f} N'''.format(A_steel, StWeight,
                                                                                                        A_Al, AlWeight))
```

## References

1. <https://en.wikipedia.org/wiki/Direct_stiffness_method>

```{code-cell} ipython3

```
