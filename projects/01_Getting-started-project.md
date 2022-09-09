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

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
#Ambient Temperature (*F)
T_a = 65

# Initial Temperature (*F)
T_0 = 85

# Temperature after 2 hours (*F)
T = 75

# Elapsed Time (hours)
dt = 2

#Calculate K
K = (T_0-T)/(dt*(T-T_a))

print("The empirical constant K is {}".format(K))
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def findK(initialTemp, finalTemp, ambientTemp, t_elapsed):
    """
    This function calculates K based on initial and final temperatures, the ambient temperature,
    and the elapsed time between measurements
    """
    return (initialTemp-finalTemp)/(t_elapsed*(finalTemp-ambientTemp))
```

```{code-cell} ipython3
print("findK(initialTemp, finalTemp, ambientTemp, t_elapsed) calculates K as {} for the given values of: \ninitialTemp = {} *F\nfinalTemp = {} *F\nambientTemp = {} *F\nt_elapsed = {} hrs ".format(findK(T_0, T, T_a, dt),T_0, T, T_a, dt))
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
#Time step in hours
step = 0.5

#Create time range with time step 'step'
t = np.arange(0, dt, step)

#Create an array of zeors to iterate over
T_euler=np.zeros(len(t))

#Set initial temperature to the given value
T_euler[0]=T_0

for i in range(1,len(t)):
        T_euler[i]=T_euler[i-1]-K*(T_euler[i-1]-T_a)*step

#Plot Analytical Values
plt.plot(t, T_euler, "o", label="Numerical Solution")
plt.legend()
```

```{code-cell} ipython3

```
