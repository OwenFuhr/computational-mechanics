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

+++

Part a:

```{code-cell} ipython3
#Solving Analytically
#Create a time range with fine steps from zero to the given elapsed time
t_an = np.arange(0,dt+.1,0.1)
#Solve Analytically
T_analytical = T_a +(T_0-T_a)*np.exp(-K*t_an)

#Solving Numerically
step = 8
Graphs = 4
for i in range(Graphs):
    #Time step in hours for euler integration
    step = step/4

    #Create new time range with time step 'step'
    t_nu = np.arange(0, dt+step,step)

    #Create array of zeros to place temperature values in
    T_euler = np.zeros(len(t_nu))

    #Set the initial temperature to the given value
    T_euler[0] = T_0

    for i in range(1,len(t_nu)):
            T_euler[i]=T_euler[i-1]-K*(T_euler[i-1]-T_a)*step
    #Create a new figure each loop with a larger size than default
    plt.figure(i+1,figsize=(8,5))
    plt.plot(t_nu, T_euler,"ro",label="Numerical Solution with {} steps".format(len(t_nu)))

    #Plot Analytical Solution
    plt.plot(t_an, T_analytical, "b-", label="Analyitical Solution")
    
    #Plot Measured Values
    plt.plot(0,T_0,"ks")
    plt.plot(2,T, "ks",label="Measured Values")

    #Label Axes
    plt.xlabel("Time elapsed (hours)")
    plt.ylabel("Temperature ($\degree$F)")
    plt.title("Temperature vs. Time")
    plt.legend()
    
    #Print Error
    print(str(step)+ " hour step size:\nThe error between the analytical solution and measured value is " + str(T-T_analytical[-1]) + "*F\n",
         "The error between the numerical solution and the measured value is " + str(T-T_euler[-1]) + "*F\n",
         "The error between the analytical and numerical solutions is " + str(T_analytical[-1]-T_euler[-1]) + "*F\n\n")
```

As seen above, the numerical solution converges to the analytical solution as the space between the time steps decreases, and has the same error with the measured value.

+++

Part b:

```{code-cell} ipython3
#Extend time range on analytical solution and solve
timebound = 20
step = 0.1
t_an = np.arange(0,timebound+step,step)
#Solve Analytically
T_analytical = T_a +(T_0-T_a)*np.exp(-K*t_an)

plt.plot(t_an,T_analytical,"b-",label="Analytical Solution")

#Label Axes
plt.xlabel("Time elapsed (hours)")
plt.ylabel("Temperature ($\degree$F)")
plt.title("Temperature vs. Time")
plt.legend()
```

The temperature approaches the ambient temperature $T_a$ = 65$^{o}$F as t$\rightarrow\infty$

+++

Part c:

The time between death and temperature measurement can be found by solving the analyitical solution for time at $T = 98.6^{o}F$.
The equation:
$T(t) =T_a+(T(0)-T_a)e^{-Kt}$
can be rearranged as:
$t = -\frac {\ln{\frac{T(nominal)-T_a}{T(measured)-T_a}}}{K}$

Which will give the number of hours elapsed between normal body temperature and the temperature measurement after death as a negative number

```{code-cell} ipython3
def TimeofDeath(measurementTime,measuredTemp, ambientTemp, K):
    '''
    This function takes measurementTime, the time of measurement in 24 hour string format HH:MM,
    measuredTemp, the measured body temperature after death, the ambient temperature ambientTemp, and empirical
    constant K and returns the time elapsed since death assuming a nominal body temperature of 98.6*F
    '''

    #Nominal body temp in degrees F
    nominalTemp = 98.6
    
    elapsed = np.log((nominalTemp-ambientTemp)/(measuredTemp-ambientTemp))/K
    
    #Handle clock times
    meastimehm = measurementTime.split(":")
    timehm = str(round(elapsed,2)).split(".")
    
    elhr, elmin = int(timehm[0]), int(timehm[1])
    mehr, memin = int(meastimehm[0]), int(meastimehm[1])
    
    
    if elmin > memin:
        death_min = 60 + memin-elmin
        elhr += 1
    else:
        death_min = memin-elmin
    if elhr > mehr:
        death_hr = 24 + mehr-elhr
    else:
        death_hr = mehr-elhr
    
    time24 = str(death_hr) + ":" + (("0" + str(death_min)) if len(str(death_min))==1 else str(death_min))
    
    return time24, elapsed

Meastime = "11:00"
time, elapsed = TimeofDeath(Meastime,T_0,T_a,K)
print("The time of death in 24 hour format was {}. {:.2f} hours have elapsed since from time of death to the time of measurement.".format(time, elapsed))
```

```{code-cell} ipython3

```
