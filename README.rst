Atractor project
################

This is a bundle of tools for dealing with time series analysis togheter with aplications to a set of interesting out-in-the-world systems including world temperature, solar spots, heartbeat and stock-market time-series.

All the documentation and explanation of these examples are located in the examples_docs file. Here you will find a guide for installation and the description of the tools used in those analysis (the package functions), so you can use them to build your own analysis of some interesting phenomenon.

Installation
************

Tools
*****
For the sake of reference, all the tools listed here are found in src/attarctor_project/tools.py file and tested with pytest located in tests/test_tools.py

parametric_diferential_equations class
======================================
This class takes no arguments and is a bundle of known parametrized differential equations to generate the related time series.

pendulum_ode(x, y, b=0.25, c=5.0)
---------------------------------
This function takes four arguments, two of which mandatory, and returns an numpy array containing two floats. The **x** and **y** arguments are the cartesian "position" of the system, **b** and **c** are parameters and are set to default in an attractor on (0, 0). It returns 

``[y, -b*y - c*np.sin(x)]`` 

witch is the parametrized form of the ODE

 ``o''(t) + bo'(t) + csin(o(t)) = 0``,
 
 with  
 
 ``y(t) = o'(t)``

rossler_ode(x, y, z, a=0.15, b=0.2, c=10.0)
-------------------------------------------
This function takes six arguments, three cartesian coordinates and three parameters which default corresponds to a known strange attractor of the system and returns the updated cartesian values:

``[-y -z, x + ay, b + z(x - c)]``

characterizing the Rossler attractor


lorenz_ode(x, y, z, sigma=10.0, beta=8/3.0, rho=28.0)
-----------------------------------------------------
The famous Lorenz ODE. This function takes six arguments, three cartesian coordinates and three parameters which default corresponds to a known strange attractor of the system and returns the updated cartesian values:

``[sigma(y - x), x(rho - z) - y, xy - betaz]``

iterated_maps class
===================
Another colection of objects, but instead of ODEs we have here iterated maps. Iterated maps differ operationally from ODEs because we do not just solve the equation numerically for a given initial condition, but instead of discretize time in the calculations to a value depending on the resolution of the solution, the time is inherently descreet. From initial conditions, we calculate the next and puting it back in the function we calculated the next values, iterating it to produce the series.

quadratic_map(x, A=1, B=0, C=0, n=1)
------------------------------------
This function takes five arguments, a initial values and four parameters and returns the (float) updated value. 
The quadractic map is given by:
``Ax(x + B) + C``representing a full quadractic map. The ``n=1`` means that only a value is returned after a single iteraction. On the other hand, if is a integer greater than 1 the n'th iteration is returned. For examples
if ``n=2`` is calculated:

``A(Ax(x + B) + C)(Ax(x + B) + C + B) + C``
which is a fourth order polynomial that do not embedds all possible fourth order polynomials. The A, B, C parameters are default to a known chaotic behavior of this map. This is convenient for finding more interesting maps and cycles of theses maps as x^n = x with ^n meaning the n-th iteration of the value x under the map in question, is the condition to identify a cycle of ^n order of this map. Therefore with the extra parameter ``n`` it is possible to find those cycles in the same way as one finds a fixed point: x^1 = x

henon_map(x, y, a=1.4, b=0.3, n=1)
------------------------------------
This function takes five arguments, two initial values and three parameters and returns a numpy array with two entries containing the updated values. The arguments a and b are default to values studied by Henon showing a chaotic attractor. The ``n`` argument has the same meaning as the quadractic map.
The values returned are:

``[y + 1 - axx, bx]``, 


simple_equation_solvers class
=============================
Is a bundle of some numerical methods to solve ODEs. 

rk4(ode, state, parameters, dt=0.001)
-------------------------------------
Is the fourth order Runge-Kutta method, it can also be rk1 and rk2 with the same parameters to access the first and second order methods. 
the ``ode`` parameter is an function object which is the ode to integrate. ``state`` and ``parameters`` are arrays (iterables) containing the state and the parameters of the ODEs.
``dt`` is the time interval to use in the integration. It returns the calculations of one Runge-Kutta iteration. 

time_series_generators class
============================
This class contains a generators to build time series from ODEs and iterated maps.

generate_series_from_ODE(data_length, ode, state, parameters, dt, transient)
----------------------------------------------------------------------------
This function takes the integer argument ``data_length``, the function o integrate ``ode``, the arrays of initial conditions of this function ``state``, the parameters of this function ``parameters`` the time delta to integrate ``dt`` and a integer ``transient`` which is the number of steps between the integration starting point to the instant to start the measurements which is necessary when one wishes to analyse time series nearest to the attractors, and returns a float array of length ``data_length`` containing the trajectories constructed with the numerical integration.

generate_series_from_iterated_maps(data_length, iter_map, initial_state, parameters, transient=0)
-------------------------------------------------------------------------------------------------
This function takes the same parameters as the ``generate_series_from_ODE`` but the argument ``ode`` changed to the argument ``iter_map`` which is the iterated map of interest and the parameter ``transient`` is set to 0. 


spectral_analysis class
=======================
This class contains a set of tools to analyse time series with simplified Fourier analysis. 

fourier_discreet_transform(data, sample_rate, duration)
-------------------------------------------------------
It returns a tuple of two numpy arrays. A row array containing frequencies measured and a row array containing the amplitudes.