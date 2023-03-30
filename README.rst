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
This function receives three arguments. The time series ``data``. The rate each sample was measured (time delta between entries) ``sample_rate``. And the duration of the signal: ``duration``.
It returns a tuple of two numpy arrays. A row array containing frequencies measured and a row array containing the amplitudes.

fft_filter(percentual, spectrum)
--------------------------------
Takes a float between 0 and 1: ``percentual``; and the amplitudes of the frequencies coposing the frequency spectrum of the time series. This function sets to zero (cut off) the amplitudes lower than percentual% of the maximum âmplitude in the signal and returns the spectrum with this modification.
It is used to filter the spectrum for low amplitude frequencies.

filtered_signal(perc, spectrum)
-------------------------------
This function takes the float ``percentual`` and the amplitudes of the frequencies of the signal Fourier descreet spectrum ``spectrum`` to call the function ``fft_filter(percentual, spectrum)`` to get a filtered spectrum and build a filtered signal from it. 
It returns a filtered spectryum and the filtered signal.

best_scale(data, inf=0.001, sup=0.5, p_threshold=0.005, grafics=False)
----------------------------------------------------------------------
To decide which percentage to use to filterthe signal, this function performs a variation of percentage values measuring the correlation of the residual (true signal minus filtered signal correlation to the true signal) and its p-value. It selects the pecentage value under the imputed p-value limiar having the lower correlation measured.
It takes the signal: ``data``; the range of percentages to consider: ``inf`` and ``sup``; the p-values threshold ``p_threshold``, and; a parameter ``grafics`` that if True the function plots the correlation and p-values measured against the percentage variation.


non_linear_methods class
========================
This class is a set of tools used for signal processing but originating from non-linear dynamics. They concentrate on different characteristics of the signal compared to Fourier analysis.

cobweb_diagram(imap, init_condit, params, iter=1000, xlim=[-3, 3], ylim=[-3,3], show=True, ax=None)
---------------------------------------------------------------------------------------------------

This function takes a unidimensional iterated map, ``imap``, its initial condition, ``init_condit``, its parameters ``params``, and iterate it ``iter`` times and if the ``show`` argument is True it displays a cobweb diagram of the trajectory of the orbits in a box of limits ``xlim`` and ``ylim``. Else, if the ``ax`` argument is not None, it returns the ax object with the diagram for further customization.

orbit_diagram(imap, measuring_time, init_cond_range, params_range, param_index, args_index, args, params, points=1000, thresshold=4)
----------------------------------------------------------------------------------------------------------------------
This function prints the orbit diagram of the iterated map. It takes the map, ``imap``, the number of itractions to submit the map, ``measuring_time``, two arrays containing the range of the initial conditions and parameters to consider, ``init_cond_range`` and ``params_range``, which argument and parameter to vary in ``args_index`` and ``param_index``, the set of all parameters and arguments of the iterated map, ``args`` and ``params``, the "resolution" of the plot in the argument ``points``, wich is the number of steps in the parameter and argument variation, and finally the value of reference to consider or discard the solutions ``thresshold``.
This function varyies a parameter and a initial condition (argument) of the map and iterate it ``measuring_time``. If the absolute of the value does not exceed the limit ``thresshold``, than we plot the values of the argument and the parameter in the graph ``param_index`` times ``args_index``. In the limit of ``measuring_time`` big, this means to keep all values where the orbits converge and not scape elsewhere.

lorentz_map(Signal, lag=1, plot=True)
-------------------------------------
This tool plots the relation between a value and the value of the series ``lag`` steps behind. If ``lag`` is None, this function will find the relation between the local maximuns and plot them. It returns the lagged series togheter with the respective related values and if ``plot`` is True it prints the map.


minimum_info_tau(data, tau_max=100, graph=False)
------------------------------------------------
This function finds the lag interval which composition with the original returns the least information (correlation). It takes teh signal, ``data``, the maximum lag to consider, ``tau_max``, and wheter to plot the correlation measured or not with the argument ``graph``.

attractor_reconstructor(data, tau_to_use=None, how_many_plots=1, scatter=False, plot=True)
------------------------------------------------------------------------------------------
This function recostructs the attractor of a time series using the method of lags. It uses the signal, ``data``, the lag to consider ``tau_to_use``, which cam be an integer, None (in which case the lag returned by ``minimum_info_tau`` function) or an array of integers to be plotted in association of the argument ``how_many_plots`` which say how many recostructed attractors to plot using the available lags. The function retruns three series to generate a 3D map of the attractor and the lag used and if ``plot`` is True it shows the attractor with teh default configurations.