""" Main tools used in this package. Here one will find tools
related to build time series from known diferential equations
or iterated maps as well as spectral decomposition tools
and non-linear dynamics tools"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq

class parametric_diferential_equations():
    """Set of diferential equations in parametric form which
    we use in the main code

    Parameters
    ----------

    Returns
    -------

    
    """
    def pendulum_ode(x, y, b=0.25, c=5.0):
        """Pendulum ODE

        Parameters
        ----------
        x : a float. The x position
            
        y : a float. The y position
            
        b : a float. The dumping coeficient
            (Default value = 0.25)
        c : a float. The drive: g/l
            (Default value = 5.0)

        Returns
        -------

        
        """
        return np.array([y, -b*y - c*np.sin(x)])

    def rossler_ode(x, y, z, a=0.15, b=0.2, c=10.0):
        """Rossler attractor ODE

        Parameters
        ----------
        x : a float. The x parameter
            
        y : a float. The y parameter
            
        z : a float. The z parameter
            
        a : a float. A parameter of the equation
            (Default value = 0.15)
        b : a float. A parameter of the equation
            (Default value = 0.2)
        c : a float. A parameter of the equation
            (Default value = 10.0)

        Returns
        -------

        
        """
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def lorenz_ode(x, y, z, sigma=10.0, beta=8/3.0, rho=28.0):
        """Lorenz attractor ODE

        Parameters
        ----------
        x : a float. The x parameter
            
        y : a float. The y parameter
            
        z : a float. The z parameter
            
        sigma : a float. A parameter of the equation
            (Default value = 10.0)
        beta : a float. A parameter of the equation
            (Default value = 8/3.0)
        rho : a float. A parameter of the equation
            (Default value = 28.0)

        Returns
        -------

        
        """
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

class iterated_maps():
    """Iterated maps used in the main code"""
    def quadratic_map(x, A=1, B=0, C=0, n=1):
        """Quadractic iterated map. It receives the value of the
        function at t=t and returns it at the instant t=t+1 following
        a quadractic recepee.

        Parameters
        ----------
        x : a float. The initial value of the function
            
        A : a float. The second order coefficient
            (Default value = 1)
        B : a float. A*B is the first order coefficient
            (Default value = 0)
        C : a float. The free parameter
            (Default value = 0)
        n :
            (Default value = 1)

        Returns
        -------

        
        """
        if n < 1:
            return None
        elif n == 1:
            return A*x*(x + B) + C
        else:
            return A*iterated_maps.quadratic_map(
                x, A=A, B=B, C=C, n=n-1)*(iterated_maps.quadratic_map(
                x, A=A, B=B, C=C, n=n-1) + B) + C

class simple_equation_solvers():
    """Simple Runge-Kutta solvers for ODEs"""
    def rk1(ode, state, parameters, dt=0.001):
        """Runge-Kutta of order 1

        Parameters
        ----------
        ode : a function object. The ODE to integrate
            
        state : a array of floats. The state of the system
            
        parameters : an arrays of floats. The parameters of the ODE
            
        dt : a float. The time-interval (resolution)
            (Default value = 0.001)

        Returns
        -------

        
        """
        return state + dt * ode(*state, *parameters)

    def rk2(ode, state, parameters, dt=0.001):
        """Runge-Kutta of order 2

        Parameters
        ----------
        ode : a function object. The ODE to integrate
            
        state : a array of floats. The state of the system
            
        parameters : an arrays of floats. The parameters of the ODE
            
        dt : a float. The time-interval (resolution)
            (Default value = 0.001)

        Returns
        -------

        
        """
        f1 = ode(*state, *parameters)
        return state + dt * ode(*(state + dt * f1 * 0.5), *parameters)
    
    def rk4(ode, state, parameters, dt=0.001):
        """Runge-Kutta of order 4

        Parameters
        ----------
        ode : a function object. The ODE to integrate
            
        state : a array of floats. The state of the system
            
        parameters : an arrays of floats. The parameters of the ODE
            
        dt : a float. The time-interval (resolution)
            (Default value = 0.001)

        Returns
        -------

        
        """
        k1 = dt * ode(*state, *parameters)
        k2 = dt * ode(*(state + 0.5 * k1), *parameters)
        k3 = dt * ode(*(state + 0.5 * k2), *parameters)
        k4 = dt * ode(*(state + k3), *parameters)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
class time_series_generators():
    """Generate time series from the solution of the ODEs and iterated
    maps"""
    def generate_series_from_ODE(data_length, ode, state, parameters, dt, transient):
        """ Generte a time series from ODEs
        Parameters
        ----------
        data_length : a integer. The length of the time series
     
        ode : a function object. The ODE to get the time series from
            
        state : a list of floats. The system initial state
            
        parameters : a list of floats. The parameters of the function
            
        dt : a float. The time interval between evaluations
            
        transient : a integer. The steps to wait before starts measuring
            

        Returns
        -------

        """
        data = np.zeros((data_length, len(state)))
        for i in range(transient):
            state = simple_equation_solvers.rk4(ode, state, parameters, dt)
        for i in range(data_length):
            state = simple_equation_solvers.rk4(ode, state, parameters, dt)
            data[i,:] = state
        return data

    def generate_series_from_iterated_maps(data_length, iter_map, initial_state,
                                           transient=0, parameters=[1, 0, -1.5, 1]):
        """ Generte a time series from iterated maps
        Parameters
        ----------
        data_length : a integer. The length of the time series
     
        iter_map : a function object. The iterated map to get
        the time series from
            
        initial_state : a list of floats. The system initial state
            
        parameters : a list of floats. The parameters of the function
            
        transient : a integer. The steps to wait before starts measuring
            
        Returns
        -------

        """
        state = initial_state
        for i in range(transient):
            state = [iter_map(*state, *parameters)]
        data = [state]
        for i in range(data_length):
            state = [iter_map(*state, *parameters)]
            data.append(state)
        return data

class spectral_analysis():
    """Spectral analysis of time series with Fourier transforms"""
    def fourier_discreet_transform(data, sample_rate, duration):
        """ Generte a the discreet Fourier transform of the data
        ----------
        data : an array of floats. The data to get the Transform

        sample_rate : a float. The time interval of the collected data

        duration : a float. The duration of the series (sample_rate
        times the array size)

        Returns
        -------
        A tuple of two arrays of floats. The array of frequencies and
        the Fourier spectrum of the data
        """
        return rfftfreq(duration*sample_rate,
                        1/sample_rate), np.abs(rfft(data))
    
    def fft_filter(percentual, data):
        """ Generte a time series from the raw data where frequencies
        whit low enough amplitude are set to zero (filtered signal)
        ----------
        data : an array of floats. The spectrum to filter
     
        percentual : a float. The percentual from the amplitude of the
        top frequency which to cut the other frequencies
            
        Returns
        -------
        An array of floats. The spectrum resulting of the filtering
        """
        # taking the absolute of the signal
        data_abs = np.abs(data)
        # defining the cutoff
        th = percentual*(data_abs.max())
        data_tof = data_abs.copy()
        # filtering
        data_tof[data_tof <= th] = 0
        return data_tof
    
    def filter_signal(perc, data):
        """ Generte a filtered signal
        ----------
        data : an array of floats. The signal to filter
     
        percentual : a float. The percentual from the amplitude of the
        top frequency which to cut the other frequencies
            
        Returns
        -------
        A tuple of two arrays of floats. The filtered spectrum and the
        filtered time series.
        """
        f_s = spectral_analysis.fft_filter(perc, data)
        return f_s, np.real(np.fft.ifft(f_s))

    def best_scale(data, inf=0.001, sup=0.3, p_threshold=0.005,
                   grafics=False):
        """ Among many percentual values, it produces and filters the
        signals and compare the correlation and p-value and considers
        the percentual of least correlation with p-value lower than
        the stipulated thresshold
        ----------
        data : an array of floats. The time series to consider

        inf : a float. The minimum percentual to consider

        sup : a float. The maximum percentual to consider

        p_threshold : a float. The p-value limiar to consider the results

        grafics : a bolean. If True, shows a graph of the correlation and
        p-value calculated
            
        Returns
        -------
        An tuple of three floats. The percentual value of minimum correlation,
        the associated p-value and correlation
        """
        th_list = np.linspace(inf, sup, 10000)
        p_values = []
        corr_values = []
        new_th_list = []
        for t in th_list:
            filt_signal = spectral_analysis.filter_signal(t, data)
            res = stats.spearmanr(data, data-filt_signal)
            if abs(res.pvalue) < p_threshold:
                p_values.append(res.pvalue)
                corr_values.append(res.correlation)
                new_th_list.append(t)
        if grafics:
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            plt.scatter(new_th_list,corr_values,s=2,color='navy')
            plt.ylabel('Correlation Value')
            plt.xlabel('Threshold Value')
            plt.subplot(1,2,2)
            plt.scatter(new_th_list,p_values,s=2,color='navy')
            plt.ylabel('P-Value')
            plt.xlabel('Threshold Value')
            plt.show()
        corr_values_abs = [abs(x) for x in corr_values]
        aux = np.argmin(np.min(corr_values_abs))
        return new_th_list[aux], p_values[aux], corr_values[aux]

SAMPLE_RATE = 1000
DURATION = 10
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y
x_1, y_1 = generate_sine_wave(10, SAMPLE_RATE, DURATION)
x_2, y_2 = generate_sine_wave(5, SAMPLE_RATE, DURATION)
mix = y_1 + 0.5*y_2

xf, yf = spectral_analysis.fourier_discreet_transform(mix,
                                                      SAMPLE_RATE,
                                                      DURATION)

plt.plot(xf[0:200], np.abs(yf[0:200]))
plt.show()