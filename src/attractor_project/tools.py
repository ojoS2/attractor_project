"""Main tools used in this package. Here one will find tools
related to build time series from known diferential equations
or iterated maps as well as spectral decomposition tools
and non-linear dynamics tools"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy import stats
from scipy.fft import rfft, irfft, rfftfreq
from sklearn import metrics
from tensorflow import keras


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
        data = [state[0]]
        for i in range(data_length):
            state = [iter_map(*state, *parameters)]
            data.append(state[0])
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
        return rfftfreq(int(duration*sample_rate),
                        1/sample_rate), np.abs(rfft(data))
    
    def fft_filter(percentual, spectrum):
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
        data_abs = np.abs(spectrum)
        # defining the cutoff
        th = percentual*(np.max(data_abs))
        data_tof = data_abs.copy()
        # filtering
        data_tof[data_tof <= th] = 0
        return data_tof
    
    def filtered_signal(perc, spectrum):
        """ Generte a filtered signal
        ----------
        spectrum : an array of floats. The signal to filter
     
        percentual : a float. The percentual from the amplitude of the
        top frequency which to cut the other frequencies
            
        Returns
        -------
        A tuple of two arrays of floats. The filtered spectrum and the
        filtered time series.
        """
        f_s = spectral_analysis.fft_filter(perc, spectrum)
        return f_s, irfft(f_s)

    def best_scale(data, inf=0.001, sup=0.5, p_threshold=0.005,
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
        plot_p_values = []
        plot_corr_values = []
        plot_new_th_list = []
        _, spectrum = spectral_analysis.\
        fourier_discreet_transform(data, sample_rate=1, duration=len(data))
        for t in th_list:
            filt_signal, _ = spectral_analysis.filtered_signal(t, spectrum)
            res = stats.spearmanr(spectrum, filt_signal)
            plot_p_values.append(res.pvalue)
            plot_corr_values.append(res.correlation)
            plot_new_th_list.append(t)
            if abs(res.pvalue) < p_threshold:
                p_values.append(res.pvalue)
                corr_values.append(res.correlation)
                new_th_list.append(t)
        if grafics:
            fig, ax1 = plt.subplots(figsize=(20,10))
            plt.axvline(new_th_list[0], 0, 1, c='black',
                        linestyle=':', alpha = 0.8)
            plt.annotate('threshold', xy=(new_th_list[0], 1),
                          xytext=(6, -100), textcoords='offset points',
                          rotation=90, va='bottom', ha='center')
            ax1.scatter(plot_new_th_list, plot_corr_values,s=1,color='navy')
            ax1.set_ylabel('Correlation Value')
            ax1.tick_params(axis='y', labelcolor='navy')
            ax1.set_xlabel('Scale')
            ax2 = ax1.twinx()   
            ax2.scatter(plot_new_th_list, plot_p_values,s=1,color='red')
            ax2.set_ylabel('P-Value')
            ax2.tick_params(axis='y', labelcolor='red')
            plt.show()
        corr_values_abs = [abs(x) for x in corr_values]
        aux = corr_values_abs.index((np.min(corr_values_abs)))
        return new_th_list[aux], p_values[aux], corr_values[aux]

class non_linear_methods():
    """Non-linear methods applyied to time series analysis"""
    def cobweb_diagram(imap, init_condit, params, iter=1000, xlim=[-3, 3],
                       ylim=[-3,3]):
        """Produces a cobweb diagram of the iterated map under the initial
        condition and parameters

        Parameters
        ----------
        imap : a iterated map object. The function which to use to produce
        the diagram

        init_condit : an integer. The initial condition of the system

        params : an aray of floats. The parameters of the iterated map

        iter : an integer. The number of iterations to consider
            (Default value = 1000)
        xlim : an array of floats. The limits of the x axis to plot
            (Default value = [-3, 3])
        ylim : an array of floats. The limits of the y axis to plot
            (Default value = [-3, 3])

        Returns
        -------

        
        """
        # plot the identity and the map
        x = np.linspace(xlim[0], xlim[1], num=100)
        y = [imap(i, *params) for i in x]
        plt.plot(x, y, c='black')
        plt.plot(x, x, c='blue')
        plt.axvline(x=0, linewidth=0.5, color='gray')
        plt.axhline(y=0, linewidth=0.5, color='gray')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        # calculation and ploting of the cobweb
        args = init_condit
        y_0 = [imap(*args, *params)]
        for i in range(iter):
            if abs(y_0[0]) > ylim[1]:
                print(f'{i} iteractions before getting out.')
                plt.show()
                break
            plt.plot((args[0], args[0]), (args[0], y_0[0]), scaley = False, linewidth=0.5, color='red')
            plt.plot((args[0], y_0[0]), (y_0[0], y_0[0]), scaley = False, linewidth=0.5, color='red')
            args = y_0
            y_0 = [imap(*args, *params)]
        plt.show()

    def cobweb_projection(imap, params, points= 1000, iter=10000,
                          xlim=[-2, 2], ylim=[-2,2], kept=True):
        """Produces the projection of the cobweb diagram of the
        initial conditions that did not got out of the system
        under the considered parameters.

        Parameters
        ----------
        imap : a iterated map object. The function which to use to produce
        the diagram

        init_condit : an integer. The initial condition of the system

        params : an aray of floats. The parameters of the iterated map

        points : an integer. The number of initial conditions to consider
            (Default value = 1000)
        iter : an integer. The number of iterations to consider
            (Default value = 10000)
        xlim : an array of floats. The limits of the x axis to plot
            (Default value = [-2, 2])
        ylim : an array of floats. The limits of the y axis to plot
            (Default value = [-2, 2])
        kept : a boolean. If true shows the initial conditions that
        remained. If False shows the initial conditions that scaped
            (Default value = True)

        Returns
        -------

        
        """
        remained = np.ones(points, dtype=bool)
        step = (xlim[1] - xlim[0])/points
        for i in range(points):
          args = [xlim[0] + i*step]
          y_0 = [imap(*args, *params)]
          for _ in range(iter):
               if abs(y_0[0]) > ylim[1]:
                    remained[i] = False
                    break
               args = y_0
               y_0 = [imap(*args, *params)]
        init_condit = np.linspace(xlim[0], xlim[1], points)
        #plt.hist(1.*np.array(remained))
        #plt.show()
        if kept:
          return init_condit[remained]
        else:
          return init_condit[~remained]
     
    def orbit_diagram(imap, measuring_time, init_cond_range, params_range, 
                      param_index, args_index, args,
                      params, points=1000):
        """Print the orbit diagram of the iterated map

        Parameters
        ----------
        imap : a iterated map object. The function which to use to produce
        the diagram

        measuring_time : an integer. The number of iterations to consider

        init_cond_range : an array of floats. The range of initial conditions
        to initialize the system

        params_range : an array of floats. The range of the parameter to
        consider

        param_index : an integer. The index of the parameter to vary

        args_index : an integer. The index of the argument to vary

        args : an aray of floats. The arguments of the iterated map
        (including the varying one)

        params : an aray of floats. The parameters of the iterated map
        (including the varying one)

        points : an integer. The number of iterations to decide
            (Default value = 1000)

        Returns
        -------

        
        """
        init_cond_list = np.linspace(params_range[0], params_range[1], num=points)
        params_list = np.linspace(init_cond_range[0], init_cond_range[1], num=points)
        aux_x = []
        aux_y = []
        for i in params_list:
            params[param_index] = i   
            for j in init_cond_list:
                args[args_index] = j
                for k in range(measuring_time): 
                    args = [iterated_maps.quadratic_map(*args, *params)]
                    if abs(args[args_index]) > 4:
                        break
                if abs(args[args_index]) < 4:
                    aux_x.append(i)
                    aux_y.append(args[args_index])
        plt.scatter(aux_x, aux_y, marker='o', s=0.001, c='black')
        plt.show()

    def lorentz_map(Signal, lag=1, plot=True):
        """Print the orbit diagram of the iterated map

        Parameters
        ----------
        imap : a iterated map object. The function which to use to produce
        the diagram

        Signal : an array of floats. The time series to plot the lorentz map

        lag : an integer. The lag to consider consecutive measurements
        (Default value = 1)

        plot : a boolean. Wheter to plot the lorenzt map here or just return
        the data
        (Default value = True)  
        Returns
        -------

        
        """ 
        def is_max(vec):
               aux_a = vec[1] - vec[0]
               aux_p = vec[2] - vec[1]
               if aux_a >= 0 and aux_p <= 0:
                    return True
               else:
                    return False

        if lag is None:
            y_0 = []
            for i in range(1, len(Signal) - 1):
                if is_max([Signal[i-1], Signal[i], Signal[i+1]]):
                    y_0.append(Signal[i])
            z_0 = np.roll(y_0, -1)[:-1]
            y_0 = y_0[:-1]
            aux = 'Max to max'
        else:
            aux = str(lag)
            y_0 = []
            for i in range(lag, len(Signal) - lag):
                if is_max([Signal[i-lag], Signal[i], Signal[i+lag]]):
                    y_0.append(Signal[i])
            z_0 = np.roll(y_0, -lag)[:-lag]
            y_0 = y_0[:-lag]
        if plot:
          plt.plot(y_0, y_0, c='red', linewidth=0.5, label='f(x)=x curve')
          plt.scatter(y_0, z_0, marker='.', c='black', s=1, label='data')
          plt.xlabel('f(x-lag)')
          plt.ylabel('f(x)')
          plt.title(f'Lorentz Map (lag = {aux})')
          plt.legend(loc='best')
          plt.show()
        return y_0, z_0

    def minimum_info_tau(data, tau_max=100, graph=False):
        """find the minimum lag interval which composition returns
        the least information (correlation)

        Parameters
        ----------
        data : an array of floats. The time series to find the lag of
        minimum information

        tau_max : an integer. The maximum lag to consider

        graph : a bolean. If true plots the information score against the
        lag
            (Default value = False)

        Returns
        -------

        
        """ 
        hist, bin_edges = np.histogram(data, bins=1000, density=True)
        bin_indices = np.digitize(data, bin_edges)
        data_discrete = [data[index] for index in bin_indices if index < len(data)]
        mis = []
        tau_to_use = None
        for tau in range(1, tau_max):
            unlagged = data_discrete[:-tau]
            lagged = np.roll(data_discrete, -tau)[:-tau]
            joint = np.vstack((unlagged, lagged))
            mis.append(metrics.normalized_mutual_info_score(unlagged, lagged))
            if tau_to_use is None and len(mis) > 1 and mis[-2] < mis[-1]: # return first local minima
                tau_to_use = tau - 1
        if graph:
            # Print mutual information vs time delay
            plt.plot(list(range(1, tau_max)), mis)
            # Blocks until window is closed
            plt.show()
        return tau_to_use 

    def attractor_reconstructor(data, tau_to_use=None, how_many_plots=1,
                                scatter=False, plot=True):
        """Recostruct the attractor of a time series using the 
        method of lags

        Parameters
        ----------
        data : an array of floats. The time series to reconstruct
        the attractor

        tau_to_use : an integer, None or an array of integers. The lag to use.
        If none it will be decided based on the
        non_linear_methods.minimum_info_tau function. The array option must be
        used with how_many_plots set to how_many_plots = len(tau_to_use)
        (Default value = None)

        how_many_plots : an integer. Controls what to plot. If one, only the
        chosen tau_to_use related reconstructor will be used. Otherwise, all
        the elements in tau_to_use array will generate a graph ploted in the
        same plot
        (Default value = 1)

        scatter : a boolean. If True, the function generates scatter plots.
         It generates line plots othewise.
        (Default value = False)

        plot : a boolean. Wheter to plot the lorenzt map here or just return
        the data
        (Default value = True)  
        
        Returns
        -------

        
        """ 
        if not tau_to_use:
            tau_to_use = non_linear_methods.minimum_info_tau(data, tau_max=1000)
            data_lag0 = np.array(data[:-tau_to_use]).flatten()
            data_lag1 = np.array(np.roll(data, -tau_to_use)[:-tau_to_use]).flatten()
            data_lag2 = np.array(np.roll(data, -2 * tau_to_use)[:-tau_to_use]).flatten()
        if plot:
          # Plot time delay embedding
          if how_many_plots == 1:
               fig = plt.figure()
               ax = fig.add_subplot(111, projection='3d')
               if scatter:
                    ax.scatter(data_lag0, data_lag1, data_lag2,
                                   c='black', marker='.', s=1)    
               else:
                    ax.plot(data_lag0, data_lag1, data_lag2,
                         c='black')
               ax.set_title(f'reconstructed attractor of lag {tau_to_use}')
               plt.show()
          else:
               fig, ax = plt.subplots(1, how_many_plots)
               for index, i in enumerate(tau_to_use):
                    data_lag0 = np.array(data[:-i]).flatten()
                    data_lag1 = np.array(np.roll(data, -i)[:-i]).flatten()
                    data_lag2 = np.array(np.roll(data, -2 * i)[:-i]).flatten()
                    ax[index].scatter(data_lag0, data_lag1, data_lag2,
                    marker='.', c='black')
                    ax[index].set_title(f'reconstructed attractor (lagg {i})')
               plt.show()
        return data_lag0, data_lag1, data_lag2       

class series_model_keras():

    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        return ds.batch(batch_size).prefetch(1)

    def recursive_neural_network_lr_optimization(train_set, metrics='mae'):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)
        # model having:
        # -> a 1D-convolutional filter with a width of 5 (2 samples before,
        # two after the selected sample)
        # -> two LSTM layers do the sequence learning work
        # -> two dense layers, and
        #  an output layer (1 neuron) 
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                            strides=1, padding="causal",
                            activation="relu",
                            input_shape=[None, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
        ])
        # learning rate is increased sucessively over the 100 epochs
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=[metrics])
        history = model.fit(train_set, epochs=100, callbacks=[lr_schedule], verbose=0)
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([1e-8, 1e-4, 0, 60])
        plt.title('Learing Rate Optimization')
        plt.xlabel("Learing Rate")
        plt.ylabel("Loss")
        plt.show()
        return model, history

    def plot_optimized_model_progress(train_set, lr):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                            strides=1, padding="causal",
                            activation="relu",
                            input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
        ])
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

        checkpoint_path = "examples/solar_cycles_data/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

        model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=["mae"])
        
        
        history = model.fit(train_set, epochs=500, verbose=0, callbacks=[cp_callback])
        #import matplotlib.image  as mpimg
        loss = history.history['loss']
        epochs = range(len(loss))
        # full graph
        plt.plot(epochs, loss, 'r')
        plt.title('Training loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Loss"])
        #plt.figure()
        plt.show()
        # zoomed graph
        zoomed_loss = loss[200:]
        zoomed_epochs = range(200,500)
        plt.plot(zoomed_epochs, zoomed_loss, 'r')
        plt.title('Training loss (zoomed)')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Loss"])
        #plt.figure()
        plt.show()
        return model

    def create_model():
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                            strides=1, padding="causal",
                            activation="relu",
                            input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
        ])
        model.compile(loss=tf.keras.losses.Huber(),
                    metrics=["mae"])
        return model

    def model_forecast(model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    def find_best(series_list, test_values, window_size, plot=True):

        test = test_values[window_size-1:-window_size]
        predict = np.roll(series_list[:, 0, 0], 0)[window_size:]
        aux = tf.keras.metrics.mean_absolute_error(test, predict).numpy()
        index = 0
        for i in range(1, series_list.shape[1]):
            predict = np.roll(series_list[:, i, 0], i)[window_size:]
            print('Mean Absolute Error (MAE): ', i,
                tf.keras.metrics.mean_absolute_error(test, predict).numpy())   
            temp = tf.keras.metrics.mean_absolute_error(test, predict).numpy()
            if temp < aux:
                aux = temp
                index = i
        predict = np.roll(series_list[:, index, 0], index)[window_size:]
        if plot:
            plt.plot(predict, label='best prediction', c='red')
            plt.plot(test, label='validation set', linestyle='--', c='blue')
            plt.show()
        return predict

    def load_model(checkpoint_path):
        model = series_model_keras.create_model()
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        model.load_weights(checkpoint_path)
        return model

