import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
from tabulate import tabulate
from src.attractor_project.tools import parametric_diferential_equations as pde
from src.attractor_project.tools import iterated_maps as im
from src.attractor_project.tools import time_series_generators as tsg
from src.attractor_project.tools import spectral_analysis as sa
from src.attractor_project.tools import non_linear_methods as nlm

def build_data():

    series_length = 100000
    transient = 1000
    quadratic = {'init_state': [0],
                 'params': [1, 0, -1.7, 1]}
    henon = {'init_state': [0.1, .1],
             'params': [1.4, 0.3, 1]}
    quadratic_map = tsg.\
        generate_series_from_iterated_maps(data_length=series_length,
        iter_map=im.quadratic_map, initial_state=quadratic['init_state'],
        transient=transient, parameters=quadratic['params'])
    henon_map = tsg.\
        generate_series_from_iterated_maps(data_length=series_length,
        iter_map=im.henon_map, initial_state=henon['init_state'],
        transient=transient, parameters=henon['params'])
    df_im_ts = pd.DataFrame({'quadractic map': quadratic_map[:-1],
                             'henon map x': [i[0] for i in henon_map][:-1],
                             'henon map y': [i[1] for i in henon_map][:-1]})
    dt = 0.01
    pendulum = {'init_state': [np.pi - 0.1, 0.0],
                'params':[.25, 5.]}
    rossler = {'init_state': [5., 5., 5.],
                'params':[0.2, 0.2, 5.7]}
    lorenz = {'init_state': [1., 1., 1.],
                'params':[10., 8/3., 28.]}
    ODE_s_pendulum = tsg.generate_series_from_ODE(data_length=series_length,
                                                ode=pde.pendulum_ode,
                                                state=pendulum['init_state'],
                                                parameters=pendulum['params'],
                                                dt=dt,
                                                transient=transient)
    ODE_s_rossler = tsg.generate_series_from_ODE(data_length=series_length,
                                                ode=pde.rossler_ode,
                                                state=rossler['init_state'],
                                                parameters=rossler['params'],
                                                dt=dt,
                                                transient=transient)
    ODE_s_lorenz = tsg.generate_series_from_ODE(data_length=series_length,
                                                ode=pde.lorenz_ode,
                                                state=lorenz['init_state'],
                                                parameters=lorenz['params'],
                                                dt=dt,
                                                transient=transient)
    df_ode_ts = pd.DataFrame({'pendulum x': [i[0] for i in ODE_s_pendulum],
                              'pendulum y': [i[1] for i in ODE_s_pendulum],
                              'rossler x': [i[0] for i in ODE_s_rossler],
                              'rossler y': [i[1] for i in ODE_s_rossler],
                              'rossler z': [i[2] for i in ODE_s_rossler],
                              'lorenz x': [i[0] for i in ODE_s_lorenz],
                              'lorenz y': [i[1] for i in ODE_s_lorenz],
                              'lorenz z': [i[2] for i in ODE_s_lorenz]})
    return df_im_ts, df_ode_ts

def plot_hist_im(df_im, name=None):
    sns.histplot(data=df_im, stat='density',
                x='quadractic map', bins=100, alpha=0.4,
                color='blue', label='quadractic map')
    sns.histplot(data=df_im, stat='density',
                x='henon map x', bins=100, alpha=0.4,
                color='red', label='henon map x coordinate')
    sns.histplot(data=df_im, stat='density',
                x='henon map y', bins=35, alpha=0.4,
                color='yellow', label='henon map y coordinate')
    plt.legend(loc='best')
    plt.xlabel('Values')
    if name is not None:
        plt.savefig(name+'.png')
    plt.show()

def plot_hist_ode(df_ode, name=None):
    '''
    sns.histplot(data=df_ode_ts, stat='density',
                x='pendulum x', bins=100, alpha=0.4,
                color='blue', label='pendulum x coordinate')
    sns.histplot(data=df_ode_ts, stat='density',
                x='pendulum x', bins=100, alpha=0.4,
                color='purple', label='pendulum y coordinate')
    '''
    sns.histplot(data=df_ode, stat='density',
                x='rossler x', bins=50, alpha=0.4,
                color='pink', label='rossler x coordinate')
    sns.histplot(data=df_ode, stat='density',
                x='rossler y', bins=50, alpha=0.4,
                color='red', label='rossler y coordinate')
    #sns.histplot(data=df_ode, stat='density',
    #            x='rossler z', bins=5, alpha=0.4,
    #            color='orange', label='rossler z coordinate')
    sns.histplot(data=df_ode, stat='density',
                x='lorenz x', bins=100, alpha=0.4,
                color='yellow', label='lorenz x coordinate')
    sns.histplot(data=df_ode, stat='density',
                x='lorenz y', bins=100, alpha=0.4,
                color='green', label='lorenz y coordinate')
    sns.histplot(data=df_ode, stat='density',
                x='lorenz z', bins=100, alpha=0.4,
                color='cyan', label='lorenz z coordinate')
    plt.legend(loc='best')
    plt.xlabel('Values')
    if name is not None:
        plt.savefig(name+'.png')
    plt.show()

def plot_1D(df, name=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    nlm.cobweb_diagram(imap=im.quadratic_map,
                       init_condit=[0],
                       params=[1, 0, -1.7, 1],
                       iter=100, xlim=[-3, 3],
                       ylim=[-3, 3],
                       show=False, ax=ax0)
    ax0.set_title('Quadractic map \n cobweb diagram')
    ax1 = fig.add_subplot(122)
    ax1.plot(df['quadractic map'][:100], c='blue')
    ax1.set_title('Quadractic map \n time series')
    if name is not None:
        plt.savefig(name+'.png')
    plt.show()

def plot_2D(df_im, df_ode, name=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(221)
    ax0.scatter(df_im['henon map x'], df_im['henon map y'],
                color='black',
                label='Henon map attractor', marker='.', s=0.0001)
    ax0.set_title('Henon attractor')
    ax1 = fig.add_subplot(222)
    ax1.plot(df_im['henon map x'][-50:], c='blue', label='x coordinate')
    ax1.plot(df_im['henon map y'][-50:], c='red', label='y coordinate')
    ax1.set_title('Henon map \n coordinates time series')
    ax3 = fig.add_subplot(223)
    plt.scatter(df_ode['pendulum x'], df_ode['pendulum y'],
                color='black',
                label='Pendulum attractor', marker='.', s=0.01)
    ax3.set_title('Pendulum \n attractor')
    ax4 = fig.add_subplot(224)
    ax4.plot(df_ode['pendulum x'][-1000:], c='blue', label='x coordinate')
    ax4.plot(df_ode['pendulum y'][-1000:], c='red', label='y coordinate')
    ax4.set_title('Pendulum \n coordinates time series')
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)
    plt.legend(loc='best')
    if name is not None:
        plt.savefig(name+'.png')
    plt.show()

def plot_3D(df_ode, name=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(221, projection = '3d')
    ax0.plot(df_ode['rossler x'], df_ode['rossler y'],
            df_ode['rossler z'], c='black', lw=0.1)
    ax0.set_title('Rossler attractor')
    ax1 = fig.add_subplot(222)
    ax1.plot(df_ode['rossler x'][-3000:], c='blue', label='x coordinate')
    ax1.plot(df_ode['rossler y'][-3000:], c='red', label='y coordinate')
    ax1.plot(df_ode['rossler z'][-3000:], c='green', label='z coordinate')
    ax1.set_title('Rossler attractor \n coordinates time series')
    ax2 = fig.add_subplot(223, projection = '3d')
    ax2.plot(df_ode['lorenz x'], df_ode['lorenz y'],
            df_ode['lorenz z'], c='black', lw=0.1)
    ax2.set_title('Lorenz attractor')
    ax3 = fig.add_subplot(224)
    ax3.plot(df_ode['lorenz x'][-700:], c='blue', label='x coordinate')
    ax3.plot(df_ode['lorenz y'][-700:], c='red', label='y coordinate')
    ax3.plot(df_ode['lorenz z'][-700:], c='green', label='z coordinate')
    ax3.set_title('Lorenz attractor \n coordinates time series')
    plt.subplots_adjust(left=0.2,
                            bottom=0.1,
                            right=0.8,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.6)
    plt.legend(loc='best')
    if name is not None:
        plt.savefig(name+'.png')
    plt.show()

def spectral_decomposition(data):
    temp = data
    temp = temp - np.mean(data)
    frequencies, amplitudes = sa.\
    fourier_discreet_transform(data=temp,
                            sample_rate=1,
                            duration=len(temp))
    filtered_spectrum, filtered_signal=sa.\
    filtered_signal(perc=0, spectrum=amplitudes)
    return temp, frequencies, amplitudes, filtered_spectrum, filtered_signal

def plot_spectral_decomposition(axA, axB, data_name, signal,
                                frequencies, amplitudes,
                                filtered_signal, xmin=100,
                                xmax=200):
    axA.plot(frequencies, amplitudes, c='black', alpha=0.3)
    axA.set_title(f'{data_name} \n descreete Fourier spectrum')
    axB.plot(signal[xmin: xmax], c='blue', label='generated series', alpha=0.5)
    axB.plot(filtered_signal[xmin: xmax], c='red', label='inverse transform',
            alpha=0.9)
    axB.set_title(f'{data_name} \n time series')

def plot_spectral_analysis_maps(df_im, name=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(321)
    ax1 = fig.add_subplot(322)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(324)
    ax4 = fig.add_subplot(325)
    ax5 = fig.add_subplot(326)
    signal, frequencies, amplitudes, _, filtered_signal = \
    spectral_decomposition(data = list(df_im['quadractic map'].values))
    plot_spectral_decomposition(axA = ax0, axB = ax1, data_name='Quadractic map',
                                signal=signal, frequencies=frequencies,
                                amplitudes=np.abs(amplitudes),
                                filtered_signal=filtered_signal)

    signal, frequencies, amplitudes, _, filtered_signal = \
    spectral_decomposition(data = list(df_im['henon map x'].values))
    plot_spectral_decomposition(axA=ax2, axB=ax3,
                                data_name='Henon map x coordinate',
                                signal=signal, frequencies=frequencies,
                                amplitudes=np.abs(amplitudes),
                                filtered_signal=filtered_signal)
    signal, frequencies, amplitudes, _, filtered_signal = \
    spectral_decomposition(data = list(df_im['henon map y'].values))
    plot_spectral_decomposition(axA=ax4, axB=ax5,
                                data_name='Henon map y coordinate',
                                signal=signal, frequencies=frequencies,
                                amplitudes=np.abs(amplitudes),
                                filtered_signal=filtered_signal)
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.8)
    ax3.legend(bbox_to_anchor=(0.465, 0.5))
    if name is not None:
        plt.savefig(name + '.png')
    plt.show()

def plot_spectral_analysis_ODEs(df_ode, name=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(421)
    ax1 = fig.add_subplot(422)
    ax2 = fig.add_subplot(423)
    ax3 = fig.add_subplot(424)
    ax4 = fig.add_subplot(425)
    ax5 = fig.add_subplot(426)
    ax6 = fig.add_subplot(427)
    ax7 = fig.add_subplot(428)
    signal, frequencies, amplitudes, _, filtered_signal = \
    spectral_decomposition(data = list(df_ode['rossler x'].values))
    plot_spectral_decomposition(axA = ax0, axB = ax1, data_name='Rossler x coordinate',
                                signal=signal, frequencies=frequencies,
                                amplitudes=amplitudes,
                                filtered_signal=filtered_signal,
                                xmin=1000, xmax=5000)

    signal, frequencies, amplitudes, _, filtered_signal = \
    spectral_decomposition(data = list(df_ode['rossler z'].values))
    plot_spectral_decomposition(axA=ax2, axB=ax3,
                                data_name='Rossler z coordinate',
                                signal=signal, frequencies=frequencies,
                                amplitudes=amplitudes,
                                filtered_signal=filtered_signal,
                                xmin=1000, xmax=5000)
    signal, frequencies, amplitudes, _, filtered_signal = \
    spectral_decomposition(data = list(df_ode['lorenz x'].values))
    plot_spectral_decomposition(axA=ax4, axB=ax5,
                                data_name='Lorenz x coordinate',
                                signal=signal, frequencies=frequencies,
                                amplitudes=amplitudes,
                                filtered_signal=filtered_signal,
                                xmin=500, xmax=2000)
    signal, frequencies, amplitudes, _, filtered_signal = \
    spectral_decomposition(data = list(df_ode['lorenz z'].values))
    plot_spectral_decomposition(axA=ax6, axB=ax7,
                                data_name='Lorenz z coordinate',
                                signal=signal, frequencies=frequencies,
                                amplitudes=amplitudes,
                                filtered_signal=filtered_signal,
                                xmin=500, xmax=2000)
    plt.subplots_adjust(left=0.2,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=1.3)
    ax3.legend(bbox_to_anchor=(0.17, 0.45))
    if name is not None:
        plt.savefig(name+'.png')
    plt.show()

def Lorentz_map_plots(df_im, df_ode, name=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(321)
    ax1 = fig.add_subplot(322)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(324)
    ax4 = fig.add_subplot(325)
    ax5 = fig.add_subplot(326)
    x, y = nlm.lorentz_map(Signal=list(df_im['quadractic map'].values),
                        lag=1, plot=False)
    ax0.scatter(x, y, c='black', marker='.', s=0.1, label='data')
    ax0.plot(x, x, c='red', linewidth=0.5, label='f(x)=x curve')
    ax0.set_title('Lorenz map of \n the quadractic map')
    ax0.set_ylabel('f(x+1)')
    x, y = nlm.lorentz_map(Signal=list(df_im['henon map x'].values),
                        lag=1, plot=False)
    ax1.scatter(x, y, c='black', marker='.', s=0.1, label='data')
    ax1.set_title('Lorenz map of \n the Henon map coordinate x')
    ax1.plot(x, x, c='red', linewidth=0.5, label='f(x)=x curve')
    x, y = nlm.lorentz_map(Signal=list(df_ode['rossler x'].values),
                        lag=1, plot=False)
    ax2.scatter(x, y, c='black', marker='.', s=0.1, label='data')
    ax2.plot(x, x, c='red', linewidth=0.5, label='f(x)=x curve')
    ax2.set_title('Lorenz map of \n the Rossler map coordinate x')
    ax2.set_ylabel('f(x+1)')
    x, y = nlm.lorentz_map(Signal=list(df_ode['rossler z'].values),
                        lag=1, plot=False)
    ax3.scatter(x, y, c='black', marker='.', s=0.1, label='data')
    ax3.plot(x, x, c='red', linewidth=0.5, label='f(x)=x curve')
    ax3.set_title('Lorenz map of \n the Rossler map coordinate z')
    x, y = nlm.lorentz_map(Signal=list(df_ode['lorenz x'].values),
                        lag=1, plot=False)
    ax4.scatter(x, y, c='black', marker='.', s=0.1, label='data')
    ax4.plot(x, x, c='red', linewidth=0.5, label='f(x)=x curve')
    ax4.set_title('Lorenz map of \n the Lorenz map coordinate x')
    ax4.set_ylabel('f(x+1)')
    ax4.set_xlabel('f(x)')
    x, y = nlm.lorentz_map(Signal=list(df_ode['lorenz z'].values),
                        lag=1, plot=False)
    ax5.scatter(x, y, c='black', marker='.', s=0.1, label='data')
    ax5.plot(x, x, c='red', linewidth=0.5, label='f(x)=x curve')
    ax5.set_title('Lorenz map of \n the Lorenz map coordinate z')
    ax5.set_xlabel('f(x)')
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.8)
    if name is not None:
        plt.savefig(name+'.png')
    plt.show()

def plot_attractors(df_im, df_ode, name=None):
    fig = plt.figure()
    x, y, z, tau = nlm.attractor_reconstructor(data=list(\
        df_im['quadractic map'].values),
        tau_to_use=1, how_many_plots=1,
        scatter=True, plot=False)
    ax0 = fig.add_subplot(121, projection = '3d')
    ax0.scatter(x, y, z, marker='.', c='black', s=0.1)
    ax0.set_title('Reconstructed attractor \n\
                quadractic map (lag=1)')
    x, y, z, tau = nlm.attractor_reconstructor(data=list(\
        df_im['henon map x'].values),
        tau_to_use=1, how_many_plots=1,
        scatter=True, plot=False)
    ax1 = fig.add_subplot(122, projection = '3d')
    ax1.scatter(x, y, z, marker='.', c='black', s=0.1)
    ax1.set_title('Reconstructed attractor \n\
                henon map (x coordinate, lag=1)')
    if name is not None:
        plt.savefig(name[0]+'.png')
    plt.show()
    fig = plt.figure()
    x, y, z, tau = nlm.attractor_reconstructor(data=list(\
        df_ode['rossler x'].values),
        tau_to_use=None, how_many_plots=1,
        scatter=True, plot=False)
    ax0 = fig.add_subplot(221, projection = '3d')
    ax0.scatter(x, y, z, marker='.', c='black', s=0.1)
    ax0.set_title(f'Reconstructed attractor \n\
                Rossler ODE (x coordinate, lag={tau})')
    x, y, z, tau = nlm.attractor_reconstructor(data=list(\
        df_ode['rossler z'].values),
        tau_to_use=None, how_many_plots=1,
        scatter=True, plot=False)
    ax1 = fig.add_subplot(222, projection = '3d')
    ax1.scatter(x, y, z, marker='.', c='black', s=0.1)
    ax1.set_title(f'Reconstructed attractor \n\
                Rossler ODE (z coordinate, lag={tau})')
    x, y, z, tau = nlm.attractor_reconstructor(data=list(\
        df_ode['lorenz x'].values),
        tau_to_use=None, how_many_plots=1,
        scatter=True, plot=False)
    ax2 = fig.add_subplot(223, projection = '3d')
    ax2.scatter(x, y, z, marker='.', c='black', s=0.1)
    ax2.set_title(f'Reconstructed attractor \n\
                Lorenz ODE (x coordinate, lag={tau})')
    x, y, z, tau = nlm.attractor_reconstructor(data=list(\
        df_ode['lorenz z'].values),
        tau_to_use=None, how_many_plots=1,
        scatter=True, plot=False)
    ax3 = fig.add_subplot(224, projection = '3d')
    ax3.scatter(x, y, z, marker='.', c='black', s=0.1)
    ax3.set_title(f'Reconstructed attractor \n\
                Lorenz ODE (z coordinate, lag={tau})')
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.8)
    if name is not None:
        plt.savefig(name[1]+'.png')
    plt.show()

# building the data from iterated maps and diferential equations
df_im, df_ode = build_data()

print(tabulate(df_im.head(10), headers = 'keys', tablefmt = 'rst'))
print(tabulate(df_ode.head(10), headers = 'keys', tablefmt = 'rst'))


# bidimensional maps plots

plot_1D(df_im)#, name='examples/figures/1D_visualization')
plot_2D(df_im, df_ode)#, name='examples/figures/2D_visualization')
plot_3D(df_ode)#, name='examples/figures/3D_visualization')


# ploting histograms of the iterated maps
plot_hist_im(df_im)#, name='examples/figures/tm_hist')
plot_hist_ode(df_ode)#, name='examples/figures/ode_hist')



#spectral analysis 
plot_spectral_analysis_maps(df_im, name='examples/figures/im_spectrum')
plot_spectral_analysis_ODEs(df_ode, name='examples/figures/ode_spectrum')






#attractor reconstructor

plot_attractors(df_im, df_ode, name=['examples/figures/rec_im_attractor', 'examples/figures/rec_ode_attractor'])

# non-linear methods
# Lorenz map

Lorentz_map_plots(df_im, df_ode, name='examples/figures/lorentz_map' )
