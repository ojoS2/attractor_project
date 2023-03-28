import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
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

def plot_hist_im():
    sns.histplot(data=df_im_ts, stat='density',
                x='quadractic map', bins=100, alpha=0.4,
                color='blue', label='quadractic map')
    sns.histplot(data=df_im_ts, stat='density',
                x='henon map x', bins=100, alpha=0.4,
                color='red', label='henon map x coordinate')
    sns.histplot(data=df_im_ts, stat='density',
                x='henon map y', bins=35, alpha=0.4,
                color='yellow', label='henon map y coordinate')
    plt.legend(loc='best')
    plt.show()

def plot_hist_ode():
    '''
    sns.histplot(data=df_ode_ts, stat='density',
                x='pendulum x', bins=100, alpha=0.4,
                color='blue', label='pendulum x coordinate')
    sns.histplot(data=df_ode_ts, stat='density',
                x='pendulum x', bins=100, alpha=0.4,
                color='purple', label='pendulum y coordinate')
    '''
    sns.histplot(data=df_ode_ts, stat='density',
                x='rossler x', bins=50, alpha=0.4,
                color='pink', label='rossler x coordinate')
    sns.histplot(data=df_ode_ts, stat='density',
                x='rossler y', bins=50, alpha=0.4,
                color='red', label='rossler y coordinate')
    sns.histplot(data=df_ode_ts, stat='density',
                x='rossler z', bins=10, alpha=0.4,
                color='orange', label='rossler z coordinate')
    sns.histplot(data=df_ode_ts, stat='density',
                x='lorenz x', bins=100, alpha=0.4,
                color='yellow', label='lorenz x coordinate')
    sns.histplot(data=df_ode_ts, stat='density',
                x='lorenz y', bins=100, alpha=0.4,
                color='green', label='lorenz y coordinate')
    sns.histplot(data=df_ode_ts, stat='density',
                x='lorenz z', bins=100, alpha=0.4,
                color='cyan', label='lorenz z coordinate')
    plt.legend(loc='best')
    plt.show()

# building the data from iterated maps and diferential equations
df_im_ts, df_ode_ts = build_data()
# ploting histograms of the iterated maps
#plot_hist_im()
#plot_hist_ode()


# bidimensional maps plots
def plot_2D():
    fig = plt.figure()
    ax0 = fig.add_subplot(221)
    ax0.scatter(df_im_ts['henon map x'], df_im_ts['henon map y'],
                color='black',
                label='Henon map attractor', marker='.', s=0.0001)
    ax0.set_title('Henon attractor')
    ax1 = fig.add_subplot(222)
    ax1.plot(df_im_ts['henon map x'][-50:], c='blue', label='x coordinate')
    ax1.plot(df_im_ts['henon map y'][-50:], c='red', label='y coordinate')
    ax1.set_title('Henon map \n coordinates time series')
    ax3 = fig.add_subplot(223)
    plt.scatter(df_ode_ts['pendulum x'], df_ode_ts['pendulum y'],
                color='black',
                label='Pendulum attractor', marker='.', s=0.01)
    ax3.set_title('Pendulum \n coordinates time series')
    ax4 = fig.add_subplot(224)
    ax4.plot(df_ode_ts['pendulum x'][-1000:], c='blue', label='x coordinate')
    ax4.plot(df_ode_ts['pendulum y'][-1000:], c='red', label='y coordinate')
    ax4.set_title('Pendulum \n coordinates time series')
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)
    plt.legend(loc='best')
    plt.show()

def plot_3D():
    fig = plt.figure()
    ax0 = fig.add_subplot(221, projection = '3d')
    ax0.plot(df_ode_ts['rossler x'], df_ode_ts['rossler y'],
            df_ode_ts['rossler z'], c='black', lw=0.1)
    ax0.set_title('Rossler attractor')
    ax1 = fig.add_subplot(222)
    ax1.plot(df_ode_ts['rossler x'][-3000:], c='blue', label='x coordinate')
    ax1.plot(df_ode_ts['rossler y'][-3000:], c='red', label='y coordinate')
    ax1.plot(df_ode_ts['rossler z'][-3000:], c='green', label='z coordinate')
    ax1.set_title('Rossler attractor \n coordinates time series')
    ax2 = fig.add_subplot(223, projection = '3d')
    ax2.plot(df_ode_ts['lorenz x'], df_ode_ts['lorenz y'],
            df_ode_ts['lorenz z'], c='black', lw=0.1)
    ax2.set_title('Lorenz attractor')
    ax3 = fig.add_subplot(224)
    ax3.plot(df_ode_ts['lorenz x'][-700:], c='blue', label='x coordinate')
    ax3.plot(df_ode_ts['lorenz y'][-700:], c='red', label='y coordinate')
    ax3.plot(df_ode_ts['lorenz z'][-700:], c='green', label='z coordinate')
    ax3.set_title('Lorenz attractor \n coordinates time series')

    plt.subplots_adjust(left=0.2,
                            bottom=0.1,
                            right=0.8,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.6)
    plt.legend(loc='best')
    plt.show()

#plot_2D()
plot_3D()