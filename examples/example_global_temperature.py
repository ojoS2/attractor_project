import sys
import pandas as pd
import numpy as np
import datapackage
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
from src.attractor_project.tools import spectral_analysis as sa
from src.attractor_project.tools import non_linear_methods as nlm
from datetime import timedelta
from scipy.optimize import curve_fit
from scipy import stats

def read_file(name, separator, line_to_start):
    return pd.read_csv(name, sep=separator, skiprows=line_to_start)

def load_from_datahub():
    data_url = 'https://datahub.io/core/global-temp/datapackage.json'
    package = datapackage.Package(data_url)
    resources = package.resources
    for resource in resources:
        if resource.tabular:
            data = pd.read_csv(resource.descriptor['path'])
    return data

def organizing_and_preprocessing():
    df = load_from_datahub()
    df['Date'] = pd.to_datetime(np.array(df['Date'].values))
    GIS = df[df['Source'] == 'GISTEMP']
    GCA = df[df['Source'] == 'GCAG']
    temp = GIS.merge(GCA, how='inner', on='Date')
    df = temp[['Date', 'Mean_x', 'Mean_y']]
    df.columns = ['Date', 'GISTEMP', 'GCAG']
    df = df.sort_values(by='Date', ascending=True, ignore_index=True)
    return df

def time_series_interval_processing(df, days=100):
    init = df.loc[0, 'Date']
    end = df.loc[len(df) - 1, 'Date']
    aux = init
    date = []
    value_GI = []
    value_GC = []
    while aux < end:
        bot = aux
        top = aux + timedelta(days=days)
        value_GI.append(df[df['Date'].between(bot, top)]['GISTEMP'].mean())
        value_GC.append(df[df['Date'].between(bot, top)]['GCAG'].mean())
        date.append(top)
        aux = top
    return pd.DataFrame({'Date': date, 'GISTEMP_AVG': value_GI,
                         'GCAG_AVG': value_GC})

def temperature_data_processing(df):
    aux=lambda date: (date - df.iloc[0, 0])/pd.Timedelta(1, 'd')
    df['time_index'] = df['Date'].apply(aux)
    popt, pcov = curve_fit(exp_func, xdata=df['time_index'],
                           ydata=df['GISTEMP_AVG'])
    fit = lambda x: popt[0] + popt[1]*np.exp(0.0001*popt[2]*x)
    df['GISTEMP_exp_fit'] = df['time_index'].apply(fit)
    popt, pcov = curve_fit(exp_func, xdata=df['time_index'],
                           ydata=df['GCAG_AVG'])
    fit = lambda x: popt[0] + popt[1]*np.exp(0.0001*popt[2]*x)
    df['GCAG_exp_fit'] = df['time_index'].apply(fit)

    df['norm_results_GISTEMP'] = df['GISTEMP_AVG'] - df['GISTEMP_exp_fit']
    df['norm_results_GCAG'] = df['GCAG_AVG'] - df['GCAG_exp_fit']
    df['norm_results_GISTEMP'] = np.array(df['norm_results_GISTEMP'].values\
                                 - df['norm_results_GISTEMP'].mean())
    df['norm_results_GCAG'] = np.array(df['norm_results_GCAG'].values\
                              - df['norm_results_GCAG'].mean())
    return df

def temperature_regression(df, save=None):
    sns.regplot(data=df, x='time_index', y='GISTEMP_AVG',
                    order=12, scatter_kws={'marker': 'o', 'color': 'red',
                                           's': 0.5, 'label': 'Data'},
                    line_kws={'color':'red', 'linewidth': 1,
                        'label':'Order 12 polynomial regression(GISTEMP)'})
    sns.regplot(data=df, x='time_index', y='GCAG_AVG',
                    order=12, scatter_kws={'marker': 'x', 'color': 'blue',
                                           's': 0.5, 'label': 'Data'},
                    line_kws={'color':'blue', 'linewidth': 1,
                        'label':'Order 12 polynomial regression(GCAG)'})
    sns.lineplot(data=df, x='time_index', y='GISTEMP_exp_fit', color='red', 
                 label='exponential regression(GISTEMP)', linestyle='--')
    sns.lineplot(data=df, x='time_index', y='GCAG_exp_fit', color='blue', 
                 label='exponential regression(GCAG)', linestyle='--')
    plt.legend(loc='best')
    aux = len(df) - 1
    plt.xticks(ticks = [df.loc[0, 'time_index'],
                          df.loc[int(aux/2), 'time_index'],
                          df.loc[int(aux), 'time_index']],
                 labels = [df.loc[0, 'Date'],
                           df.loc[int(aux/2), 'Date'],
                           df.loc[int(aux), 'Date']])
    plt.xlabel("Time")
    plt.ylabel("Temperature anomally (in Ceusius)")
    plt.title("Data regression")
    if save:
        plt.savefig(save)
    plt.show()

def exp_func(x, a_0, a_1, a_2):
    return a_0 + a_1*np.exp(0.0001*a_2*x)

def stationary_signal_plot(df, save=None):
    fig, axs = plt.subplots(2, 1)
    plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.8,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    sns.lineplot(data=df, x='time_index', y='norm_results_GISTEMP', color='blue',
                 ax=axs[0])
    sns.lineplot(data=df, x='time_index', y='norm_results_GCAG', color='red',
                 ax=axs[1])
    axs[0].set(xlabel='Date', ylabel='Signal', title='stationary signal(GISTEMP)')
    axs[1].set(xlabel='Date', ylabel='Signal', title='stationary signal(GCAG)')
    aux = len(df) - 1
    axs[0].set_xticks(ticks = [df.loc[0, 'time_index'], df.loc[int(aux), 'time_index']],
                labels = [df.loc[0, 'Date'], df.loc[int(aux), 'Date']])
    axs[1].set_xticks(ticks = [df.loc[0, 'time_index'], df.loc[int(aux), 'time_index']],
                labels = [df.loc[0, 'Date'], df.loc[int(aux), 'Date']])
    
    if save:
        plt.savefig(save)
    

    plt.show()

def plot_peaks_regression(peaks_location):
    x = np.arange(1, len(peaks_location) + 1)
    y = peaks_location*scale/365
    z = [np.log(i) for i in y]
    slope, intercept, r, p, se = stats.linregress(x, z)
    w = [intercept + slope*i for i in x]
    k = [np.exp(intercept) * np.exp(slope*i) for i in x]
    fig, ax = plt.subplots(1, 2, sharey=False)
    ax[0].scatter(x, z, c='black', label='peaks frequency\nlocation')
    ax[0].plot(x, w, c='red', label=f'linear regression\n\
    intercept {round(intercept, 3)}\nslope {round(slope, 3)}')
    ax[0].set_xlabel("peak number")
    ax[0].set_ylabel("logarithm of the peak location")
    ax[0].set_title("Fourier peaks distance regression")
    ax[0].legend(loc='best')
    ax[1].scatter(x, y, c='black', label='peaks frequency\nlocation')
    ax[1].plot(x, k, c='red', label=f'exp regression')
    ax[1].set_xlabel("peak number")
    ax[1].set_ylabel("peak location (in years)")
    ax[1].set_title("Fourier peaks distance regression")
    ax[1].legend(loc='best')
    plt.show()

def plot_frequencies_spectrum_pieces(data_GI, data_GC, scope, label,
                                     duration, sample_rate, save=None):
    #df.loc[int(len(df) - 1), 'time_index']
    #sample_rate=1/40
    x, y_GI = sa.fourier_discreet_transform(data_GI,
                                        sample_rate=sample_rate,
                                        duration=duration)

    _, y_GC = sa.fourier_discreet_transform(data_GC,
                                            sample_rate=sample_rate,
                                            duration=duration)
    plt.axvline(1./365, np.min(y_GC), np.max(y_GC), c='red', linestyle=':',
                alpha = 0.8 )
    plt.annotate('1 year frequency',
                xy=(1./365, np.max(y_GC)), xytext=(6, -100),
                textcoords='offset points',
                rotation=90, va='bottom', ha='center')
    plt.axvline(0.0055, np.min(y_GC), np.max(y_GC), c='black', linestyle=':',
                alpha = 0.8 )
    plt.annotate('half a year frequency',
                xy=(0.0055, np.max(y_GC)), xytext=(6, -100),
                textcoords='offset points',
                rotation=90, va='bottom', ha='center')

    plt.axvline(0.00081, np.min(y_GC), np.max(y_GC), c='black', linestyle=':',
                alpha = 0.8 )
    plt.annotate('3.429 years frequency',
                xy=(0.00081, np.max(y_GC)), xytext=(6, -100),
                textcoords='offset points',
                rotation=90, va='bottom', ha='center')
    plt.axvline(0.00032, np.min(y_GC), np.max(y_GC), c='black', linestyle=':',
                alpha = 0.8 )
    plt.annotate('8,96 years frequency',
                xy=(0.00032, np.max(y_GC)), xytext=(6, -100),
                textcoords='offset points',
                rotation=90, va='bottom', ha='center')
    
    #plt.axvline(0.00017, np.min(y_GC), np.max(y_GC), c='black', linestyle=':',
    #            alpha = 0.8 )
    #plt.annotate('16,33 years frequency',
    #            xy=(0.00017, np.max(y_GC)), xytext=(6, -300),
    #            textcoords='offset points',
    #            rotation=90, va='bottom', ha='center')
    
    plt.axvline(0.00005, np.min(y_GC), np.max(y_GC), c='black', linestyle=':',
                alpha = 0.8 )
    plt.annotate('55,55 years frequency',
                xy=(0.00005, np.max(y_GC)), xytext=(-6, -100),
                textcoords='offset points',
                rotation=90, va='bottom', ha='center')
    

    plt.subplots_adjust(left=0.2,
                        bottom=0.2,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    inf, sup = int(len(x)*scope[0]), int(len(x)*scope[1])
    if sup >= len(x):
        sup = len(x) - 1
    plt.plot(x[inf: sup], y_GI[inf: sup], label='frequencies spectrum GISTEMP', c='red', alpha=0.5)
    plt.plot(x[inf: sup], y_GC[inf: sup], label='frequencies spectrum GCAG', c='blue', alpha=0.5)
    locs, _ = plt.xticks()
    locs = locs[:-1]
    #plt.xticks(locs[locs>0.], [str(365*i) for i in locs[locs>0.]], rotation=90)
    plt.xlabel("Frequency in inverse of years")
    plt.ylabel("Absolute amplitude")
    plt.title("Frequency's spectrum" + label)
    plt.legend(loc='best')
    plt.xlim(x[inf], x[sup])
    if save is not None:
        plt.savefig('examples/figures/'+ save + '.png')
    plt.show()

def load_data():
    temp = organizing_and_preprocessing()
    scale=40
    df = time_series_interval_processing(temp, days=scale)
    df = temperature_data_processing(df)
    temperature_regression(df, save=None)
    stationary_signal_plot(df, save=None)
    data_GI = list(df['norm_results_GISTEMP'].values)
    data_GC = list(df['norm_results_GCAG'].values)
    return data_GI, data_GC

def plot_all(data_GI, data_GC):

    plot_frequencies_spectrum_pieces(data_GI, data_GC,
                                 scope=[0, 1], label='',
                                 save=None)
    perc, p_val, corr = sa.best_scale(data_GI, inf=0.001, sup=0.5, p_threshold=0.005,
                    grafics=True)
    print(perc, p_val, corr)
    perc, p_val, corr = sa.best_scale(data_GC, inf=0.001, sup=0.5, p_threshold=0.005,
                    grafics=True)
    print(perc, p_val, corr)
    nlm.lorentz_map(data_GI)
    nlm.lorentz_map(data_GC)
    nlm.attractor_reconstructor(data_GI, tau_to_use=None, how_many_plots=1,
                                    scatter=True)
    nlm.attractor_reconstructor(data_GC, tau_to_use=None, how_many_plots=1,
                                    scatter=True)
    
pd.options.display.max_columns = None #print all columns    
pd.options.display.max_rows = None #print all rows  
data_GI, data_GC = load_data()
plot_all(data_GI, data_GC)
# build a prediction



