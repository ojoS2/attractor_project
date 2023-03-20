#https://ecg.mit.edu/time-series/hr.11839
import sys
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
from src.attractor_project.tools import parametric_diferential_equations as pde
from src.attractor_project.tools import iterated_maps as im
from src.attractor_project.tools import time_series_generators as tsg
from src.attractor_project.tools import spectral_analysis as sa
from src.attractor_project.tools import non_linear_methods as nlm

class MIT():
    def load_data_MIT():
        def include_time(df):
            time = [0.5*i + 0.5 for i in range(df.shape[0])]
            df['time'] = time
            return df
        series_1 = pd.read_csv('https://ecg.mit.edu/time-series/hr.11839', names=['HR'])
        include_time(series_1)
        series_2 = pd.read_csv('https://ecg.mit.edu/time-series/hr.7257', names=['HR'])
        include_time(series_2)
        series_3 = pd.read_csv('https://ecg.mit.edu/time-series/hr.207', names=['HR'])
        include_time(series_3)
        series_4 = pd.read_csv('https://ecg.mit.edu/time-series/hr.237', names=['HR'])
        include_time(series_4)
        return series_1, series_2, series_3, series_4


pd.options.display.max_columns = None #print all columns    
pd.options.display.max_rows = None #print all rows  

def plot_original_and_filtered(series_1, series_2, series_3, series_4):
    long_df, short_df = [series_1, series_2], [series_3, series_4]
    long_titles, short_titles = ['Series_1', 'Series_2'], ['Series_3', 'Series_4']
    long_duration, short_duration = 900, 475
    for i in range(len(long_df)):
        data = list(long_df[i]['HR'].values - np.mean(long_df[i]['HR']))
        _, spectrum = sa.fourier_discreet_transform(
        data, sample_rate=2,
        duration=long_duration)
        scale, p_value, corr = sa.best_scale(data, inf=0.001, sup=0.5, p_threshold=0.005,
                    grafics=False)
        print(scale, p_value, corr)
        filtered_spectrum, filtered_signal = sa.filtered_signal(scale, spectrum)
        _, axs = plt.subplots(2,1)
        plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8,
                            top=0.9, wspace=0.4, hspace=0.4)
        axs[0].plot(data, c='blue', alpha=0.5, label='original')
        axs[0].plot(filtered_signal, c='red', alpha=0.5, label='filtered')
        axs[0].set_title('Signal ' + long_titles[i])
        axs[0].set_xticks([])

        axs[1].plot(spectrum[:200], c='blue', alpha=0.5, label='original')
        axs[1].plot(filtered_spectrum[:200], c='red', alpha=0.5, label='filtered')
        axs[1].set_title('Spectrum ' + long_titles[i])
        axs[1].set_xticks([])
        plt.show()

    for i in range(len(short_df)):
        data = list(short_df[i]['HR'].values - np.mean(short_df[i]['HR']))
        _, spectrum = sa.fourier_discreet_transform(
        data, sample_rate=2,
        duration=short_duration)
        scale, p_value, corr = sa.best_scale(data, inf=0.001, sup=0.5,
                                            p_threshold=0.005,
                                            grafics=False)
        print(scale, p_value, corr)
        filtered_spectrum, filtered_signal = sa.filtered_signal(scale, spectrum)
        _, axs = plt.subplots(2,1)
        plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
        axs[0].plot(data, c='blue', alpha=0.5, label='original')
        axs[0].plot(filtered_signal, c='red', alpha=0.5, label='filtered')
        axs[0].set_title('Signal ' + short_titles[i])
        axs[0].set_xticks([])

        axs[1].plot(spectrum[:200], c='blue', alpha=0.5, label='original')
        axs[1].plot(filtered_spectrum[:200], c='red', alpha=0.5, label='filtered')
        axs[1].set_title('Spectrum ' + short_titles[i])
        axs[1].set_xticks([])
        plt.show()

def plot_non_linear_measurements(series_1, series_2, series_3, series_4):

    data = [list(series_1['HR'].values - np.mean(series_1['HR'])),
            list(series_2['HR'].values - np.mean(series_2['HR'])),
            list(series_3['HR'].values - np.mean(series_3['HR'])),
            list(series_4['HR'].values - np.mean(series_4['HR']))]

    _, axs = plt.subplots(2,2)
    aux = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8,
                        top=0.9, wspace=0.4, hspace=0.4)
    for index, series in enumerate(data):
        y, z = nlm.lorentz_map(series, lag=None, plot=False)
        aux[index].plot(y, y, c='red', linewidth=0.5, label='f(x)=x curve')
        aux[index].scatter(y, z, marker='.', c='black', s=1, label='data')
        aux[index].set(xlabel='f(x-lag)', ylabel='f(x)',
                    title=f'Series {index + 1} lorentz map')    
    plt.show()    


    fig, axs = plt.subplots(2,2,figsize=(10,10), subplot_kw=dict(projection='3d'))
    aux = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
    for index, series in enumerate(data):
        #aux[index] = fig.add_subplot(1, 2, 2, projection='3d')
        x, y, z = nlm.attractor_reconstructor(series, tau_to_use=None, how_many_plots=1,
                                    scatter=False, plot=False)
        aux[index].plot(x, y, z, c='red', linewidth=0.5)
        aux[index].scatter(x, y, z, marker='.', c='black', s=1)
        aux[index].set(title=f'Series {index + 1} reconstructed attractor')
    plt.show()

#series_1, series_2, series_3, series_4 = MIT.load_data_MIT()
#plot_original_and_filtered(series_1, series_2, series_3, series_4)
#plot_non_linear_measurements(series_1, series_2, series_3, series_4)

class PhysioNet():

    def verify_distribution_differences(A, B):
        stat, p_value = ttest_ind(A, B)
        print(f" populations t-test: statistic={stat:.4f}, p-value={p_value:.4f}")
        temp = "Considering a p-value threshold of 0.05 we consider "
        if p_value <= 0.05:
            print(temp + "the distributions distinct.")
        else:
            print(temp + "that the distinctions of the distributions are not statistically relevant")
        return stat, p_value

    def load_data():
        os.system('mkdir -p examples/heartbeat_data')
        os.system('wget -r -N -c -np -P examples/heartbeat_data https://physionet.org/files/rr-interval-healthy-subjects/1.0.0/ ')
        os.system('ls examples/heartbeat_data/physionet.org/files/rr-interval-healthy-subjects/1.0.0 > examples/heartbeat_data/names.txt')
        with open('examples/heartbeat_data/names.txt') as f:
            temp = f.readlines()
        files = []
        for i in temp:
            files.append(i.strip('\n'))
        path = 'examples/heartbeat_data/physionet.org/files/rr-interval-healthy-subjects/1.0.0/'
        TimeSeries = {}
        for filename in files:
            id = filename[:-4]
            if id.isdigit():
                temp = pd.read_csv(path+filename, names=[filename[:-4]])
                TimeSeries[int(id)] = list(temp[filename[:-4]][:35000].values)
            elif filename[-4:] == '.csv':
                features = pd.read_csv(path+filename, names=['Id', 'age', 'gender'], skiprows=1)
        # for some reason this file reads as char
        TimeSeries[8] = [int(i) for i in TimeSeries[8]]
        # drop columns not found in the data
        to_drop = [i in TimeSeries.keys() for i in features.Id] 
        features = features[to_drop]
        os.system('rm -R examples/heartbeat_data')
        return TimeSeries, features

    def age_gender_exploration():
        fig, axs = plt.subplots(1,2)
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                            top=0.8, wspace=0.5, hspace=0.5)
        sns.boxplot(data=features, x='gender', y='age', ax = axs[0])
        sns.histplot(data=features, x='age', hue = 'gender', ax = axs[1], alpha=0.5)
        plt.show()
        stat, p_value = PhysioNet.verify_distribution_differences(list(features[features['gender']=='M']['age'].values),
                                list(features[features['gender']=='F']['age'].values))

    def plot_frequencies(TimeSeries, features):
        void_index = features.gender.isna()
        known_features = features[~void_index]
        unknown_features = features[void_index]
        intervals = [[0, 0.5], [0.5, 1], [1,10], [10, 30], [30, 60]]
        labels = ['0-0.5', '0.5-1', '1-10', '10-30', '30-60']
        for i in range(5):
            temp = known_features[known_features['age'] >= intervals[i][0]]
            temp = temp[temp['age'] < intervals[i][1]]
            M_idex = temp[temp['gender']=='M']['Id']
            F_idex = temp[temp['gender']=='F']['Id']
            #print(M_idex)
            temp_dict = {}
            for j in M_idex:
                temp_dict[j] = TimeSeries[j]
            M_data = pd.DataFrame(temp_dict)
            temp_dict = {}
            for j in F_idex:
                temp_dict[j] = TimeSeries[j]
            F_data = pd.DataFrame(temp_dict)

            data = list(M_data.apply(np.mean, axis=1).values)
            data = data - np.mean(data)
            M_frequency, M_spectrum = sa.fourier_discreet_transform(data=data,
                                                                    sample_rate=1,
                                                                    duration=len(data))
            data = list(F_data.apply(np.mean, axis=1).values)
            data = data - np.mean(data)
            F_frequency, F_spectrum = sa.fourier_discreet_transform(data=data,
                                                                    sample_rate=1,
                                                                    duration=len(data))
            _, axs = plt.subplots(2,2)
            aux = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                                top=0.8, wspace=0.7, hspace=0.7)
            aux[0].plot(list(M_data.apply(np.mean, axis=1).values)[:500], c='black')
            aux[0].set(xlabel='', ylabel='measurements',
                            title=f'population {labels[i]} fragment\nMales') 
            
            aux[1].hist(M_spectrum[:200], color='red', bins=100, density=True)
            aux[1].set(xlabel='', ylabel='relative\namplitude',
                            title=f'population {labels[i]} spectrum\nMales') 
            
            aux[2].plot(list(F_data.apply(np.mean, axis=1).values)[:500], c='black')
            aux[2].set(xlabel='time', ylabel='measurements',
                            title=f'population {labels[i]} fragment\nFemales') 
            
            aux[3].hist(F_spectrum[:200], color='red', bins=100, density=True)
            aux[3].set(xlabel='frequency', ylabel='relative\namplitude',
                            title=f'population {labels[i]} spectrum\nFemales') 
            plt.show()



            gender = list(np.repeat('M', len(M_spectrum)))\
                     + list(np.repeat('F', len(F_spectrum)))
            spectrum = list(M_spectrum) + list(F_spectrum)
            avg = pd.DataFrame({'gender': gender, 'spec': spectrum})

            spec = avg['spec'].values
            
            df_spec = pd.DataFrame()
            df_spec['Males'] = np.percentile(list(M_spectrum), range(100))
            df_spec['Females'] = np.percentile(list(F_spectrum), range(100))
            

            fig, axs = plt.subplots(1,2)
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                                top=0.8, wspace=0.5, hspace=0.5)
            axs[0].scatter(x='Females', y='Males', data=df_spec,
                           marker='.', c='black', s=5, label='')
            sns.lineplot(x='Females', y='Females', data=df_spec, color='r',
                         ax=axs[0])
            axs[0].set(xlabel='Quantile of frequency\n spectrum, Females',
                       ylabel='Quantile of frequency spectrum, Males',
                       title=f"QQ plot of the\n population {labels[i]}")
            
            sns.histplot(data=avg, x='spec', hue='gender', bins=len(avg),
                         stat="density", element="step", fill=False,
                         cumulative=True, common_norm=False, ax=axs[1])
            axs[1].set(xlabel='', ylabel='Cumulative distribution',
                       title=f"Cumulative distribution of the\n frequencies\
 (population {labels[i]})")
            axs[1].set_xlim([0, 30000])
            axs[1].set_xticks([])
            plt.show()

            stat, p_value = PhysioNet.verify_distribution_differences(list(M_spectrum),
                                                                      list(F_spectrum))
            
TimeSeries, features = PhysioNet.load_data()
PhysioNet.age_gender_exploration()
PhysioNet.plot_frequencies(TimeSeries, features)
# predict the gender of the other distributions