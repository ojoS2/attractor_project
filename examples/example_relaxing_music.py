import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')

from src.attractor_project.tools import spectral_analysis as sa
from src.attractor_project.tools import non_linear_methods as nlm

def import_a_file_from_kaggle_as_dataframe(data_identifyer,
                                           temp_directory, file_name,
                                           zip_name):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(data_identifyer, path=temp_directory)
    try:
        with zipfile.ZipFile(temp_directory + "/" + zip_name,
                             mode="r") as archive:
            archive.extract(file_name,
                            path=temp_directory + "/")
    except zipfile.BadZipFile as error:
        print(error)
    df = pd.read_csv(temp_directory + "/" + file_name)
    os.system('rm ' + temp_directory + "/"  + file_name)
    os.system('rm ' + temp_directory + "/"  + zip_name)
    
    return df

def plot_lorentz_map(files_list, name=None):
    fig, axs = plt.subplots(5, 2, sharex=True, sharey=True,
                            figsize=(15, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.5, hspace=0.8)
    aux = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0],
        axs[2, 1], axs[3, 0], axs[3, 1], axs[4, 0], axs[4, 1]]
    for i in range(len(files_list)):
        signal = list(df[files_list[i]].values)
        y_0, z_0 = nlm.lorentz_map(signal, lag=1, plot=False)
        aux[i].scatter(y_0, z_0, marker='.', c='black', s=0.1, alpha=0.3)
        aux[i].plot(y_0, y_0, c='red', lw=0.5)
        title = files_list[i].replace("_", " ")
        title = title.replace("-", "\n")
        aux[i].set_title(title)
    if name is not None:
        plt.savefig(name)
    plt.show()

def plot_attractors(name=None):
    fig, axs = plt.subplots(5, 2, figsize=(10,10),
                            subplot_kw=dict(projection='3d'))#,
                            #sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.1, hspace=0.1)
    aux = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0],
        axs[2, 1], axs[3, 0], axs[3, 1], axs[4, 0], axs[4, 1]]
    titles=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for i in range(len(files_list)):
        signal = list(df[files_list[i]].values)
        signal = signal[:100000]
        x, y, z, tau = nlm.attractor_reconstructor(signal, tau_to_use=1,
                                                plot=False)
        aux[i].plot(x, y, z, c = 'black', lw=0.1)
        title = files_list[i].replace("_", " ")
        title = title.replace("-", ",")
        aux[i].set_title(title + f' lagg={tau}')
    if name is not None:
        plt.savefig(name)
    plt.show()


data_identifyer = 'ricardosimo/relaxing-music-spectrum'
temp_directory = "examples"
file_name = "amplitudes.csv"
zip_name = "relaxing-music-spectrum.zip"

df = import_a_file_from_kaggle_as_dataframe(data_identifyer,
                                           temp_directory,
                                           file_name, zip_name)
print(df.head())
files_list = ['Adele-Someone_Like_You',
              'Airstream-Electra',
              'All_Saints-Pure_Shores',
              'Coldplay-Strawberry_Swing',
              'Dj_Shah-Mellomaniac',
              'Enya-Watermark',
              'Barcelona-Please_Don_t_Go',
              'Mozart-Canzonetta_Sull_aria',
              'Weightless']
plot_lorentz_map(files_list, name='examples/figures/lorentz_map_relaxing_music.png')
plot_attractors(name='examples/figures/rec_attractors_relaxing_music.png')