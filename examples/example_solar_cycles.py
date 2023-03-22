import os
import zipfile
import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
from datetime import timedelta
from tensorflow import keras
from src.attractor_project.tools import parametric_diferential_equations as pde
from src.attractor_project.tools import iterated_maps as im
from src.attractor_project.tools import time_series_generators as tsg
from src.attractor_project.tools import spectral_analysis as sa
from src.attractor_project.tools import non_linear_methods as nlm


def import_a_file_from_kaggle_as_dataframe(data_identifyer,
                                         temp_directory, file_name):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(data_identifyer, path=temp_directory)
    try:
        with zipfile.ZipFile(temp_directory + "/" + file_name,
                             mode="r") as archive:
            archive.extract(file_name,
                            path=temp_directory + "/")
    except zipfile.BadZipFile as error:
        print(error)
    df = pd.read_csv(temp_directory + "/" + file_name)
    os.system('rm -R ' + temp_directory)
    return df

def column_datetime_to_float(df, initial_value, conversion_unity,
                             date_column):
    aux = lambda date: (date - initial_value)\
                        /pd.Timedelta(1, conversion_unity)
    df['float_time'] = df[date_column].apply(aux)
    return df

def plot_time_interval_occurencies(df, column):
    aux1 = list(df[column].values)
    aux2 = np.roll(aux1, 1)
    temp = [i - j for i, j in zip(aux1, aux2)]
    temp = temp[1:-1]
    plt.hist(temp, density=True, bins=len(list(set(temp))))
    plt.xticks(list(set(temp)), [str(i) for i in list(set(temp))], rotation=90)
    plt.xlabel('occurrencies')
    plt.ylabel('density')
    plt.title('distribution of occurencies')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                        top=0.8, wspace=0.7, hspace=0.7)
    plt.show()

def normalize_time_intervals(df, date_column, value_column, period=31):
    init = df.loc[0, date_column]
    end = df.loc[len(df) - 1, date_column]
    aux = init
    date = []
    value = []
    while aux < end:
        bot = aux
        top = aux + timedelta(days=period)
        value.append(df[df[date_column].\
                    between(bot, top)]\
                    [value_column].mean())
        date.append(top)
        aux = top
    return pd.DataFrame({'date': date, 'values': value})

def plot_series(time, series, format="-", start=0, end=None,
                label=None, show=True):
    myplot = plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time (Months)")
    plt.ylabel("Average Monthly Sunpots")
    if label != None:
        plt.legend()
    plt.grid(True)
    if show:
        plt.show()

def plot_windows(series, time, window_size):
    s = np.array(series)
    t = np.array(time)
    def plot_split(array, window_size):
        number_windows = len(array) // window_size + 1
        return np.array_split(array, number_windows, axis=0)
    plt.figure(figsize=(15, 9))
    plt.title("Windows")
    for series_window,time_window in zip(plot_split(s, window_size), plot_split(t, window_size)):
        plot_series(time_window, series_window, show=False)
    plt.show()

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
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    #checkpoint_path = "examples/solar_cycles_data/cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                             save_weights_only=True,
    #                                             verbose=1)

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

def find_best(series_list, test_values, window_size):

    test = test_values[window_size-1:-window_size]
    #plt.plot(test, label='validation set', linestyle='--')
    predict = np.roll(series_list[:, 0, 0], 0)[window_size:]
    aux = tf.keras.metrics.mean_absolute_error(test, predict).numpy()
    index = 0
    for i in range(1, 64):
        predict = np.roll(series_list[:, i, 0], i)[window_size:]
        print('Mean Absolute Error (MAE): ', i,
              tf.keras.metrics.mean_absolute_error(test, predict).numpy())   
        temp = tf.keras.metrics.mean_absolute_error(test, predict).numpy()
        if temp < aux:
            aux = temp
            index = i
    predict = np.roll(series_list[:, index, 0], index)[window_size:]
    #plt.plot(predict)
    #plt.show()
    return predict

def load_single_file_from_zip(temp_directory, file_name):
    try:
        with zipfile.ZipFile(temp_directory + "/" + file_name,
                                mode="r") as archive:
            archive.extract(file_name,
                            path=temp_directory + "/")
    except zipfile.BadZipFile as errors:
        print(errors)
    df = pd.read_csv(temp_directory + "/" + file_name)
    return df 

def sort_and_convert_datetime_columns(df, column_to_sort='Date', new_column_name='float_time'):
    df[column_to_sort] = pd.to_datetime(df[column_to_sort])
    df.sort_values(by=column_to_sort, ascending=True, inplace=True)
    aux=lambda date: (date - df[column_to_sort][0])/pd.Timedelta(1, 'd')
    df[new_column_name] = (df[column_to_sort].apply(aux)).astype('float')
    return df

def plot_frequency_spectrum(series, period, duration, plot=False):
    frequencies, amplitudes = sa.fourier_discreet_transform(data=series - np.mean(series),
                                                            sample_rate=1/period,
                                                            duration=duration)
    if plot:
        plt.plot(frequencies, amplitudes)
        plt.show()
    return frequencies, amplitudes

def spectral_analysis():
    time = list(df['float_time'].values)
    series = list(df['values'].values - np.mean(list(df['values'].values)))
    duration = (time[-1] - time[0])
    plot_series(time, series, format="-", start=0, end=None, label=None)
    frequencies, amplitudes = plot_frequency_spectrum(series=series,
                                                      period=period_to_normalize,
                                                      duration=duration,
                                                      plot=False)
    scale, p_value, corr = sa.best_scale(data=series, inf=0.001, sup=0.5, p_threshold=0.005,
                                         grafics=True)
    filtered_spectrum, filtered_signal = sa.filtered_signal(0.0, amplitudes)
    plt.plot(series)
    plt.plot(filtered_signal)
    plt.show()
    plt.plot(amplitudes)
    plt.plot(filtered_spectrum)
    plt.show()

def plot_dataspliting(time_train, values_train, 
                      time_test, values_test,
                      series, time, window_size):
    plt.figure(figsize=(15, 9))
    plt.title("Train/Test Split")
    plot_series(time_train, values_train, label="training set", show=False)
    plot_series(time_test, values_test, label="validation set", show=False)
    plt.show()
    plot_windows(series, time, window_size)

def experiments(model, values_test, window_size):
    list_of_best_fit = []
    for i in range(10):
        rnn_forecast = model_forecast(model,
                                      np.array(values_test)[..., np.newaxis],
                                      window_size)
        list_of_best_fit.append(find_best(series_list=rnn_forecast, test_values=values_test, window_size=window_size))
    return list_of_best_fit

data_identifyer = 'robervalt/sunspots'
temp_directory = "examples/solar_cycles_data"
checkpoint_directory = "examples/models_checkpoint"
file_name = "Sunspots.csv"
column_to_sort='Date'
period_to_normalize = 31
df = import_a_file_from_kaggle_as_dataframe(data_identifyer,
                                            temp_directory, file_name)

df = column_datetime_to_float(df,
                              initial_value = df['Date'][0],
                              conversion_unity='d',
                              date_column='Date')
df = load_single_file_from_zip(temp_directory, file_name)
df = sort_and_convert_datetime_columns(df, column_to_sort='Date')
plot_time_interval_occurencies(df, column='float_time')
# the intervals need normalization
df = normalize_time_intervals(df,
                              date_column='Date',
                              value_column="Monthly Mean Total Sunspot Number",
                              period=period_to_normalize)
df = column_datetime_to_float(df,
                              initial_value = df['date'][0],
                              conversion_unity='d',
                              date_column='date')
plot_time_interval_occurencies(df, column='float_time')
time = list(df['float_time'].values)
series = list(df['values'].values)
#spliting the data
split = int(0.75*len(time))
time_train, values_train = np.array(time[:split]), np.array(series[:split])
time_test, values_test = np.array(time[split:]), np.array(series[split:])
# the window size is important for training results
window_size = 100
batch_size = 120
shuffle_buffer_size = 1000
plot_dataspliting(time_train, values_train, 
                  time_test, values_test,
                  series, time, window_size)


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 100
batch_size = 256
train_set = windowed_dataset(values_train, window_size,
                             batch_size, shuffle_buffer_size)
model, history = recursive_neural_network_lr_optimization(train_set)
model = plot_optimized_model_progress(train_set, lr=1e-5)
print('Model mounted')
#load_model()


rnn_forecast = model_forecast(model,
                              np.array(values_test)[..., np.newaxis],
                              window_size)

print(rnn_forecast.shape)


list_of_best_fit = experiments(model, values_test, window_size)
for i in list_of_best_fit:
    print(i)
    plt.plot(i)
plt.plot(values_test[window_size-1:-window_size], label='validation set', linestyle='--')
plt.show()

print('Mean Absolute Error (MAE): ',tf.keras.metrics.mean_absolute_error(values_test, rnn_forecast).numpy())
