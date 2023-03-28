import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
from src.attractor_project.tools import parametric_diferential_equations as pde
from src.attractor_project.tools import iterated_maps as im
from src.attractor_project.tools import time_series_generators as tsg
from src.attractor_project.tools import spectral_analysis as sa

def test_pendulum_ode():
    assert abs(pde.pendulum_ode(x=0, y=0, b=0.25, c=5.0)[0]) < 0.000001
    assert abs(pde.pendulum_ode(x=0, y=0, b=0.25, c=5.0)[1]) < 0.000001
    assert abs(pde.pendulum_ode(x=10, y=10, b=0.0, c=0.0)[0]) == 10
    assert abs(pde.pendulum_ode(x=10, y=10, b=0.0, c=0.0)[1]) < 0.000001

def test_rossler_ode():
    assert abs(pde.rossler_ode(x=0, y=0, z=0, a=0.15, b=0.2, c=10.0)[0]) < 0.000001
    assert abs(pde.rossler_ode(x=0, y=0, z=0, a=0.15, b=0.2, c=10.0)[1]) < 0.000001
    assert abs(pde.rossler_ode(x=0, y=0, z=0, a=0.15, b=0.2, c=10.0)[2]) - 0.2 < 0.000001
    assert pde.rossler_ode(x=10, y=10, z=10, a=0., b=0., c=0.0)[0] == -20
    assert pde.rossler_ode(x=10, y=10, z=10, a=0., b=0., c=0.0)[1] == 10
    assert pde.rossler_ode(x=10, y=10, z=10, a=0., b=0., c=0.0)[2] == 100

def test_lorenz_ode():
    assert abs(pde.lorenz_ode(x=0, y=0, z=0, sigma=10.0, beta=8/3.0, rho=28.0)[0]) < 0.000001
    assert abs(pde.lorenz_ode(x=0, y=0, z=0, sigma=10.0, beta=8/3.0, rho=28.0)[1]) < 0.000001
    assert abs(pde.lorenz_ode(x=0, y=0, z=0, sigma=10.0, beta=8/3.0, rho=28.0)[2]) < 0.000001
    assert abs(pde.lorenz_ode(x=10, y=10, z=10, sigma=.0, beta=.0, rho=.0)[0]) < 0.000001
    assert pde.lorenz_ode(x=10, y=10, z=10, sigma=.0, beta=.0, rho=.0)[1] == -110
    assert pde.lorenz_ode(x=10, y=10, z=10, sigma=.0, beta=.0, rho=.0)[2] == 100

def test_quadratic_map():
    result_1 = [4, 16, 256, 65536]
    result_2 = [16, 65536, 18446744073709551616,
                115792089237316195423570985008687907853269984665640564039457584007913129639936]
    aux_1, aux_2 = 2, 2
    for i in range(4):
        aux_1 = im.quadratic_map(x=aux_1, A=1, B=0, C=0, n=1)
        aux_2 = im.quadratic_map(x=aux_2, A=1, B=0, C=0, n=2)
        assert aux_1 == result_1[i]
        assert aux_2 == result_2[i]

def test_generate_series_from_ODE():
    results = [14.94311394, 21.22697266, 26.23106149]
    data = tsg.generate_series_from_ODE(data_length=10,
                                        ode=pde.lorenz_ode,
                                        state=[10, 10, 10],
                                        parameters=[],
                                        dt=0.01, transient=0)
    for i in range(3):
        assert abs(data[-1, i] - results[i]) < 0.00001

def test_generate_series_from_iterated_maps():
    results_1 = [-0.6174310703743311, -1.118778873336408, -0.24833383257611774]
    results_3 = [0.26579675123891344, -1.205099533626746, 0.7431692688644809]
    results_13 = [0.4733090581565429, -1.2211994279038247, 0.729103837057933]
    data_1 = tsg.generate_series_from_iterated_maps(data_length=100,
                                                  iter_map=im.quadratic_map,
                                                  initial_state=[0], transient=0,
                                                  parameters=[1, 0, -1.5, 1])
    data_3 = tsg.generate_series_from_iterated_maps(data_length=100,
                                                  iter_map=im.quadratic_map,
                                                  initial_state=[0], transient=0,
                                                  parameters=[1, 0, -1.5, 3])
    data_13 = tsg.generate_series_from_iterated_maps(data_length=100,
                                                  iter_map=im.quadratic_map,
                                                  initial_state=[0], transient=0,
                                                  parameters=[1, 0, -1.5, 13])
    for i in range(3):
        assert abs(data_1[98+i][0] - results_1[i]) < 0.00001
        assert abs(data_3[98+i][0] - results_3[i]) < 0.00001
        assert abs(data_13[98+i][0] - results_13[i]) < 0.00001

def fourier_discreet_transform():
    SAMPLE_RATE = 1000
    DURATION = 10
    frequencies = [1, 5, 7, 13, 21]
    amplitudes = [1, 2, 3, 4, 5]
    def generate_sine_wave(amp, freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        y = amp*np.sin((2 * np.pi) * frequencies)
        return x, y
    def generate_mixed_signal(f, A, SAMPLE_RATE, DURATION):
        y = []
        x, aux = generate_sine_wave(amplitudes[0],
                                    frequencies[0],
                                    SAMPLE_RATE, DURATION)
        y.append(aux)
        for i in range(1, 5):
            _, aux = generate_sine_wave(amplitudes[i],
                                        frequencies[i],
                                        SAMPLE_RATE, DURATION)
            y.append(aux)
        mix = y[0] + y[1] + y[2] + y[3] + y[4]
        return x, mix
    def get_maximuns(x, y):
        z = []
        w = []
        for index, value in enumerate(y):
            if value > 100:
                z.append(x[index])
                w.append(value)
        return z, w

    x, mix = generate_mixed_signal(f=frequencies, A=amplitudes,
                                   SAMPLE_RATE=SAMPLE_RATE,
                                   DURATION=DURATION)
    x, y = sa.fourier_discreet_transform(data=mix,
                                    sample_rate=SAMPLE_RATE,
                                    duration=DURATION)
    z, w = get_maximuns(x, y)
    for i in range(len(z)):
        assert abs(z[i] - frequencies[i]) < 0.0001
    for i in range(1, len(z)):
        assert w[i] > w[i-1]

def test_fft_filter():
    SAMPLE_RATE = 1000
    DURATION = 10
    frequencies = [1, 5, 7, 13, 21]
    amplitudes = [1, 2, 3, 4, 5]
    def generate_sine_wave(amp, freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        y = amp*np.sin((2 * np.pi) * frequencies)
        return x, y
    def generate_mixed_signal(f, A, SAMPLE_RATE, DURATION):
        y = []
        x, aux = generate_sine_wave(amplitudes[0],
                                    frequencies[0],
                                    SAMPLE_RATE, DURATION)
        y.append(aux)
        for i in range(1, 5):
            _, aux = generate_sine_wave(amplitudes[i],
                                        frequencies[i],
                                        SAMPLE_RATE, DURATION)
            y.append(aux)
        mix = y[0] + y[1] + y[2] + y[3] + y[4]
        return x, mix
    def get_maximuns(x, y):
        z = []
        w = []
        for index, value in enumerate(y):
            if value > 100:
                z.append(x[index])
                w.append(value)
        return z, w
    x, mix = generate_mixed_signal(f=frequencies, A=amplitudes,
                                   SAMPLE_RATE=SAMPLE_RATE,
                                   DURATION=DURATION)
    x, y = sa.fourier_discreet_transform(data=mix,
                                    sample_rate=SAMPLE_RATE,
                                    duration=DURATION)
    filter = [0.1, 0.25, 0.45, 0.65, 0.85]
    for i in filter:
        new_y = sa.fft_filter(percentual=i, spectrum=y)
        z, w = get_maximuns(x, new_y)
        for j in range(len(z)):
            assert abs(z[j] - frequencies[5 - len(z) + j]) < 0.0001

def test_filtered_signal():
    SAMPLE_RATE = 1000
    DURATION = 10
    frequencies = [1, 5, 7, 13, 21]
    amplitudes = [1, 2, 3, 4, 5]
    def generate_sine_wave(amp, freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        y = amp*np.sin((2 * np.pi) * frequencies)
        return x, y
    def generate_mixed_signal(f, A, SAMPLE_RATE, DURATION):
        y = []
        x, aux = generate_sine_wave(amplitudes[0],
                                    frequencies[0],
                                    SAMPLE_RATE, DURATION)
        y.append(aux)
        for i in range(1, 5):
            _, aux = generate_sine_wave(amplitudes[i],
                                        frequencies[i],
                                        SAMPLE_RATE, DURATION)
            y.append(aux)
        mix = y[0] + y[1] + y[2] + y[3] + y[4]
        return x, mix
    def get_maximuns(x, y):
        z = []
        w = []
        for index, value in enumerate(y):
            if value > 100:
                z.append(x[index])
                w.append(value)
        return z, w
    x, mix = generate_mixed_signal(f=frequencies, A=amplitudes,
                                   SAMPLE_RATE=SAMPLE_RATE,
                                   DURATION=DURATION)
    x, y = sa.fourier_discreet_transform(data=mix,
                                    sample_rate=SAMPLE_RATE,
                                    duration=DURATION)
    filter = [0.1, 0.25, 0.45, 0.65, 0.85]
    for i in filter:
        new_y, new_series = sa.filtered_signal(i, y)
        z, w = get_maximuns(x, new_y)
        _, new_new_y = sa.fourier_discreet_transform(data=new_series,
                                    sample_rate=SAMPLE_RATE,
                                    duration=DURATION)
        new_z, new_w = get_maximuns(x, new_new_y)
        for j in range(len(z)):
            assert abs(z[j] - new_z[j]) < 0.0001
        