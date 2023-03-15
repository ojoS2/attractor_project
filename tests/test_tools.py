import pytest
import sys
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
from src.attractor_project.tools import parametric_diferential_equations as pde
from src.attractor_project.tools import iterated_maps as im
from src.attractor_project.tools import time_series_generators as tsg

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