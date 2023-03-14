import pytest
import sys
sys.path.insert(0, '/home/ricardo/Desktop/AttractorProject/attractor_project')
from src.attractor_project.tools import parametric_diferential_equations as pde


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
    

#def test_generate_series_from_ODE():
#    assert True
