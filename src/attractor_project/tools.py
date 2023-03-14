""" Main tools used in this package. Here one will find tools
related to build time series from known diferential equations
or iterated maps as well as spectral decomposition tools
and non-linear dynamics tools"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from watermark import watermark
from scipy import stats
from sklearn import metrics
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from selenium import webdriver
import csv
import datapackage


class parametric_diferential_equations():
    """ Set of diferential equations in parametric form which
    we use in the main code"""
    def pendulum_ode(x, y, b=0.25, c=5.0):
        """ Pendulum ODE

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
        An array containing the parametric equation of the ODE
        """
        return np.array([y, -b*y - c*np.sin(x)])

    def rossler_ode(x, y, z, a=0.15, b=0.2, c=10.0):
        """
        Rossler attractor ODE 

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
        An array containing the parametric equation of the ODE
        """
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def lorenz_ode(x, y, z, sigma=10.0, beta=8/3.0, rho=28.0):
        """
        Lorenz attractor ODE
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
        An array containing the parametric equation of the ODE
        """
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    
