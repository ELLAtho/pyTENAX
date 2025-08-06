# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:44:54 2025

@author: ellar
"""

import numpy as np
import pandas as pd
import xarray as xr
import time

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from scipy.stats import norm
from scipy.optimize import minimize
from scipy import fft


from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob



class STENAX():
    """
    A class used to represent the temperature model in TENAX using sine generators
    
    
    Attributes:
    -----------
    return_period : list
        Return periods [y]
    durations : list
        Duration of interest [min]; same as precipitation input in this example.
    beta : float
        Shape parameter of the Generalized Normal for g(T)
    temp_time_hour : int (negative)
        Time window to compute T [h]
    alpha : float
        Significance level for the dependence of the shape on T [-]
        alpha=0 --> dependence of shape on T is always allowed
        alpha=1 --> dependence of shape on T is never allowed
        alpha   --> dependence of shape on T depends on stat. significance at the alpha-level
    n_monte_carlo : int
        Number of elements in the MC samples [-]
    tolerance : float
        Max fraction of missing data in one year [-]
    min_ev_dur : int
        Minimum event duration [min]
    separation : int
        Separation time between idependent storms [min]
    left_censoring : list
        Left-censoring threshold [percentile]; see Marra et al. 2023 (https://doi.org/10.1016/j.advwatres.2023.104388)
    niter_smev : int
        Number of iterations for uncertainty for the SMEV model [-]
    niter_tnx :int 
        Number of iterations for uncertainty for the TENAX model [-]; in the paper we used 1e3
    temp_res_monte_carlo  : float
        Resolution in T for the MC samples [-]
    temp_delta : int
        Range in T of MC samples [-]; explores temperatures up to Tdelt degrees higher and lower of the observed ones
    init_param_guess : list
        Initial values of Weibull parameters for fminsearch [-]
    
    Methods:
    --------
    __init__(self, return_period, durations, beta=4, temp_time_hour, alpha, 
             n_monte_carlo, tolerance, min_ev_dur, separation, left_censoring, 
             niter_smev, niter_tnx,  temp_res_monte_carlo , temp_delta, init_param_guess):
        Initializes the TENAX class with the provided parameters.
    
    
    """
    def __init__(self, 
                 
                 # the variables for mu
                 A = 13, #average yearly temperature
                 B = 5, # range of temperature going up and down
                 p_mu = 365.25/(2*np.pi), #period (aka a year)
                 shift = 180, # shift of starting point
                 daysize = 3,#if a daily contribution is included

                 # the variables for sigma
                 var = 2, #variance of the normal distribution around
                 delta = 0.5, #dependence on the day
                 ave_loc = 0, #equivalent to shift, where the changes start in the year
                 p_sigma = 365.25/(2*np.pi), #period (aka a year)
                 
                 # the same as TENAX stuff
                 n_monte_carlo = int(2e4),): 
        """
        Initialize the STENAX model with the specified parameters.
            
        """

        self.A = A
        self.B = B
        self.p_mu = p_mu
        self.shift = shift
        self.daysize = daysize
        
        self.var = var
        self.delta = delta
        self.ave_loc = ave_loc
        self.p_sigma = p_sigma
        
        self.n_monte_carlo = n_monte_carlo
        
        
    def __str__(self):
        return "Welcome in the jugnle, this is THe Object of STENAX class"  
## maybe i actually don't need a class...


# definition of the function for mu in the temperature distribution over the year
def yearly_mu(x, A, B, shift = 0, p = 365.25/(2*np.pi), daysize = 0, daylength = 1/(2*np.pi)):
    return A + B*np.sin((x + shift)/p) + daysize*np.sin(x/daylength)

def yearly_sigma(x, var, delta, ave_loc, p = 365.25/(2*np.pi)):
    return (1 + delta * np.sin(x/p + ave_loc/p))*var

def storm_filter(mu, sigma, x = np.arange(0,365)): #basically removes a chunk of days, following a normal distribution centered at day mu and spread by day sigma
    pdf = norm.pdf(x, loc = mu, scale = sigma)
    weights = (1-pdf)/len(x)
    return weights


def gen_sine_temperature_pdf(eT, A, B, var, delta, ave_loc, x= np.arange(0,365), p_mu = 365.25/(2*np.pi),  p_sigma = 365.25/(2*np.pi), daysize = 0, daylength = 1/(2*np.pi)):
    mu = yearly_mu(x, A, B, p = p_mu, daysize = daysize, daylength = daylength)
    sigma = yearly_sigma(x, var, delta, ave_loc, p = p_sigma)#*np.sqrt(2)
    
    norms = [norm.pdf(eT, loc = mu[i], scale = sigma[i])/len(x) for i in x] #TODO: here you can put weighting as a storm filter
    pdf = sum(norms) 
    return pdf



def sine_temperature_loglik_day_incl(theta, T, p = 365.25/(2*np.pi), daylength = 1/(2*np.pi)):
   
    A, B, var, delta = theta[0], theta[1], theta[2], theta[3] 
    
    ave_loc, daysize =theta[4], theta[5]
    
    pdf = gen_sine_temperature_pdf(T, A, B, var, delta, ave_loc, x= np.arange(0,365), p = p, daysize = daysize, daylength = daylength)
    
    return sum(np.log(pdf + 1e-10))

def sine_temperature_loglik(theta, T, p_mu = 365.25/(2*np.pi), p_sigma = 365.25/(2*np.pi)):
   
    A, B, var, delta, ave_loc = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    
    pdf = gen_sine_temperature_pdf(T, A, B, var, delta, ave_loc, x= np.arange(0,365), p_mu = p_mu, p_sigma = p_sigma)
    
    return sum(np.log(pdf + 1e-10))

def sine_temperature_model(T_obs, init_params = [13, 4, 3, 0.5, 90, 0], day_incl = False):
    if day_incl:
        phat = minimize(lambda theta: -sine_temperature_loglik_day_incl(theta, T_obs),
                        init_params,
                        method = 'Nelder-Mead')
        param_names = ['A', 'B', 'var', 'delta', 'ave_loc', 'daysize']
    else:
        init_params = init_params[0:-1]
        phat = minimize(lambda theta: -sine_temperature_loglik(theta, T_obs),
                        init_params,
                        method = 'Nelder-Mead')
        param_names = ['A', 'B', 'var', 'delta', 'ave_loc']
        
        
    return dict(zip(param_names, phat.x))



################################################################################
# functions to fit the two seperately
def datetime_series_to_array(series): #series = oe.oe_time
    dates = pd.to_datetime(series).dt.date
    start_date = dates[0]
    diffs = np.array([(day - start_date).days for day in dates])
    return diffs
    

# fits a sine wave to the observed T by minimizing the residuals
def yearly_mu_fit(x,T_obs,init_params = [13, 5, 0, 365.25/(2*np.pi)]): # x in days as integer array
    
    def residuals(theta, x, T_obs):
        mu_sim = yearly_mu(x, A = theta[0], B = theta[1], shift = theta[2], p = theta[3])
        
        diffs = mu_sim - T_obs
        
        return np.nansum(diffs**2)
    
    phat = minimize(lambda theta: residuals(theta, x, T_obs),
                    init_params,
                    method = 'L-BFGS-B')
    
    
    param_names = ['A', 'B', 'shift', 'p_mu']
    return dict(zip(param_names, phat.x))


def yearly_sigma_fit(std_obs_cycle, init_params = [5, 0.5, 100, 365.25/(2*np.pi)]): # std_obs_cycle is a pd.Series
    
    x = std_obs_cycle.index.astype("int")
    
    def residuals(theta, x, obs):
        sigma_sim = yearly_sigma(x,theta[0],theta[1],theta[2], p = theta[3])
        diffs = sigma_sim - obs
        
        return np.nansum(diffs**2)
    
    
    phat = minimize(lambda theta: residuals(theta, x, std_obs_cycle.to_numpy()),
                    init_params,
                    method = 'L-BFGS-B')
    
    param_names = ['var', 'delta', 'ave_loc', 'p_sigma']
    return dict(zip(param_names, phat.x))

# functions as francesco said
def sine_temperature_loglik_v2(theta, x, T):
   
    A, B, shift, p_mu = theta[0], theta[1], theta[2], theta[3]
    var, delta, ave_loc, p_sigma = theta[4], theta[5], theta[6], theta[7]
    
    mu = yearly_mu(x, A, B, shift = shift, p = p_mu)
    sigma = yearly_sigma(x, var, delta, ave_loc, p = p_sigma)
    
    pdf = norm.pdf(T, mu, sigma)
    
    return sum(np.log(pdf + 1e-10))


def sine_temperature_model_v2(x, T_obs, 
                              init_params = [13, 4, 0, 365.25/(2*np.pi), 3, 0.5, 90, 365.25/(2*np.pi)],
                              bounds = [(-50, 50), (0, 30), (None, None), (50/(2*np.pi), 366/(2*np.pi)),  # A, B, shift, p_mu
          (1e-3, 30), (0, 1), (0, 365.25/(2*np.pi)), (50/(2*np.pi), 366/(2*np.pi))] ):
    
    phat = minimize(lambda theta: -sine_temperature_loglik_v2(theta, x, T_obs),
                    init_params,
                    method = 'L-BFGS-B',
                    bounds = bounds)
    
    param_names = ['A', 'B', 'shift', 'p_mu', 'var', 'delta', 'ave_loc', 'p_sigma']
    return dict(zip(param_names, phat.x))



#################################################################################
# fourier transformsss
def FT_temp_model(days, T_obs):
    mean = np.mean(T_obs)
    resids = T_obs - mean
    
    N = len(T_obs)
    T = 1 # one day... for now
    yf = fft.fft(resids)
    xf = fft.fftfreq(N, T)[:N//2]
    
    Fyy = abs(yf)
    
    guess_freq = abs(xf[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(resids) * 2.**0.5
    
    #set the frequency to what we xpect it to be... so issues with resolution are reduced
    if (1/(guess_freq*2*np.pi) < 370/(2*np.pi)) & (1/(guess_freq*2*np.pi) > 350/(2*np.pi)):
        p_mu = 365.25/(2*np.pi)
    elif (1/(guess_freq*2*np.pi) < 370/(4*np.pi)) & (1/(guess_freq*2*np.pi) > 350/(4*np.pi)):
        p_mu = 365.25/(4*np.pi)
    else:
        p_mu = 1/(guess_freq*2*np.pi)
        print("WARNING: p not a factor 1 or 2 of the year length")
    
    
    phat_list = [mean, guess_amp, p_mu, 1/(guess_freq*2*np.pi)]
    
    param_names = ['A', 'B', 'p_mu', 'p_mu_actual_calculated']
    
    phat = dict(zip(param_names, phat_list))
    return phat



#################################################################################

def generate_temperature_mc(A, B, var, delta, ave_loc, Ts, x= np.arange(0,365), p_mu = 365.25/(2*np.pi),  p_sigma = 365.25/(2*np.pi), n_monte_carlo = int(2e4)):
    """
    Generate monte carlo temperature values using the distribution defined above.

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.
    ave_loc : TYPE
        DESCRIPTION.
    Ts : TYPE
        DESCRIPTION.
    x : TYPE, optional
        DESCRIPTION. The default is np.arange(0,365).
    p_mu : TYPE, optional
        DESCRIPTION. The default is 365.25/(2*np.pi).
    p_sigma : TYPE, optional
        DESCRIPTION. The default is 365.25/(2*np.pi).
    n_monte_carlo : TYPE, optional
        DESCRIPTION. The default is int(2e4).

    Returns
    -------
    T_mc : TYPE
        DESCRIPTION.

    """
    
    pdf_values = gen_sine_temperature_pdf(Ts, A, B, var, delta, ave_loc, x = x, p_mu = p_mu ,  p_sigma = p_sigma)
    df = np.vstack([pdf_values, Ts])
    
    T_mc = randdf(n_monte_carlo, df, 'pdf').T 
    
    return T_mc

#TODO: need to do the above but in a sinusoidal way so can recalculate based on the std and mean seperately


    
    


