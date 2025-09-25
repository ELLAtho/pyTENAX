# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 08:41:55 2025

@author: ellar
"""

from os.path import dirname, abspath, join
from os import getcwd
import sys
#run this fro src folder, otherwise it doesn't work
THIS_DIR = dirname(getcwd())
CODE_DIR = join(THIS_DIR, 'src')
RES_DIR =  join(THIS_DIR, 'res')
sys.path.append(CODE_DIR)
sys.path.append(RES_DIR)
sys.path.append('D:')
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


# %% Defining parameters and simulation functions
#### a simulation of all the temperatures during the year

### define the easy functions
A = 13 #average yearly temperature
B = 5 # range of temperature going up and down
p = 365.25/(2*np.pi) #period (aka a year)
shift = 180 # shift of starting point


var = 2 #variance of the normal distribution around
delta = 0.5 #dependence on the day
ave_loc = 0 #equivalent to shift, where the changes start in the year
daysize = 3

x = np.arange(0,365)

xhour = np.arange(0,365,1/24)

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


# %% Functions for fitting observations
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


# %% functions to fit the two seperately
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

def yearly_sigma_fit_rolling(oe_dataframe, window = 40, init_params = [5, 0.5, 100, 365.25/(2*np.pi)]): # std_obs_cycle is a pd.Series
    
    oe_dataframe["days_from_start"] = ((oe_dataframe.oe_time - oe_dataframe.oe_time[0]).astype("int")/(24*60*60*10e8)).round().astype("int")
    
    # calculate rolling std according to the day
    rolling_std = [np.nan]*len(oe_dataframe)
    
    for i in np.arange(0,len(oe_dataframe)):
        day = oe_dataframe.days_from_start.iloc[i]
        if day >= window-1:
            df_section = oe_dataframe[(oe_dataframe.days_from_start<=day)&(oe_dataframe.days_from_start>day-window)]
            rolling_std[i] = df_section["T"].std()
        else:
            pass
    
    x = oe_dataframe.days_from_start
    
    def residuals(theta, x, obs):
        sigma_sim = yearly_sigma(x,theta[0],theta[1],theta[2], p = theta[3])
        diffs = sigma_sim - obs
        
        return np.nansum(diffs**2)
    
    
    phat = minimize(lambda theta: residuals(theta, x, rolling_std),
                    init_params,
                    method = 'L-BFGS-B')
    
    param_names = ['var', 'delta', 'ave_loc', 'p_sigma']
    return dict(zip(param_names, phat.x)), rolling_std





  # EG
# plt.plot(eT_hist, hist, '--')
# plt.plot(eT, gen_sine_temperature_pdf(eT, phat_chack[0],
#                                       phat_chack[1], 
#                                       phat_sigma[0],
#                                       phat_sigma[1], 
#                                       phat_sigma[2] - phat_chack[2], 
#                                       p_mu = phat_chack[3],
#                                       p_sigma = phat_sigma[3],))  
    


# %%  functions as francesco said
def sine_temperature_loglik_v2(theta, x, T):
   
    A, B, shift, p_mu = theta[0], theta[1], theta[2], theta[3]
    var, delta, ave_loc, p_sigma = theta[4], theta[5], theta[6], theta[7]
    
    mu = yearly_mu(x, A, B, shift = shift, p = p_mu)
    sigma = yearly_sigma(x, var, delta, ave_loc, p = p_sigma)
    
    pdfs = np.log(norm.pdf(T, mu, sigma) + 1e-20)
    
    return sum(pdfs)


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



# %%  fourier transformsss
def FT_temp_model(T_obs):
    mean = np.mean(T_obs)
    resids = T_obs - mean
    
    N = len(T_obs)
    T = 1 # one day... for now
    yf = fft.fft(resids)
    xf = fft.fftfreq(N, T)[:N//2]
    
    Fyy = abs(yf)
    
    guess_freq = abs(xf[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(resids) * 2.**0.5
    
    
    c = np.real(yf[np.argmax(Fyy[1:])+1])
    s = -np.imag(yf[np.argmax(Fyy[1:])+1])
    angle = np.arctan2(s, c)
    
    #set the frequency to what we xpect it to be... so issues with resolution are reduced
    if (1/(guess_freq*2*np.pi) < 370/(2*np.pi)) & (1/(guess_freq*2*np.pi) > 350/(2*np.pi)):
        p_mu = 365.25/(2*np.pi)
    elif (1/(guess_freq*2*np.pi) < 370/(4*np.pi)) & (1/(guess_freq*2*np.pi) > 350/(4*np.pi)):
        p_mu = 365.25/(4*np.pi)
    else:
        p_mu = 1/(guess_freq*2*np.pi)
        print("WARNING: p not a factor 1 or 2 of the year length")
    
    
    phat_list = [mean, guess_amp, p_mu, 1/(guess_freq*2*np.pi), angle]
    
    param_names = ['A', 'B', 'p_mu', 'p_mu_actual_calculated','angle']
    
    phat = dict(zip(param_names, phat_list))
    return phat

def FT_std_model(T_series, window = 40):
    rolling_std = T_series.rolling(window).std()[window-1:]
    
    mean = np.mean(rolling_std)
    
    resids = rolling_std - mean
    
    N = len(rolling_std)
    T = 1 # one day... for now
    yf = fft.fft(resids)
    xf = fft.fftfreq(N, T)[:N//2]
    
    
    Fyy = abs(yf)
    
    guess_freq = abs(xf[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(resids) * 2.**0.5
    
    c = np.real(yf[np.argmax(Fyy[1:])+1])
    s = -np.imag(yf[np.argmax(Fyy[1:])+1])
    angle = np.arctan2(s, c)
    
    
    if (1/(guess_freq*2*np.pi) < 370/(2*np.pi)) & (1/(guess_freq*2*np.pi) > 350/(2*np.pi)):
        p_sigma = 365.25/(2*np.pi)
    elif (1/(guess_freq*2*np.pi) < 370/(4*np.pi)) & (1/(guess_freq*2*np.pi) > 350/(4*np.pi)):
        p_sigma = 365.25/(4*np.pi)
    else:
        p_sigma = 1/(guess_freq*2*np.pi)
        print("WARNING: p not a factor 1 or 2 of the year length")
    
    
    ave_loc = (-angle+np.pi/2) * p_sigma/(2*np.pi)
    
    phat_list = [mean, guess_amp/mean, p_sigma, 1/(guess_freq*2*np.pi), ave_loc, angle]
    
    param_names = ['var', 'delta', 'p_sigma', 'p_sigma_actual_calculated', 'ave_loc','angle']
    
    phat = dict(zip(param_names, phat_list))
    return phat
    

def FT_plot(T_series, window = 40): #TODO: the shift is still wrong... don't know if it is relatively correct
    
    phat_mu = FT_temp_model(T_series[window-1:].to_numpy())
    
    phat_sigma = FT_std_model(T_series, window = window)
    
    rolling_std = T_series.rolling(window).std()[window-1:]
    
    shift = (-phat_mu["angle"]+np.pi/2)*phat_mu["p_mu"]*2
    
    x = np.arange(len(rolling_std))
    
    
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,3,1)
    ax.plot(T_series[window-1:],label = "observations")
    
    ax.plot(T_series.index[window-1:],
            yearly_mu(x,phat_mu["A"],phat_mu["B"],
                      shift,
                      phat_mu["p_mu"]
                      ),label = "FT")
    
    ax.set_title("Temperature")
    plt.legend()
    
    ax = fig.add_subplot(1,3,2)
    ax.plot(rolling_std, label = "std")
    
    ax.plot(T_series.index[window-1:],
            yearly_sigma(x,phat_sigma["var"],phat_sigma["delta"],
                         phat_sigma["ave_loc"],phat_sigma["p_sigma"]))
    
    ax.set_title(f"Standard deviation, window = {window} days")
    plt.legend()
    
def FT_model(T_series, window = 40, plot_dist = True): 
    
    phat_mu = FT_temp_model(T_series[window-1:].to_numpy())
    
    phat_sigma = FT_std_model(T_series, window = window)
    
    del_angle = phat_sigma["angle"] - phat_mu["angle"]
    ave_loc_out = del_angle *365.25/(2*np.pi)
    
    if plot_dist == True:
        eT = np.arange(-12,40,0.2)
        eT_hist = np.arange(-20,40)
        eT_edges = np.concatenate([np.array([eT_hist[0]-(eT_hist[1]-eT_hist[0])/2]),(eT_hist + (eT_hist[1]-eT_hist[0])/2)]) #convert bin centres into bin edges
        hist, bin_edges = np.histogram(T_series.to_numpy(), bins=eT_edges, density=True)
        
        pdf = gen_sine_temperature_pdf(eT,phat_mu["A"],phat_mu["B"],phat_sigma["var"],phat_sigma["delta"],ave_loc_out)
        
        
        plt.plot(eT_hist, hist, "b--", label = "observations")
        plt.plot(eT, pdf, "r", label = "pdf")
        plt.show()
    else:
        pass
    
    return phat_mu, phat_sigma, ave_loc_out
    


# %% plot the generated distributions from parameters
A = 13 #average yearly temperature
B = 7 # range of temperature going up and down
p = 365.25/(2*np.pi) #period (aka a year)
shift = 0 # shift of starting point


var = 2 #variance of the normal distribution around
delta = 0.3 #dependence on the day
ave_loc = 180 #equivalent to shift, where the changes start in the year
daysize = 3

theta = [A, B, var, delta, ave_loc, 0]

plt.plot(xhour,yearly_mu(xhour, A, B, shift, daysize = daysize))
plt.fill_between(xhour, 
                 yearly_mu(xhour, A, B, shift, daysize = daysize) - yearly_sigma(xhour, var, delta, ave_loc),
                 yearly_mu(xhour, A, B, shift, daysize = daysize) + yearly_sigma(xhour, var, delta, ave_loc),
                 alpha = 0.5)
# plt.title(f"mu = {A} + {B}sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + {ave_loc})*2pi/365.25))*{var}")
plt.ylabel("temperature [C]")
plt.xlabel("day of year")
plt.xlim(0,365)
plt.show()


eT = np.arange(-12,40,0.2)
norms_day = [gen_norm_pdf(eT, yearly_mu(xhour[i], A, B, shift, daysize = daysize),
                      yearly_sigma(xhour[i], var, delta, ave_loc)*np.sqrt(2), 2) for i in range(365*24)]

norms = [gen_norm_pdf(eT, yearly_mu(x[i], A, B, shift),
                      yearly_sigma(x[i], var, delta, ave_loc)*np.sqrt(2), 2) for i in range(365)]

norms2 = [norm.pdf(eT,loc = yearly_mu(xhour[i], A, B, shift, daysize = daysize), scale = yearly_sigma(xhour[i], var, delta, ave_loc))for i in range(365*24)]

plt.plot(eT,sum(norms_day)/(365*24), label = "incl diurnal cycle")
plt.plot(eT,sum(norms)/(365), label = "daily ave")
# plt.plot(eT, gen_sine_temperature_pdf(eT, A, B, var, delta, ave_loc), "--")
# plt.plot(eT, sum(norms2)/(365*24), "--")
plt.title(f"mu = {A} + {B}sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + {ave_loc})*2pi/365.25))*{var}")
plt.xlim(-3,30)
plt.xlabel("Temperature [C]")
plt.ylabel("pdf")
plt.legend()
plt.show()

## make a grid with some options
Bs = [2,4,6]
ave_locs = [0,90,180]


fig = plt.figure(figsize = (12,12))
for i in range(3):
    B = Bs[i]
    for j in range(3):
        ave_loc = ave_locs[j]
        
        ax = fig.add_subplot(3,3,1+i+3*j)
        ax.plot(x,yearly_mu(x, A, B, shift))
        plt.fill_between(x, 
                         yearly_mu(x, A, B, shift) - yearly_sigma(x, var, delta, ave_loc),
                         yearly_mu(x, A, B, shift) + yearly_sigma(x, var, delta, ave_loc),
                         alpha = 0.5)
        plt.title(f"B = {B}, φ = {ave_loc}")
        plt.xlim(0,365)
        plt.ylabel("Temperature [C]")
        
        plt.ylim(6,22)

plt.suptitle(f"mu = {A} + B*sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + ave_loc)*2pi/365.25))*{var}")
plt.show()



# eT = np.arange(0,30,0.2)

fig = plt.figure(figsize = (12,12))
for i in range(3):
    B = Bs[i]
    for j in range(3):
        ave_loc = ave_locs[j]
        
        # norms = [gen_norm_pdf(eT, yearly_mu(x[i], A, B, shift),
        #                       yearly_sigma(x[i], var, delta, ave_loc), 2) for i in range(365)]
        
        ax = fig.add_subplot(3,3,1+i+3*j)
        ax.plot(eT,gen_sine_temperature_pdf(eT,A,B,var,delta,ave_loc))
        plt.xlim(0,30)
        plt.title(f"B = {B}, φ = {ave_loc}")

plt.suptitle(f"mu = {A} + B*sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + ave_loc)*2pi/365.25))*{var}")

plt.show()


# %% sliders
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import itertools
from dash import Dash, dcc, html, Output, Input

As = np.arange(10,21)
Bs = np.arange(2,11)
sigs = np.arange(2,6)
dels = np.arange(0.2,1,0.2)
phis = np.arange(0,181,30)


app = Dash(__name__)

app.layout = html.Div([
    html.H2("Temperature model"),
    
    dcc.Graph(id="climate-plot"),
    
    html.Div([
        html.Label("B"),
        dcc.Slider(min=min(Bs), max=max(Bs), step=1, value=Bs[0],
                   marks={int(b): str(int(b)) for b in Bs}, id="B-slider")
    ], style={"margin": "20px"}),

    html.Div([
        html.Label("σ"),
        dcc.Slider(min=min(sigs), max=max(sigs), step=1, value=sigs[0],
                   marks={int(s): str(int(s)) for s in sigs}, id="sigma-slider")
    ], style={"margin": "20px"}),

    html.Div([
        html.Label("δ"),
        dcc.Slider(min=float(min(dels)), max=float(max(dels)), step=0.2, value=dels[0],
                   marks={round(d,1): str(round(d,1)) for d in dels}, id="delta-slider")
    ], style={"margin": "20px"}),

    html.Div([
        html.Label("φ"),
        dcc.Slider(min=min(phis), max=max(phis), step=30, value=phis[0],
                   marks={int(p): str(int(p)) for p in phis}, id="phi-slider")
    ], style={"margin": "20px"})
])


@app.callback(
    Output("climate-plot", "figure"),
    Input("B-slider", "value"),
    Input("sigma-slider", "value"),
    Input("delta-slider", "value"),
    Input("phi-slider", "value")
)
def update_plot(B, sigma, delta, phi):
    mu_vals = yearly_mu(x, A, B, shift)
    sigma_vals = yearly_sigma(x, sigma, delta, phi)

    # yearly cycle traces
    traces = [
        go.Scatter(x=x, y=mu_vals, name="Mean"),
        go.Scatter(x=x, y=mu_vals - sigma_vals, name="-σ"),
        go.Scatter(x=x, y=mu_vals + sigma_vals, fill="tonextx", name="+σ")
    ]
    
    # distribution subplot
    norms = [gen_norm_pdf(eT, mu_vals[i], sigma_vals[i], 2) for i in range(365)]
    traces.append(go.Scatter(x=eT, y=sum(norms)/365, name="Distribution"))

    # Create subplot layout
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Yearly Cycle", "Distribution"))
    fig.add_traces(traces[:3], rows=[1,1,1], cols=[1,1,1]).update_layout(xaxis_title="Day of year", yaxis_title="Temperature [°C]")

    fig.add_trace(traces[3], row=1, col=2)
    
    fig.update_xaxes(title_text="Day of Year", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="pdf", row=1, col=2)

    fig.update_layout(title=f"B={B}, σ={sigma}, δ={delta}, φ={phi}",
                      showlegend=False)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
    import webbrowser
    webbrowser.open("http://127.0.0.1:8050/")
    app.run(debug=True, use_reloader=False)



# %% fitting to observations, reading data


drive = 'D'

country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30

region_lats = [minlat,37.5,maxlat]
region_lons = [minlon,-116,-105,-90,maxlon]




save_name = f"{drive}:/outputs/{country_save}\\average_temp_shape.csv"
df = pd.read_csv(save_name,dtype = {0:str})


df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'
df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 

#merging the dataframes to ensure station consistency
missing_rows = pd.merge(df_parameters.station, df.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
else:
    pass

min_yrs = 10

info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})


# shouldn't need this anymore as changed files
# if name_len!=0:
#     info.station = info['station'].apply(lambda x: f'{int(x):0{name_len}}') #need to edit this according to file
# else:
#     pass

info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

#select stations


val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min


if 'min_startdate' in locals():    
    val_info = val_info[val_info['startdate']>=min_startdate]
else:
    pass

if 'minlat' in locals():
    
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
    
else:
    pass
val_info = val_info.reset_index()


s = 3
fontsize = 15


n_lat = len(region_lats)-1
n_lon = len(region_lons)-1

###############################################################################
# %% select stations (longest) in each grid

numb_per_grid = 1

station_names = []
station_lats = []
station_lons = []
for i in range(len(region_lats)-1):
    for j in range(len(region_lons)-1):
        minlat_now = region_lats[i]
        maxlat_now = region_lats[i+1]
        minlon_now = region_lons[j]
        maxlon_now = region_lons[j+1]
        
        
        mask = (val_info.latitude >= minlat_now)&(
            val_info.latitude < maxlat_now)&(
                val_info.longitude >= minlon_now)&(
                    val_info.longitude < maxlon_now)
                    
        info_now = val_info[mask].sort_values(by="cleaned_years",ascending=False).reset_index()
        
        for k in range(numb_per_grid):
            station_names.append(info_now.station.iloc[k])
            station_lats.append(info_now.latitude.iloc[k])
            station_lons.append(info_now.longitude.iloc[k])


###############################################################################
# plot the station locations

s = 3
fontsize = 15

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1,1,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c="g",
    s = s,
    label = "all stations"
)

sc = ax.scatter(
    station_lons,
    station_lats,
    c="r",
    s = s*3,
    label = "chosen stations"
)


for lat_i in range(n_lat-1):
    ax.plot([minlon-3,maxlon+3],[region_lats[lat_i+1],region_lats[lat_i+1]],  'r', linewidth=2, transform=ccrs.PlateCarree())

for lon_i in range(n_lon-1):
    ax.plot([region_lons[lon_i+1],region_lons[lon_i+1]],[minlat-3,maxlat+3],  'r', linewidth=2, transform=ccrs.PlateCarree())


plt.legend()
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

plt.xlim(-125,-70)
plt.ylim(25,50)

plt.show()

###############################################################################
# %% run model
phat2s = []

for i in range(len(station_names)):
    station = station_names[i]
    
    T_ = np.genfromtxt(f"D:/ordinary_events/US_main/T_{station}.csv")
    P_ = np.genfromtxt(f"D:/ordinary_events/US_main/P_{station}.csv")
    times = pd.read_csv(f"D:/ordinary_events/US_main/time_{station}.csv",parse_dates = ["oe_time"])
    
    
    oe = pd.DataFrame({
        "oe_time": times.oe_time,
        "T" : T_,
        "P" : P_
        })
    
    oe["date"] = pd.to_datetime(oe.oe_time).dt.date
    oe['days_of_year'] = pd.to_datetime(oe['date']).dt.dayofyear
    oe["days_of_year10"] = np.trunc(oe.days_of_year/10)*10
    
    
    cycle_mean = oe.groupby("days_of_year")["T"].mean()
    cycle_std = oe.groupby("days_of_year")["T"].std()
    
    cycle_std10 = oe.groupby("days_of_year10")["T"].std()
    
    phat = sine_temperature_model(T_)
    pdf = gen_sine_temperature_pdf(eT,phat["A"],phat["B"],phat["var"],phat["delta"],phat["ave_loc"])
    
    eT_hist = np.arange(-20,40)
    eT_edges = np.concatenate([np.array([eT_hist[0]-(eT_hist[1]-eT_hist[0])/2]),(eT_hist + (eT_hist[1]-eT_hist[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(T_, bins=eT_edges, density=True)
    
    # the opposite way round
    days = datetime_series_to_array(oe.oe_time)
    
    phat_sigma = yearly_sigma_fit(cycle_std)
    # phat_sigma10 = yearly_sigma_fit(cycle_std10) # this is std calculated in 10 day chunks
    phat_mu = yearly_mu_fit(days, T_)
    
    
    deviations_from_yearly_mu = oe["T"] - yearly_mu(days, phat_mu["A"], phat_mu["B"], shift = phat_mu["shift"], p = phat_mu["p_mu"])
    
    eT_hist_deviation = np.arange(-20,20)
    eT_edges_deviation = np.concatenate([np.array([eT_hist_deviation[0]-(eT_hist_deviation[1]-eT_hist_deviation[0])/2]),(eT_hist_deviation + (eT_hist_deviation[1]-eT_hist_deviation[0])/2)]) #convert bin centres into bin edges
    hist_deviation, bin_edges = np.histogram(deviations_from_yearly_mu, bins=eT_edges_deviation, density=True)
    
    
    plt.plot(eT_hist_deviation, hist_deviation, '--') #TODO: fit a normal distribution to this...
    plt.title("distribution of the deviations")
    plt.show()
    
    
    pdf_back = gen_sine_temperature_pdf(eT, phat_mu["A"], phat_mu["B"],
                                        phat_sigma["var"], phat_sigma["delta"],
                                        phat_sigma["ave_loc"] - phat_mu["shift"],
                                        p_mu = phat_mu["p_mu"], p_sigma = phat_sigma["p_sigma"])
    
    # pdf_back10 = gen_sine_temperature_pdf(eT, phat_mu["A"], phat_mu["B"],
    #                                     phat_sigma10["var"], phat_sigma10["delta"],
    #                                     phat_sigma10["ave_loc"] - phat_mu["shift"],
    #                                     p_mu = phat_mu["p_mu"], p_sigma = phat_sigma10["p_sigma"])
    
    
    ## new version as francesco said
    # phat2 = sine_temperature_model_v2(days, T_)
    # phat2s.append(phat2)
    
    
    # pdf2 = gen_sine_temperature_pdf(eT,phat2["A"],phat2["B"],
    #                                 phat2["var"],phat2["delta"],
    #                                 phat2["ave_loc"]-phat2["shift"],       
    #                                 p_mu = phat2["p_mu"], p_sigma = phat2["p_sigma"])
 
    
    
    day_difference = oe.days_of_year.iloc[0]
    
    
    fig = plt.figure(figsize=(18,6))
    
    ax = fig.add_subplot(1,3,1)
    plt.plot(eT_hist, hist, '--')
    
    plt.plot(eT,pdf, label = "fitted on pdf")
    
    plt.plot(eT, pdf_back,
             label = "backwards fit")
    
    # plt.plot(eT,pdf2, label = "new version")

    # plt.plot(eT, pdf_back10,
             # label = "backwards fit, std smoothed over 10 days")
    
    plt.legend()
    
    plt.title(f"{i} {station}. ({station_lats[i]},{station_lons[i]})")
    
    
    
    xs = np.arange(1,len(cycle_std)+1)
    
    
    shift = minimize(lambda theta: np.sum((yearly_mu(xs, phat["A"], phat["B"], shift = theta) - cycle_mean)**2),
                    0,
                    method='Nelder-Mead').x[0]
    
    day_difference = oe.days_of_year.astype("int").iloc[0]/(3600*24*10e8)
    
    
    
    
    
    ax = fig.add_subplot(1,3,2)
    
    #plot observed averaged cycle
    ax.plot(xs,cycle_mean)
    
    #plot simulated cycle
    plt.plot(xs,yearly_mu(xs, phat["A"], phat["B"], shift = shift), label = "fitted on pdf")
    plt.plot(xs,yearly_mu(xs, phat_mu["A"], phat_mu["B"], shift = phat_mu["shift"] - day_difference, p = phat_mu["p_mu"]), label = "backwards fit")
    # plt.plot(xs,yearly_mu(xs, phat2["A"], phat2["B"], shift = phat2["shift"] - day_difference, p = phat2["p_mu"]), label = "new version")
    
    plt.legend()
    
    ax.set_xlabel("day of year")
    ax.set_ylabel("Temperature [C]")
    plt.title(f"{i} Mean yearly cycle")
    
    ax = fig.add_subplot(1,3,3)
    ax.plot(xs,cycle_std)
    plt.plot(xs,yearly_sigma(xs,phat["var"],phat["delta"],shift+phat["ave_loc"])) 
    plt.plot(xs,yearly_sigma(xs,phat_sigma["var"],phat_sigma["delta"],phat_sigma["ave_loc"],p=phat_sigma["p_sigma"])) 
    # plt.plot(xs,yearly_sigma(xs,phat2["var"],phat2["delta"],phat2["ave_loc"],p=phat2["p_sigma"])) 
    # plt.plot(xs,yearly_sigma(xs,phat_sigma10["var"],phat_sigma10["delta"],phat_sigma10["ave_loc"],p=phat_sigma10["p_sigma"])) 
    
    ax.set_xlabel("day of year")
    ax.set_ylabel("Temperature [C]")  
    plt.title("standard deviation yearly cycle")
    plt.show()
    
    # plot scatter
    s = 2
    plt.scatter(oe['days_of_year'],T_,s = s,alpha = 0.5, color = "r")
    
    plt.plot(xs,yearly_mu(xs, phat_mu["A"], phat_mu["B"],phat_mu["shift"] - day_difference, p = phat_mu["p_mu"]),color ="b")
    plt.fill_between(xs,
                     yearly_mu(xs, phat_mu["A"], phat_mu["B"],phat_mu["shift"] - day_difference, p = phat_mu["p_mu"]) - 3*yearly_sigma(xs,phat_sigma["var"],phat_sigma["delta"],phat_sigma["ave_loc"],p=phat_sigma["p_sigma"]),
                     yearly_mu(xs, phat_mu["A"], phat_mu["B"],phat_mu["shift"] - day_difference, p = phat_mu["p_mu"]) + 3*yearly_sigma(xs,phat_sigma["var"],phat_sigma["delta"],phat_sigma["ave_loc"],p=phat_sigma["p_sigma"]),
                     alpha = 0.2,
                     color = "b",
                     label = "3 times std"
                     )
    plt.legend()
    plt.xlim(0,365)
    plt.xlabel("day of year")
    plt.ylabel("Temperature (C)")
    plt.show()
    
    ###########################################################################
    # do it with cut version of none oe
    
    
    full_temp_xr = xr.load_dataarray(f"D:/US_temp/US_{station}.nc")
    full_temp = full_temp_xr.to_numpy().squeeze() - 273.15
    full_temp_24hr = full_temp_xr.squeeze().to_pandas().resample("d").mean() - 273.15
    
    full_temp_24hr_shortened = pd.DataFrame(full_temp_24hr[-5000:-1])
    full_temp_24hr_shortened["days_of_year"] = full_temp_24hr_shortened.index.dayofyear
    
    cycle_mean = full_temp_24hr_shortened.groupby("days_of_year")["t2m"].mean()
    cycle_std = full_temp_24hr_shortened.groupby("days_of_year")["t2m"].std()
    rolling_std = full_temp_24hr_shortened.t2m.rolling(10).std()
    
    T_full = full_temp_24hr_shortened.t2m.to_numpy()
    phat_full = sine_temperature_model(T_full)
    pdf = gen_sine_temperature_pdf(eT, phat_full["A"],phat_full["B"],phat_full["var"],phat_full["delta"],phat_full["ave_loc"])
    
    eT_hist = np.arange(-20,40)
    eT_edges = np.concatenate([np.array([eT_hist[0]-(eT_hist[1]-eT_hist[0])/2]),(eT_hist + (eT_hist[1]-eT_hist[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(T_full, bins=eT_edges, density=True)
    
    
    days = np.arange(0,len(T_full))
    
    phat_sigma = yearly_sigma_fit(cycle_std)
    phat_mu = yearly_mu_fit(days, T_full)
    pdf_back = gen_sine_temperature_pdf(eT, phat_mu["A"], phat_mu["B"],
                                        phat_sigma["var"], phat_sigma["delta"],
                                        phat_sigma["ave_loc"] - phat_mu["shift"],
                                        p_mu = phat_mu["p_mu"], p_sigma = phat_sigma["p_sigma"])
    
    ## new version as francesco said
    # phat2 = sine_temperature_model_v2(days, T_full)
    
    
    
    # pdf2 = gen_sine_temperature_pdf(eT,phat2["A"],phat2["B"],
    #                                 phat2["var"],phat2["delta"],
    #                                 phat2["ave_loc"]-phat2["shift"],       
    #                                 p_mu = phat2["p_mu"], p_sigma = phat2["p_sigma"])
 
    

    
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,3,1)
    
    plt.plot(eT_hist, hist, '--')
    
    plt.plot(eT,pdf, label = "fitted on pdf")
    
    plt.plot(eT, pdf_back,
             label = "backwards fit")
    
    # plt.plot(eT,pdf2, label = "new version")
    
    plt.title(f"{i} full temperature {station}. ({station_lats[i]},{station_lons[i]})")
    plt.legend()
    
    
    
    xs = np.arange(1,len(cycle_std)+1)
    
    
    shift = minimize(lambda theta: np.sum((yearly_mu(xs, phat_full["A"], phat_full["B"], shift = theta) - cycle_mean)**2),
                    0,
                    method='Nelder-Mead').x[0]
    
    day_difference = full_temp_24hr_shortened.days_of_year.iloc[0] # this is because the shift is based on the difference from where the cycle starts, rather than the beginning of the year
    
    ax = fig.add_subplot(1,3,2)
    
    #plot observed averaged cycle
    ax.plot(xs,cycle_mean)
    
    #plot simulated cycle
    plt.plot(xs,yearly_mu(xs, phat_full["A"], phat_full["B"], shift = shift), label = "fitted on pdf")
    plt.plot(xs,yearly_mu(xs, phat_mu["A"], phat_mu["B"], shift = phat_mu["shift"]-day_difference, p = phat_mu["p_mu"]), label = "backwards fit")
    # plt.plot(xs,yearly_mu(xs, phat2["A"], phat2["B"], shift = phat2["shift"] - day_difference, p = phat2["p_mu"]), label = "new version")
    
    
    ax.set_xlabel("day of year")
    ax.set_ylabel("Temperature [C]")
    plt.title(f"{i} Mean yearly cycle, full")
    plt.legend()
    
    ax = fig.add_subplot(1,3,3)
    ax.plot(xs,cycle_std)
    plt.plot(xs,yearly_sigma(xs,phat_full["var"],phat_full["delta"],shift+phat_full["ave_loc"])) 
    plt.plot(xs,yearly_sigma(xs,phat_sigma["var"],phat_sigma["delta"],phat_sigma["ave_loc"],p=phat_sigma["p_sigma"])) 
    # plt.plot(xs,yearly_sigma(xs,phat2["var"],phat2["delta"],phat2["ave_loc"],p=phat2["p_sigma"])) 
    
    ax.set_xlabel("day of year")
    ax.set_ylabel("Temperature [C]")  
    plt.title("standard deviation yearly cycle, full")
    plt.show()
    
# %% run model with rolling std
for i in range(len(station_names)):
    station = station_names[i]
    
    T_ = np.genfromtxt(f"D:/ordinary_events/US_main/T_{station}.csv")
    P_ = np.genfromtxt(f"D:/ordinary_events/US_main/P_{station}.csv")
    times = pd.read_csv(f"D:/ordinary_events/US_main/time_{station}.csv",parse_dates = ["oe_time"])
    
    
    oe = pd.DataFrame({
        "oe_time": times.oe_time,
        "T" : T_,
        "P" : P_
        })
    
    oe["date"] = pd.to_datetime(oe.oe_time).dt.date
    oe['days_of_year'] = pd.to_datetime(oe['date']).dt.dayofyear
    
    days = datetime_series_to_array(oe.oe_time)
    
    phat_sigma_rolling30, rolling_std30 = yearly_sigma_fit_rolling(oe, window = 30)
    phat_sigma_rolling40, rolling_std40 = yearly_sigma_fit_rolling(oe, window = 40)
    phat_sigma_rolling50, rolling_std50 = yearly_sigma_fit_rolling(oe, window = 50)
    
    phat_mu = yearly_mu_fit(days, T_)
    
    eT_hist = np.arange(-20,40)
    eT_edges = np.concatenate([np.array([eT_hist[0]-(eT_hist[1]-eT_hist[0])/2]),(eT_hist + (eT_hist[1]-eT_hist[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(T_, bins=eT_edges, density=True)
    
    pdf30 = gen_sine_temperature_pdf(eT, phat_mu["A"], phat_mu["B"],
                                        phat_sigma_rolling30["var"], phat_sigma_rolling30["delta"],
                                        phat_sigma_rolling30["ave_loc"] - phat_mu["shift"],
                                        p_mu = phat_mu["p_mu"], p_sigma = phat_sigma_rolling30["p_sigma"])
    
    
    pdf40 = gen_sine_temperature_pdf(eT, phat_mu["A"], phat_mu["B"],
                                        phat_sigma_rolling40["var"], phat_sigma_rolling40["delta"],
                                        phat_sigma_rolling40["ave_loc"] - phat_mu["shift"],
                                        p_mu = phat_mu["p_mu"], p_sigma = phat_sigma_rolling40["p_sigma"])
    
    pdf50 = gen_sine_temperature_pdf(eT, phat_mu["A"], phat_mu["B"],
                                        phat_sigma_rolling50["var"], phat_sigma_rolling50["delta"],
                                        phat_sigma_rolling50["ave_loc"] - phat_mu["shift"],
                                        p_mu = phat_mu["p_mu"], p_sigma = phat_sigma_rolling50["p_sigma"])
    
    
    fig = plt.figure(figsize = (12,5))
    
    ax = fig.add_subplot(1,3,1)
    ax.plot(eT_hist, hist, '--')
    ax.plot(eT,pdf30)
    ax.set_title("window = 30 days")
    
    ax = fig.add_subplot(1,3,2)
    ax.plot(eT_hist, hist, '--')
    ax.plot(eT,pdf40)
    ax.set_title("window = 40 days")
    
    ax = fig.add_subplot(1,3,3)
    ax.plot(eT_hist, hist, '--')
    ax.plot(eT,pdf50)
    ax.set_title("window = 50 days")
    plt.show()
    
    # plot the yearly cycle
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,3,1)
    
    ax.scatter(oe['days_of_year'],T_,s = s)
    plt.plot(x,yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]),color ="b")
    plt.fill_between(x,
                     yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]) - 3*yearly_sigma(x,phat_sigma_rolling30["var"],phat_sigma_rolling30["delta"],phat_sigma_rolling30["ave_loc"]+phat_mu["shift"]),
                     yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]) + 3*yearly_sigma(x,phat_sigma_rolling30["var"],phat_sigma_rolling30["delta"],phat_sigma_rolling30["ave_loc"]+phat_mu["shift"]),
                     alpha = 0.2,
                     color = "b",
                     label = "3 times std"
                     )
    plt.legend()
    
    ax = fig.add_subplot(1,3,2)
    
    ax.scatter(oe['days_of_year'],T_,s = s)
    
    plt.plot(x,yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]),color ="b")
    plt.fill_between(x,
                     yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]) - 3*yearly_sigma(x,phat_sigma_rolling40["var"],phat_sigma_rolling40["delta"],phat_sigma_rolling40["ave_loc"]),
                     yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]) + 3*yearly_sigma(x,phat_sigma_rolling40["var"],phat_sigma_rolling40["delta"],phat_sigma_rolling40["ave_loc"]),
                     alpha = 0.2,
                     color = "b",
                     label = "3 times std"
                     )
    plt.legend()
    
    ax = fig.add_subplot(1,3,3)
    
    ax.scatter(oe['days_of_year'],T_,s = s)
    
    plt.plot(x,yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]),color ="b")
    plt.fill_between(x,
                     yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]) - 3*yearly_sigma(x,phat_sigma_rolling50["var"],phat_sigma_rolling50["delta"],phat_sigma_rolling50["ave_loc"]),
                     yearly_mu(x, phat_mu["A"], phat_mu["B"],phat_mu["shift"]) + 3*yearly_sigma(x,phat_sigma_rolling50["var"],phat_sigma_rolling50["delta"],phat_sigma_rolling50["ave_loc"]),
                     alpha = 0.2,
                     color = "b",
                     label = "3 times std"
                     )
    plt.legend()
    
    plt.show()
    
    













