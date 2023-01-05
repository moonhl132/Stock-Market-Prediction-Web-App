#import libraries
import pandas as pds
import numpy as nmp
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
import math, random
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plot
from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
plot.style.use('fivethirtyeight')
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

st.title("Stock Market Prediction")



