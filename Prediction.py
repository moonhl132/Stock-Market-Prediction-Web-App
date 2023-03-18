#import libraries
import pandas as pds
import numpy as nmp
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
import math, random
from datetime import datetime
import datetime 
from datetime import date
import matplotlib.pyplot as plot
from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pmdarima.arima.utils import ndiffs
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
plot.style.use('fivethirtyeight')
import plotly.graph_objects as go
import streamlit as st
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
import warnings
warnings.filterwarnings("ignore")
     

with st.sidebar:
    ticker_list = pds.read_csv('https://raw.githubusercontent.com/moonhl132/Stock-Market-Prediction-Web-App/main/Stock%20Market%20Prediction/Companies.txt')
    st.markdown("<h1 style='text-align: center; color: White;'>Index LQ45 Stock Ticker  </h1>", unsafe_allow_html=True)
    tickerlist = st.selectbox('',ticker_list)
    tickerData = yf.Ticker(tickerlist)

    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days = 1)

    start_date = st.date_input("Start date", datetime.date(2021, 1, 1))
    end_date = st.date_input("End date", tomorrow)

    predict = st.button("Predict")

#start_date= datetime(datetime.now().year-2,datetime.now().month,datetime.now().day)
#end_date=  datetime(datetime.now().year,datetime.now().month,datetime.now().day+1)

#ticker_list = pds.read_csv('https://raw.githubusercontent.com/moonhl132/Stock-Market-Prediction-Web-App/main/Stock%20Market%20Prediction/Companies.txt')
#st.markdown("<h5 style='text-align: center; color: White;'>Index LQ45 Stock Ticker  </h5>", unsafe_allow_html=True)
#tickerlist = st.selectbox('',ticker_list)
#tickerdata = yf.Ticker(tickerlist)
if predict:
    st.markdown("<h1 style='text-align: center; color: White;'>Stock Market Prediction</h1>", unsafe_allow_html=True)

    tickerDf = pdr.get_data_yahoo(tickerlist, data_source='yahoo', start= start_date , end= end_date)

    colm1, colm2, colm3, colm4 = st.columns(4)

    with colm1:
        today_open_price  = tickerDf.iloc[-1]['Open']
        st.metric('Open', today_open_price.round(0))
    
    with colm2:
        today_close_price  = tickerDf.iloc[-1]['Close']
        st.metric('Close', today_close_price.round(0))

    with colm3:
        today_high_price  = tickerDf.iloc[-1]['High']
        st.metric('High', today_high_price.round(0))

    with colm4:
        today_low_price  = tickerDf.iloc[-1]['Low']
        st.metric('Low', today_low_price.round(0))

    layout = go.Layout(
        title = "Ticker",
        xaxis = {'title' : 'Date'},
        yaxis = {'title' : 'Price'},
    )

    fig = go.Figure(
        layout = layout,
        data = [
            go.Candlestick(
                open = tickerDf['Open'],
                high = tickerDf['High'],
                low = tickerDf['Low'],
                close = tickerDf['Close'],
                name = 'Status'
            ),
        ]
    )
    fig.update_layout(height = 650)
    st.plotly_chart(fig, use_container_width=True)

    data = tickerDf.iloc[:,3:4].values


    col1, col2 = st.columns(2)

    with col1:
        # Random Forest Algorithm
        forecast_out_rf = 1
        tickerDf['Prediction'] = tickerDf['Close'].shift(-forecast_out_rf)
        data_frame_rf = tickerDf[['Close' , 'Prediction']]

        x_rf = nmp.array(data_frame_rf.iloc[:-forecast_out_rf,0:-1])
        x_prediction_rf = nmp.array(data_frame_rf.iloc[-forecast_out_rf:,0:-1])
        y_rf = nmp.array(data_frame_rf.iloc[:-forecast_out_rf,-1])

        X_train_rf, X_test_rf, Y_train_rf, Y_test_rf = train_test_split(x_rf , y_rf, test_size = 0.2)

        rf_regressor = RandomForestRegressor(n_estimators=100)
        rf_regressor.fit(X_train_rf, Y_train_rf)

        prediction_rf = rf_regressor.predict(X_test_rf)

        st.markdown("<h3 style='text-align: center; color: White;'>Random Forest Prediction</h3>", unsafe_allow_html=True)


        fig = plot.figure(figsize=(7.2,4.8),dpi=65)
        plot.grid(True)
        plot.plot(prediction_rf, color = 'red', label = 'Predicted Price')
        plot.plot(Y_test_rf,color='blue', label="Actual Price")
        plot.title('Random Forest Regression')
        plot.legend(['Prediction','Train'], loc='lower right')
        st.pyplot(fig)

        rmse = math.sqrt(mean_squared_error(Y_test_rf, prediction_rf))
        st.write("RMSE : " + str(rmse))

        
        forecast_rf = rf_regressor.predict(x_prediction_rf)
        st.write("Tomorrow Price : " + str(forecast_rf[-1].round()))

    with col2:
        #Linear Regression Model

        forecast_out = 1
        tickerDf['Prediction'] = tickerDf['Close'].shift(-forecast_out)
        data_frame = tickerDf[['Close' , 'Prediction']]

        x_lr = nmp.array(data_frame.iloc[:-forecast_out,0:-1])
        x_prediction_lr = nmp.array(data_frame.iloc[-forecast_out:,0:-1])
        y_lr = nmp.array(data_frame.iloc[:-forecast_out,-1])
        y_lr = nmp.reshape(y_lr,(-1,1))

        x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(x_lr,y_lr, test_size = 0.2)

        scale_lr = StandardScaler()
        x_train_lr = scale_lr.fit_transform(x_train_lr)
        x_test_lr = scale_lr.transform(x_test_lr)
        x_prediction_lr = scale_lr.transform(x_prediction_lr)

        linear_regression = LinearRegression()
        linear_regression.fit(x_train_lr,y_train_lr)

        y_test_pred = linear_regression.predict(x_test_lr)

        st.markdown("<h3 style='text-align: center; color: White;'>Linear Regression Prediction</h3>", unsafe_allow_html=True)

        fig = plot.figure(figsize=(7.2,4.8),dpi=65)
        plot.grid(True)
        plot.plot(y_test_pred, color = 'red', label = 'Predicted Price')
        plot.plot(y_test_lr,color='blue', label="Actual Price")
        plot.title('Linear Regression')
        plot.legend(['Prediction','Train'], loc='lower right')
        st.pyplot(fig)

        rmse = math.sqrt(mean_squared_error(y_test_lr, y_test_pred))
        st.write("RMSE : " + str(rmse))

        forecasting = linear_regression.predict(x_prediction_lr)

        st.write("Tomorrow Price : " + str(forecasting[0,0].round()))

    col3, col4 = st.columns(2)

    with col3:

        #ARIMA Model Algorithm
        data_arima = int(math.ceil(len(tickerDf)*0.8))
        data_train_arima = list(tickerDf[0:data_arima]['Close'])
        data_test_arima = list(tickerDf[data_arima:]['Close'])

        result = adfuller(data)
        diff = tickerDf.Close.diff().dropna()
        diff = tickerDf.Close.diff().diff().dropna()
        d_value = ndiffs(data, test="adf")

        model_predictions = []
        n_test_obser = len(data_test_arima) 

        for i in range (n_test_obser):
            model = ARIMA(data_train_arima, order = (6,1,0))
            modelfit = model.fit()
            output = modelfit.forecast()
            yhat = list(output)[0]
            model_predictions.append(yhat)
            actual_test_value = data_test_arima[i]
            data_train_arima.append(actual_test_value)

        st.markdown("<h3 style='text-align: center; color: White;'>ARIMA Prediction</h3>", unsafe_allow_html=True)


        fig = plot.figure(figsize=(7.2,4.8),dpi=65)
        plot.grid(True)
        plot.plot(model_predictions, color = 'red', marker ='o', label = 'Predicted Price')
        plot.plot(data_test_arima,color='blue', label="Actual Price")
        plot.title('ARIMA Model')
        plot.legend(['Prediction','Train'], loc='lower right')
        st.pyplot(fig)

        rmse = math.sqrt(mean_squared_error(model_predictions, data_test_arima))
        st.write("RMSE : " + str(rmse))


        next_day = model_predictions[-1]
        st.write("Tomorrow Price : " + str(next_day.round()))


    with col4:
        # LSTM Algorithm
        data_split = math.ceil(len(tickerDf)*.8)
        data_train = tickerDf.iloc[0:data_split,:]
        data_test = tickerDf.iloc[data_split:,:]

        scale = MinMaxScaler(feature_range=(0,1))
        scale_data =  scale.fit_transform(data)

        X_train = []
        Y_train = []
        for i in range (30,len(scale_data)):
            X_train.append(scale_data[i-30:i,0])
            Y_train.append(scale_data[i,0])

        X_train = nmp.array(X_train)
        Y_train = nmp.array(Y_train)
        Forecast = nmp.array(X_train[-1,1:])
        Forecast = nmp.append(Forecast,Y_train[-1])

        X_train = nmp.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
        Forecast = nmp.reshape(Forecast, (1,Forecast.shape[0],1))

        model = Sequential()

        model.add(LSTM(units=100, return_sequences = True, input_shape= (X_train.shape[1],1)))
        model.add(Dropout(rate = 0.2))

        model.add(LSTM(units=100, return_sequences = True))
        model.add(Dropout(rate = 0.2))

        model.add(LSTM(units=100, return_sequences = True))
        model.add(Dropout(rate = 0.2))

        model.add(LSTM(units=100))
        model.add(Dropout(rate = 0.2))

        model.add(Dense(1))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        model.fit(X_train, Y_train, batch_size = 1 , epochs = 20)

        dataset_test = data_test.iloc[:,3:4].values
        dataset = pds.concat((data_train['Close'], data_test['Close']),axis=0)
        testing = dataset[len(dataset) - len(data_test) -30:].values
        testing = testing.reshape(-1,1)
        testing = scale.transform(testing)

        X_test = []
        for i in range (30,len(testing)):
            X_test.append(testing[i-30:i,0])

        X_test = nmp.array(X_test)
        X_test = nmp.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

        Prediction = model.predict(X_test)
        Prediction = scale.inverse_transform(Prediction)

        st.markdown("<h3 style='text-align: center; color: White;'>LSTM Prediction</h3>", unsafe_allow_html=True)

        fig = plot.figure(figsize=(7.2,4.8),dpi=65)
        plot.title('LSTM Model')
        plot.plot(dataset_test, color = 'red', label = 'Actual Price')
        plot.plot(Prediction, color = 'blue', label = 'Predicted Price')
        plot.legend(['Train','Prediction'], loc='lower right')
        st.pyplot(fig)

        rmse = math.sqrt(mean_squared_error(dataset_test, Prediction))
        st.write("RMSE : " + str(rmse))

        forecasted_price = model.predict(Forecast)
        forecasted_price = scale.inverse_transform(forecasted_price)
        st.write("Tomorrow Price : " + str(forecasted_price[0,0].round()))


    
    mean_lstm = forecasted_price[0,0]
    mean_arima = next_day
    mean_lr = forecasting[0,0]
    mean_rf = forecast_rf[-1]
    today_stock = tickerDf.iloc[-1:]

    
    if ((today_stock.iloc[-1]['Close'] < mean_lstm) & (today_stock.iloc[-1]['Close'] < mean_arima) & (today_stock.iloc[-1]['Close'] < mean_lr)):
        st.header("DECISION FOR TOMORROW IS : BUY !!!")
    
    elif ((today_stock.iloc[-1]['Close'] < mean_lstm) & (today_stock.iloc[-1]['Close'] < mean_arima) & (today_stock.iloc[-1]['Close']< mean_rf)):
        st.header("DECISION FOR TOMORROW IS : BUY !!!")
    
    elif ((today_stock.iloc[-1]['Close'] < mean_lr) & (today_stock.iloc[-1]['Close'] < mean_arima) & (today_stock.iloc[-1]['Close'] < mean_rf)):
        st.header("DECISION FOR TOMORROW IS : BUY !!!")
    
    elif ((today_stock.iloc[-1]['Close'] < mean_lstm) & (today_stock.iloc[-1]['Close'] < mean_arima) & (today_stock.iloc[-1]['Close'] < mean_rf) & (today_stock.iloc[-1]['Close'] < mean_lr) ):
        st.header("DECISION FOR TOMORROW IS : BUY !!!")

    
    
    elif ((today_stock.iloc[-1]['Close'] <= mean_lstm) & (today_stock.iloc[-1]['Close'] <= mean_arima)):
        st.header("DECISION FOR TOMORROW IS : HOLD !!!")
    
    elif ((today_stock.iloc[-1]['Close'] <= mean_lstm) & (today_stock.iloc[-1]['Close'] <= mean_lr)):
        st.header("DECISION FOR TOMORROW IS : HOLD !!!")

    elif ((today_stock.iloc[-1]['Close'] <= mean_lstm) & (today_stock.iloc[-1]['Close'] <= mean_rf)):
        st.header("DECISION FOR TOMORROW IS : HOLD !!!")

    elif ((today_stock.iloc[-1]['Close'] <= mean_arima) & (today_stock.iloc[-1]['Close'] <= mean_lr)):
        st.header("DECISION FOR TOMORROW IS : HOLD !!!")

    elif ((today_stock.iloc[-1]['Close'] <= mean_arima) & (today_stock.iloc[-1]['Close'] <= mean_rf)):
        st.header("DECISION FOR TOMORROW IS : HOLD !!!")

    elif ((today_stock.iloc[-1]['Close'] <= mean_lr) & (today_stock.iloc[-1]['Close'] <= mean_rf)):
        st.header("DECISION FOR TOMORROW IS : HOLD !!!")

    
    else :
        st.header("DECISION FOR TOMORROW IS : SELL !!!")



else :


    tickerDf = pdr.get_data_yahoo('^JKSE', data_source='yahoo', start= start_date , end= end_date)

   
    st.markdown("<h2 style='text-align: center; color: White;'>INDEX HARGA SAHAM GABUNGAN (IHSG)</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        today_open_price  = tickerDf.iloc[-1]['Open']
        st.metric('Open', today_open_price.round(4))
    
    with col2:
        today_close_price  = tickerDf.iloc[-1]['Close']
        st.metric('Close', today_close_price.round(4))

    with col3:
        today_high_price  = tickerDf.iloc[-1]['High']
        st.metric('High', today_high_price.round(4))

    with col4:
        today_low_price  = tickerDf.iloc[-1]['Low']
        st.metric('Low', today_low_price.round(4))
    
    layout = go.Layout(
        title = "IHSG Candlestick",
        xaxis = {'title' : 'Date'},
        yaxis = {'title' : 'Price'},
    )

    fig = go.Figure(
        layout = layout,
        data = [
            go.Candlestick(
                open = tickerDf['Open'],
                high = tickerDf['High'],
                low = tickerDf['Low'],
                close = tickerDf['Close'],
                name = 'Status'
            ),
        ]
    )
    fig.update_layout(height = 650)
    st.plotly_chart(fig, use_container_width=True)



    
    
   
