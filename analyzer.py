import streamlit as st

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import math
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates

def Supertrend(df, atr_period, multiplier):
    
    high = df['High']
    low = df['Low']
    close = df['Close']

    sma130 = close.rolling(130).mean()
    
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    # df['atr'] = df['tr'].rolling(atr_period).mean()
    
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband,
        'SMA 130': sma130
    }, index=df.index)

def generateCandleStickPlot(data, name, style='ggplot'):
  plt.style.use(style)
  data.reset_index(level=0, inplace=True)
  ohlc = data.loc[:, ['Date','Open', 'High', 'Low', 'Close','Final Lowerband','Final Upperband', 'SMA 130']]
  ohlc['Date'] = pd.to_datetime(ohlc['Date'])
  ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
  ohlc = ohlc.astype(float)
  # Creating Subplots
  fig, ax = plt.subplots()

  candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

  ax.plot(ohlc['Date'], ohlc['Final Lowerband'], color='green', label='Final Lowerband')
  ax.plot(ohlc['Date'], ohlc['Final Upperband'], color='red', label='Final Upperband')
  ax.plot(ohlc['Date'], ohlc['SMA 130'], color='blue', label='SMA 130')
 

  # Setting labels & titles
  ax.set_xlabel('Date')
  ax.set_ylabel('Price')
  plt.title(f'Daily Candlestick Chart of {name}')

  # Formatting Date
  date_format = mpl_dates.DateFormatter('%d-%m-%Y')
  ax.xaxis.set_major_formatter(date_format)
  fig.autofmt_xdate()

  fig.tight_layout()
  return plt, fig

def calculateSuperTrend(symbol, symbol_name, style, plot = True, weeks = 52, atr_period = 10, atr_multiplier = 3):
  start_date = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')
  df = yf.download(symbol, start=start_date)
  supertrend = Supertrend(df, atr_period, atr_multiplier)
  df = df.join(supertrend)
  # visualization
  # plt.plot(df['Close'], label='Close Price')
  # plt.plot(df['Final Lowerband'], 'g', label = 'Final Lowerband')
  # plt.plot(df['Final Upperband'], 'r', label = 'Final Upperband')
  if plot == True:
    plt, fig = generateCandleStickPlot(df.iloc[-40:,:], symbol_name, style)
  else:
    plt = None
    fig = None
  return df, plt, fig


symbols_list = [('^GDAXI', 'DAX'), ('BTC-EUR','Bitcoin'), ('ETH-EUR','Ethereum'), ('LIN.DE','Linde plc'), ('TSLA','Tesla'), ('VWRL.AS','FTSE All-World'), ('IBC0.F','MSCI Europe'), ('GC=F','Gold'), ('4GLD.SG','Xetra-Gold'), ('CL=F', 'Crude Oil'), ('ZB=F','U.S. Treasury Bond Futures'), ('MSF.DE','Microsoft'),('FB2A.F','Facebook'), ('NFC.DE','Netflix'),('NVD.DE','NVIDIA'), ('ASML.AS','ASML'),('APC.F','Apple'),('GOOG','Google'),('^GSPC','S&P 500'),('C090.DE','LYXOR COMMO EX AGRI ETF I'),('SPYV.DE','SPDR S&P EME.MKTS DIV.ARIS.ETF')]

st.sidebar.title("Supertrend Analyzer")

st.sidebar.subheader('Single Stocks')
name = st.sidebar.selectbox('Select a stock', [x[1] for x in symbols_list])
symbol = [x[0] for x in symbols_list if x[1] == name][0]


plt.clf()
df, plt, fig = calculateSuperTrend(symbol, name, 'seaborn-whitegrid')
if (df.iloc[-1,:]['Supertrend'] == True) & (df.iloc[-2,:]['Supertrend'] == False):
    message=f'Buy signal for {name}. New bullish supertrend detected.'
elif (df.iloc[-1,:]['Supertrend'] == False) & (df.iloc[-2,:]['Supertrend'] == True):
    message=f'Sell signal for {name}. New bearish supertrend detected.'
else:
    if df.iloc[-1,:]['Supertrend'] == True:
        message=f'{name} is currently bullish. \n**Stop:** {int(df.iloc[-1,:]["Final Lowerband"]*100)/100}'
    else:
        message=f'{name} is currently bearish. \n**Stop:** {int(df.iloc[-1,:]["Final Upperband"]*100)/100}'
st.subheader(message)
st.pyplot(fig)