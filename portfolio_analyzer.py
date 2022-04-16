# https://github.com/ranaroussi/quantstats
# https://github.com/ssantoshp/Empyrial/blob/master/Empyrial.py
# https://medium.com/@tzjy/a-great-tool-to-portfolio-optimization-riskfolio-lib-python-code-included-d4e4d503541c

import pandas as pd
import yfinance as yf
import quantstats as qs
import warnings
import riskfolio as rp
import matplotlib.pyplot as plt

import datetime as dt

from config import symbols_list, obj_list

TODAY = dt.date.today()

# extend pandas functionality with metrics, etc.
qs.extend_pandas()


def get_returns(stocks, wts, start_date, end_date=TODAY):
    if len(stocks) > 1:
        assets = yf.download(stocks, start=start_date, end=end_date, progress=False)["Adj Close"]
        assets = assets.filter(stocks)
        ret_data = assets.pct_change()[1:]
        returns = (ret_data * wts).sum(axis=1)
        return returns
    else:
        df = yf.download(stocks, start=start_date, end=end_date, progress=False)["Adj Close"]
        df = pd.DataFrame(df)
        ret_data = df.pct_change()
        returns = (ret_data * wts).sum(axis=1)
        return returns

class analyzer:
    def __init__(self, st):
        with st.sidebar.expander('Portfolio Analyzer'):
            # describes the supertrend indicator in stock market analysis
            st.write('helps to determine the optimal portfolio balance')
            self.stock_names = st.multiselect('Select stocks', [x[1] for x in symbols_list])
            self.start_date = st.date_input('Start Date', dt.date(2020, 1, 1))
            self.end_date = st.date_input('End Date', dt.date.today())
            self.obj_name = st.selectbox('Select objective', [x[1] for x in obj_list])
            self.general_weight_limits = st.slider('General weight limits', 0.0, 1.0, (0.00, 0.5))
            self.equity_weight_limits = st.slider('Shares weight limits', 0.0, 1.0, (0.00, 0.1))
            self.commodities_weight_limits = st.slider('Commodities weight limits', 0.0, 1.0, (0.00, 0.1))
            print(self.general_weight_limits, self.equity_weight_limits)
            checked = len(self.stock_names) >= 1
            show_portfolio = st.checkbox('Show Portfolio Analysis', value=checked)
        if show_portfolio:
            self.weights=[]
            self.stock_symbols = []
            self.weights = []
            self.classes = []
            for i in range(len(self.stock_names)):
                self.stock_symbols.append([x[0] for x in symbols_list if x[1] == self.stock_names[i]][0])
                self.classes.append([x[2] for x in symbols_list if x[1] == self.stock_names[i]][0])
            self.obj = []
            self.obj =[x[0] for x in obj_list if x[1] == self.obj_name][0]            
            print(self.obj)
            self.plot(st)

    def plot(self, st):
        with st.spinner('Optimizing Portfolio...'):
            if len(self.stock_symbols)>1:
                data = yf.download(self.stock_symbols, start=self.start_date, end=self.end_date, progress=False)
                data = data.loc[:, ('Adj Close', slice(None))]
                data.columns = self.stock_names
                Y = data[self.stock_names].pct_change().dropna()
                port = rp.Portfolio(returns=Y)
                method_mu = 'hist'
                method_cov = 'hist'
                port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
                model="Classic" # Classic (historical), BL (Black Litterman), FM (Factor Model)
                rm = 'MV' # Risk Measure
                obj = self.obj # Objective Function -> MinRisk, MaxRet, Utility, Sharpe
                hist = True # historical scenarios for risk measures
                rf = 0 # Risk free rate
                l = 0 # risk aversion factor, only useful when obj is "Utility"

                # Setup constraints
                asset_classes = {
                    'Assets': self.stock_symbols,
                    'Class 1': self.classes,
                    }
                # print(asset_classes)
                asset_classes = pd.DataFrame(asset_classes)
                asset_classes = asset_classes.sort_values(by=['Assets'])
                constraints = {
                    'Disabled': [False, False, False, False, False, False],
                    'Type': ['All Assets','All Assets', 'Classes', 'Classes', 'Classes', 'Classes'],
                    'Set':['','','Class 1', 'Class 1','Class 1', 'Class 1'],
                    'Position': ['','','Equity', 'Equity','Commodities', 'Commodities'],
                    'Sign': ['<=', '>=', '<=', '>=', '<=', '>='],
                    'Weight': [self.general_weight_limits[1], self.general_weight_limits[0], self.equity_weight_limits[1], self.equity_weight_limits[0], self.commodities_weight_limits[1], self.commodities_weight_limits[0]],
                    'Type Relative': ['','','','','',''],
                    'Relative Set': ['','','','','',''],
                    'Relative': ['','','','','',''],
                    'Factor': ['','','','','','']
                    }
                constraints = pd.DataFrame(constraints)
                #print(constraints)
                A,B = rp.assets_constraints(constraints, asset_classes)
                # w_max, w_min = rp.hrp_constraints(constraints, asset_classes)
                # print(w_max)
                # print(w_min)
                port.ainequality = A
                port.binequality = B

                w = port.optimization(model=model, rm=rm, obj=obj, hist=hist, rf=rf, l=l)
                print(w)
                self.weights=w['weights'].tolist()
            else:
                self.weights=[1]

        with st.spinner('Plotting Portfolio Composition...'):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(self.stock_names, self.weights, color='blue')
            ax.set_title('Portfolio Composition')
            ax.set_ylabel('Weight')
            ax.set_xlabel('Stock')
            ax.set_xticklabels(self.stock_names, rotation=45)
            with st.expander('Portfolio Composition'):
                st.pyplot(fig)

        with st.spinner('Downloading data...'):
            # get the returns for the selected stocks
            returns = get_returns(self.stock_symbols, self.weights, self.start_date)
        with st.spinner('Plotting Performance Diagram...'):
            fig = qs.plots.snapshot(returns, title='Portfolio Returns', show=False)
            with st.expander('Performance Diagram'):
                st.pyplot(fig)

        with st.spinner('Calculating Portfolio Metrics...'):
            with st.expander('Metrics'):
                metrics = qs.reports.metrics(returns, mode='full', display=False)
                print(metrics.to_dict())
                col1,col2,col3=st.columns([1,1,2])
                col1.metric('Sharpe Ratio', metrics.loc['Sharpe'][0])
                col2.metric('Sortino Ratio', metrics.loc['Sortino'][0])
                perf_df = (metrics.loc[['YTD ', 'MTD ', '3M ', '6M ', '1Y ', '3Y (ann.) ', '5Y (ann.) ', '10Y (ann.) ']]*100).round(2)
                perf_df.columns = ['%']
                col3.table(perf_df)
                col1.metric('CAGR%', str(round(metrics.loc['CAGR﹪'][0]*100,2))+'%')
                col2.metric('Cumulative Return', str(round(metrics.loc['Cumulative Return '][0]*100,2))+'%')
                col1.metric('Volatility (ann.)', str(round(metrics.loc['Volatility (ann.) '][0]*100,2))+'%')
                col2.metric('Max Drawdown ', str(round(metrics.loc['Max Drawdown '][0]*100,2))+'%')
            
            fig = qs.plots.monthly_heatmap(returns, show=False)
            with st.expander('Monthly Returns'):
                st.pyplot(fig)
            