import modules.financialmodelling as fm
import plotly.graph_objects as go
from functools import reduce
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np


class DataProcessing:
    def __init__(self):
        self.fmp_class=fm.FinancialModellingPrep()
        self.fmp_class.api_key="<<YOUR API KEY>>"

    def get_patches(self,df,col="long"):
        patches = []
        start_date = None
        for i, row in df.iterrows():
            if row[col]:
                if start_date is None:
                    start_date = row['date']
            else:
                if start_date is not None:
                    patches.append((start_date, row['date']))
                    start_date = None
        if start_date is not None:
            patches.append((start_date, df.iloc[-1]['date']))
        return patches

    def get_intersections(self,df):
        # find the intersections of the moving averages
        # and return the dates
        # return a list of dates
        intersections=[]
        last_row=None

        for i, row in df.iterrows():
            if last_row is not None:
                last_period=((last_row["sma_200"]>last_row["sma_50"] and last_row["sma_50"]>last_row["sma_21"]) or 
                            (last_row["sma_200"]<last_row["sma_50"] and last_row["sma_50"]<last_row["sma_21"]))
                
                current_period=((row["sma_200"]>row["sma_50"] and row["sma_50"]>row["sma_21"]) or 
                            (row["sma_200"]<row["sma_50"] and row["sma_50"]<row["sma_21"]))
                
                # if the last period is true and the current period is false
                # that means that the last period didnt follow the pattern
                # and the current period is following the pattern
                # so we have an intersection
                if last_period ==False and current_period==True:
                    intersections.append(row["date"])

            last_row=row


        return intersections

    def get_slope(self,df,col,period=2):
        # find the slope of the moving averages
        # return a list of slopes
        df[f'{col}_slope'] = df[col].rolling(period).apply(lambda x: np.polyfit(np.arange(period), x, 1)[0])

        return df

    def get_three_line_strikes(self,df):
        # apply three line strikes
        # if the last 4 candles are green and the last candle is red
        # then we have a three line strike
        # return a list of dates
        last_row=None

        df["candle_type"]=np.where(df["close"]>df["open"],"bullish","bearish")
        df["indicator_bullish_three_line_strike"]=None
        df["indicator_bearish_three_line_strike"]=None

        firstcandle=None
        secondcandle=None
        thirdcandle=None
        fourthcandle=None
        for i, row in df.iterrows():

            if firstcandle is not None and secondcandle is not None and thirdcandle is not None and fourthcandle is not None:
                if (firstcandle["candle_type"]=="bullish" and 
                    secondcandle["candle_type"]=="bullish" and 
                    thirdcandle["candle_type"]=="bullish" and 
                    fourthcandle["candle_type"]=="bearish" and 
                    firstcandle["close"]<secondcandle["close"] and
                    secondcandle["close"]<thirdcandle["close"] and
                    firstcandle["close"]>fourthcandle["close"]):

                    df["indicator_bullish_three_line_strike"].iloc[i]=True

                if (firstcandle["candle_type"]=="bearish" and 
                    secondcandle["candle_type"]=="bearish" and 
                    thirdcandle["candle_type"]=="bearish" and 
                    fourthcandle["candle_type"]=="bullish" and 
                    firstcandle["close"]>secondcandle["close"] and
                    secondcandle["close"]>thirdcandle["close"] and
                    firstcandle["close"]<fourthcandle["close"]):

                    df["indicator_bearish_three_line_strike"].iloc[i]=True

            if firstcandle is None:
                firstcandle=row["candle_type"]
            if firstcandle is not None:
                secondcandle=firstcandle
            if secondcandle is not None:
                thirdcandle=secondcandle
            if thirdcandle is not None:
                fourthcandle=thirdcandle



        return df



    def get_technical_analysis(self,ticker,from_date="2020-01-01",timestep="1day"):

        df_sma_200=self.fmp_class.technical_indicator(ticker=ticker,timestep=timestep,technical_indicator_type="sma",period=200,from_date=from_date)
        df_sma_50=self.fmp_class.technical_indicator(ticker=ticker,timestep=timestep,technical_indicator_type="sma",period=50,from_date=from_date)
        df_sma_21=self.fmp_class.technical_indicator(ticker=ticker,timestep=timestep,technical_indicator_type="sma",period=21,from_date=from_date)
        df_rsi=self.fmp_class.technical_indicator(ticker=ticker,timestep=timestep,technical_indicator_type="rsi",period=14,from_date=from_date)
        df_williams=self.fmp_class.technical_indicator(ticker=ticker,timestep=timestep,technical_indicator_type="williams",period=2,from_date=from_date)

        # dropping columns that are duplicated
        cols_to_drop=["open","close","high","low","volume"]

        df_sma_50.drop(columns=cols_to_drop,inplace=True)
        df_sma_21.drop(columns=cols_to_drop,inplace=True)
        df_rsi.drop(columns=cols_to_drop,inplace=True)
        df_williams.drop(columns=cols_to_drop,inplace=True)

        # renaming columns
        df_sma_200.rename(columns={"sma":"sma_200"},inplace=True)
        df_sma_50.rename(columns={"sma":"sma_50"},inplace=True)
        df_sma_21.rename(columns={"sma":"sma_21"},inplace=True)

        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), [df_sma_200, df_sma_50, df_sma_21,df_rsi,df_williams])

        # sorting the dataframe by date
        df_merged.sort_values(by=["date"],inplace=True)

        # calculating slope of the moving averages
        df_merged=self.get_slope(df_merged,"sma_200")
        df_merged=self.get_slope(df_merged,"sma_50")
        df_merged=self.get_slope(df_merged,"sma_21")

        return df_merged


    def apply_strategy_1(self,df):
        """
        Simple screening rules:

        Long position:
        1. All moving averages are sloping upwards
        2. The 21 day moving average is above the 50 day moving average
        3. The 50 day moving average is above the 200 day moving average
        4. The RSI is above 50
        5. The Williams %R is below -80 (oversold)
        6. The 200 day moving average is sloping upwards
        7. The 50 day moving average is sloping upwards
        8. The 21 day moving average is sloping upwards


        Short position:
        1. All moving averages are sloping downwards
        2. The 21 day moving average is below the 50 day moving average
        3. The 50 day moving average is below the 200 day moving average
        4. The RSI is below 50
        5. The Williams %R is above -20 (overbought)
        6. The 200 day moving average is sloping downwards
        7. The 50 day moving average is sloping downwards
        8. The 21 day moving average is sloping downwards

        """
        # apply screening rules
        df["long"] = ((df['sma_200'] < df['sma_50']) & 
                            (df['sma_50'] < df['sma_21']) & 
                            (df['rsi'] > 50) & 
                            (df['williams'] < -80) &
                            (df['sma_200_slope'] > 0) &
                            (df['sma_50_slope'] > 0) &
                            (df['sma_21_slope'] > 0))
        
        df["short"] = ((df['sma_200'] > df['sma_50']) & 
                            (df['sma_50'] > df['sma_21']) & 
                            (df['rsi'] < 50) & 
                            (df['williams'] > -20) &
                            (df['sma_200_slope'] < 0) &
                            (df['sma_50_slope'] < 0) &
                            (df['sma_21_slope'] < 0))

        return df

    def plot_chart(self,df,ticker,plot2="rsi",plot3="williams"):
        # apply date filter    
        fig = make_subplots(rows=3, cols=1,specs=[[{"type": "scatter"}], [{"type": "scatter"}],[{"type": "scatter"}]],shared_xaxes=True)

        chartfigure=go.Figure(data=[
                go.Candlestick(x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']),
                go.Scatter(x=df['date'],y=df['sma_200'],name="SMA 200"),
                go.Scatter(x=df['date'],y=df['sma_50'],name="SMA 50"),
                go.Scatter(x=df['date'],y=df['sma_21'],name="SMA 21")])

        for t in chartfigure.data:
            fig.append_trace(t, row=1, col=1)


        fig.add_trace(go.Scatter(x=df['date'],y=df[plot2],name=plot2), row=2, col=1),
        
        fig.add_trace(go.Scatter(x=df['date'],y=df[plot3],name=plot3), row=3, col=1)
        
        # apply patches
        for patch in self.get_patches(df,"long"):
            fig.add_shape(type="rect",
                xref="x",
                yref="paper",
                x0=patch[0],
                y0=0,
                x1=patch[1],
                y1=1,
                fillcolor="green",
                opacity=0.5,
                layer="below",
                line_width=0,
                name="Long")

        for patch in self.get_patches(df,"short"):
            fig.add_shape(type="rect",
                xref="x",
                yref="paper",
                x0=patch[0],
                y0=0,
                x1=patch[1],
                y1=1,
                fillcolor="red",
                opacity=0.5,
                layer="below",
                line_width=0,
                name="Long")

        for intersection in self.get_intersections(df):
            fig.add_vline(type="rect",
                x=intersection,
                fillcolor="black",
                layer="below",
                line_width=3,
                name="Intersection")


        fig.update_layout(
            yaxis=dict(
                autorange = True,
                fixedrange= False
            ))

        fig.update_xaxes(rangeslider_visible=False)
        
        # adding a buy today text to the figure based on the first row
        title="Do nothing today on {}".format(ticker)
        action=0

        if df.iloc[-1]["long"]:
            title = "Long today on {}".format(ticker)
            fig.add_annotation(x=df.iloc[-1]["date"], y=0.9, text="Long today", showarrow=True, arrowhead=1)
            action=1
        elif df.iloc[-1]["short"]:
            title = "Short today on {}".format(ticker)
            fig.add_annotation(x=df.iloc[-1]["date"], y=0.9, text="Short today", showarrow=True, arrowhead=1)
            action=-1
        
        fig.update_layout(title=title)
        return fig,title,action