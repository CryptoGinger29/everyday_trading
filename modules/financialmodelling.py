import requests
import pandas as pd

class InvalidTechnicalIndicator(Exception):
    pass

class InvalidTimestep(Exception):
    pass

class InvalidEndpoint(Exception):
    pass

class FinancialModellingPrep:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.version="v3"
        self.timesteps=["1min", "5min", "15min", "30min", "1hr", "4hr", "1day"]
        self.endpoints=["technical_indicator","historical_chart"]
        self.baseurl="https://financialmodelingprep.com/api/{version}/{endpoint}/{timestep}/{ticker}"
        self.technical_indicator_types=["sma","ema","wma","dema","tema","williams","rsi"]

    # decorator to check the syntax on the input to the function
    def _syntaxtest(func):
        def inputcheck(self,*args, **kw):
            if "timestemp" in kw:
                if kw["timestep"] not in self.timesteps:
                    raise InvalidTimestep("Invalid timestep")
            
            if "technical_indicator_type" in kw:
                if kw["technical_indicator_type"] not in self.technical_indicator_types:
                    raise InvalidTechnicalIndicator("Invalid technical indicator")
            
            return func(self,*args, **kw)
        return inputcheck
    
    def format_df(self,r):
        df=pd.json_normalize(r.json())
        df["date"]=pd.to_datetime(df["date"])
        return df


    @_syntaxtest
    def  technical_indicator(self,ticker=None,technical_indicator_type=None,timestep=None,endpoint="technical_indicator",period=None,from_date=None):
        url=self.baseurl.format(version=self.version,endpoint=endpoint,timestep=timestep,ticker=ticker)

        r=requests.get(url,params={"apikey":self.api_key,"type":technical_indicator_type,"period":period,"from":from_date})

        return self.format_df(r)


    @_syntaxtest
    def chart(self,ticker=None,timestep=None,endpoint="chart"):
        url=self.baseurl.format(version=self.version,endpoint=endpoint,timestep=timestep,ticker=ticker)

        r=requests.get(url,params={"apikey":self.api_key})
        return url,r.json()