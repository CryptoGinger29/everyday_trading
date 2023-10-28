
import modules.dataprocessing as dp
import plotly.graph_objects as go

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, LSTM,Flatten,TimeDistributed
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.express as px

from sklearn.decomposition import PCA

class TradeModelling:
    def __init__(self):
        self.dp_class=dp.DataProcessing()
        self.ticker="NOVO-B.CO"
        self.df=None
        self.dataset=None
        self.lookback=10 # svarende til 2 arbejdsuger, størrelsen på input dataen
        self.tradinghorizon=5 # svarende til 1 arbejdsuge, maks leve tid for en trade

        self.stoploss=0.01
        self.takeprofit=0.02

        self.X_values=[]
        self.y_values=[]
        self.unscaled_X_values=[]

        self.X=None
        self.y=None
        self.unscaled_X=None

        self.model=None
        self.latentmodel=None
        self.kmeans=None
        self.timestep="1day"
        self.from_date="2011-01-01"
        self.epoch=1000
        
    def dataload(self):
        self.df=self.dp_class.get_technical_analysis(self.ticker,timestep=self.timestep,from_date=self.from_date)

        # find and print all rows with NaN values
        temp_df=self.df.copy()
        temp_df.drop("date",axis=1,inplace=True)
        temp_df.dropna(inplace=True)

        scaler = StandardScaler()

        for i in range(self.lookback,len(temp_df)):
            unscaledvalues=temp_df.iloc[i-self.lookback:i].values
            values=scaler.fit_transform(unscaledvalues)

            tradinghorizon=temp_df.iloc[i:i+self.tradinghorizon].copy()
            # looper frem i tiden for at finde ud af om traden først møder stop loss eller take profit

            startprice=temp_df.iloc[i]["close"]

            stoplossprice=startprice*(1-self.stoploss)
            takeprofitprice=startprice*(1+self.takeprofit)

            action=0
            for j,row in tradinghorizon.iterrows():

                if stoplossprice>row["low"]:
                    break
                elif takeprofitprice<row["high"]:
                    action=1
                    break
            
            
            self.unscaled_X_values.append(unscaledvalues)
            self.X_values.append(values)
            self.y_values.append([action])
        
        self.unscaled_X=np.array(self.unscaled_X_values)
        self.X=np.array(self.X_values)
        self.y=np.array(self.y_values)


    def createdataset(self):
        self.dataset=self.dp_class.apply_strategy_1(self.df)

    def layers(self,input):
        encoder = TimeDistributed(Dense(200, activation='tanh'))(input)
        encoder = TimeDistributed(Dense(50, activation='tanh'))(encoder)
        encoder = TimeDistributed(Dense(10, activation='tanh'))(encoder)
        latent= Flatten()(encoder)
        decoder = TimeDistributed(Dense(50, activation='tanh'))(encoder)
        decoder = TimeDistributed(Dense(200, activation='tanh'))(decoder)
        out = TimeDistributed(Dense(self.X.shape[2]))(decoder)

        return out,latent
    
    def trainautoencodermodel(self):

        es = EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True, verbose=1, patience=0)

        input = Input(shape=(self.X.shape[1:]))
        model_layers,__=self.layers(input)
        
        model = Model(inputs=input, outputs=model_layers)

        model.compile(optimizer=Adam(learning_rate=0.00005), loss="mse")
        model.summary()

        X_train, X_test, _, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


        model.fit(X_train,X_train,epochs=self.epoch,verbose=2,batch_size=32,callbacks=[es],validation_data=(X_test,X_test))

        self.model=model

        return model
    
    def setuplatentmodel(self):
        input = Input(shape=(self.X.shape[1:]))
        _,latent=self.layers(input)

        model = Model(inputs=input, outputs=latent)

        model.compile(optimizer=Adam(learning_rate=0.00005), loss="mse")
        model.summary()

        for layer1, layer2 in zip(self.model.layers, model.layers):
            if layer1.name==layer2.name:
                layer2.set_weights(layer1.get_weights())

        self.latentmodel=model

        return model


    def kmeansoptimization(self):
        for n_cluster in range(2, 10):
            kmeans = KMeans(n_clusters=n_cluster, random_state=0)
            kmeans.fit(self.X)
            print(n_cluster, silhouette_score(self.X, kmeans.labels_))


    def pattern_recognition(self):
        predictions = self.model.predict(self.X)
        reduced_dimensions=self.latentmodel.predict(self.X)

        bestscore=9999999
        bestclustercount=0
        scores=[]
        for n_cluster in range(5, 25):
            kmeans = KMeans(n_clusters=n_cluster, random_state=0,n_init=10)
            kmeans.fit(reduced_dimensions)

            score=silhouette_score(reduced_dimensions, kmeans.labels_)
            scores.append(score)
            if score<bestscore:
                bestscore=score
                bestclustercount=n_cluster


        fig=px.line(x=range(len(scores)),y=scores)
        fig.show()

        # Define the number of components you want to keep (e.g., 2 for 2D visualization)
        n_components = 2

        # Initialize the PCA model
        pca = PCA(n_components=n_components)

        # Fit the PCA model to your data and transform the data
        data_pca = pca.fit_transform(reduced_dimensions)


        self.kmeans = KMeans(n_clusters=bestclustercount, random_state=0,n_init=10)
        cluster_labels = kmeans.fit_predict(data_pca)   

        cluser_labels_str=[str(x) for x in cluster_labels]

        fig=px.scatter(x=data_pca[:,0],y=data_pca[:,1],color=cluser_labels_str)
        fig.show()

        fig=px.scatter(x=data_pca[:,0],y=data_pca[:,1],color=self.y[:,0])
        fig.show()

        # investiagte the clusters
        # figure out if the distribution of the y-values are different in the clusters
        # find clusters where number of y value distribution is skwed
        means=[]
        for i in range(bestclustercount):
            clusterdata=self.y[cluster_labels==i]
            means.append(clusterdata.mean(axis=0))

        means=np.array(means)

        YMEAN=self.y.mean(axis=0)[0]
        ULIM=YMEAN+means.std(axis=0)[0]
        LLIM=YMEAN-means.std(axis=0)[0]

        fig=px.bar(x=range(bestclustercount),y=means[:,0],labels={"x":"cluster","y":"mean of y"})
        fig.add_hline(y=YMEAN,annotation_text="mean of y")
        fig.add_hline(y=ULIM,line_dash="dash",annotation_text="mean of y")
        fig.add_hline(y=LLIM,line_dash="dash",annotation_text="mean of y")

        fig.show()

        # find the cluster with the highest mean of y
        for i in range(bestclustercount):
            if means[i,0]>ULIM:
                # kigger på open price
                open=self.unscaled_X[cluster_labels==i][0,:,0]
                high=self.unscaled_X[cluster_labels==i][0,:,1]
                low=self.unscaled_X[cluster_labels==i][0,:,2]
                close=self.unscaled_X[cluster_labels==i][0,:,3]

                sma200=self.unscaled_X[cluster_labels==i][0,:,5]
                sma50=self.unscaled_X[cluster_labels==i][0,:,6]
                sma21=self.unscaled_X[cluster_labels==i][0,:,7]

                fig=go.Figure(data=[go.Candlestick(x=np.arange(self.X.shape[1]),open=open,high=high,low=low,close=close)])

                fig.add_trace(go.Scatter(x=np.arange(self.X.shape[1]),y=sma200,name="sma200"))
                fig.add_trace(go.Scatter(x=np.arange(self.X.shape[1]),y=sma50,name="sma50"))
                fig.add_trace(go.Scatter(x=np.arange(self.X.shape[1]),y=sma21,name="sma21"))
                
                fig.show()