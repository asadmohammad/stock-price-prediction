# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 19:02:38 2021

@author: Mohammad Asad
"""

#Importing all important libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
#from pmdarima.arima import auto_arima
import math
#from nsepy import get_history
from numpy import array , hstack







@st.cache
def read_data(path):
    df = pd.read_excel(path,engine = 'openpyxl')
    return df

# Function to Cast Str Column to DateTime
def Cast_Str_To_DateTime(self_Df):
    self_Df.Date = pd.to_datetime(self_Df.Date, format = '%d-%b-%y')
    self_Df = self_Df.sort_values(by=['Date'])
    return self_Df


def Interpolate(series,method):
    series = series.interpolate(method=method)
    return series


def preprocessing_for_prediction(data,AORD,DAX,CAC,FTSE,HSI,NASDAQ,SNP,KOSPI,SHCOMP,NIKKEI):
    # convert to [rows, columns] structure
    AORD = AORD.reshape((len(AORD), 1))
    DAX = DAX.reshape((len(DAX), 1))
    CAC = CAC.reshape((len(CAC), 1))
    FTSE = FTSE.reshape((len(FTSE), 1))
    HSI = HSI.reshape((len(HSI), 1))
    NASDAQ = NASDAQ.reshape((len(NASDAQ), 1))
    SNP = SNP.reshape((len(SNP), 1))
    KOSPI = KOSPI.reshape((len(KOSPI), 1))
    SHCOMP = SHCOMP.reshape((len(SHCOMP), 1))
    NIKKEI = NIKKEI.reshape((len(NIKKEI), 1))
    # normalization features
    scaler = MinMaxScaler(feature_range=(0, 1))    
    AORD_scaled = scaler.fit_transform(AORD)
    DAX_scaled = scaler.fit_transform(DAX)
    CAC_scaled = scaler.fit_transform(CAC)
    FTSE_scaled = scaler.fit_transform(FTSE)
    HSI_scaled = scaler.fit_transform(HSI)
    NASDAQ_scaled = scaler.fit_transform(NASDAQ)
    SNP_scaled = scaler.fit_transform(SNP)
    KOSPI_scaled = scaler.fit_transform(KOSPI)
    SHCOMP_scaled = scaler.fit_transform(SHCOMP)
    NIKKEI_scaled = scaler.fit_transform(NIKKEI)
    
    # horizontally stack columns
    dataset_stacked = hstack((DAX_scaled, CAC_scaled,FTSE_scaled,HSI_scaled,NASDAQ_scaled,SNP_scaled,KOSPI_scaled,KOSPI_scaled,NIKKEI_scaled))
    return AORD_scaled,DAX_scaled,CAC_scaled,FTSE_scaled,HSI_scaled,NASDAQ_scaled,SNP_scaled,KOSPI_scaled,SHCOMP_scaled,NIKKEI_scaled, dataset_stacked

def preprocessing(data,AORD,DAX,CAC,FTSE,HSI,NASDAQ,SNP,KOSPI,SHCOMP,NIKKEI):
    # convert to [rows, columns] structure
    AORD = AORD.reshape((len(AORD), 1))
    DAX = DAX.reshape((len(DAX), 1))
    CAC = CAC.reshape((len(CAC), 1))
    FTSE = FTSE.reshape((len(FTSE), 1))
    HSI = HSI.reshape((len(HSI), 1))
    NASDAQ = NASDAQ.reshape((len(NASDAQ), 1))
    SNP = SNP.reshape((len(SNP), 1))
    KOSPI = KOSPI.reshape((len(KOSPI), 1))
    SHCOMP = SHCOMP.reshape((len(SHCOMP), 1))
    NIKKEI = NIKKEI.reshape((len(NIKKEI), 1))
    # normalization features
    scaler = MinMaxScaler(feature_range=(0, 1))    
    AORD_scaled = scaler.fit_transform(AORD)
    DAX_scaled = scaler.fit_transform(DAX)
    CAC_scaled = scaler.fit_transform(CAC)
    FTSE_scaled = scaler.fit_transform(FTSE)
    HSI_scaled = scaler.fit_transform(HSI)
    NASDAQ_scaled = scaler.fit_transform(NASDAQ)
    SNP_scaled = scaler.fit_transform(SNP)
    KOSPI_scaled = scaler.fit_transform(KOSPI)
    SHCOMP_scaled = scaler.fit_transform(SHCOMP)
    NIKKEI_scaled = scaler.fit_transform(NIKKEI)
    
    # horizontally stack columns
    dataset_stacked = hstack((DAX_scaled, CAC_scaled,FTSE_scaled,HSI_scaled,NASDAQ_scaled,SNP_scaled,KOSPI_scaled,KOSPI_scaled,NIKKEI_scaled,AORD_scaled))
    return AORD_scaled,DAX_scaled,CAC_scaled,FTSE_scaled,HSI_scaled,NASDAQ_scaled,SNP_scaled,KOSPI_scaled,SHCOMP_scaled,NIKKEI_scaled, dataset_stacked

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
       	# find the end of this pattern
       	end_ix = i + n_steps_in
       	out_end_ix = end_ix + n_steps_out-1
       	# check if we are beyond the dataset
       	if out_end_ix > len(sequences):
       		break
       	# gather input and output parts of the pattern
       	seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
       	X.append(seq_x)
       	y.append(seq_y)
    return array(X), array(y)


@st.cache(allow_output_mutation=True)
def build_lstm(train_X,train_y,test_X, test_y,n_steps_in,n_steps_out,n_features):
    #optimizer learning rate
    opt = keras.optimizers.Adam(learning_rate=0.01)
    
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.add(Activation('linear'))
    model.compile(loss='mse' , optimizer=opt , metrics=['mse'])
    
    history = model.fit(train_X , train_y , epochs=10 , steps_per_epoch=25 , verbose=1 ,validation_data=(test_X, test_y) ,shuffle=False)
    return history,model


def main():
    st.title("Stock Price Prediction")
    path = 'stock_price_brian_1.1.xlsx'
    data = read_data(path)
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file,engine = 'openpyxl')
    
    if st.checkbox('Show Raw Data'):
        st.subheader("Showing raw data---->>>")
        st.write(data.head())
    
    ### Cast str to DateTime ###
    data = Cast_Str_To_DateTime(data)
    data.set_index('Date',inplace=True)

    ### Change columns type from 'Str' to 'Numeric'
    data["DAX"] = pd.to_numeric(data["DAX"], downcast="float",errors='coerce')
    data["CAC"] = pd.to_numeric(data["CAC"], downcast="float",errors='coerce')
    data["FTSE"] = pd.to_numeric(data["FTSE"], downcast="float",errors='coerce')

    ### Interpolate missing values
    data.DAX = Interpolate(data.DAX,'linear')
    data.CAC = Interpolate(data.CAC,'linear')
    data.FTSE = Interpolate(data.FTSE,'linear')
    for i in data.columns:
        a = data[i].values
        globals()[i] = a
    SNP = data['S&P'].values    
    AORD_scaled,DAX_scaled,CAC_scaled,FTSE_scaled,HSI_scaled,NASDAQ_scaled,SNP_scaled,KOSPI_scaled,SHCOMP_scaled,NIKKEI_scaled,dataset_stacked = preprocessing(data,AORD,DAX,CAC,FTSE,HSI,NASDAQ,SNP,KOSPI,SHCOMP,NIKKEI)
    #ML Section 
    
    choose_model = st.sidebar.selectbox("Choose the Prediction Window",["None","Seventh Day","Thirtieth Day","Ninetieth Day"])
    if(choose_model == "Seventh Day"):
        #choose_model  == "Seven"
        # choose a number of time steps #change this accordingly
        n_steps_in_07, n_steps_out_07 = 30 , 7
        
        data_to_predict_07 = data.tail(n_steps_in_07)
        
        for i in data_to_predict_07.columns:
            a = data_to_predict_07[i].values
            globals()[i + '_07'] = a
        SNP_07 = data_to_predict_07['S&P'].values
        
        AORD_scaled_07,DAX_scaled_07,CAC_scaled_07,FTSE_scaled_07,HSI_scaled_07,NASDAQ_scaled_07,SNP_scaled_07,KOSPI_scaled_07,SHCOMP_scaled_07,NIKKEI_scaled_07,dataset_stacked_07 = preprocessing_for_prediction(data_to_predict_07,AORD_07,DAX_07,CAC_07,FTSE_07,HSI_07,NASDAQ_07,SNP_07,KOSPI_07,SHCOMP_07,NIKKEI_07)
        
        
        dataset_stacked_07_new = dataset_stacked_07.reshape(1, dataset_stacked_07.shape[0] , dataset_stacked_07.shape[1])
        
        # covert into input/output
        X_07, y_07 = split_sequences(dataset_stacked, n_steps_in_07, n_steps_out_07)
        

        split = math.ceil(len(X_07)*0.7)
        train_X_07 , train_y_07 = X_07[:split, :] , y_07[:split, :]
        test_X_07 , test_y_07 = X_07[split:, :] , y_07[split:, :]
    
        n_features_07 = train_X_07.shape[2]
        
        """## **Modeliing**"""
        history_07, model_07 = build_lstm(train_X_07,train_y_07,test_X_07, test_y_07,n_steps_in_07,n_steps_out_07,n_features_07)
        
        
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        y_07 = data_to_predict_07['AORD'].values
        y_07 = y_07.reshape((len(y_07), 1))
        scaler1.fit(y_07)
        
        y_pred_07 = model_07.predict(dataset_stacked_07_new)
        y_pred_inv_07 = scaler1.inverse_transform(y_pred_07)
        y_pred_inv_07 = y_pred_inv_07.reshape(n_steps_out_07,1)
        y_pred_inv_07 = y_pred_inv_07[:,0]
        st.write('07th day prediction is: ')
        st.write(y_pred_inv_07[-1])
        
    elif(choose_model == "Thirtieth Day"):
        n_steps_in_30, n_steps_out_30 = 90 , 30
        
        data_to_predict_30 = data.tail(n_steps_in_30)
        
        for i in data_to_predict_30.columns:
            a = data_to_predict_30[i].values
            globals()[i + '_30'] = a
        SNP_30 = data_to_predict_30['S&P'].values
        
        AORD_scaled_30,DAX_scaled_30,CAC_scaled_30,FTSE_scaled_30,HSI_scaled_30,NASDAQ_scaled_30,SNP_scaled_30,KOSPI_scaled_30,SHCOMP_scaled_30,NIKKEI_scaled_30,dataset_stacked_30 = preprocessing_for_prediction(data_to_predict_30,AORD_30,DAX_30,CAC_30,FTSE_30,HSI_30,NASDAQ_30,SNP_30,KOSPI_30,SHCOMP_30,NIKKEI_30)
        
        
        dataset_stacked_30_new = dataset_stacked_30.reshape(1, dataset_stacked_30.shape[0] , dataset_stacked_30.shape[1])
        
        # covert into input/output
        X_30, y_30 = split_sequences(dataset_stacked, n_steps_in_30, n_steps_out_30)
        

        split = math.ceil(len(X_30)*0.7)
        train_X_30 , train_y_30 = X_30[:split, :] , y_30[:split, :]
        test_X_30 , test_y_30 = X_30[split:, :] , y_30[split:, :]
    
        n_features_30 = train_X_30.shape[2]
        
        """## **Modeliing**"""
        history_30, model_30 = build_lstm(train_X_30,train_y_30,test_X_30, test_y_30,n_steps_in_30,n_steps_out_30,n_features_30)
        
        
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        y_30 = data_to_predict_30['AORD'].values
        y_30 = y_30.reshape((len(y_30), 1))
        scaler1.fit(y_30)
        
        y_pred_30 = model_30.predict(dataset_stacked_30_new)
        y_pred_inv_30 = scaler1.inverse_transform(y_pred_30)
        y_pred_inv_30 = y_pred_inv_30.reshape(n_steps_out_30,1)
        y_pred_inv_30 = y_pred_inv_30[:,0]
        st.write('30th day prediction is: ')
        st.write(y_pred_inv_30[-1])
        
        
    elif(choose_model == "Ninetieth Day"):
        n_steps_in_90, n_steps_out_90 = 180 , 90
        
        data_to_predict_90 = data.tail(n_steps_in_90)
        
        for i in data_to_predict_90.columns:
            a = data_to_predict_90[i].values
            globals()[i + '_90'] = a
        SNP_90 = data_to_predict_90['S&P'].values
        
        AORD_scaled_90,DAX_scaled_90,CAC_scaled_90,FTSE_scaled_90,HSI_scaled_90,NASDAQ_scaled_90,SNP_scaled_90,KOSPI_scaled_90,SHCOMP_scaled_90,NIKKEI_scaled_90,dataset_stacked_90 = preprocessing_for_prediction(data_to_predict_90,AORD_90,DAX_90,CAC_90,FTSE_90,HSI_90,NASDAQ_90,SNP_90,KOSPI_90,SHCOMP_90,NIKKEI_90)
        
        
        dataset_stacked_90_new = dataset_stacked_90.reshape(1, dataset_stacked_90.shape[0] , dataset_stacked_90.shape[1])
        
        # covert into input/output
        X_90, y_90 = split_sequences(dataset_stacked, n_steps_in_90, n_steps_out_90)
        

        split = math.ceil(len(X_90)*0.7)
        train_X_90 , train_y_90 = X_90[:split, :] , y_90[:split, :]
        test_X_90 , test_y_90 = X_90[split:, :] , y_90[split:, :]
    
        n_features_90 = train_X_90.shape[2]
        
        """## **Modeliing**"""
        history_90, model_90 = build_lstm(train_X_90,train_y_90,test_X_90, test_y_90,n_steps_in_90,n_steps_out_90,n_features_90)
        
        
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        y_90 = data_to_predict_90['AORD'].values
        y_90 = y_90.reshape((len(y_90), 1))
        scaler1.fit(y_90)
        
        y_pred_90 = model_90.predict(dataset_stacked_90_new)
        y_pred_inv_90 = scaler1.inverse_transform(y_pred_90)
        y_pred_inv_90 = y_pred_inv_90.reshape(n_steps_out_90,1)
        y_pred_inv_90 = y_pred_inv_90[:,0]
        st.write('90th day prediction is: ')
        st.write(y_pred_inv_90[-1])
    
    st.write('Feature Importance')
    
    X_For_Feature_Importance = data
    y_For_Feature_Importance = X_For_Feature_Importance.pop('AORD')
    
    model_For_Feature_Importance = RandomForestRegressor()
    model_For_Feature_Importance.fit(X_For_Feature_Importance, y_For_Feature_Importance)
    
    feature_importance = pd.Series(model_For_Feature_Importance.feature_importances_, index=X_For_Feature_Importance.columns)
#    feature_importance_plot = feature_importance.nlargest(10)
#    st.pyplot(fig = feature_importance_plot)
    
    fig, ax = plt.subplots()
    feature_importance.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.pyplot(fig)


        
        
        

if __name__ == "__main__":
	main()

    
