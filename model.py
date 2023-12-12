
import numpy as np
import pandas as pd
# import mne
from scipy import signal

# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM ,Add, Input, Reshape
from tensorflow.keras.layers import Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



subjects = ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12']

def ResBlock(X,f,filters):
    F1,F2,F3 = filters
    X_shortcut = X
    
    X = Conv1D(filters = F1,kernel_size = f,padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    
    X = Conv1D(filters = F2,kernel_size = f,padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    
    X = Conv1D(filters = F3,kernel_size =f,padding = 'same')(X)
    X = BatchNormalization()(X)
    
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    
    return X

def ResNet(n0,k1,k2,n1,n2,n3,n4,n5,n6):
    X_input = Input((18))
    
    X = Dense(n0)(X_input)
    X = Reshape((n0,1))(X)
    
    X = Conv1D(n1, k1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    X = AveragePooling1D(pool_size=(k2))(X)
    
    X = LSTM(n2,return_sequences = True)(X)
    
    X = ResBlock(X,k1,[n3,n3,n2])
    X = AveragePooling1D(pool_size=(k2))(X)
    
    X = Conv1D(n4, k1, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    X = AveragePooling1D(pool_size=(k2))(X)
        
    X = ResBlock(X,k1,[n5,n5,n4])
    X = AveragePooling1D(pool_size=(k2))(X)
    
    X = Conv1D(n6, k1, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    X = AveragePooling1D(pool_size=(k2))(X)
    
    X = Flatten()(X)
    X = Dense(18)(X)
    
    model = Model(inputs = X_input,outputs = X)
#     model.summary()
    
    return model

def split(Data):
    np.random.seed(42)
    Data = Data.sample(frac=1).reset_index(drop=True)
    n = Data.shape[0]
    n1 = int(n*0.8)
    n2 = int(n1+(n*0.1))
    train = Data.loc[0:n1:1,:]
    train.drop("Unnamed: 0",axis=1,inplace = True)
    val = Data.loc[n1+1:n2:1,:]
    val.drop("Unnamed: 0",axis=1,inplace = True)
    test = Data.loc[n2+1:n-1:1,:]
    test.drop("Unnamed: 0",axis=1,inplace = True)
    return (train,val,test)


pcc = dict()


    
    
for sub in subjects:
    ##############################################################################
    ############################Data Handling#########################################
    inputs = pd.read_csv(r'C:\Users\Lenovo\OneDriveIITDelhi\Desktop\SI_Shubh\Data\csv_data\raw_input_{}.csv'.format(sub))
    X_train,X_val,X_test = split(inputs)
    outputs = pd.read_csv(r'C:\Users\Lenovo\OneDriveIITDelhi\Desktop\SI_Shubh\Data\csv_data\ica_input_{}.csv'.format(sub))
    y_train,y_val,y_test = split(outputs)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    globals()[f"{sub}_test"] = pd.DataFrame(y_test)
    ###############################################################################
    ############### Resnet Model ##################################
    model = ResNet(128,3,2,64,64,32,128,64,256)
    tf.keras.backend.clear_session()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=['mse'])
    model_history=model.fit(X_train, y_train, batch_size=512, epochs=500, validation_data=(X_val, y_val),
                            callbacks=[callback])
    model.save('model_{}'.format(sub))
    ###############################################################################
    ############### PCC ##################################

    y_pred = np.asarray(model.predict(X_test))
    
    globals()[f"{sub}_Resnet_pred"] = pd.DataFrame(model.predict(X_test))
    
    y_measured_channels = []
    y_pred_channels = []
    for i in range(0,18):
        y_m_l = []
        y_pred_l = []
        for j in range(0,y_test.shape[0]):
            y_m_l.append(y_test[j][i])
            y_pred_l.append(y_pred[j][i])
        y_measured_channels.append(y_m_l)
        y_pred_channels.append(y_pred_l)
        
    pcc_subject = []
    for i in range(18):
        coeff = np.corrcoef(y_measured_channels[i],y_pred_channels[i])
        pcc_subject.append(coeff[0][1])
    pcc[sub] = pcc_subject

PCC_ALL = pd.DataFrame(pcc)

with pd.ExcelWriter("Tests_All_SUBJECTS.xlsx") as writer:
        for sub in subjects:
                globals()[f"{sub}_test"].to_excel(writer, sheet_name=(f"{sub}_test"))

with pd.ExcelWriter("Predictions_All_SUBJECTS.xlsx") as writer:
        for sub in subjects:
                globals()[f"{sub}_Resnet_pred"].to_excel(writer, sheet_name=(f"{sub}_Resnet_pred"))

with pd.ExcelWriter("PCC_ALL_SUBJECTS_RESNET.xlsx") as writer:  
        PCC_ALL.to_excel(writer, sheet_name='PCC_ALL_SUBJECTS_RESNET')