#!/usr/bin/env python

import numpy as np
#import joblib
from Heyu_get_12ECG_features import get_12ECG_features
from keras.models import load_model

#修改：
def run_12ECG_classifier(data,header_data,classes,model):  # 将分类代码添加到此函数

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    data = get_12ECG_features(data)   # 调用get_12ECG_features()函数返回处理完的单条数据
    #修改
    data_lead0 = data[0,:].reshape([1,3000,1])
    data_lead1 = data[1,:].reshape([1,3000,1])
    data_lead2 = data[2,:].reshape([1,3000,1])
    data_lead3 = data[3,:].reshape([1,3000,1])
    data_lead4 = data[4,:].reshape([1,3000,1])
    data_lead5 = data[5,:].reshape([1,3000,1])
    data_lead6 = data[6,:].reshape([1,3000,1])
    data_lead7 = data[7,:].reshape([1,3000,1])
    data_lead8 = data[8,:].reshape([1,3000,1])
    data_lead9 = data[9,:].reshape([1,3000,1])
    data_lead10 = data[10,:].reshape([1,3000,1])
    data_lead11 = data[11,:].reshape([1,3000,1])

    #label = model.predict([data_lead0, data_lead1, data_lead2, data_lead3, data_lead4, data_lead5, \
#                            data_lead6, data_lead7, data_lead8, data_lead9, data_lead10, data_lead11])
#                            
    score = model.predict([data_lead0, data_lead1, data_lead2, data_lead3, data_lead4, data_lead5, \
                            data_lead6, data_lead7, data_lead8, data_lead9, data_lead10, data_lead11])
    '''
    feats_reshape = features.reshape(1,-1)
    label = model.predict(feats_reshape)
    score = model.predict_proba(feats_reshape)
    '''
    label = np.argmax(score,axis=1)
#    label = label.astype('int64')
    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score
#修改
# data_lead0 = data[0,:].reshape([1,7500,1])
# data_lead1 = data[1,:].reshape([1,7500,1])
# data_lead2 = data[0,:].reshape([1,7500,1])
# data_lead3 = data[1,:].reshape([1,7500,1])
# data_lead4 = data[0,:].reshape([1,7500,1])
# data_lead5 = data[1,:].reshape([1,7500,1])
# data_lead6 = data[0,:].reshape([1,7500,1])
# data_lead7 = data[1,:].reshape([1,7500,1])
# data_lead8 = data[0,:].reshape([1,7500,1])
# data_lead9 = data[1,:].reshape([1,7500,1])
# data_lead10 = data[0,:].reshape([1,7500,1])
# data_lead11 = data[1,:].reshape([1,7500,1])

# label = model.predict([data_lead0, data_lead1, data_lead2, data_lead3, data_lead4, data_lead5, \
                        # data_lead6, data_lead7, data_lead8, data_lead9, data_lead10, data_lead11])
                        
# score = model.predict_proba([data_lead0, data_lead1, data_lead2, data_lead3, data_lead4, data_lead5, \
                        # data_lead6, data_lead7, data_lead8, data_lead9, data_lead10, data_lead11])

def load_12ECG_model():
    # load the model from disk 
    #filename='model.h5'
    loaded_model = load_model('model.h5')

    return loaded_model