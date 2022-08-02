import scipy
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from tflite_runtime.interpreter import Interpreter
import tqdm
import numpy as np
import pandas as pd 
import pickle

with open('/home/pi/tesr/X_test_0802.pickle', 'rb') as f:
    X_test = pickle.load(f)

with open('/home/pi/tesr/X_valid_label_0802.pickle', 'rb') as f:
    X_valid_label = pickle.load(f)
    
model_path='/home/pi/tesr/AE_974_0802.tflite'    

if __name__=='__main__':
    
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    prediction_digits = np.empty((0,21))

    for i in tqdm.tqdm(range(len(X_test))):
        test = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test)

        interpreter.invoke()

        output = interpreter.tensor(output_index)
        asd = np.array(output())
        prediction_digits= np.append(prediction_digits,asd,axis=0)


    X_pred = prediction_digits
    X_pred = pd.DataFrame(X_pred)

    scored = pd.DataFrame()
    #Xtest = X_test.reshape(X_pred.shape[0], X_pred.shape[1])
    X_test = pd.DataFrame(X_test)
    # scored['loss mae'] = np.mean(np.abs(X_pred-Xtest), axis=1)
    scored['loss mae'] = np.mean(np.abs(X_pred-X_test), axis=1)
    scored['threshold'] = np.median(scored['loss mae'])
    scored['anomaly'] = scored['loss mae'] > scored['threshold']
    scored.head()

    scored = scored.reset_index()

    recall_list = []
    precision_list =[]
    thresh_list = []
    acc_list = []
    f1_list = []
    thresh = np.mean(np.mean(scored['loss mae']))
    label_val_window = X_valid_label
    for i in tqdm.tqdm(range(500)):
        vv_thresh = i*(thresh/100)

        label_pred = []
        for j in range(len(label_val_window)):
            if scored['loss mae'][j] >= vv_thresh:
                label_pred.append(1)
            else:
                label_pred.append(0)
        recall = recall_score(label_val_window, label_pred)+0.0001
        precision = precision_score(label_val_window, label_pred, zero_division=1)+0.0001

        acc = accuracy_score(label_val_window, label_pred)+0.0001
        f1 = f1_score(label_val_window, label_pred)+0.0001
        recall_list.append(recall)
        precision_list.append(precision)
        acc_list.append(acc)
        f1_list.append(f1)
        thresh_list.append(vv_thresh)
        
    total = []
    total.append(acc_list)
    total.append(recall_list)
    total.append(precision_list)
    total.append(f1_list)
    total.append(thresh_list)
    total = pd.DataFrame(total).T
    total.columns = ['acc', 'recall', 'precision', 'f1', 'thresh_pair_wise']
    #total.to_csv('/home/ubuntu/internship/hankyung/3. totalnormal/dtw*critic.csv')
        
    import matplotlib.pyplot as plt
    plt.plot(thresh_list,acc_list,'g',label='acc')    
    plt.plot(thresh_list,recall_list,'b',label='recall')
    plt.plot(thresh_list,precision_list,'r',label='precision')
    plt.plot(thresh_list,f1_list, 'y', label='f1')
    plt.xlabel('thresh axis')
    plt.ylabel('recall,precision axis')
    plt.title('sample')
    plt.legend(loc='upper right')
    print("기준 threshold: {}".format(thresh))
    print(round(total[total['f1']==np.max(total['f1'])],3))
    print(total.loc[total[abs(total['recall']-total['precision']) == np.min(abs(total['recall'] - total['precision'])) ].index])
    plt.show()   

