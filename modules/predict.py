import os
import librosa
import pandas as pd
import numpy as np
from keras.models import load_model



def prepare_test():
    mfcc = pd.read_csv("..test_data/public_features/mfcc_features.csv")
    test_df = mfcc.iloc[:, :]
    X_test = test_df.iloc[:,:].values.reshape(test_df.shape[0],13,-1)
    print(test_df.shape)
    X_test = X_test[...,np.newaxis]
    print(X_test.shape)
    return X_test

def predict(re_model,results):
  test_df = pd.read_csv("..data/public_metadata.csv")
  X_test = prepare_test()  
  preds = re_model.predict(X_test)
  submission = pd.DataFrame()
  submission["uuid"] = test_df["uuid"]
  submission["assessment_result"] = preds
  submission.to_csv(f"..test_results/{results}.csv", index=0)  
  
  
def predict_mean():
    test_df = pd.read_csv("..data/public_metadata.csv")
    X_test = prepare_test()
    res = np.zeros(X_test.shape[0])
    res = res[...,np.newaxis]
    print(res.shape)
    model_list = os.listdir("..test_weights/models/")

    for name in model_list:
        model = load_model("..test_weights/models/" + name)
        preds = model.predict(X_test)
        res += model.predict(X_test) 
    res /= len(model_list)
    submission = pd.DataFrame()
    submission["uuid"] = test_df["uuid"]
    submission["assessment_result"] = res
    return submission

  
if __name__ == "__main__":
    re_model = load_model("..test_weights/models/model-kfold-1.h5")
    predict(re_model,"result1")
    re_model = load_model("..test_weights/models/model-kfold-2.h5")
    predict(re_model,"result2")
    re_model = load_model("..test_weights/models/model-kfold-3.h5")
    predict(re_model,"result3")
    re_model = load_model("..test_weights/models/model-kfold-4.h5")
    predict(re_model,"result4")
    re_model = load_model("..test_weights/models/model-kfold-5.h5")
    predict(re_model,"result5")
    submission = predict_mean()
    submission.to_csv("..test_results/results_mean.csv", index=0)
    
