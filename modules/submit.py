import os
import pandas as pd
import numpy as np
import zipfile
from tensorflow.keras.models import load_model
from configs.config import Config
from modules.dataset import pre_process
from modules.feature import mfcc_feature
from modules.model import CNNModel

def create_submission():
    test_df = pd.read_csv(
        str(Config.ROOT_TEST_DIR / "private_test_sample_submission.csv"))
    test_df['path'] = test_df['uuid'].apply(lambda x: str(
        Config.ROOT_TEST_DIR / f"private_test_audio_files/{x}.wav"))
    xmfcc = []
    for index, fname in enumerate(test_df["path"]):
        try:
            source, sr = pre_process(fname)
            mfcc = mfcc_feature(source, sr)
            mfcc = mfcc.reshape(-1,)
            xmfcc.append(mfcc)
        except:
            mfcc = np.zeros(2808)
            mfcc = mfcc.reshape(-1,)
            xmfcc.append(mfcc)

    mfcc_df = pd.DataFrame(xmfcc)
    X_test = mfcc_df.iloc[:, :].values.reshape(mfcc_df.shape[0], 13, -1)
    X_test = X_test[..., np.newaxis]
    model_list = os.listdir(Config.WEIGHT_PATH)
    res = np.zeros(X_test.shape[0])
    res = res[..., np.newaxis]
    input_shape = (X_test.shape[1],X_test.shape[2],1)
    cnn = CNNModel(input_shape)
    model = cnn.define()
    for name in model_list:
        model.load_weights(str(Config.WEIGHT_PATH/f"{name}"))
        res += model.predict(X_test)
    res /= len(model_list)
    submission = pd.DataFrame()
    submission["uuid"] = test_df["uuid"]
    submission["assessment_result"] = res
    submission.to_csv("results.csv", index=False)

    Config.SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(Config.SUBMISSION_PATH / "results.zip"), 'w') as zf:
        zf.write('results.csv')
    os.remove('results.csv')
