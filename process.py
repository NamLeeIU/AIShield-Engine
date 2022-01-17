import gdown
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from configs.config import Config
from modules.dataset import pre_process
from modules.feature import mfcc_feature
from modules.model import CNNModel


def convert_to_wav(file_path):
    """
    This function is to convert an audio file to .wav file

    Args:
        file_path (str): paths of audio file needed to be convert to .wav file

    Returns:
        new path of .wav file
    """
    ext = file_path.split(".")[-1]
    assert ext in [
        "mp4", "mp3", "acc"], "The current API does not support handling {} files".format(ext)

    sound = AudioSegment.from_file(file_path, ext)
    wav_file_path = ".".join(file_path.split(".")[:-1]) + ".wav"
    sound.export(wav_file_path, format="wav")

    os.remove(file_path)
    return wav_file_path


urls = {
    "model-kfold-1.h5": "1uY5Lxz5JmKdHHtv9ABk7KQNlrdxFpqJI",
    "model-kfold-2.h5": "1r87t8-LvNK6ydqy2LImyBadfKCJyhRHG",
    "model-kfold-3.h5": "1eQVZBEE1FC6VbC1BK5IhJw_MwUCD58wC",
    "model-kfold-4.h5": "18JVZ8qmC2iT8-jA7TifdnOtzqXFLlt7P",
    "model-kfold-5.h5": "19mSvojmm5Ye6l2vi2-hYKOapM7Vd3p6a",
}

for name, id in urls.items():
    if (not(os.path.isfile(str(Config.WEIGHT_PATH/f"{name}")))):
        url = f"https://drive.google.com/uc?id={id}"
        output = str(Config.WEIGHT_PATH/f"{name}")
        gdown.download(url, output, quiet=False)
        print(f"Loaded {name}")

model_list = os.listdir(Config.WEIGHT_PATH)
loaded_model = []
input_shape = (13, 216, 1)
cnn = CNNModel(input_shape)
model = cnn.define()
for name in model_list:
    model.load_weights(str(Config.WEIGHT_PATH/f"{name}"))
    loaded_model.append(model)


def predict(df):
    global loaded_model
    meta_df = pd.DataFrame()
    meta_df["path"] = df["file_path"]
    xmfcc = []
    try:
        source, sr = pre_process(meta_df.at[0, "path"])
        mfcc = mfcc_feature(source, sr)
        mfcc = mfcc.reshape(-1,)
        xmfcc.append(mfcc)
    except:
        return -1
    mfcc_df = pd.DataFrame(xmfcc)
    X_test = mfcc_df.iloc[:, :].values.reshape(mfcc_df.shape[0], 13, -1)
    X_test = X_test[..., np.newaxis]
    res = np.zeros(X_test.shape[0])
    res = res[..., np.newaxis]
    for model in loaded_model:
        res += model.predict(X_test)
    res /= len(loaded_model)
    positive_proba = res
    positive_proba = np.asscalar(positive_proba[0])
    return positive_proba
