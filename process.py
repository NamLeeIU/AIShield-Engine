import gdown
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from configs.config import Config
from modules.dataset import pre_process
from modules.feature import mfcc_feature


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


def download_models():
    urls = {
        "model-kfold-1.h5": "1REzAtMyw7YE1g80JUUaEv92i4RVr1Mx4",
        "model-kfold-2.h5": "1JSZGuTbogZsMIF0Q9jQ9vY7Jo16fNQjS",
        "model-kfold-3.h5": "14tLtyF1OL5bDpwLw0I15bQjG_kT2K8bH",
        "model-kfold-4.h5": "1q7Ju7vCKb0d2GdgDw2_5xlGuvWTz_GXa",
        "model-kfold-5.h5": "1AlFNzlr-XSmyu20ozsDrX-2D4LkV8-y7",
    }
    for name, id in urls.items():
        if (not(os.path.isfile(str(Config.WEIGHT_PATH/f"{name}")))):
            url = f"https://drive.google.com/uc?id={id}"
            output = str(Config.WEIGHT_PATH/f"{name}")
            gdown.download(url, output, quiet=False)
            print(f"Loaded {name}")

init_model = True
def model_loading(init_model):
    if(init_model):
        model_list = os.listdir(Config.WEIGHT_PATH)
        loaded_model = []
        for name in model_list:
            model = load_model(str(Config.WEIGHT_PATH/f"{name}"))   
            loaded_model.append(model)
        init_model = False
        return loaded_model

def predict(df):
    download_models()
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
    loaded_model = model_loading(init_model)
    for model in loaded_model:
        res += model.predict(X_test)
    res /= len(loaded_model)
    positive_proba = res
    positive_proba = np.asscalar(positive_proba[0])
    return positive_proba
