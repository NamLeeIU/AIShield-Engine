import os
from pydub import AudioSegment

from pydub import AudioSegment
import os
import librosa
import pandas as pd
import numpy as np
import malaya_speech
from pydub.silence import split_on_silence
import soundfile as sf
from keras.models import load_model
import os
from pydub import AudioSegment
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


def predict(df):
    meta_df = pd.DataFrame()
    meta_df["path"] = df["file_path"]
    try:
        source, sr = pre_process(meta_df.at[0, "path"])
    except:
        return 0.0
    mfcc = mfcc_feature(source, sr)
    mfcc = mfcc.reshape(-1,)
    test_df = mfcc.iloc[:, :]
    X_test = test_df.iloc[:, :].values.reshape(test_df.shape[0], 13, -1)
    X_test = X_test[..., np.newaxis]
    res = np.zeros(X_test.shape[0])
    res = res[..., np.newaxis]
    model_list = os.listdir(Config.WEIGHT_PATH)
    for name in model_list:
        model = load_model(str(Config.WEIGHT_PATH/f"{name}"))
        res += model.predict(X_test)
    res /= len(model_list)
    positive_proba = res
    return positive_proba[0]
