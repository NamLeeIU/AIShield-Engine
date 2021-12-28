import librosa
import pandas as pd
import numpy as np
import malaya_speech
from pydub import AudioSegment
from pydub.silence import split_on_silence
from modules.feature import mfcc_feature


def zero_pad(y, sr):
    if y.shape[0] <= 5*sr:
        pad_width = 5*sr - y.shape[0]
        y = np.pad(y, pad_width=((0, pad_width)), mode='constant')
    return y


def pre_process(fname):
    y, sr = librosa.load(fname, res_type='kaiser_fast')
    y_ = librosa.effects.trim(y, top_db=20)[0]
    y_int = malaya_speech.astype.float_to_int(y)
    audio = AudioSegment(
        y_int.tobytes(),
        frame_rate=sr,
        sample_width=y_int.dtype.itemsize,
        channels=1
    )
    audio_chunks = split_on_silence(
        audio,
        min_silence_len=200,
        silence_thresh=-30,
        keep_silence=100,
    )
    y_ = sum(audio_chunks)
    y_ = np.array(y_.get_array_of_samples())
    y_ = malaya_speech.astype.int_to_float(y_)
    y_ = zero_pad(y_, sr)
    return y_, sr


def get_train_data(train_meta_df, train_extra_df):
    train_meta_df = train_meta_df.loc[train_meta_df["audio_noise_note"].isnull(
    )]
    train_extra_df = train_extra_df.loc[train_extra_df["assessment_result"] == '1']

    train_df = pd.DataFrame()
    train_df["path"] = train_meta_df["path"]
    train_df["label"] = train_meta_df["assessment_result"]

    # ==============================================================================
    extra_df = pd.DataFrame()
    extra_df["path"] = train_extra_df["path"]
    extra_df["label"] = 1

    # ==============================================================================
    train_df = pd.concat([train_df, extra_df], axis=0)
    train_df["silence"] = 0
    # ==============================================================================
    xmfcc = []
    for index, fname in enumerate(train_df["path"]):
        try:
            source, sr = pre_process(fname)
            mfcc = mfcc_feature(source, sr)
            mfcc = mfcc.reshape(-1,)
            xmfcc.append(mfcc)
        except:
            train_df.at[index, "silence"] = 1
    mfcc_df = pd.DataFrame(xmfcc)
    train_df = train_df.loc[train_df["silence"] == 0]
    train_df.reset_index(drop=True, inplace=True)
    mfcc_df["label"] = train_df["label"]
    return mfcc_df
