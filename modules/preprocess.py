from pydub import AudioSegment
import os
import librosa
import pandas as pd
import numpy as np
import malaya_speech
from malaya_speech import Pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf
import matplotlib.pyplot as plt
import IPython.display as ipd


def pre_process(path,fname,dir_path):
  y, sr = librosa.load(path,res_type='kaiser_fast')
  y_ = librosa.effects.trim(y, top_db = 20)[0]
  y_int = malaya_speech.astype.float_to_int(y)
  audio = AudioSegment(
      y_int.tobytes(),
      frame_rate = sr,
      sample_width = y_int.dtype.itemsize,
      channels = 1
  )
  audio_chunks = split_on_silence(
      audio,
      min_silence_len = 200,
      silence_thresh = -30,
      keep_silence = 100,
  )
  y_ = sum(audio_chunks)
  y_ = np.array(y_.get_array_of_samples())
  y_ = malaya_speech.astype.int_to_float(y_)
  dir_path = dir_path + fname
  sf.write(dir_path, data = y_, samplerate = sr)
  
  
def pre_process_all():
  train_df = pd.read_csv("..data/test_train.csv")
  dir_path = "..test_data/train/"
  silence_index = []
  silence_fname = []
  for num,fname in enumerate(train_df["path"]):
    path = "..data/train/" + fname
    try:
      pre_process(path,fname,dir_path)
    except:
      silence_index.append(num)
      silence_fname.append(fname)
  train_silence_ls = pd.concat([pd.Series(silence_index), pd.Series(silence_fname)],
                        axis=1)
  train_silence_ls.columns = ['index_in_train', 'fname']
  train_silence_ls.to_csv("..test_data/train_silence.csv",index=False)

  # ==============================================================================
  private_df = pd.read_csv("..data/test_private.csv")
  dir_path = "..test_data/private/"
  silence_index = []
  silence_fname = []
  for num,fname in enumerate(private_df["path"]):
    path = "..data/private/" + fname
    try:
      pre_process(path,fname,dir_path)
    except:
      silence_index.append(num)
      silence_fname.append(fname)
  private_silence_ls = pd.concat([pd.Series(silence_index), pd.Series(silence_fname)],
                        axis=1)
  private_silence_ls.columns = ['index_in_private', 'fname']
  private_silence_ls.to_csv("..test_data/private_silence.csv",index=False)

  # ==============================================================================
  public_df = pd.read_csv("..data/test_public.csv")
  dir_path = "..test_data/public/"
  silence_index = []
  silence_fname = []
  for num,fname in enumerate(public_df["path"]):
    path = "..data/public/" + fname
    try:
      pre_process(path,fname,dir_path)
    except:
      silence_index.append(num)
      silence_fname.append(fname)
  public_silence_ls = pd.concat([pd.Series(silence_index), pd.Series(silence_fname)],
                        axis=1)
  public_silence_ls.columns = ['index_in_public', 'fname']
  public_silence_ls.to_csv("..test_data/public_silence.csv",index=False)
  

if __name__ == "__main__":
   pre_process_all()
