from pydub import AudioSegment
import os
import librosa
import pandas as pd
import numpy as np
import malaya_speech
from pydub.silence import split_on_silence
import soundfile as sf
from keras.models import load_model



def pre_process(path,fname,dir_path):
  y, sr = librosa.load(path,res_type='kaiser_fast')
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
  train_df = pd.read_csv("..raw_data/data.csv")
  dir_path = "..process_data/data/"
  silence_index = []
  silence_fname = []
  for num,fname in enumerate(train_df["name"]):
    path = "..raw_data/data/" + fname
    try:
      pre_process(path,fname,dir_path)
    except:
      silence_index.append(num)
      silence_fname.append(fname)
  train_silence_ls = pd.concat([pd.Series(silence_index), pd.Series(silence_fname)],
                        axis=1)
  train_silence_ls.columns = ['index_in_rawdata', 'fname']
  train_silence_ls.to_csv("..raw_data/silence.csv",index=False)

    
def zero_pad(path):
  pad_ms = 5000  # Add here the fix length you want (in milliseconds)
  audio = AudioSegment.from_wav(path)
  if pad_ms > len(audio):
    silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
    padded = audio + silence  # Adding silence after the audio
    os.remove(path)
    padded.export(path, format='wav')

def zpad_all():
  data_dir = os.listdir("..process_data/data")
  for fname in data_dir:
    path = "..process_data/" + fname
    zero_pad(path)  
    
def mean_feature(path):
    source, sr = librosa.load(path, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=source[0:110250], sr=sr, n_mfcc=13)      
    return mfcc
    
def extract_all(df):
    Xmfcc = []
    for path in df["path"]:
        mfcc = mean_feature(path)
        mfcc = mfcc.reshape(-1,)
        Xmfcc.append(mfcc)
    return Xmfcc
  
def extract_data():
    df = pd.read_csv("..data/data.csv")
    Xmfcc = extract_all(train_df) 
    mfcc_df = pd.DataFrame(Xmfcc)
    mfcc_df.to_csv("..process_data/features/mfcc_features.csv", index=False)
    
    
def prepare_test():
    mfcc = pd.read_csv("..process_data/features/mfcc_features.csv")
    test_df = mfcc.iloc[:, :]
    X_test = test_df.iloc[:,:].values.reshape(test_df.shape[0],13,-1)
    print(test_df.shape)
    X_test = X_test[...,np.newaxis]
    print(X_test.shape)
    return X_test
    
  
def predict_mean(df):
    test_df = df
    X_test = prepare_test()
    res = np.zeros(X_test.shape[0])
    res = res[...,np.newaxis]
    model_list = os.listdir("..weights/")
    for name in model_list:
        model = load_model("..weights/" + name)
        preds = model.predict(X_test)
        res += model.predict(X_test) 
    res /= len(model_list)
    positive_proba = res
    return positive_proba
    

    
def build_path(df): 
    meta_df = df
    df = pd.DataFrame()
    df["name"] = meta_df["uuid"].apply(lambda uuid: f"{uuid}.wav")
    df["path"] = meta_df["uuid"].apply(lambda uuid: f"..raw_data/data/{uuid}.wav")
    df.to_csv("..raw_data/data.csv", index=False)
 
def predict(df):
  build_path(df)
  pre_process_all()
  zpad_all()
  extract_data()
  positive_proba = predict_mean(df)

   
  
