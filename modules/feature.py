import librosa


def mfcc_feature(source, sr):
    mfcc = librosa.feature.mfcc(y=source[0:5*sr], sr=sr, n_mfcc=13)
    return mfcc
