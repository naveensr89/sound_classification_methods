import numpy as np
import librosa
import os
from multiprocessing import Pool
import time
import sys

# user imports
from utils import *

DATASET_DIR = "../audio/"
DEST_ROOT_DIR = '../features/'

# one of ['spectral_feat', 'mel_stft', 'mel_stft_db', 'stft', mfcc_zcr', 'audioset_em']
FEATURE_TYPE = 'stft'
NUM_PARALLEL_PROCESS = 8

SAMPLE_RATE = 44100
HOP_SIZE = 512
WINDOW_SIZE = 1024
N_MELS = 128

#
DEST_DIR_NAME = "{}_fs_{}_window_{}_hop_{}".format(FEATURE_TYPE, SAMPLE_RATE,
                                                   WINDOW_SIZE, HOP_SIZE)
if 'mel_' in FEATURE_TYPE:
    DEST_DIR_NAME += '_mel_{}'.format(N_MELS)
if FEATURE_TYPE == 'audioset_em':
    DEST_DIR_NAME = FEATURE_TYPE

DEST_DIR = os.path.join(DEST_ROOT_DIR, DEST_DIR_NAME)


def mfcc_zcr(x):
    # MFCC
    melspectrogram = librosa.feature.melspectrogram(
        x, sr=SAMPLE_RATE, hop_length=HOP_SIZE, n_fft=WINDOW_SIZE)
    logamplitude = librosa.core.amplitude_to_db(melspectrogram)
    mfcc = librosa.feature.mfcc(S=logamplitude, n_mfcc=13)
    # mfcc_mean = mfcc.mean(axis=0)
    # mfcc_std = mfcc.std(axis=0)
    # zcr
    zcr = librosa.feature.zero_crossing_rate(
        x, frame_length=WINDOW_SIZE, hop_length=HOP_SIZE)
    # zcr_mean = zcr.mean()
    # zcr_std = zcr.std()
    # concatenate
    # features = np.concatenate([mfcc_mean, mfcc_std, [zcr_mean], [zcr_std]])
    features = np.vstack([mfcc, zcr])
    return features


def mel_stft(x, db=True):
    features = librosa.feature.melspectrogram(
        y=x,
        sr=SAMPLE_RATE,
        n_fft=WINDOW_SIZE,
        hop_length=HOP_SIZE,
        n_mels=N_MELS)
    if db:
        features = librosa.core.amplitude_to_db(features)
    return features


def spectral_feat(x):
    # stft
    X = librosa.stft(x, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE)
    X_mag = np.abs(X)

    # features
    centroid = librosa.feature.spectral_centroid(S=X_mag).squeeze()
    bandwidth = librosa.feature.spectral_bandwidth(S=X_mag).squeeze()
    flatness = librosa.feature.spectral_flatness(S=X_mag).squeeze()
    rolloff = librosa.feature.spectral_rolloff(S=X_mag).squeeze()

    return np.vstack([centroid, bandwidth, flatness, rolloff])


def audioset_em(x):
    import sys
    sys.path.append('../audioset/')
    from get_audioset_embeddings import AudiosetEmbedding

    audioset_em = AudiosetEmbedding()

    features = audioset_em.get_audioset_embeddings(samples=x, sr=SAMPLE_RATE)
    return features


def get_features(audio_file):
    """
    Return feature extraction
    Globals:
        DEST_DIR: Destination directory to save feature extraction result
        FEATURE_TYPE : one of the following ['spectral_feat', 'stft', 'mel_stft', 
         'mel_stft_db', 'mfcc_zcr', 'audioset_em']
        SAMPLE_RATE : audio sample rate
        WINDOW_SIZE : fft size
        HOP_SIZE : hop length
        N_MELS: number of mel filters
    Parameters:
        audio_file: Path to audio file(wav)
    Return:
        Feature extraction stored as a file in DEST_DIR
    """
    print(audio_file)
    # load audio
    x, fs = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

    if FEATURE_TYPE == 'spectral_feat':
        features = spectral_feat(x)
    elif FEATURE_TYPE == 'mfcc_zcr':
        features = mfcc_zcr(x)
    elif FEATURE_TYPE == 'stft':
        X = librosa.stft(x, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE)
        features = np.abs(X)
    elif FEATURE_TYPE == 'mel_stft':
        features = mel_stft(x, db=False)
    elif FEATURE_TYPE == 'mel_stft_db':
        features = mel_stft(x, db=True)
    elif FEATURE_TYPE == 'audioset_em':
        features = audioset_em(x)

    # save output
    _, fname, _ = get_path_fname_ext(audio_file)
    out_file = os.path.join(DEST_DIR, fname + '.npy')
    np.save(out_file, features)


if __name__ == "__main__":

    # checks
    if not os.path.exists(DATASET_DIR):
        print("Dataset directory %s doesn't exist " % (DATASET_DIR))
        exit()
    # create destination diretory
    try:
        os.makedirs(DEST_DIR, exist_ok=False)
    except:
        print('Error: destination directory %s already exists' % (DEST_DIR))
        exit()

    # multiprocessing pool
    p = Pool(NUM_PARALLEL_PROCESS)

    # get all audio files
    audio_file_list = get_file_list(DATASET_DIR)

    start = time.time()
    # map
    p.map(get_features, audio_file_list)
    end = time.time()
    print('Total time taken = %.2f' % (end - start))
