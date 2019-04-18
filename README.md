# Sound Classification Exploration

Exploring sound classification methods

# Requirements
Create a python virtual environment and install requirements  
``` 
pip install -r requirements.txt 
```

Clone the repository [ESC-50](https://github.com/karoldvl/ESC-50)

Download Audioset embeddings models below and place them in `audioset/` directory
* [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt),
  in TensorFlow checkpoint format.
* [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz),
  in NumPy compressed archive format.


## Feature extraction
Edit `src/feature_extract.py` and set the following values
```
DATASET_DIR = "ESC_50/audio/"
DEST_ROOT_DIR = '../features/'

# one of ['spectral_feat', 'mel_stft', 'mel_stft_db', 'stft', mfcc_zcr', 'audioset_em']
FEATURE_TYPE = 'stft'
NUM_PARALLEL_PROCESS = 8

SAMPLE_RATE = 44100
HOP_SIZE = 512
WINDOW_SIZE = 1024
N_MELS = 128
```

Running `python feature_extract.py` will save the features for each audio file into a numpy array saved in .npy format in `DEST_ROOT_DIR`.

# Sound classification methods

Explore sound classification methods in following notebooks

- `src/spectral_features.ipynb` : Spectral features (centroid, bandwidth, flatness, rolloff) 
- `src/mfcc.ipynb` : Mel-frequency cepstral coefficients (MFCC) 
- `src/stft.ipynb` : short-time Fourier transform (STFT)
- `src/audioset_embeddings.ipynb`: Audio embeddings as feautres