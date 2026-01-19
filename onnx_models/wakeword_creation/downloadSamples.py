# @title  { display-mode: "form" }
# @markdown # 2. Download Data
# @markdown Training custom models requires downloading a wide variety of data
# @markdown that will help make the model perform well in real-world scenarios.
# @markdown This example notebook will download small samples of background noise,
# @markdown music, and Room Impulse Responses (to add echo). This will still produce
# @markdown a custom model that performs well, but if you are interested in adding even more,
# @markdown feel free to extend this notebook to download the full datasets and even add
# @markdown your own!
# @markdown
# @markdown Downloading this example data will usually take about 15 minutes.

# @markdown **Important note!** The data downloaded here has a mixture of different
# @markdown licenses and usage restrictions. As such, any custom models trained with this
# @markdown data should be considered as appropriate for **non-commercial** personal use only.

# ## Install all dependencies
# !pip install datasets
# !pip install scipy
# !pip install tqdm

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# install openwakeword (full installation to support training)
!git clone https://github.com/dscripka/openwakeword
!pip install -e ./openwakeword --no-deps
# !cd openwakeword

# install other dependencies
!pip install mutagen==1.47.0
!pip install torchinfo==1.8.0
!pip install torchmetrics==1.2.0
!pip install speechbrain==0.5.14
!pip install audiomentations==0.33.0
!pip install torch-audiomentations==0.11.0
!pip install acoustics==0.2.6
# !pip uninstall tensorflow -y
# !pip install tensorflow-cpu==2.8.1
# !pip install protobuf==3.20.3
# !pip install tensorflow_probability==0.16.0
# !pip install onnx_tf==1.10.0
!pip install onnxruntime==1.22.1 ai_edge_litert==1.4.0 onnxsim
!pip install onnx2tf
!pip install onnx==1.19.1
# !pip install ai_edge_litert==1.2.0
!pip install onnx_graphsurgeon
!pip install sng4onnx
!pip install pronouncing==0.2.0
!pip install datasets==2.14.6
!pip install deep-phonemizer==0.0.19

# Download required models (workaround for Colab)
import os
os.makedirs("./openwakeword/openwakeword/resources/models", exist_ok=True)
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite

# Imports
import sys

if "piper-sample-generator/" not in sys.path:
    sys.path.append("piper-sample-generator/")
from generate_samples import generate_samples

import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm

## Download all data

## Download MIR RIR data (takes about ~2 minutes)
output_dir = "./mit_rirs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    !git lfs install
    !git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses
    rir_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")]}).cast_column("audio", datasets.Audio())
    # Save clips to 16-bit PCM wav files
    for row in tqdm(rir_dataset):
        name = row['audio']['path'].split('/')[-1]
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

## Download noise and background audio (takes about ~3 minutes)

# Audioset Dataset (https://research.google.com/audioset/dataset/index.html)
# Download one part of the audioset .tar files, extract, and convert to 16khz
# For full-scale training, it's recommended to download the entire dataset from
# https://huggingface.co/datasets/agkphysics/AudioSet, and
# even potentially combine it with other background noise datasets (e.g., FSD50k, Freesound, etc.)

if not os.path.exists("audioset"):
    os.mkdir("audioset")

    fname = "bal_train09.tar"
    out_dir = f"audioset/{fname}"
    link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
    !wget -O {out_dir} {link}
    !cd audioset && tar -xvf bal_train09.tar

    output_dir = "./audioset_16k"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save clips to 16-bit PCM wav files
    audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]})
    audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(audioset_dataset):
        name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

# Free Music Archive dataset
# https://github.com/mdeff/fma

output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    # Save clips to 16-bit PCM wav files
    n_hours = 1  # use only 1 hour of clips for this example notebook, recommend increasing for full-scale training
    for i in tqdm(range(n_hours*3600//30)):  # this works because the FMA dataset is all 30 second clips
        row = next(fma_dataset)
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
        i += 1
        if i == n_hours*3600//30:
            break

# Download pre-computed openWakeWord features for training and validation

# training set (~2,000 hours from the ACAV100M Dataset)
# See https://huggingface.co/datasets/davidscripka/openwakeword_features for more information
if not os.path.exists("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
    !wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy

# validation set for false positive rate estimation (~11 hours)
if not os.path.exists("validation_set_features.npy"):
    !wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
