from load_dataset import load_dataset
from transformers import set_seed
from ACD import train_ACD_model
from E2E import train_E2E_model
from ACSA import train_ACSA_model
from TASD import train_TASD_model
import pandas as pd
import numpy as np
import constants
import warnings
import random
import shutil
import torch
import json
import sys
import os

# Parameters

TARGET = sys.argv[1]
MODEL_TYPE = sys.argv[2]

if TARGET not in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
    raise ValueError("Error: Not a valid target")

print(TARGET)


# Set seeds
torch.device(constants.DEVICE)
torch.manual_seed(constants.RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(constants.RANDOM_SEED)
set_seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(constants.RANDOM_SEED)

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="transformers.optimization")

# Disable Pycache
sys.dont_write_bytecode = True

# Code

# Create Folders for Results
folders = ['split_results', 'results_csv', 'results_json']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load Dataset
train_dataset, test_dataset = load_dataset()


# Load Model
if TARGET == "aspect_category":
    results = train_ACD_model(TARGET, MODEL_TYPE, train_dataset, test_dataset)

if TARGET == "aspect_category_sentiment":
    results = train_ACSA_model(TARGET, MODEL_TYPE, train_dataset, test_dataset)

if TARGET == "end_2_end_absa":
    results = train_E2E_model(TARGET, MODEL_TYPE, train_dataset, test_dataset)

if TARGET == "target_aspect_sentiment_detection":
    results = train_TASD_model(TARGET, MODEL_TYPE, train_dataset, test_dataset)

# Save Results
with open(f'results_json/results_{TARGET}_{MODEL_TYPE}.json', 'w') as json_file:
    json.dump(results, json_file)

df = pd.DataFrame([results])
df.to_csv(f'results_csv/results_{TARGET}_{MODEL_TYPE}.csv', index=False)
