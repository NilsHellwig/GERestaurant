import argparse
import json
import os
import random
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

import constants
from ACD import train_ACD_model
from ACSA import train_ACSA_model
from E2E import train_E2E_model
from load_dataset import load_dataset
from TASD import train_TASD_model


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train baseline models for GERestaurant ABSA tasks.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "aspect_category",
            "aspect_category_sentiment",
            "end_2_end_absa",
            "target_aspect_sentiment_detection",
        ],
        help="The ABSA subtask to train the model for.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The type of model (e.g., 'base', 'large').",
    )

    args = parser.parse_args()
    target = args.task
    model_type = args.model_type

    print(f"Starting training for task: {target} using model type: {model_type}")

    # Set seeds for reproducibility
    torch.manual_seed(constants.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(constants.RANDOM_SEED)
    set_seed(constants.RANDOM_SEED)
    random.seed(constants.RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(constants.RANDOM_SEED)

    # Ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")

    # Disable Pycache
    sys.dont_write_bytecode = True

    # Create output directories
    for folder in ["split_results", "results_csv", "results_json"]:
        os.makedirs(folder, exist_ok=True)

    # Load Dataset
    train_dataset, test_dataset = load_dataset()

    # Train and Evaluate Model
    if target == "aspect_category":
        results = train_ACD_model(target, model_type, train_dataset, test_dataset)
    elif target == "aspect_category_sentiment":
        results = train_ACSA_model(target, model_type, train_dataset, test_dataset)
    elif target == "end_2_end_absa":
        results = train_E2E_model(target, model_type, train_dataset, test_dataset)
    elif target == "target_aspect_sentiment_detection":
        results = train_TASD_model(target, model_type, train_dataset, test_dataset)
    else:
        raise ValueError(f"Unknown task: {target}")

    # Save Results
    json_path = os.path.join("results_json", f"results_{target}_{model_type}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    csv_path = os.path.join("results_csv", f"results_{target}_{model_type}.csv")
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"Training completed. Results saved to {json_path}")


if __name__ == "__main__":
    main()

