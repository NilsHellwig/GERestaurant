import os
import shutil
import time

from transformers import AutoTokenizer

import constants
from ACD.model import get_trainer_ACD
from ACD.preprocessing import preprocess_data_ACD
from helper import format_seconds_to_time_string


def train_ACD_model(target: str, model_type: str, train_dataset: list, test_dataset: list) -> dict:
    """Train the ACD model and return evaluation results."""
    results = {"TARGET": target}
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_ACD + model_type)

    start_time = time.time()

    # Preprocess data
    train_data = preprocess_data_ACD(train_dataset, tokenizer)
    test_data = preprocess_data_ACD(test_dataset, tokenizer)

    # Initialize and run trainer
    # We pass model_type to avoid sys.argv dependency in models/trainers
    trainer = get_trainer_ACD(train_data, test_data, model_type, tokenizer, results)
    trainer.train()

    # Collect evaluation metrics
    eval_metrics = trainer.evaluate(test_data)
    results.update(eval_metrics)

    # Cleanup temporary checkpoints
    path_output = f"{constants.OUTPUT_DIR_ACD}_{target}"
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    # Calculate runtime
    runtime = time.time() - start_time
    results.update({
        "runtime": runtime,
        "runtime_formatted": format_seconds_to_time_string(runtime),
        "n_samples_train": len(train_dataset),
        "n_samples_test": len(test_dataset),
        "log_history": trainer.state.log_history,
    })

    return results

