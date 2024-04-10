from ACSA.preprocessing import preprocess_data_ACSA
from helper import format_seconds_to_time_string
from ACSA.model import get_trainer_ACSA
from transformers import AutoTokenizer
import subprocess
import numpy as np
import constants
import time
import shutil


def train_ACSA_model(TARGET, MODEL_TYPE, train_dataset, test_dataset):
    results = {"TARGET": TARGET}
    tokenizer = AutoTokenizer.from_pretrained(
        constants.MODEL_NAME_ACSA + MODEL_TYPE)

    start_time = time.time()

    # Load Data
    train_data = preprocess_data_ACSA(train_dataset, tokenizer)
    test_data = preprocess_data_ACSA(test_dataset, tokenizer)

    n_samples_train = len(train_data)
    n_samples_test = len(test_data)

    # Train Model
    trainer = get_trainer_ACSA(
        train_data, test_data, tokenizer, results)
    trainer.train()

    # save log history
    log_history = trainer.state.log_history

    # Save Evaluation of Test Data
    eval_metrics = trainer.evaluate(test_data)
    print(f"Eval Metrics:", eval_metrics)

    loss = eval_metrics["eval_loss"]

    path_output = constants.OUTPUT_DIR_ACSA + "_" + TARGET
    shutil.rmtree(path_output)

    subprocess.call("rm -rf /home/mi/.local/share/Trash", shell=True)

    runtime = time.time() - start_time

    results.update(eval_metrics)

    results["TARGET"] = TARGET
    results["eval_loss"] = loss

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)

    results["n_samples_train"] = n_samples_train
    results["n_samples_test"] = n_samples_test
    results["log_history"] = log_history

    return results
