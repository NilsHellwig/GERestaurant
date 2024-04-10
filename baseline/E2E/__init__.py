from E2E.preprocessing import get_preprocessed_data_E2E
from helper import format_seconds_to_time_string
from E2E.model import get_trainer_E2E
from transformers import AutoTokenizer
import subprocess
import numpy as np
import constants
import shutil
import time


def train_E2E_model(TARGET, MODEL_TYPE, train_dataset, test_dataset):
    results = {"TARGET": TARGET}
    tokenizer = AutoTokenizer.from_pretrained(
        constants.MODEL_NAME_E2E + MODEL_TYPE)

    start_time = time.time()

    train_dataset, test_dataset = get_preprocessed_data_E2E(
        train_dataset, test_dataset, tokenizer)

    n_samples_train = len(train_dataset)
    n_samples_test = len(test_dataset)

    # in order to save the prediction and labels for each split, results will also be handed over
    trainer = get_trainer_E2E(
        train_dataset, test_dataset, MODEL_TYPE, tokenizer, results)
    trainer.train()

    # save log history
    log_history = trainer.state.log_history

    # Save Evaluation of Test Data
    eval_metrics = trainer.evaluate(test_dataset)
    print(f"Eval Metrics:", eval_metrics)

    eval_loss = eval_metrics["eval_loss"]

    # remove model output
    path_output = constants.OUTPUT_DIR_E2E + "_"+TARGET
    shutil.rmtree(path_output)

    subprocess.call("rm -rf /home/mi/.local/share/Trash", shell=True)

    runtime = time.time() - start_time

    results.update(eval_metrics)
    results["TARGET"] = TARGET

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    results["eval_loss"] = eval_loss
    results["n_samples_train"] = n_samples_train
    results["n_samples_test"] = n_samples_test
    results["log_history"] = log_history

    return results
