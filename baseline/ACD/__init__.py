from transformers import AutoTokenizer
from helper import format_seconds_to_time_string
from ACD.preprocessing import preprocess_data_ACD
from ACD.model import get_trainer_ACD
from transformers import AutoTokenizer
import subprocess
import numpy as np
import constants
import shutil
import time


def train_ACD_model(TARGET, MODEL_TYPE, train_dataset, test_dataset):
    results = {"TARGET": TARGET}
    tokenizer = AutoTokenizer.from_pretrained(
        constants.MODEL_NAME_ACD + MODEL_TYPE)

    start_time = time.time()

    train_data = preprocess_data_ACD(train_dataset, tokenizer)
    test_data = preprocess_data_ACD(test_dataset, tokenizer)

    n_samples_train = len(train_data)
    n_samples_test = len(test_data)

    trainer = get_trainer_ACD(train_data, test_data, tokenizer, results)
    trainer.train()

    # save log history
    log_history = trainer.state.log_history

    # Save Evaluation of Test Data
    eval_metrics = trainer.evaluate(test_data)
    print(f"Eval Metrics:", eval_metrics)

    # Save Loss
    loss = eval_metrics["eval_loss"]

    path_output = constants.OUTPUT_DIR_ACD + "_" + results["TARGET"]
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
