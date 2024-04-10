from helper import format_seconds_to_time_string
from transformers import T5Tokenizer
from TASD.preprocessing import CustomDatasetTASD, encode_example
from TASD.model import get_trainer_TASD
import subprocess
import numpy as np
import constants
import shutil
import time


def train_TASD_model(TARGET, MODEL_TYPE, train_dataset, test_dataset):
    results = {"TARGET": TARGET}

    start_time = time.time()

    tokenizer = T5Tokenizer.from_pretrained(constants.MODEL_NAME_TASD + MODEL_TYPE)

    # Load Data
    train_data = train_dataset
    test_data = test_dataset

    n_samples_train = len(train_data)
    n_samples_test = len(test_data)

    train_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in train_data],
                                       [encode_example(example, tokenizer)["attention_mask"]
                                           for example in train_data],
                                       [encode_example(example, tokenizer)["labels"] for example in train_data])
    test_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in test_data],
                                      [encode_example(example, tokenizer)["attention_mask"]
                                       for example in test_data],
                                      [encode_example(example, tokenizer)["labels"] for example in test_data])

    # Train Model
    trainer = get_trainer_TASD(train_data, test_data, MODEL_TYPE, tokenizer, results)
    trainer.train()

    # save log history
    log_history = trainer.state.log_history

    # Save Evaluation of Test Data
    eval_metrics = trainer.evaluate(test_data)
    print(f"Eval Metrics:", eval_metrics)

    # Save Evaluation of Split
    results.update(eval_metrics)

    loss = eval_metrics["eval_loss"]

    path_output = constants.OUTPUT_DIR_TASD + "_" + results["TARGET"]
    shutil.rmtree(path_output)

    subprocess.call("rm -rf /home/mi/.local/share/Trash", shell=True)

    runtime = time.time() - start_time

    results["eval_loss"] = loss

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    results["n_samples_train"] = n_samples_train
    results["n_samples_test"] = n_samples_test
    results["log_history"] = log_history

    return results
