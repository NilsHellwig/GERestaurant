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
    results = {
        "TARGET": TARGET,
        "single_split_results": []
    }

    loss = []
    n_samples_train = []
    n_samples_test = []
    log_history = {}

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        constants.MODEL_NAME_ACSA + MODEL_TYPE)
    metrics_prefixes = ["accuracy", "hamming_loss",
                        "f1_macro", "f1_micro", "f1_weighted"]
    metrics_total = {f"{m}": [] for m in metrics_prefixes}

    for cross_idx in range(constants.N_FOLDS):
        # Load Data
        train_data = preprocess_data_ACSA(train_dataset[cross_idx], tokenizer)
        test_data = preprocess_data_ACSA(test_dataset[cross_idx], tokenizer)

        n_samples_train.append(len(train_data))
        n_samples_test.append(len(test_data))

        # Train Model
        trainer = get_trainer_ACSA(
            train_data, test_data, tokenizer, results, cross_idx)
        trainer.train()

        # save log history
        log_history[cross_idx] = trainer.state.log_history

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate(test_data)
        print(f"Split {cross_idx}:", eval_metrics)

        # Save Evaluation of Split
        results["single_split_results"].append(eval_metrics)

        # Save Metrics for fold
        for m in metrics_prefixes:
            metrics_total[f"{m}"].append(eval_metrics[f"eval_{m}"])

        loss.append(eval_metrics["eval_loss"])

        path_output = constants.OUTPUT_DIR_ACSA + \
            "_" + results["TARGET"]+"_"+str(cross_idx)
        shutil.rmtree(path_output)

        subprocess.call("rm -rf /home/mi/.local/share/Trash", shell=True)

    runtime = time.time() - start_time

    results["eval_loss"] = np.mean(loss)

    results.update({f"eval_{m}": np.mean(
        metrics_total[f"{m}"]) for m in metrics_prefixes})

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)

    results["n_samples_train"] = n_samples_train
    results["n_samples_train_mean"] = np.mean(n_samples_train)
    results["n_samples_test"] = n_samples_test
    results["n_samples_test_mean"] = np.mean(n_samples_test)
    results["log_history"] = log_history

    return results
