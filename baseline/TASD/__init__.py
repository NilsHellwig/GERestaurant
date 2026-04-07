import shutil
import time

from transformers import T5Tokenizer

import constants
from helper import format_seconds_to_time_string
from TASD.model import get_trainer_TASD
from TASD.preprocessing import CustomDatasetTASD, encode_example


def train_TASD_model(target: str, model_type: str, train_dataset: list, test_dataset: list) -> dict:
    """Train the TASD model and return evaluation results."""
    results = {"TARGET": target}
    start_time = time.time()

    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(constants.MODEL_NAME_TASD + model_type)

    # Encode datasets
    train_encoded = [encode_example(ex, tokenizer) for ex in train_dataset]
    test_encoded = [encode_example(ex, tokenizer) for ex in test_dataset]

    train_data = CustomDatasetTASD(
        [ex["input_ids"] for ex in train_encoded],
        [ex["attention_mask"] for ex in train_encoded],
        [ex["labels"] for ex in train_encoded],
    )
    test_data = CustomDatasetTASD(
        [ex["input_ids"] for ex in test_encoded],
        [ex["attention_mask"] for ex in test_encoded],
        [ex["labels"] for ex in test_encoded],
    )

    # Initialize and run trainer
    # We pass model_type to handle model initialization properly
    trainer = get_trainer_TASD(train_data, test_data, model_type, tokenizer, results)
    trainer.train()

    # Collect evaluation metrics
    eval_metrics = trainer.evaluate(test_data)
    results.update(eval_metrics)

    # Cleanup temporary checkpoints
    path_output = f"{constants.OUTPUT_DIR_TASD}_{target}"
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

