import torch
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import constants
from ACD.evaluation import compute_metrics_ACD


def create_model_ACD(model_type: str):
    """Factory function for initializing the ACD model."""
    model_name = f"{constants.MODEL_NAME_ACD}{model_type}"
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=len(constants.ASPECT_CATEGORIES),
        problem_type="multi_label_classification",
    ).to(torch.device(constants.DEVICE))


def get_trainer_ACD(train_data, test_data, model_type, tokenizer, results):
    """Prepare the Trainer for the ACD task."""
    # Define Arguments
    training_args = TrainingArguments(
        output_dir=f"{constants.OUTPUT_DIR_ACD}_{results['TARGET']}",
        learning_rate=constants.LEARNING_RATE_ACD,
        num_train_epochs=constants.EPOCHS_ACD,
        per_device_train_batch_size=constants.BATCH_SIZE_ACD,
        per_device_eval_batch_size=constants.BATCH_SIZE_ACD,
        save_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH else "no",
        logging_dir="logs",
        logging_steps=100,
        logging_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1_micro",
        fp16=torch.cuda.is_available(),
        report_to="none",
        do_eval=constants.EVALUATE_AFTER_EPOCH,
        evaluation_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH else "no",
        seed=constants.RANDOM_SEED,
    )

    metrics_func = compute_metrics_ACD(results)

    trainer = Trainer(
        model_init=lambda: create_model_ACD(model_type),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=metrics_func,
    )

    return trainer

