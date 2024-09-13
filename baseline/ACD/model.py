from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from ACD.evaluation import compute_metrics_ACD
import constants
import torch
import sys


def create_model_ACD():
    MODEL_TYPE = sys.argv[2]
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=constants.MODEL_NAME_ACD + MODEL_TYPE,
        num_labels=len(constants.ASPECT_CATEGORIES),
        problem_type="multi_label_classification"
    ).to(torch.device(constants.DEVICE))


def get_trainer_ACD(train_data, test_data, tokenizer, results):
    # Define Arguments
    training_args = TrainingArguments(
        output_dir=constants.OUTPUT_DIR_ACD+"_" +
        results["TARGET"],
        learning_rate=constants.LEARNING_RATE_ACD,
        num_train_epochs=constants.EPOCHS_ACD,
        per_device_train_batch_size=constants.BATCH_SIZE_ACD,
        per_device_eval_batch_size=constants.BATCH_SIZE_ACD,
        save_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH == True else "no",
        logging_dir="logs",
        logging_steps=100,
        logging_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1_micro",
        fp16=torch.cuda.is_available(),
        report_to="none",
        do_eval=True if constants.EVALUATE_AFTER_EPOCH == True else False,
        evaluation_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH == True else "no",
        seed=constants.RANDOM_SEED
    )

    compute_metrics_ACD_fcn = compute_metrics_ACD(results)

    trainer = Trainer(
        model_init=create_model_ACD,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_ACD_fcn,
    )

    return trainer
