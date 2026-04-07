import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import constants
from TASD.evaluation import compute_metrics_TASD


def create_model_TASD(model_type: str):
    """Factory function for initializing the TASD model."""
    model_name = f"{constants.MODEL_NAME_TASD}{model_type}"
    return AutoModelForSeq2SeqLM.from_pretrained(model_name).to(constants.DEVICE)


def get_trainer_TASD(train_data, test_data, model_type, tokenizer, results):
    """Prepare the Seq2Seq trainer for the TASD task."""
    args = Seq2SeqTrainingArguments(
        output_dir=f"{constants.OUTPUT_DIR_TASD}_{results['TARGET']}",
        logging_strategy=constants.LOGGING_STRATEGY_TASD,
        save_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH else "no",
        learning_rate=constants.LEARNING_RATE_TASD,
        num_train_epochs=constants.EPOCHS_TASD,
        per_device_train_batch_size=constants.BATCH_SIZE_TASD,
        per_device_eval_batch_size=constants.BATCH_SIZE_TASD,
        predict_with_generate=True,
        load_best_model_at_end=False,
        weight_decay=constants.WEIGHT_DECAY_TASD,
        metric_for_best_model=constants.METRIC_FOR_BEST_MODEL_TASD,
        seed=constants.RANDOM_SEED,
        fp16=torch.cuda.is_available(),  # Enable fp16 if GPU is available
        report_to="none",
        do_eval=constants.EVALUATE_AFTER_EPOCH,
        evaluation_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH else "no",
        generation_max_length=256,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # Note: We need to wrap the metric function properly
    metrics_func = compute_metrics_TASD(results, model_type)

    # Use a lambda to avoid sys.argv dependency in trainer's model_init
    trainer = Seq2SeqTrainer(
        model_init=lambda: create_model_TASD(model_type),
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metrics_func,
    )
    return trainer

