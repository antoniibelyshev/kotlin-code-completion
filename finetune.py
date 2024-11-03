import pandas as pd
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler
)
from datasets import DatasetDict
from typing import Any
from itertools import product
import torch
from tqdm import tqdm

from data_utils import get_preprocess_function


DEFAULT_TRAINING_ARGS = {
    "output_dir": "./checkpoints",
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 20,
    "logging_dir": "./logs",
    "logging_strategy": "steps",
    "logging_steps": 1,
    "evaluation_strategy": "epoch",
    "eval_steps": 1,
    "save_strategy": "epoch",
    "save_steps": 4,
    "save_total_limit": 2,
    "fp16": torch.cuda.is_available(),
    "weight_decay": 0.1,
    "load_best_model_at_end": True,
}


def finetune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: DatasetDict,
    *,
    max_len: int | None = None,
    early_stopping_patience: int = 2,
    trainer_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[AutoModelForCausalLM, float]:
    """
    Preprocesses the dataset and fine-tunes the model using the specified training arguments.
    """
    assert "train" in data, "Data does not contain train split"
    assert "problem" in data["train"].features, "Dataset does not contain problems"
    assert "solution" in data["train"].features, "Dataset does not contain solutions"

    tokenized_data = data.map(
        get_preprocess_function(tokenizer, max_len),
        batched=True,
        remove_columns=["problem", "solution"]
    )

    return train_model(
        model, tokenizer,
        tokenized_data,
        early_stopping_patience=early_stopping_patience,
        trainer_kwargs=trainer_kwargs,
        **kwargs,
    )


def train_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_data: DatasetDict,
    *,
    early_stopping_patience: int = 2,
    trainer_kwargs: dict[str, Any] | None = None,
    effective_batch_size: int | None = None,
    **kwargs: Any,
) -> tuple[AutoModelForCausalLM, float]:
    """
    Fine-tunes the model on tokenized data and returns the trained model with validation loss.
    """
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if effective_batch_size is not None:
        kwargs["per_device_train_batch_size"] = min(DEFAULT_TRAINING_ARGS["per_device_train_batch_size"], effective_batch_size)
        kwargs["gradient_accumulation_steps"] = effective_batch_size // kwargs["per_device_train_batch_size"]


    training_args = TrainingArguments(**{**DEFAULT_TRAINING_ARGS, **kwargs})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data.get("test"),
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        **(trainer_kwargs or dict()),
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    val_loss = eval_metrics.get("eval_loss", float('inf'))
    return model, val_loss


def validate_hyperparameters(
    model_name: str,
    data: DatasetDict,
    hyperparams: dict[str, list[Any]],
    *,
    max_len: int | None = None,
    early_stopping_patience: int = 2,
    trainer_kwargs: dict[str, Any] | None = None,
    **default_training_args: Any,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    """
    Cross-validates the hyperparameters to find the best combination based on validation loss.
    Stores and returns all combinations and losses in a pandas DataFrame.
    """
    assert "test" in data, "Data does not contain test split"

    trainer_kwargs = trainer_kwargs or dict()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    best_combination = None
    best_loss = float('inf')
    
    results = []

    keys, values = zip(*hyperparams.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    for params in tqdm(param_combinations):
        print(f"Testing hyperparameter combination: {params}")
        
        current_args = {**default_training_args, **params}

        model = AutoModelForCausalLM.from_pretrained(model_name)

        _, val_loss = finetune(
            model,
            tokenizer,
            data,
            max_len=max_len,
            early_stopping_patience=early_stopping_patience,
            trainer_kwargs=trainer_kwargs,
            **current_args,
        )
        
        results.append({**params, "val_loss": val_loss})
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_combination = params

        print(f"Validation loss for combination {params}: {val_loss}")

    results_df = pd.DataFrame(results)
    print(f"Best combination: {best_combination} with validation loss: {best_loss}")

    return best_combination, best_loss, results_df
