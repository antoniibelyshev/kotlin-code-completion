from transformers import AutoTokenizer
from datasets import Dataset


def get_preprocess_function(tokenizer: AutoTokenizer, max_len: int | None = None):
    def preprocess_function(examples: Dataset):
        tokenized_problems = tokenizer(examples["problem"], padding=False)["input_ids"]
        tokenized_solutions = tokenizer(examples["solution"], padding=False)["input_ids"]


        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for problem, solution in zip(tokenized_problems, tokenized_solutions):
            input_ids = problem + solution
            if (max_len is None) or (len(input_ids) <= max_len):
                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append([1] * (len(input_ids)))
                model_inputs["labels"].append([-100] * len(problem) + solution)

        return model_inputs

    return preprocess_function