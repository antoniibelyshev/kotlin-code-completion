from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from datasets import Dataset
from tqdm import tqdm
from typing import Any


default_generate_kwargs = {
    "max_new_tokens": 200,
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": 0.5,
    "top_k": 100,
    "top_p": 0.95,
}


def complete(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    **generate_kwargs: Any,
) -> dict[str, Any]:
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    generated_ids = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **{**default_generate_kwargs, **generate_kwargs},
    )
    
    generated_text = tokenizer.decode(generated_ids[0][len(input_ids):], skip_special_tokens=True)
    return {'ids': generated_ids, 'code': generated_text}


def evaluate_code_completion(
    dataset: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    code_language: str = "kotlin",
    **generate_kwargs: Any,
) -> dict[str, float | dict[str, float]]:
    assert "prompt" in dataset.features, "Dataset does not contain prompts"
    assert "canonical_solution" in dataset.features, "Dataset does not contain canonical solutions"

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not hasattr(model.generation_config, "pad_token") or model.generation_config.pad_token is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []
    completions_ids = []
    reference_solutions = []
    reference_ids = []
    for sample in tqdm(dataset):
        tokenized_sample = tokenizer(sample["prompt"], return_tensors="pt")
        input_ids = tokenized_sample["input_ids"].to(model.device)
        attention_mask = tokenized_sample["attention_mask"].to(model.device)
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**default_generate_kwargs, **generate_kwargs},
        )[0][len(input_ids):].cpu()
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)

        completions_ids.append(generated_ids)
        completions.append(generated_code)

        reference_solution = "\n".join(sample["canonical_solution"][1:])
        reference_solutions.append(reference_solution)
        reference_ids.append(tokenizer(reference_solution[1:], return_tensors="pt").input_ids)

    smoothing_function = SmoothingFunction().method1
    results = {
        "symbol-bleu-score": corpus_bleu(reference_solutions, completions, smoothing_function=smoothing_function),
        "token-bleu-score": corpus_bleu(reference_ids, completions_ids, smoothing_function=smoothing_function),
    }
    return results
