from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from datasets import Dataset
from tqdm import tqdm
from typing import Any
import sacrebleu
from llm_code_eval import evaluate


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

    outputs = []
    references = []
    ice_scores = []
    for sample in tqdm(dataset):
        problem = sample["prompt"]
        output = complete(problem, model, tokenizer, **generate_kwargs)["code"][len(problem):]

        outputs.append(output)

        reference = "\n".join(sample["canonical_solution"][1:])
        references.append(reference)

        ice_scores.append(evaluate(
            problem=problem,
            output=output,
            reference=reference,
            task="code-gen", aspect="usefulness", model="gpt-4o-mini"
        ))

    smoothing_function = SmoothingFunction().method1

    bleu_score = corpus_bleu(references, outputs, smoothing_function=smoothing_function)
    chrf_score = sacrebleu.corpus_chrf(outputs, [references]).score

    results = {
        "bleu-score": bleu_score,
        "chrf-score": chrf_score,
        "ice-scores": ice_scores,
    }
    return results
