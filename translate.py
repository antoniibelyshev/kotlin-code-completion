import openai
import os
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
import json
from tqdm import trange

from typing import Any


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_prompt(
    sample: dict[str, str],
    source_language: str,
    target_language: str,
    example_inline_comment: str | None = "// Comment",
    example_multiline_comment: str | None = "/*\n*Start of the comment\n*middle of the comment*End of the comment\n*/",
) -> str:
    assert "problem" in sample, "Sample does not contain problem"
    assert "solution" in sample, "Sample does not contain solution"
    
    prompt = f"Translate the following problem - solution pair from {source_language} to {target_language}.\n"
    prompt += "Provide the translation as a JSON object with keys 'problem' and 'solution' without any explanations or extra text.\n"
    prompt += f"For in-line and multi-line comments, use syntax specific to {target_language}.\n"
    prompt += "Here is the JSON format:\n{'problem': '<translated problem>', 'solution': '<translated solution>'}.\n"
    prompt += f"\nProblem:\n\n{sample['problem']}\n\nSolution:\n\n{sample['solution']}"
    
    return prompt



def translate_problem_solution_pair(
    sample: dict[str, str],
    source_language: str = "python",
    target_language: str = "kotlin",
    **get_prompt_kwargs: Any,
) -> str:
    messages = [
        {"role": "system", "content": "You are an expert programmer."},
        {"role": "user", "content": get_prompt(sample, source_language, target_language, **get_prompt_kwargs)}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1024,
        temperature=0,
    )
    
    translated_code = response["choices"][0]["message"]["content"].strip()

    try:
        translation = json.loads(translated_code)
        if "problem" not in translation:
            print("Failed to find the problem in the response")
        elif "solution" not in translation:
            print("Failed to find the solution in the response")
        else:
            return translation
        
    except json.JSONDecodeError:
        print("Failed to parse translation as JSON.")

    except:
        print("Unknown error")
    
    return {"problem": "", "solution": ""}


def translate_dataset(
    dataset: Dataset,
    n: int= 1000,
    save_path: str = None,
    *,
    source_language: str = "python",
    target_language: str = "kotlin",
    **kwargs: Any,
) -> Dataset:
    translated_samples = [
        translate_problem_solution_pair(dataset[i], source_language, target_language, **kwargs)
        for i in trange(n)
    ]
    translated_dataset = Dataset.from_list(translated_samples).filter(lambda sample: sample["problem"] != "")

    if save_path is not None:
        translated_dataset.save_to_disk(save_path)

    return translated_dataset


def main(n: int = 10000):
    dataset = load_dataset("jinaai/code_exercises")["train"]
    path = f"data/kotlin-code-exercises-{n}"
    translate_dataset(dataset, n, path)


if __name__ == "__main__":
    main()
