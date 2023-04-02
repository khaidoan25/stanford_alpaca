import argparse
import json
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path
    )
    # model.to("cuda")
    model.eval()
    tokenizer =  AutoTokenizer.from_pretrained(
        args.model_path
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # set_seed(42)
    with open(args.data_path, "r") as f:
        eval_data = json.load(f)
        
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in eval_data
    ]
    
    model_name = args.model_path.split("/")[-1]
    with open(f"result_{model_name}.json", "w") as f:
        results = []
        for sample in sources:
            result = generator(sample, max_length=512)
            results.append(result[0])
        json.dump(results, f)
    
    
def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate finetuned gpt2")
    parser.add_argument('--data_path', type=str, default="./data/gpt_eval.json",
                        help="Path to the evaluation data", required=True)
    parser.add_argument('--model_path', type=str,
                        help="Path to model", required=True)
    parser.add_argument('--output_path', type=str, default=None,
                        help="Path to save the evaluation results")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(args)