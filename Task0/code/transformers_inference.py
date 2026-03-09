import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(model_path):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, # Qwen 系列通常使用 bfloat16
        device_map="auto",
        trust_remote_code=True
    )

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
    }

    print("Starting inference...")
    start_time = time.time()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, **sampling_params)
        print(f"Prompt: {prompt}")
        print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print("-" * 20)
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 将默认模型路径改为 Qwen3-0.6B 的容器内路径
    parser.add_argument("--model", type=str, default="/app/data/Qwen3-0.6B")
    args = parser.parse_args()
    main(args.model)
