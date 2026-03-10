import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(model_path):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型，使用 bfloat16 精度
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
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
        "temperature": 0.6,
        "top_p": 0.95,
    }

    print("\n--- Starting Transformers Inference ---")
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        print(f"\nTest {i+1}: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 记录单次生成时间
        step_start = time.time()
        outputs = model.generate(**inputs, **sampling_params)
        step_end = time.time()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        print(f"Time taken: {step_end - step_start:.2f}s")
    
    end_time = time.time()
    print(f"\nTotal time for 3 prompts: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../data/Qwen3-0.6B")
    args = parser.parse_args()
    main(args.model)
