import argparse
from vllm import LLM, SamplingParams

def main(model_path, gpu_utilization):
    # 初始化 vLLM 引擎
    # 对于 12GB 显存，建议降低 gpu_memory_utilization
    # 如果显存依然不足，可以尝试设置 dtype="half" 或使用更小的模型
    llm = LLM(
        model=model_path, 
        trust_remote_code=True,
        gpu_memory_utilization=gpu_utilization,
        max_model_len=4096 # 减小最大长度以节省显存
    )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gpu-utilization", type=float, default=0.8)
    args = parser.parse_args()
    main(args.model, args.gpu_utilization)
