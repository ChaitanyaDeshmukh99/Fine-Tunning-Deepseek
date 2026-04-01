import argparse

from src.ft_pipeline import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a LoRA adapter")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name/path")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    output = run_inference(
        base_model=args.base_model,
        adapter_dir=args.adapter,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(output)


if __name__ == "__main__":
    main()
