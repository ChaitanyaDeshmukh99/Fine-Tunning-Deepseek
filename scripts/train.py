import argparse

from src.ft_pipeline import load_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.json",
        help="Path to training config JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
