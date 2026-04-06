# DeepSeek LoRA Adaptation Lab

This repository contains my end-to-end workflow for fine-tuning `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` with LoRA adapters and running local adapter-based inference.

## What Is Included

- Notebook experimentation in `Distill_Model_FT.ipynb`
- Script-based training pipeline in `scripts/train.py`
- Adapter inference CLI in `scripts/chat.py`
- Reusable training utilities in `src/ft_pipeline.py`
- Config-driven runs with `configs/train_config.json`
- Example JSONL dataset in `data/sample_train.jsonl`

## Project Structure

```
Fine-Tune-DeepSeek-main/
|-- Distill_Model_FT.ipynb
|-- configs/
|   `-- train_config.json
|-- data/
|   `-- sample_train.jsonl
|-- scripts/
|   |-- train.py
|   `-- chat.py
|-- src/
|   |-- __init__.py
|   `-- ft_pipeline.py
|-- requirements.txt
`-- README.md
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create your dataset in JSONL format (or reuse `data/sample_train.jsonl`) and point `dataset_path` in `configs/train_config.json` to it.

Each row should follow:

```json
{"instruction":"...","input":"...","output":"..."}
```

3. Run fine-tuning:

```bash
python scripts/train.py --config configs/train_config.json
```

4. Run inference with trained adapter:

```bash
python scripts/chat.py \
	--base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
	--adapter outputs/deepseek-lora \
	--prompt "Explain gradient clipping in simple terms."
```

## Configuration

Tune these values in `configs/train_config.json`:

- `max_length`: token length per training sample
- `batch_size` and `gradient_accumulation_steps`: effective batch size
- `learning_rate`, `num_train_epochs`, `warmup_ratio`: optimization controls
- `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`: adapter behavior
- `use_wandb` and `wandb_project`: experiment tracking

## Notes

- Use GPU runtime for practical training times.
- If you use private models/datasets, set your Hugging Face token before training.
- Adapter weights are saved under the configured `output_dir`.

