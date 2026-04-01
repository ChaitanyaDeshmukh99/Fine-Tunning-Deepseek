import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    model_name: str
    dataset_path: str
    output_dir: str
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    use_wandb: bool
    wandb_project: str


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainConfig(**data)


def _format_record(record: dict[str, Any]) -> str:
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    output_text = record.get("output", "").strip()

    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{output_text}"
    )
    return prompt


def load_jsonl_dataset(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    return Dataset.from_list(rows)


def build_model_and_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    return model, tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        prompts = [_format_record(row) for row in batch["_row"]]
        tokens = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    data_with_rows = Dataset.from_dict({"_row": list(dataset)})
    tokenized = data_with_rows.map(_tokenize, batched=True, remove_columns=["_row"])
    return tokenized


def run_training(config: TrainConfig) -> None:
    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    model, tokenizer = build_model_and_tokenizer(config.model_name)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    train_dataset = load_jsonl_dataset(config.dataset_path)
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, config.max_length)

    train_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        report_to=report_to,
        fp16=torch.cuda.is_available(),
        bf16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def run_inference(
    base_model: str,
    adapter_dir: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
