# -*- coding: utf-8 -*-
# QLoRA fine‑tuning para tweets de trastornos mentales (sin Accelerate)

import os
import math
import json
import random
import time

import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# ────────────────────────────
# 🔧  Configuración general
# ────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "7"        # GPU a utilizar
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # evitar warns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Usando dispositivo: {device}")

BASE_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.1" #"meta-llama/Llama-2-7b-hf"
CHECKPOINT_DIR  = "./QLoRA_Mistral_7B_Checkpoints_v12_3"
CSV_PATH        = "/home/cardiaca/Combined Data.csv"
PROMPTS_JSON    = "prompt_examples.json"

# Longitud máxima (input + target) ──────────────────────────
MAX_LEN = 256   # 256 tokens

# ────────────────────────────
# 📄  Cargar ejemplos de prompt
# ────────────────────────────
with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
    prompt_examples = json.load(f)
# Descripciones clínicas breves y específicas
descriptions = {
    "Normal": (
        "Stable mood with no clinical distress; sleeps 7–9 h/night, normal energy and appetite; "
        "daily routines uninterrupted and emotional reactions in proportion to events."
    ),
    "Depression": (
        "Low mood or loss of interest ≥2 weeks; fatigue, guilt, slowed thinking or appetite/sleep changes; "
        "social, work or school life impaired."
    ),
    "Bi-polar": (
        "Distinct episodes of abnormally elevated mood/energy (mania/ hypomania)—euphoria, impulsivity, "
        "reduced need for sleep—alternating with major depressive episodes and big shifts in self-esteem."
    ),
    "Suicidal": (
        "Recurrent thoughts of death or suicide, with or without a plan; feelings of unbearable hopelessness "
        "or being a burden to others."
    ),
    "Anxiety": (
        "Excessive, uncontrollable worry most days ≥6 months; physical tension, palpitations, restlessness, "
        "and difficulty concentrating."
    ),
    "Stress": (
        "Short-term overload reaction to a specific pressure (deadline, exam, crisis): irritability, "
        "racing thoughts, sleep disturbance that eases once the stressor ends."
    )
}

# Pistas extra para cada categoría reforzar su singularidad
hints = {
    "Normal":       "Keep tone neutral and routine—no clinical features.",
    "Depression":   "Mention loss of pleasure, exhaustion or guilt.",
    "Bi-polar":     "Hint at racing highs and crushing lows.",
    "Suicidal":     "Use imagery of finality, exit or unbearable burden.",
    "Anxiety":      "Highlight uncontrollable worry or physical tension.",
    "Stress":       "Tie it to an external pressure (deadline, crisis)."
    
}

manual_top_words = {
    "Anxiety": [
        "anxiety", "something", "feeling", "health", "heart", "pain", "symptoms", "also"
    ],
    "Normal": [
        "normal", "good", "work", "still", "got", "today", "routine", "energy"
    ],
    "Depression": [
        "depression", "cannot", "guilt", "fatigue", "hopeless", "crying", "empty", "useless"
    ],
    "Suicidal": [
        "suicidal", "cannot", "anymore", "die", "burden", "goodbye", "end", "worthless"
    ],
    "Stress": [
        "stress", "work", "help", "anxiety", "things", "pressure", "deadline", "irritable"
    ],
    "Bi-polar": [
        "bipolar", "anyone", "meds", "years", "mania", "impulsive", "crash", "ups"
    ]
}


def create_prompt(row, max_examples=5, deterministic=True):
    status = row["status"]
    ex_list = prompt_examples.get(status, [])

    # Elegir ejemplos
    if deterministic:
        sampled = ex_list[:max_examples]
    else:
        sampled = random.sample(ex_list, min(max_examples, len(ex_list)))

    # Configura els negative examples per separar de Depression
    negative_block = ""
    if status == "Bi-polar":
        # Extraer más ejemplos de clases difíciles
        neg_exs_depr = prompt_examples.get("Depression", [])[:3]
        neg_exs_anx = prompt_examples.get("Anxiety", [])[:2]
        neg_exs_str = prompt_examples.get("Stress", [])[:2]

        # Combinar negativos
        neg_exs = (
            [(e, "Depression") for e in neg_exs_depr] +
            [(e, "Anxiety") for e in neg_exs_anx] +
            [(e, "Stress") for e in neg_exs_str]
        )

        # Construir bloque de negativos
        negative_block = (
            f"### Negative Examples (Not {status})\n"
            + "\n".join(f"- \"{e}\"  ← {cls}" for e, cls in neg_exs)
            + "\n\n"
        )


    # Bloc de paraules representatives
    typical_words = manual_top_words.get(status, [])
    typical_words_block = ""
    if typical_words:
        typical_words_block = f"### Typical Words\n{', '.join(typical_words)}\n\n"


    # Formatear ejemplos en estilo tweet
    examples_txt = "\n".join(f"- {e.strip()}" for e in sampled)
    
    desc = descriptions.get(status, "")
    hint = hints.get(status, "")

    # Prompt final
    prompt = (
        f"### Disorder = {status}\n"
        f"### Description\n{desc}\n\n"
        f"{typical_words_block}"
        f"{negative_block}"
        f"### Tweet examples\n{examples_txt}\n\n"
        f"### Instruction\n"
        f"Write **one new tweet (max 20 words)** that captures the mental state of someone with **{status}**.  "
        f"Do NOT name or allude to any other disorders; the tone and details must make **{status}** unmistakable. "
        f"{hint}\n\n"
        f"### Your tweet:"
    )
    return prompt



# ────────────────────────────
# 🗄️  Cargar y limpiar CSV
# ────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df["status"] != "Personality Disorder"]
df.drop_duplicates(inplace=True)
df.dropna(subset=["statement", "status"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Filtrar tweets extremadamente largos
lens = df["statement"].str.count(" ") + 1      # nº palabras ≈ nº espacios + 1
df   = df[lens <= 200]

# Balancear (oversampling) ─ 2 k por clase
df = (
    df
    .groupby("status")
    .apply(lambda g: g.sample(n=2500, replace=True, random_state=42))
    .reset_index(drop=True)
)

# Construir columnas input/target
df["input"]  = df.apply(create_prompt, axis=1)
df["target"] = df["statement"]

# ────────────────────────────
# 🤗  Dataset → train/val/test
# ────────────────────────────
ds = Dataset.from_pandas(df[["input", "target"]])
ds = ds.train_test_split(test_size=0.20, seed=42)
train_valid = ds["train"].train_test_split(test_size=0.125, seed=42)

dataset = DatasetDict({
    "train":      train_valid["train"],
    "validation": train_valid["test"],
    "test":       ds["test"]
})

# ────────────────────────────
# 🔤  Tokenizer y modelo base
# ────────────────────────────
print("🔤 Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID,
                                          use_fast=False,
                                          use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit           = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type    = "nf4",
    bnb_4bit_compute_dtype = torch.float16
)

print("📦 Cargando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",         
    trust_remote_code=True,
    use_auth_token=True
)

# ────────────────────────────
# 🪶  Configuración LoRA
# ────────────────────────────
lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

# ────────────────────────────
# 🏷️  Tokenización del dataset
# ────────────────────────────
def tokenize_fn(batch):
    full_texts = [f"{inp} {tgt}" for inp, tgt in zip(batch["input"], batch["target"])]

    model_inputs = tokenizer(
        full_texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )

    labels = tokenizer(
        batch["target"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["input", "target"])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ────────────────────────────
# 📊  Callback de Perplexity
# ────────────────────────────
class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            print(f"\n📉 Eval loss: {metrics['eval_loss']:.4f} | 🨯 Perplexity: {ppl:.2f}")

# ────────────────────────────
# ⚙️  Parámetros de entrenamiento
# ────────────────────────────
training_args = TrainingArguments(
    output_dir             = CHECKPOINT_DIR,
    overwrite_output_dir   = True,
    num_train_epochs       = 4,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    save_safetensors       = True,

    evaluation_strategy    = "steps",
    eval_steps             = 500,
    save_strategy          = "steps",
    save_steps             = 500,

    logging_strategy       = "steps",
    logging_steps          = 50,
    logging_dir            = "./logs",

    load_best_model_at_end = True,
    metric_for_best_model  = "eval_loss",
    greater_is_better      = False,
    save_total_limit       = 50,

    learning_rate          = 2e-4,
    warmup_steps           = 100,
    lr_scheduler_type      = "cosine",

    fp16                  = True,
    report_to             = "none",
    label_names           = ["labels"],
    optim                 = "adamw_torch_fused"
)

# ────────────────────────────
# 🏋️‍♂️  Trainer
# ────────────────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_ds["train"],
    eval_dataset    = tokenized_ds["validation"],
    data_collator   = data_collator,
    callbacks       = [PerplexityCallback()]
)

# ────────────────────────────
# 🔄  Reanudar desde último checkpoint
# ────────────────────────────
ckpts = sorted(
    [os.path.join(CHECKPOINT_DIR, d) for d in os.listdir(CHECKPOINT_DIR)
     if d.startswith("checkpoint") and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))],
    key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else -1
)
last_ckpt = ckpts[-1] if ckpts else None
print("📦 Último checkpoint:", last_ckpt or "ninguno")

# ────────────────────────────
# 🚀  Entrenamiento
# ────────────────────────────
print("🚀 Iniciando entrenamiento...")
trainer.train(resume_from_checkpoint=last_ckpt)

# ────────────────────────────
# 📊  Evaluación final
# ────────────────────────────
print("📊 Evaluando en test...")
metrics = trainer.evaluate(tokenized_ds["test"])
print(metrics)

# ────────────────────────────
# 💾  Guardar modelo
# ────────────────────────────
print("💾 Guardando modelo y tokenizer...")
trainer.save_model(CHECKPOINT_DIR)
tokenizer.save_pretrained(CHECKPOINT_DIR)
print(f"✅ Modelo guardado en: {CHECKPOINT_DIR}")
