# -*- coding: utf-8 -*-
"""Generate synthetic tweets mimicking different mental‑health states.

Removed the *Personality Disorder* category entirely.
Compatible with TF ≥2.16 when tf‑keras is used (see notes below).
"""
import os
import torch
import random
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401  # needed for TF‑Hub BERT
import pickle
from xgboost import DMatrix
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
# ============================ DEVICE CONFIG ============================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Usando dispositivo: {device}")

# ============================ PATHS ============================
model_id = "meta-llama/Llama-2-7b-hf"
trained_model_dir = "/home/cardiaca/QLoRA_LLaMA2_7B_Checkpoints_v11_3/checkpoint-4500"

# ============================ BITSANDBYTES CONFIG ============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ============================ FIXED LABEL MAPPING ============================
label_mapping = {
    "Normal": 0,
    "Depression": 1,
    "Suicidal": 2,
    "Anxiety": 3,
    "Bi-polar": 4,
    "Stress": 5,
}

# Build CLASSES list in the same order
CLASSES = [None] * len(label_mapping)

for label, idx in label_mapping.items():
    CLASSES[idx] = label
print("CLASSES order:", CLASSES)
CLASSES.append("Unknown")

# ============================ PROMPT EXAMPLES ============================
PROMPTS_JSON = "prompt_examples.json"
with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
    prompt_examples = json.load(f)

# ============================ CLINICAL DESCRIPTIONS ============================
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
        #f"### Description\n{desc}\n\n"
        #f"{typical_words_block}"
        #f"{negative_block}"
        #f"### Tweet examples\n{examples_txt}\n\n"
        #f"### Instruction\n"
        #f"Write **one new tweet (max 20 words)** that captures the mental state of someone with **{status}**.  "
        #f"Do NOT name or allude to any other disorders; the tone and details must make **{status}** unmistakable. "
        #f"{hint}\n\n"
        f"### Your tweet:"
    )
    return prompt


# ============================ TOKENIZER ============================

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# ============================ LOAD BASE MODEL ============================
print("🔄 Cargando modelo base con offload a CPU si es necesario...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# ============================ LOAD LoRA ADAPTER ============================
print("🔄 Cargando adaptadores LoRA...")
model = PeftModel.from_pretrained(base_model, trained_model_dir)
model.eval()

# ============================ BERT FOR EMBEDDINGS (CPU) ============================
bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


def get_bert_embedding(text: str) -> np.ndarray:
    inputs = bert_preprocess([text])
    embedding = bert_encoder(inputs)["pooled_output"]
    return embedding.numpy()[0]

# ============================ LOAD XGBOOST MODEL ============================
# with open("/home/cardiaca/xgboost_mental_health_model.pkl", "rb") as f:
#     xgb_model = pickle.load(f)
import xgboost as xgb
xgb_model = xgb.Booster()
xgb_model.load_model("/home/cardiaca/xboost_6class_prob.json") 
# ============================ TUNED GENERATION PARAMETERS ============================
generation_params = {
    "Anxiety": {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.2},
    "Bi-polar": {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.2},
    "Depression": {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.2},
    "Normal": {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.2},
    "Stress": {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.2},
    "Suicidal": {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.2},
}

# ============================ TWEET GENERATOR ============================

def generar_tweet(
    disorder: str,
    max_new_tokens: int = 96,
    n_candidates: int = 3,
):
    params = generation_params[disorder]
    prompt = create_prompt({"status": disorder})

    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            temperature=params["temperature"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=n_candidates,
        )

    # Decodifica y limpia cada salida generada
    candidates = [
        tokenizer.decode(out, skip_special_tokens=True).replace(prompt, "").strip()
        for out in outputs
    ]

    # =================== EMBEDDING EN BATCH ===================
    # Entradas de texto → tf.Tensor de shape (n, 768)
    inputs = bert_preprocess(candidates)
    embedding_tensor = bert_encoder(inputs)["pooled_output"]
    embeddings = embedding_tensor.numpy()  # shape (n_candidates, 768)

    # =================== CLASIFICACIÓN XGBOOST ===================
    full_probs = xgb_model.predict(DMatrix(embeddings))  # shape = (n_candidates, 6)
    probs = full_probs[:, :6]

    desired_idx = label_mapping[disorder]
    desired_probs = probs[:, desired_idx]

    best_i = int(desired_probs.argmax())
    selected_tweet = candidates[best_i]
    selected_prob = desired_probs[best_i]
    pred_idx_global = int(probs[best_i].argmax())
    selected_label = CLASSES[pred_idx_global]

    print(f"\n🧠 Generados para: {disorder}")
    for i, (t, prob_row) in enumerate(zip(candidates, probs), 1):
        pred_idx_row = int(prob_row.argmax())
        pred_label = CLASSES[pred_idx_row]
        desired_p = prob_row[desired_idx]
        print(f" Candidat {i}: {t[:80]} → pred = {pred_label} | p={desired_p:.3f}")

    print(f"\n✅ Seleccionado: {selected_tweet[:80]} → {selected_label} | p={selected_prob:.3f}")
    return selected_tweet, selected_prob


# ============================ GENERATION LOOP ============================
if __name__ == "__main__":
    rows = []
    n_samples_per_class = 100
    output_csv = "tweets_all_v2_llama.csv"

    for disorder in label_mapping.keys():
        print(f"\n🔹 Generando {n_samples_per_class} tweets para clase '{disorder}'")

        for _ in tqdm(range(n_samples_per_class), desc=f"{disorder[:10]}"):
            tweet, p_sel = generar_tweet(disorder, n_candidates=3)

            emb = get_bert_embedding(tweet)
            pred = int(xgb_model.predict(DMatrix([emb]))[0].argmax())

            rows.append({
                "status":          disorder,
                "statement":       tweet,
                "predicted_class": CLASSES[pred],
                "correct":         int(pred == label_mapping[disorder]),
                "probability":     p_sel,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ Tweets guardados en '{output_csv}'")


    # disorder = "Suicidal"
    # print(f"\n🔹 Generando {n_samples_per_class} tweets para clase '{disorder}'")

    # for _ in tqdm(range(n_samples_per_class), desc=disorder[:10]):
    #     tweet, p_sel = generar_tweet(disorder, n_candidates=3)

    #     emb = get_bert_embedding(tweet)
    #     pred = int(xgb_model.predict(DMatrix([emb]))[0].argmax())

    #     rows.append({
    #         "status":          disorder,
    #         "statement":       tweet,
    #         "predicted_class": CLASSES[pred],
    #         "correct":         int(pred == label_mapping[disorder]),
    #         "probability":     p_sel,
    #     })

    # df_out = pd.DataFrame(rows)
    # df_out.to_csv(output_csv, index=False)
    # print(f"\n✅ Tweets guardados en '{output_csv}'")
