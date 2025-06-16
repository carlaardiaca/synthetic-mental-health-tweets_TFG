import os
import torch
import random
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================ CONFIG ============================
model_id = "meta-llama/Llama-2-7b-chat-hf"  # o "meta-llama/Llama-2-7b-chat-hf"
n_samples_per_class = 100
output_dir = "outputs_RAW_LLAMA"
disorders = ["Normal", "Depression", "Suicidal", "Anxiety", "Bi-polar", "Stress"]

os.makedirs(output_dir, exist_ok=True)

# ============================ PROMPT BUILDER ============================
def build_prompt(disorder):
    return (
        f"### Disorder: {disorder}\n"
        f"Write a single tweet (max 20 words) that reflects the mental and emotional state "
        f"of a person experiencing {disorder}. The tone should be realistic and evocative.\n\n"
        f"### Tweet:"
    )

# ============================ GENERATION ============================
def generate_disorder(disorder, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🔄 Loading {model_id} on GPU {gpu_id} for {disorder}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    rows = []
    for _ in tqdm(range(n_samples_per_class), desc=disorder):
        prompt = build_prompt(disorder)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.75,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        rows.append({"status": disorder, "statement": text})

    out_path = os.path.join(output_dir, f"tweets_{disorder}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")

# ============================ MULTIPROCESS ============================
if __name__ == "__main__":
    processes = []
    for gpu_id, disorder in enumerate(disorders):
        p = Process(target=generate_disorder, args=(disorder, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ All generations complete.")
