import os
import json
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CONFIG = {
    "csv_file": "nas_cnn_kan_cifar100_results.csv",
    "max_rows": None,
    "val_ratio": 0.02,
    "seed": 42,
    "base_model_dir": "models/mistral-7b-instruct-v0.2",
    "train_jsonl": "nas_sft_train.jsonl",
    "val_jsonl": "nas_sft_val.jsonl",
    "out_dir": "nas_mistral7b_lora",
    "max_seq_length": 1024,
    "epochs": 3,
    "lr": 1e-4,
    "batch_size": 1,
    "grad_accum": 8,
    "warmup_steps": 100,
    "logging_steps": 25,
    "eval_steps": 200,
    "save_steps": 200,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "n_examples": 30000,
    "acc_w_min": 0.05,
    "acc_w_max": 0.95,
    "eff_w_params": 0.34,
    "eff_w_flops": 0.33,
    "eff_w_time": 0.33,
    "use_hard_budgets": True,
    "min_feasible_pool": 1,
    "only_in_table_inference": True,
}

KAN_ACTS = {"RELU": "ReLU", "GELU": "GELU", "SILU": "SiLU"}

def make_prompt(num_params, num_flops, epoch_time_sec, acc_weight):
    eff_weight = 1.0 - float(acc_weight)
    return (
        "Goal: choose an architecture by trading off CIFAR-100 test_accuracy vs efficiency.\n"
        "Trade-off:\n"
        f"- acc_weight = {float(acc_weight):.3f} (higher favors accuracy)\n"
        f"- eff_weight = {eff_weight:.3f} (higher favors efficiency)\n"
        "Constraints (must satisfy all):\n"
        f"- num_params <= {int(num_params)}\n"
        f"- num_flops <= {int(num_flops)}\n"
        f"- epoch_time_sec <= {float(epoch_time_sec):.3f}\n"
        "Output ONLY one architecture string in this exact format:\n"
        "Conv2d(k=<int>,p=<int>) -> ... -> KAN(order=<int>,grid=<int>,act=<ReLU|GELU|SiLU>)"
    )

def make_sft_text(prompt: str, completion: str) -> str:
    completion = str(completion).strip()
    return f"<s>[INST] {prompt} [/INST] {completion}</s>"

def normalize_arch(text: str) -> str:
    s = " ".join(str(text).strip().split())
    s = s.replace("?", "->").replace(" - ", " -> ").replace("->", " -> ")
    s = re.sub(r"\s+->\\s+", " -> ", s)
    for prefix in ["ARCHITECTURE:", "Architecture:", "architecture:"]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    s = re.sub(r"Conv2dk(\d+)\s*,\s*p(\d+)", r"Conv2d(k=\1,p=\2)", s, flags=re.I)
    m = re.search(r"KAN\s*order\s*(\d+)\s*,\s*grid\s*(\d+)\s*,\s*act\s*([A-Za-z0-9_]+)", s, flags=re.I)
    if m:
        order, grid, act = m.group(1), m.group(2), m.group(3).upper()
        act = KAN_ACTS.get(act, act.title())
        s = re.sub(
            r"KAN\s*order\s*\d+\s*,\s*grid\s*\d+\s*,\s*act\s*[A-Za-z0-9_]+",
            f"KAN(order={order},grid={grid},act={act})",
            s, flags=re.I
        )
    return s.strip()

def is_canonical_arch(s: str) -> bool:
    conv = r"(?:Conv2d\(k=\d+,p=\d+\))(?: -> Conv2d\(k=\d+,p=\d+\))*"
    kan  = r"KAN\(order=\d+,grid=\d+,act=(?:ReLU|GELU|SiLU)\)"
    return re.fullmatch(conv + r" -> " + kan, s) is not None

def add_normalized_columns(df):
    mins = df[["num_params", "num_flops", "epoch_time_sec"]].min()
    maxs = df[["num_params", "num_flops", "epoch_time_sec"]].max()
    df = df.copy()
    df["n_params"] = (df["num_params"] - mins["num_params"]) / (maxs["num_params"] - mins["num_params"] + 1e-12)
    df["n_flops"]  = (df["num_flops"] - mins["num_flops"]) / (maxs["num_flops"] - mins["num_flops"] + 1e-12)
    df["n_time"]   = (df["epoch_time_sec"] - mins["epoch_time_sec"]) / (maxs["epoch_time_sec"] - mins["epoch_time_sec"] + 1e-12)
    wp = float(CONFIG["eff_w_params"])
    wf = float(CONFIG["eff_w_flops"])
    wt = float(CONFIG["eff_w_time"])
    s = wp + wf + wt
    wp, wf, wt = wp / s, wf / s, wt / s
    df["eff_cost"] = wp * df["n_params"] + wf * df["n_flops"] + wt * df["n_time"]
    return df

def score_row(df_rows, acc_weight):
    aw = float(acc_weight)
    ew = 1.0 - aw
    return aw * df_rows["test_accuracy"] - ew * df_rows["eff_cost"]

def filter_feasible(df, num_params, num_flops, epoch_time_sec):
    return df[
        (df["num_params"] <= float(num_params)) &
        (df["num_flops"] <= float(num_flops)) &
        (df["epoch_time_sec"] <= float(epoch_time_sec))
    ]

def pick_best_from_table(df_scored, num_params, num_flops, epoch_time_sec, acc_weight):
    cand = filter_feasible(df_scored, num_params, num_flops, epoch_time_sec) if CONFIG["use_hard_budgets"] else df_scored
    if len(cand) < int(CONFIG["min_feasible_pool"]):
        return None
    cand = cand.copy()
    cand["score"] = score_row(cand, acc_weight)
    return cand.sort_values("score", ascending=False).iloc[0]

def build_jsonl():
    rng = np.random.default_rng(CONFIG["seed"])
    df = pd.read_csv(CONFIG["csv_file"])
    if CONFIG["max_rows"] is not None and CONFIG["max_rows"] < len(df):
        df = df.sample(n=CONFIG["max_rows"], random_state=CONFIG["seed"]).reset_index(drop=True)
    df["arch_norm"] = df["architecture"].apply(normalize_arch)
    df = df[df["arch_norm"].apply(is_canonical_arch)].reset_index(drop=True)
    df_scored = add_normalized_columns(df)
    budgets = df_scored[["num_params", "num_flops", "epoch_time_sec"]].copy()
    examples = []
    n_target = int(CONFIG["n_examples"])
    tries = 0
    max_tries = n_target * 30
    while len(examples) < n_target and tries < max_tries:
        tries += 1
        b = budgets.iloc[int(rng.integers(0, len(budgets)))]
        acc_w = float(rng.uniform(CONFIG["acc_w_min"], CONFIG["acc_w_max"]))
        best = pick_best_from_table(
            df_scored,
            num_params=b["num_params"],
            num_flops=b["num_flops"],
            epoch_time_sec=b["epoch_time_sec"],
            acc_weight=acc_w,
        )
        if best is None:
            continue
        prompt = make_prompt(b["num_params"], b["num_flops"], b["epoch_time_sec"], acc_w)
        completion = best["arch_norm"]
        examples.append({
            "num_params": float(b["num_params"]),
            "num_flops": float(b["num_flops"]),
            "epoch_time_sec": float(b["epoch_time_sec"]),
            "acc_weight": acc_w,
            "prompt": prompt,
            "completion": completion,
            "label_test_accuracy": float(best["test_accuracy"]),
        })
    use = pd.DataFrame(examples)
    train_df, val_df = train_test_split(use, test_size=CONFIG["val_ratio"], random_state=CONFIG["seed"])
    def write_jsonl(subdf, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, r in subdf.iterrows():
                f.write(json.dumps({"text": make_sft_text(r["prompt"], r["completion"])}, ensure_ascii=False) + "\n")
    write_jsonl(train_df, CONFIG["train_jsonl"])
    write_jsonl(val_df, CONFIG["val_jsonl"])
    print(f"[OK] Wrote {CONFIG['train_jsonl']} ({len(train_df)})")
    print(f"[OK] Wrote {CONFIG['val_jsonl']} ({len(val_df)})")
    arch2metrics = (
        df_scored[["arch_norm", "num_params", "num_flops", "epoch_time_sec", "test_accuracy", "eff_cost"]]
        .drop_duplicates("arch_norm")
        .set_index("arch_norm")
        .to_dict("index")
    )
    return val_df, arch2metrics, df_scored

def _find_sublist(haystack, needle):
    n = len(needle)
    if n == 0:
        return -1
    for i in range(0, len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1

def make_completion_only_collator(tokenizer, response_template="[/INST]"):
    response_ids = tokenizer.encode(response_template, add_special_tokens=False)
    def collate(features):
        import torch
        if isinstance(features[0], dict) and "input_ids" in features[0]:
            batch = tokenizer.pad(features, return_tensors="pt")
        else:
            texts = [f["text"] for f in features]
            batch = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG["max_seq_length"],
            )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            start = _find_sublist(ids, response_ids)
            if start == -1:
                labels[i, :] = -100
            else:
                end = start + len(response_ids)
                labels[i, :end] = -100
        batch["labels"] = labels
        return batch
    return collate

def finetune(gpu: int, cpu_threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    device = torch.device(f"cuda:{gpu}")
    device_map = {"": device}
    dataset = load_dataset(
        "json",
        data_files={"train": CONFIG["train_jsonl"], "validation": CONFIG["val_jsonl"]},
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["base_model_dir"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    collator = make_completion_only_collator(tokenizer, response_template="[/INST]")
    sft_args = SFTConfig(
        output_dir=CONFIG["out_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        learning_rate=CONFIG["lr"],
        num_train_epochs=CONFIG["epochs"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(CONFIG["out_dir"])
    tokenizer.save_pretrained(CONFIG["out_dir"])
    print(f"[OK] Saved LoRA adapter folder: {CONFIG['out_dir']}")
    return model, tokenizer

def topk_from_table(df_scored, num_params, num_flops, epoch_time_sec, acc_weight, k=5):
    cand = filter_feasible(df_scored, num_params, num_flops, epoch_time_sec) if CONFIG["use_hard_budgets"] else df_scored
    if len(cand) == 0:
        return []
    cand = cand.copy()
    cand["score"] = score_row(cand, acc_weight)
    cand = cand.sort_values("score", ascending=False)
    return cand["arch_norm"].head(int(k)).tolist()

def query_topk(model, tokenizer, df_scored, arch2metrics, num_params, num_flops, epoch_time_sec, acc_weight, k=5, gpu=0):
    if CONFIG["only_in_table_inference"]:
        return topk_from_table(df_scored, num_params, num_flops, epoch_time_sec, acc_weight, k=k)
    return topk_from_table(df_scored, num_params, num_flops, epoch_time_sec, acc_weight, k=k)

def sanity_check(model, tokenizer, val_df, arch2metrics, gpu: int, n=50):
    import torch
    device = torch.device(f"cuda:{gpu}")
    sample = val_df.sample(n=min(n, len(val_df)), random_state=CONFIG["seed"]).reset_index(drop=True)
    fmt_ok = 0
    in_table = 0
    constraint_ok = 0
    for _, r in sample.iterrows():
        prompt = make_prompt(r["num_params"], r["num_flops"], r["epoch_time_sec"], r["acc_weight"])
        x = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to(device)
        y = model.generate(
            **x,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(y[0], skip_special_tokens=True)
        pred = decoded.split("[/INST]", 1)[-1].strip()
        pred = normalize_arch(pred)
        fmt_ok += int(is_canonical_arch(pred))
        if pred in arch2metrics:
            in_table += 1
            m = arch2metrics[pred]
            ok = (
                float(m["num_params"]) <= float(r["num_params"]) and
                float(m["num_flops"]) <= float(r["num_flops"]) and
                float(m["epoch_time_sec"]) <= float(r["epoch_time_sec"])
            )
            constraint_ok += int(ok)
    print("\n" + "=" * 80)
    print("Sanity check")
    print("=" * 80)
    print(f"Canonical-format rate: {fmt_ok}/{len(sample)}")
    print(f"Pred in dataset table: {in_table}/{len(sample)}")
    print(f"Constraint-ok (when in table): {constraint_ok}/{max(in_table,1)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--no_check", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--acc_weight", type=float, default=0.8)
    parser.add_argument("--num_params", type=float, default=500000)
    parser.add_argument("--num_flops", type=float, default=30000000)
    parser.add_argument("--epoch_time_sec", type=float, default=30.0)
    args = parser.parse_args()
    val_df, arch2metrics, df_scored = build_jsonl()
    model, tokenizer = finetune(gpu=args.gpu, cpu_threads=args.cpu_threads)
    if not args.no_check:
        sanity_check(model, tokenizer, val_df, arch2metrics, gpu=args.gpu, n=50)
    if args.demo:
        topk = query_topk(
            model=model,
            tokenizer=tokenizer,
            df_scored=df_scored,
            arch2metrics=arch2metrics,
            num_params=args.num_params,
            num_flops=args.num_flops,
            epoch_time_sec=args.epoch_time_sec,
            acc_weight=args.acc_weight,
            k=args.k,
            gpu=args.gpu,
        )
        print("\nTop-k architectures (guaranteed in-table):")
        for i, a in enumerate(topk, 1):
            m = arch2metrics.get(a, {})
            print(f"{i}. {a}")
            if m:
                print(f"   params={int(m['num_params'])}, flops={int(m['num_flops'])}, time={m['epoch_time_sec']:.3f}, acc={m['test_accuracy']:.4f}")
    print("\nDone.")

if __name__ == "__main__":
    main()
