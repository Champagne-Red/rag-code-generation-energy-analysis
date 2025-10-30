# run_llama_k0_windows.py - Full 7-Run Version
# FIXED: Uses corrected power_monitor.py with 0.1s sampling and 10s pauses
import os, time, re, csv, glob, random, statistics, collections, builtins
from typing import List, Dict
import ollama
from datasets import load_dataset
from power_monitor import PowerMonitor

# ===================== CONFIGURATION =====================
MODEL = "llama3.1:8b"
K = 0  # No RAG
SEED = 1337
N_MBPP_TASKS = 50
N_HUMANEVAL_TASKS = 50
TIMEOUT_S = 120

# Sampling controls
FORCE_RESAMPLE = False
BLACKLIST_MBPP = {"mbpp_123"}
BLACKLIST_HUMANEVAL = {
    "HumanEval/137", "HumanEval/136", "HumanEval/149", "HumanEval/150",
    "HumanEval/151", "HumanEval/152", "HumanEval/159", "HumanEval/158",
    "HumanEval/157", "HumanEval/156", "HumanEval/76", "HumanEval/153",
    "HumanEval/154", "HumanEval/155", "HumanEval/162", "HumanEval/148"
}

# Outputs
OUT_DIR = "Results_power"
OUT_MBPP_IDS = os.path.join(OUT_DIR, "mbpp50_ids.txt")
OUT_HUMANEVAL_IDS = os.path.join(OUT_DIR, "humaneval50_ids.txt")

# Generation params
GEN_PARAMS = {"num_predict": 128, "temperature": 0.2, "top_p": 0.95, "keep_alive": 300}

# ===================== DATASET LOADING =====================
SIG_RE = re.compile(r"^\s*def\s+\w+\s*\(.*?\)\s*:\s*$")


def extract_signature_from_code(code: str) -> str:
    for line in code.splitlines():
        if SIG_RE.match(line):
            return line.strip()
    raise ValueError("No signature found")


def normalize_tests(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        lines = raw
    else:
        txt = str(raw).replace(";", "\n")
        lines = [x.strip() for x in txt.splitlines()]
    return [x for x in lines if x.startswith("assert")]


def _build_mbpp_pool():
    try:
        dsd = load_dataset("mbpp", "sanitized")
    except Exception:
        dsd = load_dataset("mbpp")

    pool = []
    for split in ("train", "test", "validation"):
        if split not in dsd:
            continue
        for ex in dsd[split]:
            code = ex.get("code") or ""
            text = ex.get("text") or ex.get("prompt") or ""
            tests = normalize_tests(ex.get("test_list") or ex.get("test") or None)
            try:
                sig = extract_signature_from_code(code)
            except Exception:
                continue
            if not tests or not text:
                continue
            tid = ex.get("task_id", len(pool))
            pool.append({
                "task_id": f"mbpp_{tid}",
                "signature": sig,
                "text": text.strip(),
                "tests": tests,
                "source": "mbpp"
            })
    return pool


def load_mbpp_50():
    os.makedirs(OUT_DIR, exist_ok=True)
    pool = _build_mbpp_pool()
    pool = [t for t in pool if t["task_id"] not in BLACKLIST_MBPP]

    if not FORCE_RESAMPLE and os.path.exists(OUT_MBPP_IDS):
        want = [line.strip() for line in open(OUT_MBPP_IDS, encoding="utf-8") if line.strip()]
        idset = set(want)
        sample = [t for t in pool if t["task_id"] in idset]
        order = {tid: i for i, tid in enumerate(want)}
        sample.sort(key=lambda t: order.get(t["task_id"], 1_000_000))
        print(f"MBPP: Loaded frozen {len(sample)} IDs from {OUT_MBPP_IDS}", flush=True)
    else:
        rnd = random.Random(SEED)
        sample = rnd.sample(pool, min(N_MBPP_TASKS, len(pool)))
        with open(OUT_MBPP_IDS, "w", encoding="utf-8") as f:
            for t in sample:
                f.write(t["task_id"] + "\n")
        print(f"MBPP: Freshly sampled {len(sample)} tasks (seed={SEED})", flush=True)

    return sample


def _build_humaneval_pool():
    dsd = load_dataset("openai_humaneval")
    pool = []

    for ex in dsd["test"]:
        task_id = ex.get("task_id", "")
        prompt = ex.get("prompt", "")
        test = ex.get("test", "")
        entry_point = ex.get("entry_point", "")

        if not prompt or not test or not entry_point:
            continue

        try:
            sig = extract_signature_from_code(prompt)
        except Exception:
            continue

        docstring_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
        if docstring_match:
            text = docstring_match.group(1).strip()
        else:
            text = f"Complete the function {entry_point}"

        test_lines = [line.strip() for line in test.splitlines()
                      if line.strip() and line.strip().startswith("assert")]

        if not test_lines:
            continue

        pool.append({
            "task_id": task_id,
            "signature": sig,
            "text": text,
            "tests": test_lines,
            "entry_point": entry_point,
            "source": "humaneval"
        })

    return pool


def load_humaneval_50():
    os.makedirs(OUT_DIR, exist_ok=True)
    pool = _build_humaneval_pool()
    pool = [t for t in pool if t["task_id"] not in BLACKLIST_HUMANEVAL]

    if not FORCE_RESAMPLE and os.path.exists(OUT_HUMANEVAL_IDS):
        want = [line.strip() for line in open(OUT_HUMANEVAL_IDS, encoding="utf-8") if line.strip()]
        idset = set(want)
        sample = [t for t in pool if t["task_id"] in idset]
        order = {tid: i for i, tid in enumerate(want)}
        sample.sort(key=lambda t: order.get(t["task_id"], 1_000_000))
        print(f"HumanEval: Loaded frozen {len(sample)} IDs from {OUT_HUMANEVAL_IDS}", flush=True)
    else:
        rnd = random.Random(SEED)
        sample = rnd.sample(pool, min(N_HUMANEVAL_TASKS, len(pool)))
        with open(OUT_HUMANEVAL_IDS, "w", encoding="utf-8") as f:
            for t in sample:
                f.write(t["task_id"] + "\n")
        print(f"HumanEval: Freshly sampled {len(sample)} tasks (seed={SEED})", flush=True)

    return sample


# ===================== PROMPT BUILDING =====================
def build_prompt(task_text: str, signature: str, model: str) -> str:
    if model.startswith("llama"):
        return (
            f"# TASK: {task_text}\n"
            f"# Write ONLY Python code. No explanations. No prints.\n"
            f"# Start with this exact signature on the first line:\n"
            f"{signature}\n"
        )
    return ""


# ===================== CODE EXTRACTION & TESTING =====================
def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def run_assert_tests(code: str, asserts: List[str], task_dict: dict) -> bool:
    def _disabled_input(*args, **kwargs):
        raise EOFError("Interactive input is disabled during testing.")

    ns: Dict = {}
    safe_builtins = builtins.__dict__.copy()
    safe_builtins["input"] = _disabled_input
    ns["__builtins__"] = safe_builtins

    try:
        exec(code, ns, ns)
    except Exception:
        return False

    if task_dict.get("source") == "humaneval":
        entry_point = task_dict.get("entry_point")
        if not entry_point or entry_point not in ns:
            return False

        check_code = "def check(candidate):\n"
        for stmt in asserts:
            check_code += "    " + stmt + "\n"

        try:
            exec(check_code, ns, ns)
            ns["check"](ns[entry_point])
            return True
        except Exception:
            return False

    for stmt in asserts:
        try:
            exec(stmt, ns, ns)
        except Exception:
            return False
    return True


# ===================== GENERATION =====================
def generate_code(model: str, prompt: str, timeout_s: int) -> tuple:
    try:
        resp = ollama.generate(model=model, prompt=prompt, options=GEN_PARAMS)
        return resp, "ok"
    except Exception as e:
        return {"response": ""}, "error"


# ===================== RESULTS WRITING =====================
def write_results_header(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "task_id", "source", "model", "k", "pass_at_1",
            "t_generate_s", "status", "prompt_tokens", "gen_tokens"
        ])


def append_result_row(path: str, row: List):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ===================== SUMMARIZATION =====================
def summarize_results(csv_path: str, model: str, k: int):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    if not rows:
        print("No results to summarize.")
        return

    by_source = collections.defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {model} | k={k} | File: {os.path.basename(csv_path)}")
    print(f"{'=' * 80}")
    print(f"{'Source':<12} {'Pass':<6} {'Med Gen(s)':<12} {'Total Tokens':<14} {'Count':<6}")
    print(f"{'-' * 80}")

    for source, grp in sorted(by_source.items()):
        pass_rate = sum(int(x["pass_at_1"]) for x in grp) / len(grp)
        med_gen = statistics.median(float(x["t_generate_s"]) for x in grp)
        total_tokens = sum(int(x.get("gen_tokens", 0)) for x in grp)
        print(f"{source:<12} {pass_rate:.3f}  {med_gen:.6f}    {total_tokens:<14} {len(grp):<6}")

    all_pass = sum(int(r["pass_at_1"]) for r in rows) / len(rows)
    all_gen = statistics.median(float(r["t_generate_s"]) for r in rows)
    all_tokens = sum(int(r.get("gen_tokens", 0)) for r in rows)
    print(f"{'-' * 80}")
    print(f"{'ALL':<12} {all_pass:.3f}  {all_gen:.6f}    {all_tokens:<14} {len(rows):<6}")
    print(f"{'=' * 80}\n")


# ===================== MAIN EXECUTION =====================
def main_run(run_index: int):
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {MODEL} | k={K} (NO RAG) | RUN {run_index}/7")
    print(f"{'=' * 70}\n")

    out_csv = os.path.join(OUT_DIR, f"llama_k{K}_run{run_index}_results.csv")

    # Power monitor setup - skip for run 1 (warmup)
    power_monitor = None
    if run_index > 1:
        power_log = os.path.join(OUT_DIR, f"llama_k{K}_run{run_index}_power.csv")
        power_monitor = PowerMonitor(power_log)
    else:
        print("Run 1/7: System stabilization run. Power not measured.\n")

    write_results_header(out_csv)

    # Load tasks
    mbpp_tasks = load_mbpp_50()
    humaneval_tasks = load_humaneval_50()
    all_tasks = mbpp_tasks + humaneval_tasks

    print(f"\nTotal tasks: {len(all_tasks)} (MBPP: {len(mbpp_tasks)}, HumanEval: {len(humaneval_tasks)})\n")

    # Warm up model
    print("Warming up model...", flush=True)
    for _ in range(2):
        try:
            ollama.generate(model=MODEL, prompt="pass", options={"num_predict": 1, "keep_alive": 300})
        except Exception:
            pass
    print("Model ready.\n")

    # Run all tasks
    for i, task in enumerate(all_tasks, 1):
        prompt = build_prompt(task["text"], task["signature"], MODEL)

        # Start monitoring
        if power_monitor:
            power_monitor.start(task["task_id"])
        t_start = time.perf_counter()

        # Generate
        resp, status = generate_code(MODEL, prompt, TIMEOUT_S)

        # Stop monitoring
        t_generate = time.perf_counter() - t_start
        if power_monitor:
            power_monitor.stop()

        # Test
        code = extract_code(resp.get("response", "")) if status == "ok" else ""
        passed = run_assert_tests(code, task["tests"], task) if code else False

        # Token counts
        prompt_tokens = resp.get('prompt_eval_count', 0)
        gen_tokens = resp.get('eval_count', 0)

        # Save
        append_result_row(out_csv, [
            task["task_id"], task["source"], MODEL, K,
            int(passed), f"{t_generate:.6f}", status,
            prompt_tokens, gen_tokens
        ])

        status_str = " | TIMEOUT" if status == "timeout" else (" | ERROR" if status == "error" else "")
        print(f"[{i:3d}/{len(all_tasks)}] {task['task_id']:<20} | "
              f"pass={int(passed)} | gen={t_generate:.3f}s{status_str}", flush=True)

    # Close power monitor
    if power_monitor:
        power_monitor.close()

    # Summary
    summarize_results(out_csv, MODEL, K)
    print(f"\nResults for run {run_index} saved to: {out_csv}")
    print(f"\n{'=' * 70}")
    print(f"RUN {run_index}/7 COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    for i in range(1, 8):  # 7 runs total
        main_run(i)
        if i < 7:
            print(f"\n{'*' * 70}")
            print(f"Pausing 10 seconds before next run...")  # CHANGED FROM 60
            print(f"{'*' * 70}\n")
            time.sleep(10)  # CHANGED FROM 60

    print(f"\n{'=' * 70}")
    print("ALL 7 RUNS COMPLETE!")
    print(f"{'=' * 70}\n")