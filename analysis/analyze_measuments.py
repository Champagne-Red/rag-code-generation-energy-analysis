import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
import warnings

# === CONFIG ===

RESULTS_DIR = "Results_power"  # path to your folder
MODELS = ["llama", "starcoder"]
KS = [0, 3]  # 0 = no RAG, 3 = RAG
RUNS = list(range(2, 8))  # Process runs 2 through 7
K_NAMES = {'k0': 'No RAG', 'k3': 'RAG'}
COLORS_K = {'k0': '#1f77b4', 'k3': '#2ca02c'}  # No RAG = Blue, RAG = Green
COLORS_STACKED = {
    'k0_cpu': '#1f77b4',  # Blue
    'k0_gpu': '#2ca02c',  # Green
    'k3_cpu': '#aec7e8',  # Light Blue
    'k3_gpu': '#98df8a'  # Light Green
}
COLORS_M = {'llama': '#1f77b4', 'starcoder': '#2ca02c'}  # For plots 9 & 11
COLORS_PLOT7 = {
    ('llama', 'k0'): '#1f77b4',
    ('llama', 'k3'): '#aec7e8',
    ('starcoder', 'k0'): '#2ca02c',
    ('starcoder', 'k3'): '#98df8a'
}
LINESTYLES_PLOT7 = {
    ('llama', 'k0'): '-',
    ('llama', 'k3'): '--',
    ('starcoder', 'k0'): '-',
    ('starcoder', 'k3'): '--'
}

# Define specific runs to skip
SKIP_RUNS = [
    ("starcoder", "k0", 6)
]


# To disable skipping and run all files, set: SKIP_RUNS = []

# === HELPER FUNCTIONS ===

def load_power_log(path, load_raw_df=False):
    """
    Compute total energy, avg power (total, cpu, gpu), and duration from log.
    Optionally returns the raw DataFrame for time-series plots.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return None

    if not {"time_s", "cpu_w", "gpu_w"}.issubset(df.columns):
        print(f"⚠️ Missing cols in {path}")
        return None

    df = df.sort_values(by="time_s").reset_index()

    if df.empty or len(df) < 2:
        print(f"⚠️ Not enough data in {path}")
        return None

    df["total_w"] = df["cpu_w"] + df["gpu_w"]
    avg_power = df["total_w"].mean()
    avg_cpu_power = df["cpu_w"].mean()
    avg_gpu_power = df["gpu_w"].mean()

    # --- MODIFICATION BASED ON USER FEEDBACK ---
    # User reported avg time per task is 100x too small (e.g., 0.02s instead of 2s).
    # This implies the 'time_s' column in the CSV is not in seconds,
    # but in hectoseconds (units of 100 seconds).
    # We are scaling it by 100 here to correct for that assumed data format.

    total_duration_raw = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
    total_duration_s = total_duration_raw * 100.0

    # Add a warning if the original value was already large, suggesting this fix might be wrong
    if total_duration_raw > 20:  # If original duration > 20 (i.e., 2000s)
        print(
            f"⚠️ WARNING: Original time in {path} was {total_duration_raw:.2f}. Scaling by 100 to {total_duration_s:.2f}s based on user report. If this is wrong, please edit 'load_power_log' function and remove the `* 100.0`.")
    # -------------------------------------------

    total_energy = avg_power * total_duration_s
    total_cpu_energy = avg_cpu_power * total_duration_s
    total_gpu_energy = avg_gpu_power * total_duration_s

    if total_energy < 0:
        print(f"⚠️ Negative total energy ({total_energy} J) calculated for {path}. Check data.")
        total_energy = 0
        total_cpu_energy = 0
        total_gpu_energy = 0

    raw_df_to_return = df if load_raw_df else None

    return (total_energy, avg_power, avg_cpu_power, avg_gpu_power, total_cpu_energy,
            total_gpu_energy, total_duration_s, raw_df_to_return)


def load_results(path, load_raw_pass1=False):
    """
    Load run results: tokens, accuracy, num_answers, successful_tasks.
    Optionally returns the raw pass@1 series for boxplots.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return None

    if not {"pass_at_1", "gen_tokens", "prompt_tokens"}.issubset(df.columns):
        print(f"⚠️ Missing cols in {path}")
        return None

    total_tokens = df["gen_tokens"].sum() + df["prompt_tokens"].sum()
    avg_accuracy = df["pass_at_1"].mean()
    num_answers = len(df)
    successful_tasks = df["pass_at_1"].sum()
    raw_pass1_to_return = df["pass_at_1"] if load_raw_pass1 else None

    return total_tokens, avg_accuracy, num_answers, successful_tasks, raw_pass1_to_return


# === MAIN AGGREGATION ===

records = []

for model in MODELS:
    for k in KS:
        for run in RUNS:
            k_str = f"k{k}"

            if (model, k_str, run) in SKIP_RUNS:
                print(f"🚫 SKIPPING run (as configured): {model} {k_str} run {run}")
                continue

            result_path = f"{RESULTS_DIR}/{model}_{k_str}_run{run}_results.csv"
            power_path = f"{RESULTS_DIR}/{model}_{k_str}_run{run}_power_log.csv"

            if not (os.path.exists(result_path) and os.path.exists(power_path)):
                print(f"🔍 Missing file for {model} {k_str} run {run}")
                continue

            power_data = load_power_log(power_path)
            result_data = load_results(result_path)

            if power_data and result_data:
                total_energy, avg_power, avg_cpu_power, avg_gpu_power, total_cpu_energy, total_gpu_energy, total_duration_s, _ = power_data
                total_tokens, avg_accuracy, num_answers, successful_tasks, _ = result_data

                power_per_token = None
                if total_tokens > 0:
                    power_per_token = total_energy / total_tokens
                elif total_energy > 0:
                    print(f"⚠️ 0 tokens but {total_energy} J for {model} {k_str} run {run}")

                energy_per_answer_J, avg_time_per_answer_s, tokens_per_answer = None, None, None
                if num_answers > 0:
                    energy_per_answer_J = total_energy / num_answers
                    avg_time_per_answer_s = total_duration_s / num_answers
                    tokens_per_answer = total_tokens / num_answers
                else:
                    print(f"⚠️ 0 answers found for {model} {k_str} run {run}")

                energy_per_solved_task_J = None
                if successful_tasks > 0:
                    energy_per_solved_task_J = total_energy / successful_tasks
                else:
                    energy_per_solved_task_J = np.inf

                gpu_seconds = total_duration_s

                records.append({
                    "model": model, "k": k_str, "run": run,
                    "total_energy_J": total_energy, "avg_power_W": avg_power,
                    "avg_cpu_power_W": avg_cpu_power, "avg_gpu_power_W": avg_gpu_power,
                    "total_cpu_energy_J": total_cpu_energy, "total_gpu_energy_J": total_gpu_energy,
                    "total_tokens": total_tokens, "power_per_token": power_per_token,
                    "accuracy": avg_accuracy, "total_duration_s": total_duration_s,
                    "num_answers": num_answers, "successful_tasks": successful_tasks,
                    "energy_per_answer_J": energy_per_answer_J,
                    "avg_time_per_answer_s": avg_time_per_answer_s,
                    "tokens_per_answer": tokens_per_answer,
                    "energy_per_solved_task_J": energy_per_solved_task_J,
                    "gpu_seconds": gpu_seconds,
                })

if not records:
    print("❌ No records found! Check `RESULTS_DIR` and file naming.")
    exit()

summary = pd.DataFrame(records)
summary.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\n=== Full Summary (All Runs) ===")
print(summary)

# === AVERAGE OVER RUNS (with Standard Deviation) ===
grouped_summary = summary.groupby(["model", "k"], as_index=False).agg(['mean', 'std'], numeric_only=True)
grouped_summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_summary.columns.values]
grouped_summary = grouped_summary.rename(columns={'model_': 'model', 'k_': 'k'})

print("\n=== Averages and StdDev over runs (All Metrics) ===")
print(grouped_summary)


# === NEW CONSOLE SUMMARIES ===

def format_with_std(mean, std, precision=2):
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


# --- RQ1: Effectiveness (Correctness) ---
print("\n" + "=" * 20 + " 🧩 RQ1: Effectiveness (Correctness) " + "=" * 20)
rq1_table = grouped_summary[['model', 'k']].copy()
rq1_table['Pass@1 Rate'] = grouped_summary.apply(
    lambda x: f"{x['accuracy_mean'] * 100:.2f}% ± {x['accuracy_std'] * 100:.2f}", axis=1
)
rq1_table['Total Tasks (Avg)'] = grouped_summary.apply(
    lambda x: format_with_std(x['num_answers_mean'], x['num_answers_std'], precision=2), axis=1
)
rq1_table['Successful Tasks (Avg)'] = grouped_summary.apply(
    lambda x: format_with_std(x['successful_tasks_mean'], x['successful_tasks_std'], precision=2), axis=1
)
print(rq1_table.to_string(index=False))

# --- RQ2: Cost (Computational) ---
print("\n" + "=" * 20 + " ⚙️ RQ2: Cost (Computational) " + "=" * 20)
rq2_table = grouped_summary[['model', 'k']].copy()
rq2_table['Avg Latency / Task (s)'] = grouped_summary.apply(
    lambda x: format_with_std(x['avg_time_per_answer_s_mean'], x['avg_time_per_answer_s_std'], precision=2), axis=1
)
rq2_table['Avg Tokens / Task'] = grouped_summary.apply(
    lambda x: format_with_std(x['tokens_per_answer_mean'], x['tokens_per_answer_std'], precision=0), axis=1
)
rq2_table['Avg Power (W)'] = grouped_summary.apply(
    lambda x: format_with_std(x['avg_power_W_mean'], x['avg_power_W_std'], precision=2), axis=1
)
rq2_table['Avg Energy / Task (J)'] = grouped_summary.apply(
    lambda x: format_with_std(x['energy_per_answer_J_mean'], x['energy_per_answer_J_std'], precision=2), axis=1
)
rq2_table['Total GPU Seconds (Avg)'] = grouped_summary.apply(
    lambda x: format_with_std(x['gpu_seconds_mean'], x['gpu_seconds_std'], precision=2), axis=1
)
print(rq2_table.to_string(index=False))

# --- RQ3: Trade-off and Sustainability ---
print("\n" + "=" * 20 + " 🌱 RQ3: Trade-off & Sustainability " + "=" * 20)
rq3_table = grouped_summary[['model', 'k']].copy()
grouped_summary['energy_per_solved_task_J_mean'] = grouped_summary['energy_per_solved_task_J_mean'].fillna(float('inf'))
grouped_summary['energy_per_solved_task_J_std'] = grouped_summary['energy_per_solved_task_J_std'].fillna(0)
rq3_table['Energy per Solved Task (J)'] = grouped_summary.apply(
    lambda x: format_with_std(x['energy_per_solved_task_J_mean'], x['energy_per_solved_task_J_std'], precision=2),
    axis=1
)
print(rq3_table.to_string(index=False))

# --- NEW: Latency / Time Summary ---
print("\n" + "=" * 20 + " ⏱️ Latency / Time Summary " + "=" * 20)
time_table = grouped_summary[['model', 'k']].copy()
time_table['Avg Time / Task (s)'] = grouped_summary.apply(
    lambda x: format_with_std(x['avg_time_per_answer_s_mean'], x['avg_time_per_answer_s_std'], precision=2), axis=1
)
time_table['Avg Total Time / Run (s)'] = grouped_summary.apply(
    lambda x: format_with_std(x['total_duration_s_mean'], x['total_duration_s_std'], precision=2), axis=1
)
print(time_table.to_string(index=False))

# === PLOTS ===

os.makedirs("plots", exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# --- Plot 1: Accuracy (RQ1) - COMBINED ---
print("📊 Generating Plot 1: Accuracy (RQ1) - Combined...")
plt.figure(figsize=(10, 6))
labels, means, stds, colors = [], [], [], []
for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        means.append(row['accuracy_mean'].values[0])
        stds.append(row['accuracy_std'].values[0])
        colors.append(COLORS_K[k])
x_pos = np.arange(len(labels))
bars = plt.bar(x_pos, means, yerr=stds, color=colors, capsize=5, zorder=3)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=12)
plt.title("Model Accuracy (pass@1) (Mac Machine)")
plt.ylabel("pass@1 Rate")
plt.xticks(x_pos, labels)
if means:
    plt.ylim(0, max(means) * 1.15)
plt.savefig(f"plots/1_accuracy_combined.png", bbox_inches="tight")
plt.close()

# --- Plot 2: Stacked Average Power (CPU + GPU) (RQ2) ---
print("📊 Generating Plot 2: Stacked Average Power (RQ2)...")
plt.figure(figsize=(10, 6))
labels, cpu_vals, gpu_vals, totals, totals_std, colors_cpu, colors_gpu = [], [], [], [], [], [], []
num_runs_for_title = len(RUNS) * len(MODELS) * len(KS) - len(SKIP_RUNS)
for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        cpu_vals.append(row['avg_cpu_power_W_mean'].values[0])
        gpu_vals.append(row['avg_gpu_power_W_mean'].values[0])
        totals.append(row['avg_power_W_mean'].values[0])
        totals_std.append(row['avg_power_W_std'].values[0])
        colors_cpu.append(COLORS_STACKED[f'{k}_cpu'])
        colors_gpu.append(COLORS_STACKED[f'{k}_gpu'])
x_pos = np.arange(len(labels))
plt.bar(x_pos, cpu_vals, color=colors_cpu, zorder=3)
plt.bar(x_pos, gpu_vals, bottom=cpu_vals, color=colors_gpu, zorder=3)
plt.errorbar(x_pos, totals, yerr=totals_std, fmt='none', ecolor='black', capsize=5, elinewidth=1, zorder=4)
for i, total in enumerate(totals):
    plt.text(i, total + 3, f"{total:.1f}W", ha='center', va='bottom', fontsize=12)

k0_cpu_mean = grouped_summary[grouped_summary['k'] == 'k0']['avg_cpu_power_W_mean'].mean()
k0_gpu_mean = grouped_summary[grouped_summary['k'] == 'k0']['avg_gpu_power_W_mean'].mean()
k3_cpu_mean = grouped_summary[grouped_summary['k'] == 'k3']['avg_cpu_power_W_mean'].mean()
k3_gpu_mean = grouped_summary[grouped_summary['k'] == 'k3']['avg_gpu_power_W_mean'].mean()
legend_patches = [
    mpatches.Patch(color=COLORS_STACKED['k0_cpu'], label=f"No RAG (CPU) (Avg: {k0_cpu_mean:.1f}W)"),
    mpatches.Patch(color=COLORS_STACKED['k0_gpu'], label=f"No RAG (GPU) (Avg: {k0_gpu_mean:.1f}W)"),
    mpatches.Patch(color=COLORS_STACKED['k3_cpu'], label=f"RAG (CPU) (Avg: {k3_cpu_mean:.1f}W)"),
    mpatches.Patch(color=COLORS_STACKED['k3_gpu'], label=f"RAG (GPU) (Avg: {k3_gpu_mean:.1f}W)")
]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel("Average Power (Watts)")
plt.title(f"Average Power (CPU + GPU) - Avg. over {num_runs_for_title} Runs (Mac Machine)")
plt.xticks(x_pos, labels)
if totals:
    plt.ylim(0, max(totals) * 1.15)
plt.savefig("plots/2_average_power_stacked.png", bbox_inches="tight")
plt.close()

# --- Plot 3: Stacked Total Energy (CPU + GPU) (RQ2) ---
print("📊 Generating Plot 3: Stacked Total Energy (RQ2)...")
plt.figure(figsize=(10, 6))
labels, cpu_vals, gpu_vals, totals, totals_std, colors_cpu, colors_gpu = [], [], [], [], [], [], []
for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        cpu_vals.append(row['total_cpu_energy_J_mean'].values[0])
        gpu_vals.append(row['total_gpu_energy_J_mean'].values[0])
        totals.append(row['total_energy_J_mean'].values[0])
        totals_std.append(row['total_energy_J_std'].values[0])
        colors_cpu.append(COLORS_STACKED[f'{k}_cpu'])
        colors_gpu.append(COLORS_STACKED[f'{k}_gpu'])
x_pos = np.arange(len(labels))
plt.bar(x_pos, cpu_vals, color=colors_cpu, zorder=3)
plt.bar(x_pos, gpu_vals, bottom=cpu_vals, color=colors_gpu, zorder=3)
plt.errorbar(x_pos, totals, yerr=totals_std, fmt='none', ecolor='black', capsize=5, elinewidth=1, zorder=4)
for i, total in enumerate(totals):
    offset = max(total * 0.02, totals_std[i] if not np.isnan(totals_std[i]) else 0) + (total * 0.01)
    plt.text(i, total + offset, f"{total:.0f} J", ha='center', va='bottom', fontsize=12)

k0_cpu_mean_J = grouped_summary[grouped_summary['k'] == 'k0']['total_cpu_energy_J_mean'].mean()
k0_gpu_mean_J = grouped_summary[grouped_summary['k'] == 'k0']['total_gpu_energy_J_mean'].mean()
k3_cpu_mean_J = grouped_summary[grouped_summary['k'] == 'k3']['total_cpu_energy_J_mean'].mean()
k3_gpu_mean_J = grouped_summary[grouped_summary['k'] == 'k3']['total_gpu_energy_J_mean'].mean()
legend_patches_J = [
    mpatches.Patch(color=COLORS_STACKED['k0_cpu'], label=f"No RAG (CPU) (Avg: {k0_cpu_mean_J:.0f}J)"),
    mpatches.Patch(color=COLORS_STACKED['k0_gpu'], label=f"No RAG (GPU) (Avg: {k0_gpu_mean_J:.0f}J)"),
    mpatches.Patch(color=COLORS_STACKED['k3_cpu'], label=f"RAG (CPU) (Avg: {k3_cpu_mean_J:.0f}J)"),
    mpatches.Patch(color=COLORS_STACKED['k3_gpu'], label=f"RAG (GPU) (Avg: {k3_gpu_mean_J:.0f}J)")
]
plt.legend(handles=legend_patches_J, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel("Total Energy (Joules)")
plt.title(f"Total Energy Consumed (CPU + GPU) - Avg. over {num_runs_for_title} Runs (Mac Machine)")
plt.xticks(x_pos, labels)
if totals:
    plt.ylim(0, max(totals) * 1.15)
plt.savefig("plots/3_total_energy_stacked.png", bbox_inches="tight")
plt.close()

# --- Plot 4: Accuracy vs Energy Efficiency (Pareto) (RQ3) ---
print("📊 Generating Plot 4: Accuracy vs. Efficiency (RQ3)...")
plt.figure(figsize=(10, 7))
MARKERS_K = {'k0': 'o', 'k3': 'X'}
for model in MODELS:
    for k_label in ['k0', 'k3']:
        point = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k_label)]
        if point.empty: continue
        x_val = point['power_per_token_mean'].values[0]
        y_val = point['accuracy_mean'].values[0]
        plt.errorbar(
            x=x_val, y=y_val,
            xerr=point['power_per_token_std'].values[0],
            yerr=point['accuracy_std'].values[0],
            fmt=MARKERS_K[k_label], color=COLORS_M[model], capsize=5, markersize=14, zorder=3
        )
        plt.text(x_val + (x_val * 0.01), y_val, f'{y_val:.2f}', fontsize=12, ha='left')
plt.xlabel("Energy per Token (Joules/Token)")
plt.ylabel("Accuracy (pass@1)")
legend_elements = [
    mpatches.Patch(color=COLORS_M['llama'], label='Llama'),
    mpatches.Patch(color=COLORS_M['starcoder'], label='Starcoder'),
    Line2D([0], [0], marker=MARKERS_K['k0'], color='w', label='No RAG (k=0)', markerfacecolor='grey', markersize=10),
    Line2D([0], [0], marker=MARKERS_K['k3'], color='w', label='RAG (k=3)', markerfacecolor='grey', markersize=10)
]
plt.legend(handles=legend_elements, title="Legend")
plt.title("Accuracy vs. Energy Efficiency (Pareto Plot) (Mac Machine)")
plt.savefig("plots/4_accuracy_vs_efficiency.png", bbox_inches="tight")
plt.close()

# --- Plot 5: Average Energy per Task (RQ2) ---
print("📊 Generating Plot 5: Average Energy per Task (RQ2)...")
plt.figure(figsize=(10, 6))
labels, means, stds, colors = [], [], [], []
for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        means.append(row['energy_per_answer_J_mean'].values[0])
        stds.append(row['energy_per_answer_J_std'].values[0])
        colors.append(COLORS_K[k])
x_pos = np.arange(len(labels))
bars = plt.bar(x_pos, means, yerr=stds, color=colors, capsize=5, zorder=3)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f} J', va='bottom', ha='center', fontsize=12)
plt.ylabel("Average Energy per Task (Joules)")
plt.title("Average Energy Consumed per Task (Mac Machine)")
plt.xticks(x_pos, labels)
if means:
    plt.ylim(0, max(means) * 1.15)
plt.savefig("plots/5_energy_per_task.png", bbox_inches="tight")
plt.close()

# --- Plot 6: Scatter Latency vs Energy per Task (RQ2) ---
print("📊 Generating Plot 6: Latency vs. Energy Scatter (RQ2)...")
plt.figure(figsize=(10, 7))
legend_handles = []
for model in MODELS:
    for k_label in ['k0', 'k3']:
        sub = summary[(summary['model'] == model) & (summary['k'] == k_label)]
        if sub.empty: continue

        x_mean = sub['avg_time_per_answer_s'].mean()
        y_mean = sub['energy_per_answer_J'].mean()
        label = f"{model.upper()} ({K_NAMES[k_label]}) (Avg: {x_mean:.1f}s, {y_mean:.1f}J)"

        scatter = plt.scatter(
            sub['avg_time_per_answer_s'],
            sub['energy_per_answer_J'],
            color=COLORS_M[model],
            marker=MARKERS_K[k_label],
            label=label,
            alpha=0.7, zorder=3, s=150
        )
        legend_handles.append(scatter)

plt.xlabel("Average Latency per Task (s)")
plt.ylabel("Average Energy per Task (J)")
plt.legend(handles=legend_handles, title="Legend")
plt.title("Cost Relation: Latency vs. Energy per Task (All Runs) (Mac Machine)")
plt.savefig("plots/6_latency_vs_energy_scatter.png", bbox_inches="tight")
plt.close()

# --- Plot 7: Power over Time (Sample Run) (RQ2) ---
print("📊 Generating Plot 7: Power over Time (Sample) (RQ2)...")
plt.figure(figsize=(12, 7))
sample_run = RUNS[0]
plot_legend_handles = []
for model in MODELS:
    for k in ['k0', 'k3']:
        if (model, k, sample_run) in SKIP_RUNS:
            continue
        power_path = f"{RESULTS_DIR}/{model}_{k}_run{sample_run}_power_log.csv"
        if os.path.exists(power_path):
            power_data = load_power_log(power_path, load_raw_df=True)
            if power_data and power_data[7] is not None:
                df = power_data[7]
                df['time_s'] = df['time_s'] - df['time_s'].iloc[0]

                color = COLORS_PLOT7[(model, k)]
                linestyle = LINESTYLES_PLOT7[(model, k)]
                avg_power_for_run = power_data[1]

                line, = plt.plot(df['time_s'], df['total_w'],
                                 label=f"{model.upper()} ({K_NAMES[k]}) - Run {sample_run}",
                                 color=color,
                                 linestyle=linestyle)

                mean_line = plt.axhline(y=avg_power_for_run, color=color, linestyle=':', linewidth=2,
                                        label=f"Mean ({avg_power_for_run:.1f}W) - {model.upper()} ({K_NAMES[k]})")
                plot_legend_handles.append(line)
                plot_legend_handles.append(mean_line)

if plot_legend_handles:
    plt.xlabel("Time (seconds)")
    plt.ylabel("Total Power (Watts)")
    plt.title(f"Power Draw Over Time (Sample: Run {sample_run}) (Mac Machine)")
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_legend = sorted(zip(handles, labels), key=lambda x: x[1])
    handles_sorted = [item[0] for item in sorted_legend]
    labels_sorted = [item[1] for item in sorted_legend]
    plt.legend(handles=handles_sorted, labels=labels_sorted, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("plots/7_power_over_time.png", bbox_inches="tight")
else:
    print(f"⚠️ Could not generate Plot 7: No data found for sample run {sample_run}.")
plt.close()

# --- Plot 8: Joules per Successful Task (RQ3) ---
print("📊 Generating Plot 8: Energy per Solved Task (RQ3)...")
plt.figure(figsize=(10, 6))
labels, means, stds, colors = [], [], [], []
for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        means.append(row['energy_per_solved_task_J_mean'].values[0])
        stds.append(row['energy_per_solved_task_J_std'].values[0])
        colors.append(COLORS_K[k])
x_pos = np.arange(len(labels))
bars = plt.bar(x_pos, means, yerr=stds, color=colors, capsize=5, zorder=3)
for bar in bars:
    yval = bar.get_height()
    if yval == float('inf') or np.isnan(yval):
        text = "N/A (0 Solved)"
    else:
        text = f'{yval:.2f} J'
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval if yval != float('inf') else 0, text,
             va='bottom', ha='center', fontsize=12)
plt.ylabel("Energy per Successful Task (Joules)")
plt.title("Efficiency: Joules per Solved Task (Lower is Better) (Mac Machine)")
plt.xticks(x_pos, labels)
if means:
    finite_means = [m for m in means if m != float('inf')]
    if finite_means:
        plt.ylim(0, max(finite_means) * 1.15)
plt.savefig("plots/8_energy_per_solved_task.png", bbox_inches="tight")
plt.close()

# --- Plot 9: Dual Axis: Accuracy vs. Energy Cost (RQ3) - COMBINED ---
print("📊 Generating Plot 9: Dual Axis Trade-off (RQ3) - Combined...")
fig, ax1 = plt.subplots(figsize=(10, 6))
labels, acc_means, acc_stds, nrg_means, nrg_stds, bar_colors = [], [], [], [], [], []

for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        acc_means.append(row['accuracy_mean'].values[0])
        acc_stds.append(row['accuracy_std'].values[0])
        nrg_means.append(row['total_energy_J_mean'].values[0])
        nrg_stds.append(row['total_energy_J_std'].values[0])
        bar_colors.append(COLORS_K[k])

x_pos = np.arange(len(labels))

# Bar chart for Accuracy (left axis)
color = 'tab:blue'
ax1.set_xlabel("Condition")
ax1.set_ylabel("Accuracy (pass@1)", color=color)
bars = ax1.bar(x_pos, acc_means, yerr=acc_stds, color=bar_colors, alpha=0.6, capsize=5, zorder=3)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, max(max(acc_means) * 1.2, 0.1))
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.2f}',
             va='bottom', ha='center', color=bar_colors[i], fontsize=12)

# Line chart for Energy (right axis)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Total Energy (J)", color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Plot Llama line
ax2.plot(x_pos[0:2], nrg_means[0:2], color=COLORS_M['llama'], marker='o', linestyle='--', zorder=4,
         label='Llama Energy')
ax2.errorbar(x_pos[0:2], nrg_means[0:2], yerr=nrg_stds[0:2], fmt='none', ecolor=COLORS_M['llama'], capsize=5, zorder=4)
# Plot Starcoder line
ax2.plot(x_pos[2:4], nrg_means[2:4], color=COLORS_M['starcoder'], marker='s', linestyle=':', zorder=4,
         label='Starcoder Energy')
ax2.errorbar(x_pos[2:4], nrg_means[2:4], yerr=nrg_stds[2:4], fmt='none', ecolor=COLORS_M['starcoder'], capsize=5,
             zorder=4)

ax2.set_ylim(0, max(max(nrg_means) * 1.2, 100))
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title(f"Combined Trade-off: Accuracy vs. Energy Cost (Mac Machine)")
plt.savefig(f"plots/9_dual_axis_tradeoff_combined.png", bbox_inches="tight")
plt.close()

# --- Plot 10: Average Tokens per Task (RQ2) ---
print("📊 Generating Plot 10: Average Tokens per Task (RQ2)...")
plt.figure(figsize=(10, 6))
labels, means, stds, colors = [], [], [], []
for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        means.append(row['tokens_per_answer_mean'].values[0])
        stds.append(row['tokens_per_answer_std'].values[0])
        colors.append(COLORS_K[k])
x_pos = np.arange(len(labels))
bars = plt.bar(x_pos, means, yerr=stds, color=colors, capsize=5, zorder=3)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.0f}', va='bottom', ha='center', fontsize=12)
plt.ylabel("Average Tokens per Task (Prompt + Generation)")
plt.title("Average Token Usage per Task (Mac Machine)")
plt.xticks(x_pos, labels)
if means:
    plt.ylim(0, max(means) * 1.15)
plt.savefig("plots/10_tokens_per_task.png", bbox_inches="tight")
plt.close()

# --- NEW Plot 11: Average Task Time vs. Total Run Time (RQ2) ---
print("📊 Generating Plot 11: Average Task Time vs. Total Run Time...")
fig, ax1 = plt.subplots(figsize=(10, 6))
labels, task_time_means, task_time_stds, total_time_means, total_time_stds, bar_colors = [], [], [], [], [], []

for model in MODELS:
    for k in ['k0', 'k3']:
        row = grouped_summary[(grouped_summary['model'] == model) & (grouped_summary['k'] == k)]
        if row.empty: continue
        labels.append(f"{model.upper()}\n({K_NAMES[k]})")
        task_time_means.append(row['avg_time_per_answer_s_mean'].values[0])
        task_time_stds.append(row['avg_time_per_answer_s_std'].values[0])
        total_time_means.append(row['total_duration_s_mean'].values[0])
        total_time_stds.append(row['total_duration_s_std'].values[0])
        bar_colors.append(COLORS_K[k])

x_pos = np.arange(len(labels))

# Left Axis (ax1): Average Task Time (Bars)
color = 'tab:blue'
ax1.set_xlabel("Condition")
ax1.set_ylabel("Average Time per Task (s)", color=color)
bars = ax1.bar(x_pos, task_time_means, yerr=task_time_stds, color=bar_colors, alpha=0.6, capsize=5, zorder=3)
ax1.tick_params(axis='y', labelcolor=color)
if task_time_means:
    ax1.set_ylim(0, max(max(task_time_means) * 1.2, 0.1))
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.2f}s',
             va='bottom', ha='center', color=bar_colors[i], fontsize=12)

# Right Axis (ax2): Average Total Time (Line)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Average Total Run Time (s)", color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Plot Llama line
ax2.plot(x_pos[0:2], total_time_means[0:2], color=COLORS_M['llama'], marker='o', linestyle='--', zorder=4,
         label='Llama Total Time')
ax2.errorbar(x_pos[0:2], total_time_means[0:2], yerr=total_time_stds[0:2], fmt='none', ecolor=COLORS_M['llama'],
             capsize=5, zorder=4)
# Plot Starcoder line
ax2.plot(x_pos[2:4], total_time_means[2:4], color=COLORS_M['starcoder'], marker='s', linestyle=':', zorder=4,
         label='Starcoder Total Time')
ax2.errorbar(x_pos[2:4], total_time_means[2:4], yerr=total_time_stds[2:4], fmt='none', ecolor=COLORS_M['starcoder'],
             capsize=5, zorder=4)

if total_time_means:
    ax2.set_ylim(0, max(max(total_time_means) * 1.2, 100))
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title(f"Average Task Time vs. Total Run Time (Mac Machine)")
plt.savefig(f"plots/11_time_comparison_dual_axis.png", bbox_inches="tight")
plt.close()
# ----------------------------------------------------

print("\n✅ Done! See 'plots/' for graphs and printed summary for data.")