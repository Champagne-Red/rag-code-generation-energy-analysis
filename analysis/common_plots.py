import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def setup_dataframe():
    """
    Creates the main pandas DataFrame with all the data from the paper's tables,
    including metrics for both Mac and Windows.
    """
    data = [
        # Mac Data
        {'Platform': 'Mac', 'Model': 'Llama', 'Condition': 'No RAG (k=0)',
         'Pass@1 Rate': 0.52, 'Avg Energy per Task (J)': 49.64,
         'Avg Time per Task (s)': 2.23, 'Total Energy per Run (J)': 4964,
         'Energy per Solved Task (J)': 95.44, 'Avg Total Run Time (s)': 223.00},

        {'Platform': 'Mac', 'Model': 'Llama', 'Condition': 'RAG (k=3)',
         'Pass@1 Rate': 0.4783, 'Avg Energy per Task (J)': 148.77,
         'Avg Time per Task (s)': 5.69, 'Total Energy per Run (J)': 14877,
         'Energy per Solved Task (J)': 311.29, 'Avg Total Run Time (s)': 568.67},

        {'Platform': 'Mac', 'Model': 'Starcoder', 'Condition': 'No RAG (k=0)',
         'Pass@1 Rate': 0.42, 'Avg Energy per Task (J)': 142.15,
         'Avg Time per Task (s)': 7.46, 'Total Energy per Run (J)': 14218,
         'Energy per Solved Task (J)': 338.46, 'Avg Total Run Time (s)': 745.83},

        {'Platform': 'Mac', 'Model': 'Starcoder', 'Condition': 'RAG (k=3)',
         'Pass@1 Rate': 0.42, 'Avg Energy per Task (J)': 353.26,
         'Avg Time per Task (s)': 15.46, 'Total Energy per Run (J)': 35326,
         'Energy per Solved Task (J)': 841.10, 'Avg Total Run Time (s)': 1546.00},

        # Windows Data
        {'Platform': 'Windows', 'Model': 'Llama', 'Condition': 'No RAG (k=0)',
         'Pass@1 Rate': 0.52, 'Avg Energy per Task (J)': 188.59,
         'Avg Time per Task (s)': 0.84, 'Total Energy per Run (J)': 18859,
         'Energy per Solved Task (J)': 364.29, 'Avg Total Run Time (s)': 84.00},

        {'Platform': 'Windows', 'Model': 'Llama', 'Condition': 'RAG (k=3)',
         'Pass@1 Rate': 0.49, 'Avg Energy per Task (J)': 238.79,
         'Avg Time per Task (s)': 1.06, 'Total Energy per Run (J)': 23879,
         'Energy per Solved Task (J)': 490.98, 'Avg Total Run Time (s)': 106.00},

        {'Platform': 'Windows', 'Model': 'Starcoder', 'Condition': 'No RAG (k=0)',
         'Pass@1 Rate': 0.42, 'Avg Energy per Task (J)': 685.25,
         'Avg Time per Task (s)': 3.27, 'Total Energy per Run (J)': 68525,
         'Energy per Solved Task (J)': 1631.56, 'Avg Total Run Time (s)': 327.00},

        {'Platform': 'Windows', 'Model': 'Starcoder', 'Condition': 'RAG (k=3)',
         'Pass@1 Rate': 0.45, 'Avg Energy per Task (J)': 1013.12,
         'Avg Time per Task (s)': 4.82, 'Total Energy per Run (J)': 101312,
         'Energy per Solved Task (J)': 2236.71, 'Avg Total Run Time (s)': 482.00},
    ]

    df = pd.DataFrame(data)

    # Create a combined 'Condition' label for plotting
    df['Plot_Label'] = df['Model'] + "\n(" + df['Condition'].str.replace(r' \(k=.\)', '', regex=True) + ")"
    return df


def plot_grouped_bar(df, value_col, title, ylabel, filename, log_scale=False):
    """
    Generic function to create a grouped bar chart comparing Mac and Windows
    for a specific metric.
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    # Pivot the data
    pivot_df = df.pivot(index='Plot_Label', columns='Platform', values=value_col)
    # Ensure correct order
    pivot_df = pivot_df.reindex(["Llama\n(No RAG)", "Llama\n(RAG)", "Starcoder\n(No RAG)", "Starcoder\n(RAG)"])

    ax = pivot_df.plot(kind='bar', figsize=(12, 8), width=0.7, logy=log_scale)

    # Add labels to bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    plt.xticks(rotation=0)
    ax.legend(title='Platform', loc='upper left')

    # Adjust y-limit to make space for labels
    if not log_scale:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax=ymax * 1.15)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")


def plot_comparative_scatter(df, filename="graph_5_accuracy_vs_efficiency.png"):
    """
    Creates a comparative scatter plot: Accuracy vs. Energy per Solved Task
    for all 8 conditions (both platforms).
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {'Mac': 'tab:blue', 'Windows': 'tab:orange'}
    markers = {'Llama': 'o', 'Starcoder': 's'}

    # Plot each point with a label
    for i, row in df.iterrows():
        x = row['Energy per Solved Task (J)']
        y = row['Pass@1 Rate']
        c = colors[row['Platform']]
        m = markers[row['Model']]
        label = f"{row['Plot_Label'].replace(chr(10), ' ')} ({row['Platform']})"

        ax.scatter(x, y, color=c, marker=m, s=200, alpha=0.8, edgecolors='black')

        # Add text label for clarity
        ax.text(x * 1.03, y, label, fontsize=10, verticalalignment='center')

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=0, marker='o', markersize=10, label='Mac'),
        Line2D([0], [0], color='tab:orange', lw=0, marker='s', markersize=10, label='Windows'),
        Line2D([0], [0], color='gray', lw=0, marker='o', markersize=10, label='Llama'),
        Line2D([0], [0], color='gray', lw=0, marker='s', markersize=10, label='Starcoder')
    ]
    ax.legend(handles=legend_elements, title="Legend", loc='upper right',
              bbox_to_anchor=(1.0, 0.85))

    ax.set_title('Accuracy vs. Efficiency (Mac vs. Windows)', fontsize=16, pad=20)
    ax.set_xlabel('Energy per Solved Task (Joules) (Lower is Better)', fontsize=12)
    ax.set_ylabel('Accuracy (pass@1) (Higher is Better)', fontsize=12)

    # Set x-axis to log scale for better visualization
    ax.set_xscale('log')
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    ax.grid(which='major', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")


def main():
    """
    Main function to load data and generate all plots.
    """
    df = setup_dataframe()

    # --- Graph 1: Average Energy Consumed per Task (Mac vs. Windows) ---
    # Adapts '5_energy_per_task.png'
    plot_grouped_bar(df,
                     value_col='Avg Energy per Task (J)',
                     title='Average Energy Consumed per Task (Mac vs. Windows)',
                     ylabel='Average Energy per Task (Joules)',
                     filename='graph_1_avg_energy_per_task.png',
                     log_scale=True)

    # --- Graph 2: Total Energy Consumed per Run (Mac vs. Windows) ---
    # Adapts '3_total_energy_stacked.png' (as stacked data isn't available for Windows)
    plot_grouped_bar(df,
                     value_col='Total Energy per Run (J)',
                     title='Total Energy Consumed per Run (Mac vs. Windows)',
                     ylabel='Total Energy (Joules)',
                     filename='graph_2_total_energy.png',
                     log_scale=True)  # Log scale is needed due to large value range

    # --- Graph 3: Average Time per Task (Mac vs. Windows) ---
    # Adapts bar chart part of '11_time_comparison_dual_axis.png'
    plot_grouped_bar(df,
                     value_col='Avg Time per Task (s)',
                     title='Average Time per Task (Mac vs. Windows)',
                     ylabel='Average Time per Task (s)',
                     filename='graph_3_avg_time_per_task.png')

    # --- Graph 4: Average Total Run Time (Mac vs. Windows) ---
    # Adapts line plot part of '11_time_comparison_dual_axis.png'
    plot_grouped_bar(df,
                     value_col='Avg Total Run Time (s)',
                     title='Average Total Run Time (Mac vs. Windows)',
                     ylabel='Average Total Run Time (s)',
                     filename='graph_4_avg_total_run_time.png',
                     log_scale=True)

    # --- Graph 5: Accuracy vs. Efficiency (Mac vs. Windows) ---
    # Adapts '4_accuracy_vs_efficiency.png' using a comparable metric
    plot_comparative_scatter(df, "graph_5_accuracy_vs_efficiency.png")


if __name__ == "__main__":
    main()