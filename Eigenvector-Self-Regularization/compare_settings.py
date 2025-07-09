#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate box plots to compare different experimental settings.'
    )
    parser.add_argument('--base_path', type=str, default='multiseed_output_test',
                        help='Base directory where experiment folders are located.')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                        help='List of experiments in label:folder format. Example: batch8:alex_batch8 batch16:alex_batch16')
    parser.add_argument('--fc_files', type=str, nargs='+', default=['fc1.csv', 'fc2.csv', 'fc3.csv'],
                        help='List of CSV files to analyze.')
    parser.add_argument('--output_dir', type=str, default='visualization',
                        help='Directory to save the generated box plot images.')
    parser.add_argument('--plot_type', type=str, choices=['normal','epoch','both'], default='normal',
                        help='Type of plots to generate. "normal": box plots for selected columns; "epoch": epoch-based box plots for base metrics; "both": generate both types.')
    parser.add_argument('--columns', type=str, nargs='*', default=[],
                        help='List of columns for normal box plots.')
    parser.add_argument('--epoch_bases', type=str, nargs='*', default=[],
                        help='List of base metric names for epoch-based box plots. For example: S1_RSS S2_fit')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for epoch-based box plots (assumes columns: base, base_2_100, ..., base_epochs_100).')
    parser.add_argument('--ignore_columns', type=str, nargs='*', default=['seed'],
                        help='Columns to ignore when generating normal box plots.')
    parser.add_argument('--exp_column', type=str, default='experiment',
                        help='Name of the column that will store the experiment label.')
    return parser.parse_args()

def load_combined_df(args, fc_file):
    """
    Load and combine CSV data from different experiments for a given fc_file.
    """
    combined_df = pd.DataFrame()
    for exp_info in args.experiments:
        if ':' not in exp_info:
            raise ValueError("Each experiment must be in label:folder format.")
        label, folder = exp_info.split(':', 1)
        file_path = os.path.join(args.base_path, folder, fc_file)
        if not os.path.isfile(file_path):
            print(f"[Warning] {file_path} does not exist. Skipping.")
            continue
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"[Error] Failed to load {file_path}: {e}")
            continue
        df[args.exp_column] = label
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def plot_normal_boxplots(combined_df, fc_file, args, normal_out_dir):
    """
    Generate normal box plots for the selected columns.
    """
    experiments = sorted(combined_df[args.exp_column].unique())
    if len(experiments) == 0:
        print(f"[Skip] {fc_file}: No experiment data available.")
        return

    # Determine columns to plot (only those in --columns and not in ignore_columns)
    ignore_set = set(args.ignore_columns + [args.exp_column])
    columns_for_plot = []
    for col in args.columns:
        if col in ignore_set:
            continue
        if col in combined_df.columns:
            columns_for_plot.append(col)
        else:
            print(f"[Note] {fc_file}: Column '{col}' not found.")
    
    for col in columns_for_plot:
        plt.figure(figsize=(8, 6))
        combined_df.boxplot(column=col, by=args.exp_column)
        plt.title(f'{fc_file} - {col} Comparison')
        plt.suptitle('')
        plt.xlabel(args.exp_column)
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()

        save_name = f"{os.path.splitext(fc_file)[0]}_{col}_boxplot.png"
        save_path = os.path.join(normal_out_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        print(f"[Saved] Normal box plot: {save_path}")

def plot_epoch_boxplots(combined_df, fc_file, base_metric, args, epoch_out_dir):
    """
    Generate epoch-based box plots for a given base metric.
    Assumes for epoch 1, column name is base_metric,
    and for epoch k (k>1), column name is base_metric_k_100.
    """
    exps = combined_df[args.exp_column].unique()
    if len(exps) == 0:
        print(f"[Skip] {fc_file}: No experiment data available.")
        return

    records = []
    for _, row in combined_df.iterrows():
        for e in range(1, args.epochs + 1):
            if e == 1:
                col_name = base_metric
            else:
                col_name = f"{base_metric}_{e}_100"
            if col_name not in combined_df.columns:
                continue
            value = row[col_name]
            exp_label = row[args.exp_column]
            records.append({
                'experiment': exp_label,
                'epoch': e,
                'value': value
            })

    if not records:
        print(f"[Note] {fc_file}: No valid data found for base metric '{base_metric}'.")
        return

    plot_df = pd.DataFrame(records)
    epochs = sorted(plot_df['epoch'].unique())
    experiments = sorted(plot_df['experiment'].unique())
    n_exps = len(experiments)

    plt.figure(figsize=(10, 6))
    group_width = 0.8
    box_width = group_width / n_exps
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    box_handles = []
    for i, exp in enumerate(experiments):
        data_by_epoch = []
        positions = []
        for e in epochs:
            vals = plot_df.loc[(plot_df['experiment'] == exp) & (plot_df['epoch'] == e), 'value'].values
            data_by_epoch.append(vals)
            midpoint = e
            left_edge = midpoint - group_width / 2
            pos = left_edge + i * box_width + box_width / 2
            positions.append(pos)

        bp = plt.boxplot(data_by_epoch, positions=positions, widths=box_width * 0.8, patch_artist=True, manage_ticks=False)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i % len(colors)])
        box_handles.append(bp['boxes'][0])
    
    plt.title(f'{fc_file} - {base_metric} Epoch Comparison')
    plt.xlabel('Epoch')
    plt.ylabel(base_metric)
    plt.xticks(epochs)
    plt.grid(True, axis='y')
    plt.legend(box_handles, experiments, title='Experiment')
    plt.tight_layout()

    save_name = f"{os.path.splitext(fc_file)[0]}_{base_metric}_epoch_boxplot.png"
    save_path = os.path.join(epoch_out_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Epoch box plot: {save_path}")

def main():
    args = parse_args()
    
    # Create separate output directories for normal and epoch plots
    base_out_dir = args.output_dir
    normal_out_dir = os.path.join(base_out_dir, "normal_plots")
    epoch_out_dir = os.path.join(base_out_dir, "epoch_plots")
    os.makedirs(normal_out_dir, exist_ok=True)
    os.makedirs(epoch_out_dir, exist_ok=True)

    for fc_file in args.fc_files:
        combined_df = load_combined_df(args, fc_file)
        if combined_df.empty:
            print(f"[Skip] {fc_file}: No data loaded.")
            continue

        if args.plot_type in ['normal', 'both']:
            if args.columns:
                plot_normal_boxplots(combined_df, fc_file, args, normal_out_dir)
            else:
                print(f"[Note] Normal plot type selected but no columns specified. Skipping normal plots for {fc_file}.")
        
        if args.plot_type in ['epoch', 'both']:
            if args.epoch_bases:
                for base_metric in args.epoch_bases:
                    plot_epoch_boxplots(combined_df, fc_file, base_metric, args, epoch_out_dir)
            else:
                print(f"[Note] Epoch plot type selected but no epoch base metrics specified. Skipping epoch plots for {fc_file}.")

    print(f"\n[Complete] All plots have been saved in '{base_out_dir}'.")

if __name__ == '__main__':
    main()
