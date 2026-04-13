#!/usr/bin/env python3

import sys
import os
sys.path.append("../src/")

from exo_model import ExoModel

import pysam
import argparse
import sklearn as sk
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import Counter


def calculate_template_length_distribution(bam_file, log_iter=1000000):
    # Открываем BAM файл
    bam = pysam.AlignmentFile(bam_file, "rb")
    length_dist = Counter()
    total_pairs = 0
    for read in bam:
        if read.is_paired and read.is_proper_pair and read.template_length > 0:
            tl = read.template_length
            length_dist[tl] += 1
            total_pairs += 1
            if total_pairs % log_iter == 0:
                print(f"processed {total_pairs} reads")

    bam.close()
    return length_dist, total_pairs


def optimization_task(reg_coef, max_iter, tlens):
    """Optimization task for parallel processing"""
    mu = np.argmax(tlens) // 2
    exo_model = ExoModel(reg_koef=reg_coef, max_iter=max_iter, mu=mu)
    try:
        exo_model.fit(tlens)
    except Exception as error:
        print(f"error: {error},\nreg: {reg_coef}")
        rmsd = np.inf
    else:
        linker_errors = exo_model.optimization_.x
        pred_distribution = np.convolve(linker_errors, linker_errors)
        rmsd = np.sum((pred_distribution - tlens) ** 2)
    return (exo_model, rmsd)


def check_path(path, force=False):
    """Check if path exists and handle overwrite logic"""
    if os.path.exists(path):
        if not force:
            raise ValueError(f"{path} exists. Add -f to overwrite")
        else:
            subprocess.run(f"rm -r {path}", shell=True, check=True)


def plot_results(best_model, hist, out_path, basename):
    """Plot optimization results"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(best_model.optimization_.x)
    plt.title('Linker Error Distribution')
    plt.xlabel('Position')
    plt.ylabel('Error Probability')
    
    plt.subplot(1, 2, 2)
    pred_dist = np.convolve(best_model.optimization_.x, best_model.optimization_.x)
    min_len = min(len(pred_dist), len(hist))
    plt.plot(pred_dist[:min_len], label='Model', alpha=0.7)
    plt.plot(hist[:min_len], label='Experimental', alpha=0.7)
    plt.title('Template Length Distribution')
    plt.xlabel('Template Length')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{out_path}/{basename}_errors/{basename}_analysis.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script for evaluating digesting errors of linker DNA")
    parser.add_argument("-iter", "--max_iter", help="maximum number of iteration of optimization", type=int, default=1500)
    parser.add_argument("-n", "--ntries", help="number of optimization tries", type=int, default=1)
    parser.add_argument("-i", "--bam_input", help="path to bam file", type=str, required=True)
    parser.add_argument("-@", "--njobs", help="number of processes", type=int, default=1)
    parser.add_argument("-o", "--out_dir", help="output directory", type=str, required=False)
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing files')
    parser.add_argument("-s", '--start', type=float, default=0.01, help="Start regularization coefficient")
    parser.add_argument("-e", '--end', type=float, default=0.2, help="End regularization coefficient")
    parser.add_argument("-st", '--step', type=float, default=0.05, help="Step size for regularization coefficient")
    parser.add_argument("--max_length", type=int, default=299, help="Maximum template length to consider")
    
    args = parser.parse_args()
    
    basename = os.path.basename(args.bam_input).split('.')[0]
    bam_path = os.path.abspath(args.bam_input) 

    if args.out_dir is None:
        out_path = os.path.dirname(bam_path)
    else:
        out_path = os.path.abspath(args.out_dir)

    # Create output directory
    output_dir = f"{out_path}/{basename}_errors"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input BAM: {bam_path}")
    print(f"Output directory: {output_dir}")
    print(f"Base name: {basename}")
    
    # Calculate template length distribution
    print("Calculating template length distribution...")
    tlens_dict, all_counts = calculate_template_length_distribution(bam_path)
    
    # Convert to normalized histogram (limit to max_length)
    hist = [tlens_dict.get(i, 0) / all_counts for i in range(args.max_length)]
    
    # Prepare optimization tasks
    task_args = []
    reg_values = np.arange(args.start, args.end, args.step)
    print(f"Testing regularization coefficients: {reg_values}")
    
    for reg_koef in reg_values:
        task_args.append([reg_koef, args.max_iter, hist])
    
    # Run parallel optimization
    print(f"Running {len(task_args)} optimization tasks with {args.njobs} processes...")
    with Pool(processes=args.njobs) as pool:
        results = pool.starmap(optimization_task, task_args)
    
    # Find best model
    best_task = min(results, key=lambda x: x[1])
    best_model, best_rmsd = best_task
    print(f"Best RMSD: {best_rmsd:.6f}, Regularization coefficient: {best_model.reg_koef}")
    
    # Save results
    errors_file = f"{output_dir}/{basename}_errors.csv"
    check_path(output_dir, args.force)
    np.savetxt(errors_file, best_model.optimization_.x, delimiter=',', 
               header="Linker_error_probabilities", comments='')
    
    # Plot results
    plot_results(best_model, hist, out_path, basename)
    
    # Save optimization summary
    summary_file = f"{output_dir}/{basename}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Optimization Summary\n")
        f.write(f"===================\n")
        f.write(f"Input BAM: {bam_path}\n")
        f.write(f"Best RMSD: {best_rmsd:.6f}\n")
        f.write(f"Best regularization coefficient: {best_model.reg_koef}\n")
        f.write(f"Max iterations: {args.max_iter}\n")
        f.write(f"Number of optimization tries: {args.ntries}\n")
        f.write(f"Total reads processed: {all_counts}\n")
        f.write(f"Template length range: 0-{args.max_length}\n")
    
    print(f"Results saved to {output_dir}")