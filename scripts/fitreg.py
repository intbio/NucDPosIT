import os

import multiprocessing as mp
import argparse
import signal
import numpy as np
import copy
from collections import Counter
import matplotlib.pyplot as plt

from EMmodel.em_model import StochasticEMMOdel
from EMmodel.functools import fit_model_template, make_occupancy
from script_tools import fit_regcoef_parallel
from EMmodel.bamloader import BamLoader
import script_tools


def parse_arguments():
    parser = argparse.ArgumentParser(description='NucDPosIT')
    parser.add_argument(
    "bam_file",
    type=str,
    help="Path to the input BAM file"
    )
    parser.add_argument(
    "--output_dir", "-o",
    type=str,
    default="./nucdposit_results",
    help="Output directory (default: ./nucdposit_results)"
    )
    parser.add_argument(
    "--njobs", "-@",
    help="number of workers",
    type=int,
    default=1
    )
    parser.add_argument(
    "--include",
    help="chromosomes to analise",
    type=str,
    nargs='+'
    )
    parser.add_argument(
    "--exclude",
    help="chromosomes to exclude from analysis",
    type=str,
    nargs='+'
    )
    parser.add_argument(
    "errors_file",
    type=str,
    help="Path to the input errors file"
    )
    parser.add_argument(
    "--device",
    help="torch device",
    type=str,
    default='cpu'
    )
    parser.add_argument(
        "--start", "-s",
        default=0,
        help="start for scanning",
        type=int
    )
    parser.add_argument(
        "--end", "-e",
        default=0.01,
        help="end for scanning",
        type=int
    )
    parser.add_argument(
        "--npoints",
        default=5,
        help="number of lambda points to test",
        type=int
    )
    parser.add_argument(
        "--nwindows",
        default=50,
        help="number of windows to asses regularization coefficient",
        type=int
    )
    return parser.parse_args()




def plot_regcoef_res(regcoef_res, grid):
    vals = np.asarray(regcoef_res)
    vals[np.isnan(vals)] = 0
    mean_rmsd = vals.mean(axis=0)
    median_rmsd = np.median(vals, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    for v in vals:
        axs[0].plot(grid, v)

    axs[1].plot(grid, mean_rmsd, "--", color="red", label="mean")
    axs[1].plot(grid, median_rmsd, label="median")
    plt.legend()

    axs[0].grid()
    axs[1].grid()
    return fig, axs


def main():

    signal.signal(signal.SIGINT, script_tools.signal_handler)   
    signal.signal(signal.SIGTERM, script_tools.signal_handler)  
    signal.signal(signal.SIGQUIT, script_tools.signal_handler)

    args = parse_arguments()

    model = StochasticEMMOdel(
        args.errors_file,
        15,
        device=args.device,
        tol=1e-20,
        max_iter=2000,
        alpha=0.005,
        reg_coef=0
        )

    loader = BamLoader(args.bam_file)
    chromosomes = script_tools.get_processing_chromosomes(
        args.include, args.exclude, loader.get_chromosomes()
        )
    print(f"Processing chromosomes: {chromosomes}")
    window_iterator = loader.iter_random_chromosomes(chromosomes, n_windows_total=args.nwindows, device=args.device)
    grid = np.linspace(args.start, args.end, args.npoints)
    template_occ = fit_model_template(model.errors.cpu())

    regcoef_res = fit_regcoef_parallel(model,
     window_iterator,
      args.nwindows,
       grid,
        template_occ,
         n_workers=args.njobs)

    fig, axs = plot_regcoef_res(regcoef_res, grid)
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, 'fitreg_results.png')
    fig.savefig(outpath)


if __name__ == '__main__':
    main()


