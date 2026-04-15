import os, sys

import torch
import multiprocessing as mp
import threading
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
import tempfile
import glob
import argparse
from functools import partial
import time

from EMmodel import bamloader, em_model, functools
import signal
import script_tools


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, terminating...")
    raise KeyboardInterrupt
    sys.exit(1)


def init_model(params):
    args, kwargs = params
    global model
    model = em_model.StochasticEMMOdel(*args, **kwargs)

def worker(window_data, temp_dir):
    global model
    L, R = window_data['start'], window_data['end']
    assert len(L) == len(R)
    if len(L) == 0:
        print("empty window, continue...")
        return 
    try:
        model.fit(L, R)
    except Exception as error:
        return error
    else:
        df = model.to_df()
        df.insert(0, 'chr', window_data['chromosome']) 
        temp_file = os.path.join(temp_dir, f"temp_{os.getpid()}.csv")
        df.to_csv(temp_file, header=False, index=False, mode='a')


def merge_dfs(temp_dir, outdir):
    print("Merging...")
    combined_df = []
    searching_path = os.path.abspath(os.path.join(temp_dir, '*.csv'))
    print(f"searching in {searching_path}")
    for filepath in glob.glob(searching_path):
        print(filepath)
        df = pd.read_csv(filepath)
        combined_df.append(df)

    merged_path = os.path.join(outdir, 'nucdpst.bed')
    print(f"saving to {merged_path}")
    if combined_df:
        final_df = pd.concat(combined_df, ignore_index=True)
        final_df.to_csv(merged_path, header=None, sep='\t', index=False, mode='a')


def parallel_window_processing(chromo_iterator, model_args, model_kwargs, outdir, n_workers):
    with tempfile.TemporaryDirectory(dir=outdir) as temp_dir:
        try:
            total = len(chromo_iterator) if hasattr(chromo_iterator, '__len__') else None
            print(f"tempdir: {temp_dir}, total: {total} windows")
            init_params = (model_args, model_kwargs)
            
            with mp.Pool(
                processes=n_workers,
                initializer=init_model,
                initargs=(init_params,)
            ) as pool:
                worker_func = partial(worker, temp_dir=temp_dir)
                results = pool.imap_unordered(worker_func, chromo_iterator, chunksize=10)   
                with tqdm(total=total, desc=f"processing chromosome: {chromo_iterator.chromosome}") as pbar:
                    for res in results:
                        if isinstance(res, Exception):
                            print(f"ERROR: {res}")
                        pbar.update(1)
            merge_dfs(temp_dir, outdir)   
        except KeyboardInterrupt:
            merge_dfs(temp_dir, outdir)
            time.sleep(3)
            sys.exit(1)


def load_errors(errors_path: str):
    errors = torch.from_numpy(
    np.loadtxt(
        errors_path,
        delimiter=",",
        )
    )
    errors /= errors.sum()
    return errors


def get_processing_chromosomes(include: list, exclude: list, all_chromo: list):
    include_set = set(include) if include else set()
    exclude_set = set(exclude) if exclude else set()
    all_chromoset = set(all_chromo)
    unknown_includes = include_set - all_chromoset
    if len(unknown_includes) != 0:
        raise ValueError(f"unknown include chromosomes: {list(unknown_includes)}")
    unknown_excludes = include_set - exclude_set
    if len(unknown_excludes) != 0:
        raise ValueError(f"unknown exclude chromosomes: {list(unknown_excludes)}")
    return list(include_set - exclude_set) if include else list(all_chromoset - exclude_set)


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
    "errors_file",
    type=str,
    help="Path to the input errors file"
    )
    parser.add_argument(
    "reg_coef",
    help="regularization coeffizient",
    type=float,
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
    "--device",
    help="torch device",
    type=str,
    default='cpu'
    )
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, signal_handler)   
    signal.signal(signal.SIGTERM, signal_handler)  
    signal.signal(signal.SIGQUIT, signal_handler)

    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    if 'cuda' in args.device:
        mp.set_start_method("spawn", force=True)

    errors = load_errors(args.errors_file)
    template_occupancy = functools.fit_model_template(errors)
    loader = bamloader.BamLoader(args.bam_file)

    chromosomes = script_tools.get_processing_chromosomes(
        args.include, args.exclude, loader.get_chromosomes()
        )

    for chromosome in chromosomes:
        chromo_iterator = loader.iter_chromosome(chromosome)
        parallel_window_processing(chromo_iterator, (errors, 5), 
        model_kwargs={
            "device": args.device,
             "tol": 1e-20,
              "max_iter": 1000,
               "alpha": 0.005,
                "reg_coef": args.reg_coef
                },
                 outdir=args.output_dir,
                  n_workers=args.njobs)


if __name__ == '__main__':
    main()