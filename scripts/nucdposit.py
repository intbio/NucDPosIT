import os
import sys
import torch
import multiprocessing as mp
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import tempfile
import glob
import argparse
from functools import partial
import time
import signal

from EMmodel import bamloader, em_model, functools


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, terminating...")
    raise KeyboardInterrupt
    sys.exit(1)


def init_model(params):
    args, kwargs = params
    global model
    model = em_model.StochasticEMMOdel(*args, **kwargs)


def init_worker(bam_path, chromosome, window_size, step, device, model_args, model_kwargs):
    global model, worker_iterator
    model = em_model.StochasticEMMOdel(*model_args, **model_kwargs)
    loader = bamloader.BamLoader(bam_path)
    worker_iterator = loader.iter_chromosome(
        chromosome,
        window_size=window_size,
        step=step,
        device=device
    )


def worker(window_index, temp_dir):
    global model, worker_iterator
    try:
        window_data = worker_iterator[window_index]
    except Exception as e:
        return f"Error getting window {window_index}: {e}"
    L, R = window_data['start'], window_data['end']
    assert len(L) == len(R)
    if len(L) == 0:
        return None
    try:
        model.fit(L, R)
    except Exception as error:
        return error
    else:
        df = model.to_df()
        df.insert(0, 'chr', window_data['chromosome']) 
        temp_file = os.path.join(temp_dir, f"temp_{os.getpid()}.csv")
        df.to_csv(temp_file, header=False, index=False, mode='a')
        return True


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
        final_df.to_csv(merged_path, header=None, sep='\t', index=False)


def parallel_window_processing(bam_path, chromosome, window_size, step, device, 
                               model_args, model_kwargs, outdir, n_workers):
    temp_loader = bamloader.BamLoader(bam_path)
    temp_iterator = temp_loader.iter_chromosome(
        chromosome,
        window_size=window_size,
        step=step,
        device=device
    )
    total = len(temp_iterator)
    del temp_loader
    del temp_iterator
    
    if total == 0:
        print(f"No windows to process for chromosome {chromosome}")
        return
    
    with tempfile.TemporaryDirectory(dir=outdir) as temp_dir:
        try:
            print(f"tempdir: {temp_dir}, total: {total} windows for chromosome {chromosome}")
            init_params = (bam_path, chromosome, window_size, step, device, model_args, model_kwargs)
            with mp.Pool(
                processes=n_workers,
                initializer=init_worker,
                initargs=init_params
            ) as pool:
                worker_func = partial(worker, temp_dir=temp_dir)
                indices = range(total)
                results = pool.imap_unordered(worker_func, indices, chunksize=10)
                with tqdm(total=total, desc=f"Processing chromosome {chromosome}") as pbar:
                    for res in results:
                        if isinstance(res, Exception):
                            print(f"\nERROR: {res}")
                        elif isinstance(res, str) and "Error" in res:
                            print(f"\n{res}")
                        pbar.update(1)
            
            merge_dfs(temp_dir, outdir)
            
        except KeyboardInterrupt:
            print("\nInterrupted! Merging partial results...")
            merge_dfs(temp_dir, outdir)
            time.sleep(3)
            sys.exit(1)


def load_errors(errors_path: str):
    errors = torch.from_numpy(
        np.loadtxt(errors_path, delimiter=",")
    )
    errors /= errors.sum()
    return errors


def get_processing_chromosomes(include: list, exclude: list, all_chromo: list):
    include_set = set(include) if include else set()
    exclude_set = set(exclude) if exclude else set()
    all_chromoset = set(all_chromo)
    
    unknown_includes = include_set - all_chromoset
    if unknown_includes:
        raise ValueError(f"unknown include chromosomes: {list(unknown_includes)}")
    
    if include:
        return list(include_set - exclude_set)
    else:
        return list(all_chromoset - exclude_set)


def parse_arguments():
    parser = argparse.ArgumentParser(description='NucDPosIT')
    parser.add_argument("bam_file", type=str, help="Path to the input BAM file")
    parser.add_argument("errors_file", type=str, help="Path to the input errors file")
    parser.add_argument("reg_coef", type=float, help="regularization coefficient")
    parser.add_argument("--output_dir", "-o", type=str, default="./nucdposit_results")
    parser.add_argument("--njobs", "-@", type=int, default=1, help="number of workers")
    parser.add_argument("--include", type=str, nargs='+', help="chromosomes to analyze")
    parser.add_argument("--exclude", type=str, nargs='+', help="chromosomes to exclude")
    parser.add_argument("--device", type=str, default='cpu', help="torch device")
    parser.add_argument("--window_size", "-ws", type=int, default=5000, help="window size")
    parser.add_argument("--step", type=int, default=5000, help="windows step")
    
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
    loader = bamloader.BamLoader(args.bam_file)
    
    all_chromosomes = loader.get_chromosomes()
    chromosomes = get_processing_chromosomes(args.include, args.exclude, all_chromosomes)
    
    print(f"Processing chromosomes: {chromosomes}")

    for chromosome in chromosomes:
        print(f"\n{'='*60}")
        print(f"Processing chromosome: {chromosome}")
        print(f"{'='*60}")
        
        parallel_window_processing(
            bam_path=args.bam_file,
            chromosome=chromosome,
            window_size=args.window_size,
            step=args.step,
            device=args.device,
            model_args=(errors, 5),
            model_kwargs={
                "device": args.device,
                "tol": 1e-20,
                "max_iter": 1000,
                "alpha": 0.005,
                "reg_coef": args.reg_coef
            },
            outdir=args.output_dir,
            n_workers=args.njobs
        )


if __name__ == '__main__':
    main()