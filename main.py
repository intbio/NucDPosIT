import sys
import os
sys.path.append("src/")

import argparse
import pysam
import numpy as np
import torch
from multiprocessing import Pool, Lock
import uuid

from src.EMmodel.em_model import StochasticEMMOdel, ModelOptimizer
from src.EMmodel.datasets import BamFileIterator
from src.functools import make_df


lock = Lock()


def check_path(path, force=False):
    """Check if path exists and handle overwrite logic"""
    if os.path.exists(path):
        if not force:
            raise ValueError(f"{path} exists. Add -f to overwrite")
        else:
            # Create empty file to overwrite
            open(path, 'w').close()
            
            
def worker_foo(bam_input, 
               chromosome, 
               window_size, 
               chunk_start, 
               chunk_stop,
               step, 
               device, 
               errors, 
               dyad_dist, 
               train_iter, 
               max_iter, 
               max_successful_trains,
               nretries,
               df_path, 
               dyads_df_path):
    
    bam_iterator = BamFileIterator(bam_input,
                                         chromosome,
                                         window_size,
                                         chunk_start,
                                         chunk_stop,
                                         step,
                                         device)
            
    optimizer = ModelOptimizer(StochasticEMMOdel, 
                                     errors,
                                     dyad_dist,
                                     train_iter,
                                     max_iter,
                                     max_successful_trains,
                                     nretries)    
    
    for idx, batch in enumerate(bam_iterator):
        starts, ends = batch["start"], batch["end"]
        
        try:
            best_model = optimizer.fit_data(starts, ends)
        except Exception as fit_error:
            print(f"ERROR {fit_error} on chromosome {bam_iterator.chromosome} for window {starts.min()} : {ends.max()}. SKIP")
            continue
        else:
            print(f"processed idx {idx} for window {starts.min()} : {ends.max()}")
            
        try:
            best_model.to('cpu')
            df = make_df(best_model)
            df['chr'] = chromosome
            # df['n'] = idx

            if idx != 0:               
                dyad_thold = bam_iterator.initial_start + bam_iterator.window_size + bam_iterator.step * (idx - 1) - 200
                df = df.query("dyads >= @dyad_thold")
            else:
                dyad_thold = bam_iterator.initial_start + bam_iterator.window_size - 200
                df = df.query("dyads <= @dyad_thold")

            # Create dyads BED file entries
            dyads_bed = df.groupby('dyads', as_index=False).size()
            dyads_bed['chr'] = chromosome  
            dyads_bed['stop'] = dyads_bed['dyads']
            dyads_bed.rename(columns={'dyads': 'start', 'size': 'score'}, inplace=True)
            

            # Append to output files
            with lock:
                with open(df_path, 'a') as df_file, open(dyads_df_path, 'a') as dyads_file:
                    df.to_csv(df_file, index=False, header=False, mode='a')
                    dyads_bed[['chr', 'start', 'stop', 'score']].to_csv(
                        dyads_file, index=False, header=False, sep='\t', mode='a'
                    )

        except Exception as processing_error:
            print(f"ERROR {processing_error} on chromosome {bam_iterator.chromosome} for window {starts.min()} : {ends.max()}. SKIP")
            continue


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

    parser = argparse.ArgumentParser(
        prog='NucDPosIt', 
        description='Program detects dyads position in mnase-seq NGS data'
    )

    parser.add_argument('-i', '--input', required=True, help='Input BAM file')     
    parser.add_argument('-o', '--output', required=True, help='Output directory') 
    parser.add_argument('-e', '--errors', required=True, help='Errors file') 
    parser.add_argument('-ws', '--window_size', default=3000, type=int, help='Window size')
    parser.add_argument("-s", '--step', default=2700, type=int, help='Step size')
    parser.add_argument("-d", '--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('-ddist', '--dyad_dist', default=10, type=int, help='Dyad distance')
    parser.add_argument("-nretries", default=5, type=int, help='Number of retries')
    parser.add_argument('-max_iter', "--max_iter", default=500, type=int, help='Maximum iterations')
    parser.add_argument("-exclude", '--exclude_chromosomes', default=None, nargs='+', help='Chromosomes to exclude')
    parser.add_argument('-include', '--include_chromosomes', default=None, nargs='+', help='Chromosomes to include')
    parser.add_argument('-ti', '--train_iter', default=30, type=int, help='Training iterations')
    parser.add_argument('-ntrain', '--max_successful_trains', default=3, type=int, help='Max successful trains')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing files')
    parser.add_argument('-njobs', help='number of processes', type=int, default=1)

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    df_path = os.path.join(args.output, 'templates.csv')
    dyads_df_path = os.path.join(args.output, 'dyads.bed')
    
    # Check and prepare output files
    check_path(df_path, args.force)
    check_path(dyads_df_path, args.force)
    
    # Load errors
    if not os.path.exists(args.errors):
        raise ValueError(f"Errors file {args.errors} does not exist")
    errors = torch.from_numpy(np.loadtxt(args.errors))
    
    # Open BAM file
    try:
        BAM_FILE = pysam.AlignmentFile(args.input)
    except Exception as e:
        raise ValueError(f"Could not open BAM file {args.input}: {e}")
    
    # Determine chromosomes to process
    if args.include_chromosomes:
        chromosomes = args.include_chromosomes
    else:
        chromosomes = BAM_FILE.references
    
    if args.exclude_chromosomes:
        chromosomes = [chr for chr in chromosomes if chr not in args.exclude_chromosomes]
    
    # Process each chromosome
    chromosome_lengths = dict(zip(BAM_FILE.references, BAM_FILE.lengths))
    for chromosome in chromosomes:
        print(f"Processing chromosome: {chromosome}")
        chromo_len = chromosome_lengths[chromosome]
        chunk_starts = np.linspace(0, chromo_len + 1 - args.window_size, args.njobs + 1)
        
        worker_arguments = []
        for i in range(len(chunk_starts) - 1):
            chunk_start, chunk_stop = chunk_starts[i], chunk_starts[i + 1]
            
            worker_arguments.append(
                (   args.input, 
                    chromosome,
                    args.window_size,
                    chunk_start,
                    chunk_stop,
                    args.step,
                    args.device, 
                    errors, 
                    args.dyad_dist, 
                    args.train_iter, 
                    args.max_iter, 
                    args.max_successful_trains,
                    args.nretries,
                    df_path, 
                    dyads_df_path)
            )

        
        with Pool(processes=args.njobs) as p:
            p.starmap(worker_foo, worker_arguments)
        
    
    BAM_FILE.close()
    print("Processing completed!")


if __name__ == '__main__':
    main()