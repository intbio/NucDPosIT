import sys
import os
sys.path.append("src/")

import argparse
import pysam
import numpy as np
import torch

from src.EMmodel.em_model import StochasticEMMOdel
from src.EMmodel.datasets import BamFileIterator
from src.functools import optimize_reg_coef, make_df


def check_path(path, force=False):
    """Check if path exists and handle overwrite logic"""
    if os.path.exists(path):
        if not force:
            raise ValueError(f"{path} exists. Add -f to overwrite")
        else:
            # Create empty file to overwrite
            open(path, 'w').close()


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

    parser = argparse.ArgumentParser(
        prog='NucDPosIt', 
        description='Program detects dyads position in mnase-seq NGS data'
    )

    parser.add_argument('-i', '--input', required=True, help='Input BAM file')     
    parser.add_argument('-o', '--output', required=True, help='Output directory') 
    parser.add_argument('-e', '--errors', required=True, help='Errors file') 
    parser.add_argument('-ws', '--window_size', default=4000, type=int, help='Window size')
    parser.add_argument("-s", '--step', default=3500, type=int, help='Step size')
    parser.add_argument("-d", '--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('-ddist', '--dyad_dist', default=10, type=int, help='Dyad distance')
    parser.add_argument("-nretries", default=5, type=int, help='Number of retries')
    parser.add_argument('-max_iter', "--max_iter", default=700, type=int, help='Maximum iterations')
    parser.add_argument("-exclude", '--exclude_chromosomes', default=None, nargs='+', help='Chromosomes to exclude')
    parser.add_argument('-include', '--include_chromosomes', default=None, nargs='+', help='Chromosomes to include')
    parser.add_argument('-ti', '--train_iter', default=50, type=int, help='Training iterations')
    parser.add_argument('-ntrain', '--max_successful_trains', default=3, type=int, help='Max successful trains')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing files')

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
    for chromosome in chromosomes:
        print(f"Processing chromosome: {chromosome}")
        
        try:
            bam_iterator = BamFileIterator(
                args.input, 
                chromosome, 
                window_size=args.window_size, 
                step=args.step
            )
        except Exception as e:
            print(f"Error creating iterator for chromosome {chromosome}: {e}")
            continue
        
        for idx, batch in enumerate(bam_iterator):
            starts, ends = batch["start"], batch["end"]
                
            best_model = None
            success = False
            
            try:
                best_model = optimize_reg_coef(
                    StochasticEMMOdel, 
                    starts, 
                    ends, 
                    errors,
                    {
                        "dyad_dist": args.dyad_dist,
                        "max_iter": args.max_iter,
                        "device": args.device
                    },
                    args.train_iter, 
                    args.max_successful_trains
                )
                print(f"Window {idx} on chromosome {chromosome}: dyads: {best_model.dyads.shape[0]}")
                success = True
                
            except Exception as error:
                print(f"Window {idx} on chromosome {chromosome}: ERROR - {error}, RETRYING")
                
                # Retry with reduced dyad distance
                for i in range(args.nretries):
                    new_dyad_dist = args.dyad_dist // 2 ** (i + 1)
                    if new_dyad_dist == 0:
                        new_dyad_dist = 1
                    
                    try:
                        best_model = optimize_reg_coef(
                            StochasticEMMOdel, 
                            starts, 
                            ends, 
                            errors,
                            {
                                "dyad_dist": new_dyad_dist,
                                "max_iter": args.max_iter,
                                "device": args.device
                            },
                            args.train_iter, 
                            args.max_successful_trains
                        )
                        success = True
                        print(f"Success on retry {i+1} with dyad_dist={new_dyad_dist}")
                        break
                        
                    except Exception as retry_error:
                        print(f"Retry {i+1} failed: {retry_error}")
                        continue
            
            if not success or best_model is None:
                print(f"ERROR: Skipping window {idx} on chromosome {chromosome}")
                continue
            
            # Process successful model
            try:
                best_model.to('cpu')
                
                df = make_df(best_model)
                df['chr'] = chromosome
                df['n'] = idx
                
                if idx != 0:               
                    dyad_thold = bam_iterator.initial_start + args.window_size + args.step * (idx - 1) - 200
                    df = df.query("dyads >= @dyad_thold")
                else:
                    dyad_thold = bam_iterator.initial_start + args.window_size - 200
                    df = df.query("dyads <= @dyad_thold")
                
                # Create dyads BED file entries
                dyads_bed = df.groupby('dyads', as_index=False).size()
                dyads_bed['chr'] = chromosome  # Fixed: was hardcoded to "NC_001136.10"
                dyads_bed['stop'] = dyads_bed['dyads'] + 1
                dyads_bed.rename(columns={'dyads': 'start', 'size': 'score'}, inplace=True)
                
                # Append to output files
                with open(df_path, 'a') as df_file, open(dyads_df_path, 'a') as dyads_file:
                    df.to_csv(df_file, index=False, header=False, mode='a')
                    dyads_bed[['chr', 'start', 'stop', 'score']].to_csv(
                        dyads_file, index=False, header=False, sep='\t', mode='a'
                    )
                    
            except Exception as processing_error:
                print(f"Error processing results for window {idx} on {chromosome}: {processing_error}")
                continue
    
    BAM_FILE.close()
    print("Processing completed!")


if __name__ == '__main__':
    main()