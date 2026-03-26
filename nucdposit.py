import sys
import os
sys.path.append("src/")

import argparse
import pysam
import numpy as np
import torch
from multiprocessing import Pool, Lock, Queue, Process
import logging
import uuid
import signal
import glob
import time

from src.EMmodel.em_model import StochasticEMMOdel, ModelOptimizer
from src.EMmodel.datasets import BamFileIterator
from src.functools import make_df


lock = Lock()
# shared_queue = Queue()


def sigterm_handler(signum, frame, df_path, dyads_df_path):
    print(f"Received SIGTERM ({signum}). Performing cleanup and exiting.")
    
    dftmp_files = glob.glob(f"{df_path}.*.dftmp")
    dyadtmp_files = glob.glob(f"{dyads_df_path}.*.dyadtmp")
    
    for temp_file in dftmp_files:
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            os.system(f"cat '{temp_file}' >> '{df_path}'")
            os.system(f"rm '{temp_file}'")
    
    for temp_file in dyadtmp_files:
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            os.system(f"cat '{temp_file}' >> '{dyads_df_path}'")
            os.system(f"rm '{temp_file}'")
    
    sys.exit(0)


def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(message)s',
        filename=log_path,
        filemode="w"
    )


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
               errors, 
               dyad_dist, 
               max_model_iter, 
               max_train_iter,
               max_successful_runs,
               nretries,
               device,
               df_path, 
               dyads_df_path):
    
    print(f"Worker started: {df_path}, {dyads_df_path}")  # Отладка
    
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
                               max_model_iter,
                               max_train_iter,
                               max_successful_runs,
                               nretries,
                               device)  
    
    for idx, batch in enumerate(bam_iterator):
        starts, ends = batch["start"], batch["end"]
        
        if len(starts) == 0 or len(ends) == 0:
            continue
        
        try:
            best_model = optimizer.fit_data(starts, ends)
            print(f"Processed window {starts.min()}:{ends.max()}")  # Отладка
        except Exception as fit_error:
            print(f"ERROR {fit_error} on chromosome {bam_iterator.chromosome} for window {starts.min()}:{ends.max()}. SKIP")
            continue
        

        best_model.to('cpu')
        df = make_df(best_model)
        df['chr'] = chromosome
        
        try:
            dyad_thold = bam_iterator.initial_start + 250 + bam_iterator.step * (idx + 1)
            back_thold = dyad_thold - bam_iterator.step
            df = df.query("dyads >= @back_thold and dyads < @dyad_thold")
            
            if len(df) == 0 or df.isna().any().any():
                print("Empty or NaN data, skipping")
                continue

            # Create dyads BED file entries
            dyads_bed = df.groupby('dyads', as_index=False).size()
            dyads_bed['chr'] = chromosome  
            dyads_bed['stop'] = dyads_bed['dyads'] + 1
            dyads_bed['id'] = idx
            dyads_bed.rename(columns={'dyads': 'start', 'size': 'score'}, inplace=True)

            with lock:
                df.to_csv(df_path, index=False, header=False, mode='a')
                dyads_bed[['chr', 'start', 'stop', 'id', 'score']].to_csv(
                    dyads_df_path, index=False, header=False, sep='\t', mode='a'
                )
            
        except Exception as processing_error:
            print(f"ERROR {processing_error} on chromosome {bam_iterator.chromosome} for window {starts.min()}:{ends.max()}. SKIP")
            continue
    
    print(f"Worker finished: {df_path}")  # Отладка

            
class FileProcessor:
    def __init__(self, bam_path, outdir, errors_path, force):
        self.bam_path = self.validate_inpath(bam_path)
        self.outdir = self.validate_outpath(outdir, force)
        self.errors_path = self.validate_errors(errors_path)
        # self.win_size = win_size
        # self.step = step
        # self.device = device
        # self.dyad_dist = dyad_dist
        # self.
        
    def validate_inpath(self, inpath: str):
        if not os.path.exists(inpath):
            raise ValueError(f"file {inpath} does not exist")
        if not os.path.isfile(inpath):
            raise ValueError(f"file {inpath} is not a file")
        samfile = pysam.AlignmentFile(inpath, 'rb')
        if not samfile.has_index():
            raise ValueError(f"file {inpath} has not index")
        return inpath
        
    def validate_outpath(self, outdir: str, force, outfile_name='templates.csv', out_dyads='dyads.bed'):
        if not os.path.exists(outdir):
            os.system(f"mkdir {outdir}")
        elif force:
            os.system(f"rm -r {outdir}/*")
        else:
            raise ValueError(f"{outdir} exists. Add -f to overwrite")
        if not os.path.isdir(outdir):
            raise ValueError(f"path {outpath} is not a directory")
        self.outfile = os.path.join(outdir, outfile_name)
        self.out_dyads = os.path.join(outdir, out_dyads)
        return outdir
    
    def validate_errors(self, errors_path: str):
        if not os.path.exists(errors_path):
            raise ValueError(f"file {errors_path} does not exist")
        self.errors = torch.tensor(np.loadtxt(errors_path))
        return errors_path
        
        
        
        

            
def main():
    signal.signal(signal.SIGTERM, lambda x, y: sigterm_handler(x, y, df_path, dyads_df_path))
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

    parser = argparse.ArgumentParser(
        prog='NucDPosIt', 
        description='Program detects dyads position in mnase-seq NGS data'
    )

    parser.add_argument('-i', '--input', required=True, help='Input BAM file')     
    parser.add_argument('-o', '--output', required=True, help='Output directory') 
    parser.add_argument('-e', '--errors', required=True, help='Errors file') 
    parser.add_argument('-ws', '--window_size', default=3000, type=int, help='Window size')
    parser.add_argument("-s", '--step', default=2500, type=int, help='Step size')
    parser.add_argument("-d", '--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('-ddist', '--dyad_dist', default=15, type=int, help='Dyad distance')
    parser.add_argument("-nretries", default=5, type=int, help='Number of retries')
    parser.add_argument('-max_iter', "--max_iter", default=700, type=int, help='Maximum iterations')
    parser.add_argument("-exclude", '--exclude_chromosomes', default=None, nargs='+', help='Chromosomes to exclude')
    parser.add_argument('-include', '--include_chromosomes', default=None, nargs='+', help='Chromosomes to include')
    parser.add_argument('-ti', '--train_iter', default=50, type=int, help='Training iterations')
    parser.add_argument('-ntrain', '--max_successful_trains', default=1, type=int, help='Max successful trains')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing files')
    parser.add_argument('-njobs', help='number of processes', type=int, default=1)
    
    args = parser.parse_args()
    file_processor = FileProcessor(args.input, args.output, args.errors, args.force)
    
    
    BAM_FILE = pysam.AlignmentFile(file_processor.bam_path)
    # Determine chromosomes to process
    if args.include_chromosomes:
        chromosomes = args.include_chromosomes
    else:
        chromosomes = BAM_FILE.references
    
    if args.exclude_chromosomes:
        chromosomes = [chr for chr in chromosomes if chr not in args.exclude_chromosomes]
    
    # Process each chromosome
    chromosome_lengths = dict(zip(BAM_FILE.references, BAM_FILE.lengths))
    BAM_FILE.close()
    
    for chromosome in chromosomes:
        print(f"Processing chromosome: {chromosome}")
        chromo_len = chromosome_lengths[chromosome]
        chunk_starts = np.linspace(0, chromo_len + 1 - args.window_size, args.njobs + 1)
        
        worker_arguments = []
        for i in range(len(chunk_starts) - 1):
            chunk_start, chunk_stop = chunk_starts[i], chunk_starts[i + 1]
            arg_copy = (file_processor.bam_path, 
                    chromosome,
                    args.window_size,
                    chunk_start.item(),
                    chunk_stop.item(),
                    args.step,
                    file_processor.errors, 
                    args.dyad_dist, 
                    args.max_iter,
                    args.train_iter,  
                    args.max_successful_trains,
                    args.nretries,
                    args.device,
                    file_processor.outfile, 
                    file_processor.out_dyads)
            worker_arguments.append(arg_copy)
            
        with Pool(processes=args.njobs) as p:
            p.starmap(worker_foo, worker_arguments)
            time.sleep(1)

    # BAM_FILE.close()
    print("Processing completed!")


if __name__ == '__main__':
    main()