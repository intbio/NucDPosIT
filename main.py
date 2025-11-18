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


def check_path(path):
    if os.path.exists(path) and not args.force:
        raise ValueError(f"{path} exists. Add -f to overwrite")
    elif os.path.exists(path):
        open(path, 'w').close()
        




parser = argparse.ArgumentParser(prog='NucDPosIt', 
                                 description='Program detects dyads position in mnase-seq NGS data',
                                 epilog='Text at the bottom of help'
                                )

parser.add_argument('-i', '--input', required=True)     
parser.add_argument('-o', '--output', required=True) 
parser.add_argument('-e', '--errors', required=True) 
parser.add_argument('-ws', '--window_size', default=3000, type=int)
parser.add_argument("-s", '--step', default=500)
parser.add_argument("-d", '--device', default='cpu')
parser.add_argument('-ndyads', '--init_n_dyads', default=500)
parser.add_argument('-max_iter', "--max_iter", default=600)
parser.add_argument("-exclude", '--exclude_chromosomes', default=None, nargs='+')
parser.add_argument('-include', default=None, nargs='+')
parser.add_argument('-ti', '--train_iter', default=30)
parser.add_argument('-ntrain', '--max_successful_trains', default=3)
parser.add_argument('-f', '--force', default=False)


if __name__ == '__main__':
    args = parser.parse_args()
    df_path = os.path.join(args.output, 'templates.csv')
    dyads_df_path = os.path.join(args.output, 'dyads.bed')
    
    check_path(df_path)
    check_path(dyads_df_path)
    
    if not os.path.isdir(args.output):
        raise ValueError(f"{args.output} is not a directory")
    
    if not os.path.exists(args.output):
        raise ValueError(f"{args.output} does not exist")
        
    errors = torch.from_numpy(np.loadtxt(args.errors))
    
    
    BAM_FILE = pysam.AlignmentFile(args.input)
    
    if args.include:
        chromosomes = args.include
    else:
        chromosomes = BAM_FILE.references
    
    if args.exclude_chromosomes:
        for chromo in args.exclude_chromosomes:
            chromosomes.remove(chromo)
            
    for chromosome in chromosomes:
        # records_dataset = PysamRecordsDataset(list(BAM_FILE.fetch(chromosome)))
        # loader = DataLoader(records_dataset, args.window_size, False)
        bam_iterator = BamFileIterator(args.input, chromosome, args.window_size, args.step)
        
        for idx, batch in enumerate(bam_iterator):
            starts, ends = batch["start"], batch["end"]
            model = StochasticEMMOdel(starts, ends, errors, args.init_n_dyads, reg_coef=0, max_iter=args.max_iter, device=args.device)
            best_model = optimize_reg_coef(model, args.train_iter, args.max_successful_trains)
            best_model.to('cpu')
            # df = make_df(best_model)
            
            try:
                df = make_df(best_model)
            except Exception as error:
                print("ERROR", error)
                continue

            df['chr'] = chromosome
            df['n'] = idx
            
            dyads_bed = df.groupby('dyads', as_index=False).size()
            dyads_bed['chr'] = "NC_001136.10"
            dyads_bed['stop'] = dyads_bed['dyads'] + 1
            
            with open(df_path, 'a+') as df_file, open(dyads_df_path, 'a+') as dyads_file:
                df.to_csv(df_file, index=False, header=False)
                dyads_bed[['chr', 'dyads', 'stop', 'size']].to_csv(dyads_file, index=False, header=False, sep='\t')
            
    