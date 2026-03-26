 #!/usr/bin/env python3


import sys
import os
sys.path.append("src/")

from exo_model import ExoModel

import pysam
import argparse
import sklearn as sk
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from collections import Counter


def calculate_template_length_distribution(bam_file, log_iter=1000000):
    # Открываем BAM файл
    bam = pysam.AlignmentFile(bam_file, "rb")

    # Словарь для хранения распределения длин
    length_dist = Counter()
    total_pairs = 0

    # Проходим по всем выравниваниям
    for read in bam:
        # Проверяем, что read является парным и правильно выровненным
        if read.is_paired and read.is_proper_pair and read.template_length > 0:
            # Берем абсолютное значение длины шаблона
            tl = read.template_length
            length_dist[tl] += 1
            total_pairs += 1
            if total_pairs % log_iter == 0:
                print(f"processed {total_pairs} reads")

    bam.close()
    return length_dist, total_pairs


def optimization_task(reg_coef, max_iter, tlens):
    exo_model = ExoModel(reg_koef=reg_coef, max_iter=max_iter)
    exo_model.fit(tlens)
    linker_errors = exo_model.optimization_.x
    pred_distribution = np.convolve(linker_errors, linker_errors)
    rmsd = np.linalg.norm(pred_distribution - tlens)
    return (exo_model, rmsd)


def check_path(path, force=False):
    """Check if path exists and handle overwrite logic"""
    if os.path.exists(path):
        if not force:
            raise ValueError(f"{path} exists. Add -f to overwrite")
        else:
            # Create empty file to overwrite
            open(path, 'w').close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script for evaluating digesting errors of linker DNA")
    parser.add_argument("-iter", "--max_iter", help="maximum number of iteration of optimization", type=int, default=2000)
    # parser.add_argument("-l", "--av_length", help="expected length of nucleosomal DNA in experiment", type=int, default=146)
    parser.add_argument("-n", "--ntries", help="number of optsumimization tries", type=int, default=1)
    parser.add_argument("-i", "--bam_input", help="path to bam file", type=str, required=True)
    parser.add_argument("-@", "--njobs", help="number of processes", type=int, default=1)
    parser.add_argument("-o", "--out_dir", help="output_file", type=str, required=True)
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing files')
    parser.add_argument("-s", '--start', type=float, default=0.01)
    parser.add_argument("-e", '--end', type=float, default=0.2)
    parser.add_argument("-st", '--step', type=float, default=0.05)
    
    
    args = parser.parse_args()
    
    
    basename = os.path.basename(args.bam_input).split('.')[0]
    bam_path = os.path.abspath(args.bam_input)   
    out_path = os.path.abspath(args.out_dir)
    os.system(f"mkdir {out_path}/{basename}_errors")
    
    
    
    print(bam_path, out_path, basename)
    
    # check_path(out_path, args.force)
    
    
    tlens, all_counts = calculate_template_length_distribution(bam_path)     
    print(tlens, all_counts)
    # lengths = np.arange(args.av_length - 60, args.av_length + 61)
    hist = [tlens[i] / all_counts for i in range(0, 299)]
    
    

    task_args = []
    for reg_koef in np.arange(args.start, args.end, args.step):
        for _ in range(args.ntries):
            task_args.append([reg_koef, args.max_iter, hist])
     
    with Pool(processes=args.njobs) as pool:
        results = pool.starmap(optimization_task, task_args)
        
    best_task = min(results, key=lambda x: x[1])
    best_model, best_rmsd = best_task
    print(f"best rmsd: {best_rmsd}, reg: {best_model.reg_koef}")
            
    np.savetxt(f"{out_path}/{basename}_errors/{basename}_errors.csv", best_model.optimization_.x, delimiter=',')
    
    plt.figure()
    plt.plot(best_model.optimization_.x)
    plt.savefig(f"{out_path}/{basename}_errors/{basename}_errors.png")
    
    plt.figure()
    plt.plot(np.convolve(best_model.optimization_.x, best_model.optimization_.x), label='model')
    plt.plot(hist, label='experiment')
    plt.legend(loc='upper left')
    plt.savefig(f"{out_path}/{basename}_errors/{basename}_density.png")
   


    
    
    
    
    
    

    

    
    
    


