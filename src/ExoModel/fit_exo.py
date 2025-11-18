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


def optimization_task(reg_coef, av_length, max_iter, tlens):
    exo_model = ExoModel(l0=av_length, reg_koef=reg_coef, max_iter=max_iter)
    exo_model.fit(tlens)
    linker_errors = exo_model.optimization_.x
    pred_distribution = np.convolve(linker_errors, linker_errors)
    rmsd = np.linalg.norm(pred_distribution - tlens)
    return (exo_model, rmsd)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script for evaluating digesting errors of linker DNA")
    parser.add_argument("-iter", "--max_iter", help="maximum number of iteration of optimization", type=int, default=4000)
    parser.add_argument("-l", "--av_length", help="expected length of nucleosomal DNA in experiment", type=int, default=146)
    parser.add_argument("-n", "--ntries", help="number of optsumimization tries", type=int, default=1)
    parser.add_argument("-i", "--bam_input", help="path to bam file", type=str, required=True)
    parser.add_argument("-@", "--njobs", help="number of processes", type=int, default=1)
    parser.add_argument("-o", "--out_file", help="output_file", type=str, default='exo_model_output.txt')
    
    args = parser.parse_args()
    lengths_dict, templates_all = calculate_template_length_distribution(args.bam_input)     
    lengths = np.arange(args.av_length - 60, args.av_length + 61)
    counts = np.array([lengths_dict.get(l, 0) for l in lengths]) / templates_all

    task_args = []
    for reg_koef in np.arange(0.01, 0.1, 0.01):
        for _ in range(args.ntries):
            task_args.append([reg_koef, args.av_length, args.max_iter, counts])
     
    with Pool(processes=args.njobs) as pool:
        results = pool.starmap(optimization_task, task_args)
        
    best_task = min(results, key=lambda x: x[1])
    best_model, best_rmsd = best_task
    print(f"best rmsd: {best_rmsd}")
            
    
    np.savetxt(args.out_file, best_model.optimization_.x, delimiter=',', header=os.path.basename(args.bam_input))
    pd.DataFrame(counts, index=lengths, columns=[os.path.basename(args.bam_input)]).to_csv('hist.csv', index=False)

    
    
    
    
    
    

    

    
    
    


