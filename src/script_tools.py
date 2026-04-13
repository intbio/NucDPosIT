import sys

import copy
import pandas as pd
import numpy as np
from collections import Counter
from tqdm.auto import tqdm

from EMmodel.functools import fit_model_template, make_occupancy




def get_processing_chromosomes(include: list, exclude: list, all_chromo: list):
    include_set = set(include) if include else set()
    exclude_set = set(exclude) if exclude else set()
    all_chromoset = set(all_chromo)
    
    unknown_includes = include_set - all_chromoset
    if len(unknown_includes) != 0:
        raise ValueError(f"unknown include chromosomes: {list(unknown_includes)}")
    
    unknown_excludes = exclude_set - all_chromoset
    if len(unknown_excludes) != 0:
        raise ValueError(f"unknown exclude chromosomes: {list(unknown_excludes)}")
    
    if include:
        return list(include_set - exclude_set)
    else:
        return list(all_chromoset - exclude_set)


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, terminating...")
    sys.exit(1)


def fit_regcoef_parallel(
    model,
    iterator,
    nfits,
    reg_coef_grid,
    model_template,
    verbose=True,
    n_workers=None,
    method="auto",
):
    """
    Автоматически выбирает метод параллелизации

    Parameters:
    -----------
    method : 'auto', 'joblib', 'multiprocessing', 'sequential'
    """
    def worker(templates):
        model_copy = copy.deepcopy(model)

        cur_rmsd = []
        L, R = templates["start"], templates["end"]

        for reg_coef in reg_coef_grid:
            model_copy.reg_coef = reg_coef
            try:
                model_copy.fit(L, R)
            except Exception as error:
                print(error, reg_coef)
                cur_rmsd.append(np.nan)
                continue
            else:
                model_copy.to("cpu")
                df = model_copy.to_df()
                x, oc = make_occupancy(df.start.to_numpy(), df.stop.to_numpy())
                bar_counter = Counter(df.dyads.value_counts().to_dict())
                x_bars = [bar_counter[cord] for cord in x]
                conv = np.convolve(x_bars, model_template, mode="same")
                rmsd = ((oc - conv) ** 2).sum() / len(x)
                cur_rmsd.append(rmsd)
        return cur_rmsd

    windows_index = np.random.randint(0, len(iterator), size=nfits)
    templates_list = [iterator[i] for i in windows_index]

    if method == "auto":
        try:
            from joblib import Parallel, delayed

            method = "joblib"
        except ImportError:
            try:
                from multiprocessing import Pool

                method = "multiprocessing"
            except:
                method = "sequential"

    if method == "joblib":
        from joblib import Parallel, delayed

        if n_workers is None or n_workers == -1:
            import multiprocessing as mp

            n_workers = max(1, mp.cpu_count() - 1)

        if verbose:
            print(f"Using joblib with {n_workers} workers")
            total_rmsd = Parallel(n_jobs=n_workers, verbose=10)(
                delayed(worker)(templates) for templates in templates_list
            )
        else:
            total_rmsd = Parallel(n_jobs=n_workers)(
                delayed(worker)(templates) for templates in templates_list
            )

    elif method == "multiprocessing":
        from multiprocessing import Pool, cpu_count

        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        if verbose:
            print(f"Using multiprocessing with {n_workers} workers")
            with Pool(processes=n_workers) as pool:
                total_rmsd = list(
                    tqdm(
                        pool.imap(worker, templates_list),
                        total=nfits,
                        desc="Processing windows",
                    )
                )
        else:
            with Pool(processes=n_workers) as pool:
                total_rmsd = pool.map(worker, templates_list)

    else:  
        if verbose:
            print("Using sequential processing")
            total_rmsd = []
            for templates in tqdm(templates_list, desc="Processing windows"):
                total_rmsd.append(worker(templates))
        else:
            total_rmsd = [worker(templates) for templates in templates_list]

    return total_rmsd



