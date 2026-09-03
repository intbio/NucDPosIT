import os
import sys
import argparse
import json
import shlex
import time
import glob
import tempfile
import subprocess
import numpy as np
import pysam
import pandas as pd
from simple_slurm import Slurm

def parse_args():
    parser = argparse.ArgumentParser(description='Run EM model analysis with slurm')
    parser.add_argument("--nucdposit_params", type=json.loads, default={},
                        help='JSON dict with parameters for Nucdposit.py')
    parser.add_argument('--slurm_params', type=json.loads, default={},
                        help='JSON dict of SLURM parameters, e.g. \'{"time":"01:00:00","cpus":4}\'')
    parser.add_argument('--njobs', type=int, default=1,
                        help='number of jobs to spawn (each processes a subset of chromosomes)')
    return parser.parse_args()

def build_command(params, chromo_list, nucdposit_script, chunk_id):
    params = params.copy()
    alignment = params.pop('alignment_path', None)
    errorpath = params.pop('errorpath', None)
    if alignment is None or errorpath is None:
        raise ValueError("Both 'alignment_path' and 'errorpath' must be provided")

    params.pop('accept_chromo', None)
    params.pop('except_chromo', None)

    base_out = params.pop('out_dir', './nucdpst_res')
    chunk_out_dir = f"{base_out.rstrip('/')}_chunk{chunk_id}"
    params['out_dir'] = chunk_out_dir

    cmd_parts = [f"python3 {shlex.quote(nucdposit_script)}"]
    cmd_parts.append(shlex.quote(alignment))
    cmd_parts.append(shlex.quote(errorpath))

    for key, value in params.items():
        if isinstance(value, list):
            cmd_parts.append(f"--{key}")
            for v in value:
                cmd_parts.append(shlex.quote(str(v)))
        else:
            cmd_parts.append(f"--{key}")
            cmd_parts.append(shlex.quote(str(value)))

    if chromo_list:
        cmd_parts.append("--accept_chromo")
        for chrom in chromo_list:
            cmd_parts.append(shlex.quote(chrom))

    return " ".join(cmd_parts), chunk_out_dir

def wait_for_jobs(job_ids, poll_interval=10):
    """Ожидает завершения всех заданий SLURM по их ID."""
    if not job_ids:
        return
    print(f"Waiting for jobs: {job_ids}")
    # Формируем строку для sacct: sacct -j id1,id2,... --format=State --noheader --parsable2
    job_str = ",".join(str(jid) for jid in job_ids)
    while True:
        print('check')
        cmd = ["sacct", "-j", job_str, "-X", "--format=State", "--noheader", "--parsable2"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"result: {result}")
        if result.returncode != 0:
            print("sacct failed, retrying...")
            time.sleep(poll_interval)
            continue
        # Получаем список статусов (каждая строка – одно задание)
        states = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        print(states)
        all_done = all(s in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL") or s.startswith("CANCELLED") for s in states)
        if all_done:
            break
        # Если есть ещё выполняющиеся или ожидающие – ждём
        time.sleep(poll_interval)

def merge_chunks(chunk_dirs, output_merged, base_filename=None):
    """Объединяет все CSV-файлы из папок чанков в один итоговый файл."""
    if not chunk_dirs:
        return
    # Определяем имя файла, если не задано
    if base_filename is None:
        # Берём первый попавшийся CSV в первой папке
        pattern = os.path.join(chunk_dirs[0], '*_dpst.csv')
        files = glob.glob(pattern)
        if not files:
            raise RuntimeError("No CSV files found in first chunk directory")
        base_filename = os.path.basename(files[0])

    header_written = False
    for d in chunk_dirs:
        pattern = os.path.join(d, base_filename)
        for f in glob.glob(pattern):
            df = pd.read_csv(f, sep='\t')
            if not header_written:
                df.to_csv(output_merged, sep='\t', index=False, mode='w')
                header_written = True
            else:
                df.to_csv(output_merged, sep='\t', index=False, mode='a', header=False)
    print(f"Merged {len(chunk_dirs)} chunks into {output_merged}")

def main():
    args = parse_args()
    nucdposit_params = args.nucdposit_params
    slurm_params = args.slurm_params
    njobs = args.njobs
    do_merge = args.merge

    nucdposit_script = os.path.expanduser("~/NucDPosIT/scripts/nucdposit.py")
    if not os.path.exists(nucdposit_script):
        raise FileNotFoundError(f"Nucdposit.py not found at {nucdposit_script}")

    bam_path = nucdposit_params.get('alignment_path')
    if not bam_path:
        raise ValueError("alignment_path not provided")
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        all_chromosomes = bam.references
    if not all_chromosomes:
        raise ValueError("No chromosomes found")

    chunks = np.array_split(all_chromosomes, njobs)
    chunk_lists = [chunk.tolist() for chunk in chunks]

    job_ids = []
    chunk_dirs = []

    for i, chromo_list in enumerate(chunk_lists):
        cmd, chunk_out_dir = build_command(nucdposit_params, chromo_list, nucdposit_script, i)
        os.makedirs(chunk_out_dir, exist_ok=True)

        job_slurm_params = slurm_params.copy()
        if 'output' not in job_slurm_params:
            job_slurm_params['output'] = os.path.join(chunk_out_dir, 'slurm_%A.out')
        if 'error' not in job_slurm_params:
            job_slurm_params['error'] = os.path.join(chunk_out_dir, 'slurm_%A.err')

        slurm = Slurm(**job_slurm_params)
        job_id = slurm.sbatch(cmd)
        job_ids.append(job_id)
        chunk_dirs.append(chunk_out_dir)

        print(f"Submitting job {i+1}/{njobs} with chromosomes: {chromo_list}")
        print(f"Command: {cmd}")
        print(f"Logs will be written to: {chunk_out_dir}")

    print(f"All {njobs} jobs submitted. Job IDs: {job_ids}")

    if job_ids:
        # Ожидаем завершения всех заданий
        wait_for_jobs(job_ids, poll_interval=30)

        # Определяем итоговый файл
        base_out = nucdposit_params.get('out_dir', './nucdpst_res')
        first_chunk = chunk_dirs[0]
        pattern = os.path.join(first_chunk, '*_dpst.csv')
        csv_files = glob.glob(pattern)
        if csv_files:
            base_name = os.path.basename(csv_files[0])
            output_merged = os.path.join(base_out, base_name)
        else:
            output_merged = os.path.join(base_out, 'merged_dpst.csv')

        os.makedirs(base_out, exist_ok=True)
        # Выполняем слияние
        merge_chunks(chunk_dirs, output_merged, base_name if csv_files else None)

if __name__ == '__main__':
    main()