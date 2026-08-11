import os
import sys
import argparse
import json
import shlex
import numpy as np
import pysam
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

    # Удаляем ключи, которые будем переопределять
    params.pop('accept_chromo', None)
    params.pop('except_chromo', None)

    # --- УНИКАЛЬНАЯ ВЫХОДНАЯ ДИРЕКТОРИЯ ДЛЯ КАЖДОГО ЧАНКА ---
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

    return " ".join(cmd_parts)

def main():
    args = parse_args()
    nucdposit_params = args.nucdposit_params
    slurm_params = args.slurm_params
    njobs = args.njobs

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

    slurm = Slurm(**slurm_params)

    for i, chromo_list in enumerate(chunk_lists):
        cmd = build_command(nucdposit_params, chromo_list, nucdposit_script, i)
        print(f"Submitting job {i+1}/{njobs} with chromosomes: {chromo_list}")
        print(f"Command: {cmd}")
        slurm.sbatch(cmd)

    print(f"All {njobs} jobs submitted.")

if __name__ == '__main__':
    main()