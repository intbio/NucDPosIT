#!/usr/bin/env python3
import sys
import os
import argparse
import signal
import logging
import pprint
import pandas as pd
import EMmodel.datasets as datasets
from EMmodel import em_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run EM model analysis')
    parser.add_argument('alignment_path', type=str,
                        help='Path to genome alignment file')
    parser.add_argument('--regions_path', type=str, default=None,
                        help='Path to file with specified regions to analyse')
    parser.add_argument('errorpath', type=str,
                        help='Path to error file')
    parser.add_argument('--out_dir', type=str, default='./nucdpst_res',
                        help='Output directory path')
    parser.add_argument('--dyad_dist', type=int, default=40,
                        help='Initial dyad distance')
    parser.add_argument('--nfits', type=int, default=50,
                        help='number of optimization steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device: cpu or cuda')
    parser.add_argument('--tol', type=float, default=0.01,
                        help='tolerance for sliding mean')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='coefficient of sliding mean')
    parser.add_argument('--reg_coef', type=int, default=1,
                        help='EM regularisation coefficient')
    parser.add_argument('--temperature_coef', type=int, default=20,
                        help='temperature coefficient for EM')
    parser.add_argument('--min_iter', type=int, default=100,
                        help='min iterations in optimization step')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='max iterations in optimization step')
    parser.add_argument('--window_size', type=int, default=5000,
                        help='size of scanning window')
    parser.add_argument('--step', type=int, default=4800,
                        help='step of scanning')
    parser.add_argument('--accept_chromo', nargs='+', default=None,
                        help='Space-separated list of chromosomes to include')
    parser.add_argument('--except_chromo', nargs='+', default=None,
                        help='Space-separated list of chromosomes to exclude')
    return parser.parse_args()


def setup_logging(logging_path):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(logging_path, encoding='utf-8', mode='w')
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)


def exit_gracefully(signum, frame):
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()
    sys.exit(1)


def save_dataframe(df, output_path, header=True, sep='\t'):
    """Сохраняет DataFrame в CSV, добавляя заголовок только при первой записи."""
    df.to_csv(output_path, mode='a', header=header, sep=sep, index=False)


def main():
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    args = parse_args()

    # Создаём выходную директорию (если её нет)
    os.makedirs(args.out_dir, exist_ok=True)

    # Базовое имя файла (без расширения .bam)
    bam_basename = os.path.splitext(os.path.basename(args.alignment_path))[0]

    # Лог-файл в выходной директории
    log_path = os.path.join(args.out_dir, f"{bam_basename}.log")
    setup_logging(log_path)

    logger = logging.getLogger(__name__)
    logger.info(f"Configuration:\n{pprint.pformat(vars(args), indent=2, width=120, sort_dicts=False)}")

    # --- Создание модели ---
    try:
        model = em_model.StochasticEMModel(
            args.errorpath,
            args.dyad_dist,
            args.min_iter,
            args.max_iter,
            args.nfits,
            args.reg_coef,
            args.temperature_coef,
            args.tol,
            args.device,
            args.alpha
        )
    except Exception as e:
        logger.error(f"Model creation error: {e}", exc_info=True)
        raise
    logger.debug(f"Model {type(model).__name__} loaded")

    # --- Создание Dataset / итератора ---
    try:
        factory = datasets.DatasetFactory()
        loader = factory.create_dataset(
            alignment_file=args.alignment_path,
            bed_file=args.regions_path,
            start=0,
            stop=None,
            step=args.step,              # Используем переданный step
            window_size=args.window_size
        )
    except Exception as e:
        logger.error(f"Iterator creation error: {e}", exc_info=True)
        raise
    logger.debug(f"Dataset {type(loader).__name__} loaded")
    logger.debug(f"Processing chromosomes: {', '.join(loader.processing_chromosomes)}")

    # --- Выходной файл (один на чанк) ---
    output_csv = os.path.join(args.out_dir, f"{bam_basename}_dpst.csv")
    dfs = []
    header_written = False   # флаг, чтобы записать заголовок только один раз

    for i, batch in enumerate(loader):
        L = batch['starts'].to(args.device).reshape(-1, 1)
        R = batch['ends'].to(args.device).reshape(-1, 1)
        assert len(L) == len(R)
        if len(L) == 0:
            logger.info(f"Window {i} of {batch['ref']} is empty. Skip...")
            continue

        try:
            model.fit(L, R)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt – saving buffer and exiting.")
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                # При прерывании дописываем без заголовка (или с заголовком, если ещё не было)
                save_dataframe(df, output_csv, header=not header_written)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error at {loader.chromosome}:{loader.window_start}-{loader.window_stop}: {e}", exc_info=True)
            continue
        else:
            logger.info(f"Window {i} processed: {loader.chromosome}:{loader.window_start}-{loader.window_stop}")
            cur_df = model.to_df()
            cur_df['ref'] = batch['ref']
            cur_df['batch_i'] = i
            cur_df['qid'] = batch['id']
            dfs.append(cur_df)

            # Если накопилось 10 фреймов – сбрасываем на диск
            if len(dfs) >= 10:
                df = pd.concat(dfs, ignore_index=True)
                save_dataframe(df, output_csv, header=not header_written)
                header_written = True   # после первой записи заголовок уже есть
                dfs.clear()

    # --- Остаток данных после цикла ---
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        save_dataframe(df, output_csv, header=not header_written)

    logger.info("Processing finished successfully.")


if __name__ == '__main__':
    main()