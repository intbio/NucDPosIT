import sys
import argparse
import pandas as pd
import signal 
import logging
import pprint

import EMmodel.datasets as datasets
from EMmodel import em_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run EM model analysis')
    parser.add_argument('alignment_path',
                         type=str,
                        help='Path to genome alignment file')
    parser.add_argument('--regions_path',
                         type=str,
                         help='Path to file with specified regions to analyse',
                         default=None
                         )
    parser.add_argument('errorpath', type=str,
                        help='Path to error file')
    parser.add_argument('--output', type=str,
                        help='Output file path (CSV or parquet)', default='./nucdpst_res.csv')
    parser.add_argument('--dyad_dist', type=int,
                        help='Initial dyad distance',
                       default=40)
    parser.add_argument('--nfits', type=int,
                        help='number of optimization steps',
                       default=50)
    parser.add_argument('--device', type=str,
                        help='device: cpu or cuda',
                       default='cuda')
    parser.add_argument('--tol', type=float,
                        help='tollerance for sliding mean',
                       default=0.01)
    parser.add_argument('--alpha', type=float,
                        help='coefficient of sliding mean',
                       default=0.1)
    parser.add_argument('--reg_coef', type=int,
                        help='EM regularisation coefficient',
                       default=1)
    parser.add_argument('--temperature_coef', type=int,
                        help='temperature coefficient for EM',
                       default=20)
    parser.add_argument('--min_iter', type=int,
                        help='min amount of iteration in optimization step',
                       default=100)
    parser.add_argument('--max_iter', type=int,
                        help='max amount of iteration in optimization step',
                       default=500)
    parser.add_argument('--window_size', type=int,
                        help='size of scanning window',
                       default=5000)
    parser.add_argument('--step', type=int,
                        help='step of scanning',
                       default=4800)
    return parser.parse_args()


def setup_logging():
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(
        'nucdposit.log',
         encoding='utf-8',
          mode='w'
          )
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)



def exit_gracefully(signum, frame):
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")

    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()
    raise KeyboardInterrupt
    sys.exit(1) 


def main():
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    args = parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    formatted = pprint.pformat(vars(args), indent=2, width=120, sort_dicts=False)
    logger.info(f"Configuration:\n{formatted}")



    try:
        model = em_model.StochasticEMMOdel(  
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
    except Exception as error:
        logger.error(f"Model creation error: {error}", exc_info=True)
        raise error
    logger.debug(f'model {type(model).__name__} loaded')

    try:
        factory = datasets.DatasetFactory()
        loader = factory.create_loader(alignment_file=args.alignment_path, 
                                bed_file=args.regions_path,
                                start=0,
                                stop=None,
                                step=4800,
                                window_size=args.window_size)
    except Exception as error:
        logger.error(f"Iterator creation error: {error}", exc_info=True)
        raise error
    logger.debug(f'dataset {type(loader.dataset).__name__} loaded')
    logger.debug(f'processing chromosomes are {",".join(loader.dataset.processing_chromosomes)}')

    dfs = []
    for i, batch in enumerate(loader):
        L, R = batch['starts'].to(args.device).reshape(-1, 1), batch['ends'].to(args.device).reshape(-1, 1)
        assert len(L) == len(R)
        if len(L) == 0:
            logger.info(f"window {i} of {batch['ref']}: is empty. Skip...")
            continue

        try:
            fit_res = model.fit(L, R)
        except KeyboardInterrupt:
            logger.debug('cleaning buffer')
            df = pd.concat(dfs, ignore_index=True)  
            df.to_csv(args.output)  
            sys.exit(1)
        except Exception as error:
            logger.error(f"{loader.dataset.chromosome, loader.dataset.window_start, loader.dataset.window_stop}", exc_info=True)
        else:
            logger.info(f"window {i} processed: {loader.dataset.chromosome, loader.dataset.window_start, loader.dataset.window_stop}")
            cur_df = model.to_df()
            cur_df['ref'] = loader.dataset.chromosome
            cur_df['batch_i'] = i
            cur_df['qid'] = batch['id']
            dfs.append(cur_df)
            if len(dfs) == 50:
                df = pd.concat(dfs, ignore_index=True)  
                df.to_csv(args.output, mode='a', header=None, sep='\t', index=None)  
                dfs.clear()
    if len(dfs) != 0:
        df = pd.concat(dfs, ignore_index=True)  
        df.to_csv(args.output, mode='a', header=None, sep='\t', index=None)  


if __name__ == '__main__':
    main()