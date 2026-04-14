import sys, os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import signal


from ExoModel import exo_model
from script_tools import signal_handler


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze template length distribution from BAM file"
    )
    parser.add_argument(
        "bam_file",
        type=str,
        help="Path to the input BAM file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./exomodel_results",
        help="Output directory (default: ./exomodel_results)"
    )
    parser.add_argument(
        "--no-plots", "-n",
        action="store_true",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--start", "-s",
        default=1,
        help="start for scanning",
        type=int
    )
    parser.add_argument(
        "--end", "-e",
        default=50,
        help="end for scanning",
        type=int
    )
    parser.add_argument(
        "--npoints",
        default=10,
        help="number of lambda points to test",
        type=int
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",  # Changed from store_false to store_true
        default=True,         # Added default
        help="enable verbose output"
    )
    parser.add_argument(
        "--njobs", "-@",
        help="number of workers",
        type=int,
        default=1
    )
    parser.add_argument(
        "--plot_fits",
        help="plot each optimization results",
        action="store_true",
    )
    parser.add_argument(
        "--reg_coef",
        help="fit with current regularization coeffizient value",
        type=float,
        default=None
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    signal.signal(signal.SIGINT, signal_handler)   
    signal.signal(signal.SIGTERM, signal_handler)  
    signal.signal(signal.SIGQUIT, signal_handler)

    
    os.makedirs(args.output_dir, exist_ok=True)
    loader = exo_model.TemplateLengthAnalyzer(args.bam_file)
    pairs, dist = loader.calculate_distribution()

    if args.reg_coef is not None:
        model = exo_model.HistogramDeconvolution(dist, args.reg_coef)
        res = model.fit()
        if not args.no_plots:  
            outpath = os.path.join(args.output_dir, 'optimization_results.png')
            fig, ax = model.plot()
            fig.savefig(outpath)
            plt.close(fig) 
    else:
        model = exo_model.AdaptiveHistogramDeconvolution(dist, lamb_range=(args.start, args.end), n_jobs=args.njobs)
        res = model.grid_search(n_lamb=args.npoints, verbose=args.verbose)

        if not args.no_plots:  
            outpath = os.path.join(args.output_dir, 'optimization_results.png')
            fig, ax = model.plot_results()
            fig.savefig(outpath)
            plt.close(fig) 
    
    errors_outpath = os.path.join(args.output_dir, 'errors.csv')
    model.save(errors_outpath)

    if args.plot_fits:
        plots_dir = os.path.join(args.output_dir, 'fit_plots')
        os.makedirs(plots_dir, exist_ok=True)
        figsize = (12, 5)  
        for i, row in model.get_summary_table().iterrows():
            result_x = row.result_x
            reconstructed = row.reconstructed
            lamb = row.lamb
            
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            axs[0].plot(result_x, linewidth=2)
            axs[0].set_title("Deconvolved Distribution", fontsize=14)
            axs[0].grid(True, alpha=0.3)

            axs[1].plot(reconstructed, label="Model (P * P)", linewidth=2)
            axs[1].plot(model.hist, label="Experimental", linewidth=2, alpha=0.7)
            axs[1].set_title("Histogram Comparison", fontsize=14)
            axs[1].legend(fontsize=12)
            axs[1].grid(True, alpha=0.3)

            save_path = os.path.join(plots_dir, f'fitplot{lamb}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')  
            plt.close(fig)  



if __name__ == '__main__':
    main()