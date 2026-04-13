import sys, os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


from ExoModel import exo_model


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
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    loader = exo_model.TemplateLengthAnalyzer(args.bam_file)
    pairs, dist = loader.calculate_distribution()
    model = exo_model.AdaptiveHistogramDeconvolution(dist, lamb_range=(args.start, args.end), n_jobs=args.njobs)
    res = model.grid_search(n_lamb=args.npoints, verbose=args.verbose)

    if not args.no_plots:  
        outpath = os.path.join(args.output_dir, 'optimization_results.png')
        fig, ax = model.plot_results()
        fig.savefig(outpath)
        plt.close(fig) 
    
    errors_outpath = os.path.join(args.output_dir, 'errors.csv')
    model.save(errors_outpath)


if __name__ == '__main__':
    main()