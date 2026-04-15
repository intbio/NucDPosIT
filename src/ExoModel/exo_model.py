import sys


import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import scipy as spy
from scipy.optimize import OptimizeResult
from tqdm.auto import tqdm
import pysam
from collections import Counter
from typing import Optional, Dict, Tuple, Union
from pathlib import Path
import numpy as np




class TemplateLengthAnalyzer:   
    def __init__(self, bam_path: Union[str, Path], lazy_load: bool = True):
        self.bam_path = Path(bam_path)
        self.bam_file: Optional[pysam.AlignmentFile] = None
        self.length_distribution: Optional[Counter] = None
        self.total_pairs: int = 0
        self.is_loaded: bool = False
        
        if not lazy_load:
            self.load()
    
    def load(self) -> None:
        if self.is_loaded:
            print("already loaded")
            return
        
        if not self.bam_path.exists():
            raise FileNotFoundError(f"BAM файл не найден: {self.bam_path}")
        
        self.bam_file = pysam.AlignmentFile(str(self.bam_path), "rb")
        self.is_loaded = True
        print(f"file loaded: {self.bam_path}")
    
    def calculate_distribution(self) -> Tuple[Counter, int]:
        if not self.is_loaded:
            self.load()
        
        if self.length_distribution is not None:
            print("from cache")
            return self.length_distribution, self.total_pairs
        
        length_dist = Counter()
        total_pairs = 0
        
        for read in self.bam_file:
            if read.is_paired and read.is_proper_pair and read.template_length > 0:
                tl = read.template_length
                length_dist[tl] += 1
                total_pairs += 1

        self.total_pairs = total_pairs
        self.length_distribution = np.array([length_dist[i] for i in range(max(length_dist.keys()) + 101)], dtype=float)
        self.length_distribution /= self.length_distribution.sum()
        return self.total_pairs, self.length_distribution


class HistogramDeconvolution:

    def __init__(self, hist, lamb=14, center_loc=73, center_scale=10):
        self.hist = hist
        self.lamb = lamb
        self.center_loc = center_loc
        self.center_scale = center_scale

        self.Q = hist.reshape(-1, 1)
        self.N = len(self.Q)
        self.k = self.N // 2 + 1

        self.P_init = None
        self.weights = None
        self.center = None
        self.result = None

        self._initialize_parameters()

    def _initialize_parameters(self):
        x = np.arange(0, self.k, 1)
        self.P_init = spy.stats.norm.pdf(
            x, loc=self.center_loc, scale=self.center_scale
        )
        self.P_init /= self.P_init.sum()

        self.center = np.argmax(self.Q) // 2
        x = np.arange(0, self.k)
        weights = (self.center - x) ** 2
        self.weights = weights / np.max(weights)

    def loss(self, P):
        reg = self.lamb * np.dot(P**2, self.weights)
        loss = np.sum((self.Q - np.convolve(P, P).reshape(-1, 1)) ** 2) + reg
        return loss

    def fit(self, method="SLSQP", tol=1e-30, maxiter=2000, **kwargs):
        bounds = [(0, 1) for _ in range(self.k)]
        constraints = {"type": "eq", "fun": lambda p: np.sum(p) - 1}
        self.result = spy.optimize.minimize(
            self.loss,
            bounds=bounds,
            x0=self.P_init,
            method=method,
            constraints=constraints,
            options={"maxiter": maxiter},
            tol=tol,
            **kwargs,
        )

        return self.result

    def get_deconvolved(self):
        if self.result is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.result.x

    def get_reconstructed_hist(self):
        if self.result is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return np.convolve(self.result.x, self.result.x)

    def plot(self, figsize=(16, 9)):
        if self.result is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        axs[0].plot(self.result.x, linewidth=2)
        axs[0].set_title("Deconvolved Distribution P", fontsize=14)
        axs[0].set_xlabel("Index", fontsize=12)
        axs[0].set_ylabel("Probability", fontsize=12)
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(self.get_reconstructed_hist(), label="Model (P * P)", linewidth=2)
        axs[1].plot(self.hist, label="Experimental", linewidth=2, alpha=0.7)
        axs[1].set_title("Histogram Comparison", fontsize=14)
        axs[1].set_xlabel("Index", fontsize=12)
        axs[1].set_ylabel("Counts", fontsize=12)
        axs[1].legend(fontsize=12)
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axs

    def summary(self, file_path=None):
        if self.result is None:
            print("Model not fitted yet. Call fit() first.")
            return

        if file_path:
            original_stdout = sys.stdout
            with open(file_path, 'w', encoding='utf-8') as f:
                sys.stdout = f
                self._print_summary()
                sys.stdout = original_stdout
        else:
            self._print_summary()

    def _print_summary(self):
        """Внутренний метод для печати summary"""
        print("=" * 60)
        print("DECONVOLUTION RESULTS")
        print("=" * 60)
        print(f"Success: {self.result.success}")
        print(f"Message: {self.result.message}")
        print(f"Number of iterations: {self.result.nit}")
        print(f"Final loss: {self.result.fun:.6f}")
        print(f"Lambda (regularization): {self.lamb}")
        print(f"Sum of P: {np.sum(self.result.x):.6f}")
        print(f"Max P value: {np.max(self.result.x):.6f}")
        print(f"Min P value: {np.min(self.result.x):.6f}")
        print("=" * 60)

    def save(self, outpath: str):
        np.savetxt(
            outpath,
            self.result.x, 
            delimiter=",",
            comments="", 
            fmt="%f"
        )


class AdaptiveHistogramDeconvolution(HistogramDeconvolution):

    def __init__(self, hist, lamb_range=(1, 50), n_jobs=1, **kwargs):
        super().__init__(hist, **kwargs)
        self.lamb_range = lamb_range
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.best_lamb = None
        self.best_result = None
        self.all_results = []

    def save(self, outdir: str):
        np.savetxt(
            outdir,
            self.best_result['result_x'], 
            delimiter=",",
            comments="", 
            fmt="%f"
        )

    @staticmethod
    def _evaluate_lamb(lamb, hist, center_loc, center_scale, method, tol):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = HistogramDeconvolution(
                    hist=hist,
                    lamb=lamb,
                    center_loc=center_loc,
                    center_scale=center_scale,
                )
                result = model.fit(method=method, tol=tol)

                return {
                    "lamb": lamb,
                    "loss": result.fun,
                    "success": result.success,
                    "nit": result.nit,
                    "message": result.message,
                    "result_x": result.x,
                    "reconstructed": model.get_reconstructed_hist(),
                }
        except Exception as e:
            return {
                "lamb": lamb,
                "loss": np.inf,
                "success": False,
                "error": str(e),
                "message": f"Failed: {str(e)}",
            }

    def grid_search(self, n_lamb=20, verbose=True, parallel=True):
        lamb_values = np.linspace(self.lamb_range[0], self.lamb_range[1], n_lamb)

        if verbose:
            print(f"Starting grid search with {len(lamb_values)} lambda values")
            print(f"Lambda range: [{self.lamb_range[0]:.2f}, {self.lamb_range[1]:.2f}]")
            print(f"Using {self.n_jobs if parallel else 1} processes")
            print("-" * 60)

            with Pool(processes=self.n_jobs) as pool:
                eval_func = partial(
                    self._evaluate_lamb,
                    hist=self.hist,
                    center_loc=self.center_loc,
                    center_scale=self.center_scale,
                    method="SLSQP",
                    tol=1e-30,
                )
                if verbose:
                    results = list(
                        tqdm(
                            pool.imap(eval_func, lamb_values),
                            total=len(lamb_values),
                            desc="Grid search",
                        )
                    )
                else:
                    results = pool.map(eval_func, lamb_values)

        self.all_results = results
        valid_results = [r for r in results if r["success"] and np.isfinite(r["loss"])]

        if not valid_results:
            print("Warning: No successful optimizations found!")
            return None

        best = min(valid_results, key=lambda x: x["loss"])

        self.best_lamb = best["lamb"]
        self.best_result = best

        if verbose:
            print("-" * 60)
            print(f"Best lambda found: {best['lamb']:.4f}")
            print(f"Best loss: {best['loss']:.6f}")
            print(f"Iterations: {best['nit']}")
            print("-" * 60)

        return {"lamb": self.best_lamb}

    def plot_results(self, figsize=(16, 10)):
        if not self.all_results:
            print("No results to plot. Run grid_search() or random_search() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        lambdas = [r["lamb"] for r in self.all_results]
        losses = [r["loss"] for r in self.all_results]

        sorted_indices = np.argsort(lambdas)
        lambdas_sorted = np.array(lambdas)[sorted_indices]
        losses_sorted = np.array(losses)[sorted_indices]

        axes[0, 0].plot(lambdas_sorted, losses_sorted, "b-o", linewidth=2, markersize=6)
        axes[0, 0].set_xlabel("Lambda", fontsize=12)
        axes[0, 0].set_ylabel("Loss", fontsize=12)
        axes[0, 0].set_title("Loss vs Lambda", fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        best_idx = np.argmin(losses)
        axes[0, 0].plot(
            lambdas[best_idx],
            losses[best_idx],
            "r*",
            markersize=15,
            label=f"Best: λ={lambdas[best_idx]:.3f}, loss={losses[best_idx]:.4f}",
        )
        axes[0, 0].legend()

        scatter = axes[0, 1].scatter(
            lambdas,
            losses,
            c=losses,
            cmap="viridis",
            s=80,
            alpha=0.6,
            edgecolors="black",
            linewidth=1,
        )
        axes[0, 1].set_xlabel("Lambda", fontsize=12)
        axes[0, 1].set_ylabel("Loss", fontsize=12)
        axes[0, 1].set_title("Loss Distribution", fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label="Loss")

        axes[1, 0].hist(losses, bins=20, edgecolor="black", alpha=0.7, color="skyblue")
        axes[1, 0].set_xlabel("Loss", fontsize=12)
        axes[1, 0].set_ylabel("Frequency", fontsize=12)
        axes[1, 0].set_title("Loss Distribution Histogram", fontsize=14)
        axes[1, 0].axvline(
            np.min(losses),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best: {np.min(losses):.4f}",
        )
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        if self.best_result:
            axes[1, 1].plot(
                self.best_result["result_x"],
                linewidth=2,
                color="green",
                label="Deconvolved P",
            )
            axes[1, 1].set_xlabel("Index", fontsize=12)
            axes[1, 1].set_ylabel("Probability", fontsize=12)
            axes[1, 1].set_title(f"Best Result (λ={self.best_lamb:.3f})", fontsize=14)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            axes[1, 1].text(
                0.05,
                0.95,
                f"Loss = {self.best_result['loss']:.6f}",
                transform=axes[1, 1].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        return fig, axes

    def get_summary_table(self):
        try:
            import pandas as pd
        except ImportError:
            print("Pandas not installed. Install with: pip install pandas")
            return None

        df = pd.DataFrame(self.all_results)
        df = df.sort_values("loss")
        return df

    def fit_best(self, **kwargs):
        if self.best_lamb is None:
            print("No best lambda found. Run grid_search() or random_search() first.")
            return None

        self.lamb = self.best_lamb
        result = self.fit(**kwargs)
        return result