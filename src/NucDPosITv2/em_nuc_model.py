import func_utils

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
import scipy as spy
from collections import defaultdict
import pysam
from torch.utils.data import Dataset, DataLoader
import argparse
from typing import Dict, List, Optional




class CoordinateProbs():
    def __init__(self, y_error, x_error, device="cpu", offset=43, l0=146):
        self.y_error = torch.tensor(y_error, dtype=torch.float, device=device)
        self.x_error = torch.tensor(x_error, dtype=torch.int, device=device)
        self.offset = offset
        self.__device = device
        self.l0 = l0

    @property
    def fit_res(self):
        return self.y_error

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, new_device):
        self.__device = new_device
        self.x_error = self.x_error.to(new_device)
        self.y_error = self.y_error.to(new_device)

    def get_tlen_prob(self, left_cords, right_cords, dyad_positions):
        left_errors, right_errors = (
            dyad_positions - left_cords - self.l0 // 2,
            right_cords - dyad_positions - self.l0 // 2,
        )
        left_probs, right_probs = self.get_prob(left_errors), self.get_prob(
            right_errors
        )
        return left_probs * right_probs

    def get_prob(self, error):
        indices = (error + self.offset).long()
        probs = torch.zeros_like(error, dtype=torch.float, device=self.device) + 1e-20
        mask = (indices >= 0) & (indices < self.y_error.shape[0])
        probs[mask] = self.y_error[indices[mask]]
        return probs

    
    
class EMTorch(nn.Module):
    def __init__(
        self,
        coordinats_counter,
        positions0,
        weights0,
        tau=0,
        device="cpu",
    ):
        nn.Module.__init__(self)
        self.tau = tau
        self.__device = device
        self.coordinats_counter = coordinats_counter
        self.params = nn.ParameterDict(
            {
                "positions": nn.Parameter(
                    positions0.clone().detach().float(),
                    requires_grad=False,
                ),
                "weights": nn.Parameter(
                    weights0.clone().detach().float(),
                    requires_grad=False,
                ),
                "gij": None,
                "m": None,
                "X": None,
            }
        )

    @property
    def nofreads(self):
        return self.gij.shape[0]

    @property
    def positions(self):
        return self.params["positions"].data

    @positions.setter
    def positions(self, new_pos):
        self.params["positions"].data = new_pos.clone().detach().float()

    @property
    def weights(self):
        return self.params["weights"].data

    @weights.setter
    def weights(self, new_weights):
        self.params["weights"].data = new_weights.clone().detach().float()
        

    @property
    def gij(self):
        return self.params["gij"].data

    @gij.setter
    def gij(self, new_gij):
        if self.params["gij"] is not None:
            self.params["gij"].data = new_gij
        else:
            self.params["gij"] = nn.Parameter(new_gij, requires_grad=False)

    @property
    def m(self):
        return self.params["m"].data

    @m.setter
    def m(self, new_m):
        if self.params["m"] is not None:
            self.params["m"].data = new_m
        else:
            self.params["m"] = nn.Parameter(new_m, requires_grad=False)

    @property
    def X(self):
        return self.params["X"].data

    @X.setter
    def X(self, new_X):
        if self.params["X"] is not None:
            self.params["X"].data = new_X
        else:
            self.params["X"] = nn.Parameter(new_X, requires_grad=False)

    @property
    def device(self):
        return self.__device

    @property
    def n_nucs(self):
        return self.weights.shape[0]

    def to(self, new_device, *args, **kwargs):
        self.__device = new_device
        self.coordinats_counter.device = new_device
        super().to(new_device, *args, **kwargs)

    def get_tlen_prob(self, left_cords, right_cords, dyad_positions):
        return self.coordinats_counter.get_tlen_prob(
            left_cords, right_cords, dyad_positions
        )

    def E_step(self, left_cords, right_cords):
        self.X = (
            self.get_tlen_prob(left_cords, right_cords, self.positions) * self.weights
        )
        p_x = self.weights @ self.X.T
        gij = self.X / p_x.reshape(-1, 1)
        self.gij = gij / (gij.sum(axis=1).reshape(-1, 1))

    def M_step(self):
        self.positions = (
            torch.argmax(self.gij.T.float() @ self.m.float(), dim=1).float()
            + self.offset
        )
        weights = self.gij.sum(dim=0) / self.nofreads - self.tau
        if torch.any(weights < 0):
            weights[weights < 0] = 0
            self.weights = weights / weights.sum()

    def set_m_matrix(self, left_cords, right_cords):
        max_right_cord = right_cords.max()
        min_left_cord = left_cords.min()
        self.offset = (left_cords.min()).float()
        dyads = torch.arange(min_left_cord, max_right_cord, device=self.device)
        m = self.get_tlen_prob(left_cords, right_cords, dyads)
        m = torch.log(m)
        return m

    def init_fields(self, left_cords, right_cords):
        self.coordinats_counter.device = self.device
        self.gij = torch.zeros((left_cords.size()[0], self.n_nucs), device=self.device)
        self.m = self.set_m_matrix(left_cords, right_cords).clone().detach().float()
        self.X = torch.zeros(left_cords.shape[0], self.n_nucs)

    def forward(self, left_cords, right_cords):
        self.E_step(left_cords, right_cords)
        self.M_step()
        return [self.positions, self.weights]

    def cluster_occupancy(self):
        return self.gij.sum(dim=0)
    
    
    
class ChromosomeBamDataset(Dataset):
    def __init__(
        self, bam_path: str, max_reads_per_chrom: Optional[int] = None, transform=None
    ):
        self.bam = pysam.AlignmentFile(bam_path, "rb")
        self.chromosomes = self.get_nonempty_chromosomes()
        self.max_reads = max_reads_per_chrom
        self.transform = transform

    def get_nonempty_chromosomes(self) -> List[str]:
        return [chrom for chrom in self.bam.references if self.bam.count(chrom) > 0]

    def __len__(self) -> int:
        return len(self.chromosomes)

    def __getitem__(self, idx: int):
        chrom = self.chromosomes[idx]
        reads = self._load_chromosome_reads(chrom)

        if self.transform:
            reads = self.transform(reads)

        return {
            "chromosome": chrom,
            "reads": reads,  
            "num_reads": len(reads),
        }

    def _load_chromosome_reads(self, chrom: str) -> torch.Tensor:
        reads = []
        for i, read in enumerate(self.bam.fetch(chrom)):
            if self.max_reads and i >= self.max_reads:
                break
            reads.append(read)

        return reads
        
        
        
class PysamRecordsDataset(Dataset):
    def __init__(self, pysam_records):
        self.records = self._collect_paired_reads(pysam_records)
        self.chromosome = pysam_records[0].reference_name

    def __getitem__(self, idx: int):
        record = self.records[idx]
        items = {
            "start": record[0].reference_start,
            "end": record[1].reference_end,
            "id": record[0].qname,
            'chromo': record[0].reference_name
        }
        return items

    def __len__(self):
        return len(self.records)

    def _collect_paired_reads(self, records):
        read_pairs = defaultdict(dict)
        valid_pairs = []

        for read in records:
            name = read.query_name
            if read.is_read1:
                read_pairs[name]["R1"] = read
            elif read.is_read2:
                read_pairs[name]["R2"] = read

        # Отбираем только полные пары
        for name, pair in read_pairs.items():
            if "R1" in pair and "R2" in pair:
                valid_pairs.append((pair["R1"], pair["R2"]))

        return valid_pairs
    


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for occupancy MNase-Seq profile deconvolution via EM-algorythm")
    parser.add_argument("-error_csv", help='csv file with exonuclease dissociation rates', type=str, required=True)
    parser.add_argument("-o", "--out_csv", help='output file', type=str, default='dyads.csv')
    parser.add_argument("-b", "--bam_input", help='bam alignment file', type=str, required=True)
    # parser.add_argument('-device', help='use cpu or cuda', default='cpu')
    parser.add_argument('-n_nucs', help='initial amount of nucleosomes', type=int, default=100)
    parser.add_argument('-batch_size', help='size of a batch', type=int, default=200)
    
    args = parser.parse_args()
    
    print(args.bam_input)
    
    
    ERROR_CSV = args.error_csv
    y_error = pd.read_csv(ERROR_CSV).to_numpy().flatten()
    x_error = np.arange(-len(y_error) // 2, len(y_error) // 2)
    cp = CoordinateProbs(y_error, x_error)
    device = 'cpu'
    n_nucs = args.n_nucs
    batch_size = args.batch_size
    dataset = ChromosomeBamDataset(args.bam_input)
    

    for i in range(len(dataset)):
        cur_batch = dataset[i]
        loader = DataLoader(
            PysamRecordsDataset(cur_batch["reads"]), shuffle=False, batch_size=batch_size
        )
        for record_batch in loader:
            left_cords, right_cords = (
                record_batch["start"].view(-1, 1).clone().detach(),
                record_batch["end"].view(-1, 1).clone().detach(),
            )
            p0 = (
                torch.linspace(left_cords.min(), right_cords.max(), n_nucs)
                .float()
                .to(device)
            )
            w0 = torch.tensor([1 / n_nucs] * n_nucs)
            w0 = w0 / w0.sum()
            tau_grid = np.linspace(0, 0.001, 25)
            all_rmsd = []

            for tau in tau_grid:
                model = EMTorch(cp, p0, w0, tau, device)
                func_utils.fit_model(model, left_cords, right_cords)
                rmsd = func_utils.count_loss(model)
                all_rmsd.append(rmsd)

            best_tau = tau_grid[np.argmin(all_rmsd)]
            model = EMTorch(cp, p0, w0, best_tau, device)
            model = func_utils.fit_model(model, left_cords, right_cords, 500)
            df = func_utils.make_window_df(model, torch.hstack([left_cords, right_cords]))
            df["qid"] = record_batch["id"]
            bin_heights = df.groupby('dyad', as_index=False).nuc_prob.sum()
            with open(args.out_csv, 'a+') as out_csv:
                bin_heights.to_csv(out_csv, index=False, header=False)

    


