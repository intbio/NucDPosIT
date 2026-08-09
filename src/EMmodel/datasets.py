import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pybedtools
import torch
import pysam
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging


logger = logging.getLogger(__name__)


class NucdpositDataset(Dataset, ABC):
    def __init__(
        self,
        alignment_file,
        start=0,
        stop=None,
        step=4800,
        window_size=5000,
        except_chromo: list = None,
        accept_chromo: list = None,
    ):
        self.alignment_file = alignment_file
        self.start = start
        self.stop = stop
        self.step = step
        self.window_size = window_size

        self.__except_chromo = except_chromo
        self.__accept_chromo = accept_chromo
        self.__processing_chomosomes = None

    @property
    def except_chromo(self):
        return self.__except_chromo

    @property
    def accept_chromo(self):
        return self.__accept_chromo

    @property
    def processing_chromosomes(self):
        if not self.__processing_chomosomes:
            data_chromosomes = set(self.get_chromosomes())
            accept_set = set(self.accept_chromo) if self.accept_chromo else None
            except_set = set(self.except_chromo) if self.except_chromo else set()

            if accept_set is not None:
                self.__processing_chromosomes = list(data_chromosomes & accept_set)
            else:
                self.__processing_chromosomes = list(data_chromosomes - except_set)

        return self.__processing_chromosomes

    def get_chromosome_lengths(self) -> dict:
        with pysam.AlignmentFile(self.alignment_file) as bam:
            return dict(zip(bam.references, bam.lengths))

    def get_chromosomes(self):
        with pysam.AlignmentFile(self.alignment_file) as file:
            refs = file.references
            return refs

    def fetch_records(self, chromosome, start, stop):
        with pysam.AlignmentFile(self.alignment_file, "rb") as bamfile:
            records = list(bamfile.fetch(chromosome, start, stop))
        return records

    @abstractmethod
    def get_segment_starts(self, records: list):
        pass

    @abstractmethod
    def get_segment_ends(self, records: list):
        pass

    @abstractmethod
    def get_segment_ids(self, records: list):
        pass

    @abstractmethod
    def get_segments_references(self, records: list):
        pass

    @abstractmethod
    def get_window_start(self):
        pass

    @abstractmethod
    def get_window_end(self):
        pass

    @abstractmethod
    def process_records(self) -> list:
        pass

    def __getitem__(self, idx):
        if idx < 0:
            raise ValueError("idx can not be less than 0")

        if idx >= len(self):
            raise IndexError(f"idx {idx} out of range (dataset size: {len(self)})")

        records = self.process_records(idx)
        items = {
            "ref": self.get_segments_references(records),
            "starts": self.get_segment_starts(records),
            "ends": self.get_segment_ends(records),
            "id": self.get_segment_ids(records),
            'window_start': self.get_window_start(),
            'window_end': self.get_window_end()
        }
        return items

    def _find_first_greater(self, arr, target):
        idx = bisect.bisect_right(arr, target)
        return idx if idx < len(arr) else -1

    def _collect_paired_reads(self, records):
        from collections import defaultdict

        pairs = defaultdict(dict)
        for r in records:
            if r.is_read1:
                pairs[r.query_name]["R1"] = r
            elif r.is_read2:
                pairs[r.query_name]["R2"] = r

        return [(p["R1"], p["R2"]) for p in pairs.values() if "R1" in p and "R2" in p]


class BamDataset(NucdpositDataset):
    def __init__(
        self,
        alignment_file,
        start=0,
        stop=None,
        step=4800,
        window_size=5000,
        except_chromo: list = None,
        accept_chromo: list = None,
    ):
        super().__init__(
            alignment_file, start, stop, step, window_size, except_chromo, accept_chromo
        )
        self.__index = None

    @property
    def index(self):
        if self.__index is None:
            self.__index = []
            chrom_lengths = self.get_chromosome_lengths()
            for chromo in self.processing_chromosomes:
                chrom_len = chrom_lengths[chromo]

                effective_start = self.start
                effective_stop = self.stop if self.stop is not None else chrom_len

                if effective_stop > effective_start + self.window_size:
                    num_windows = (
                        effective_stop - effective_start - self.window_size
                    ) // self.step + 1
                else:
                    num_windows = 0

                self.__index.append(num_windows)
            self.__index = np.cumsum(self.__index)
        return self.__index

    def get_segments_references(self, records):
        refname = records[0][0].reference_name
        return refname

    def get_segment_starts(self, records):
        starts = torch.tensor([pair[0].reference_start for pair in records])
        return starts

    def get_segment_ends(self, records):
        ends = torch.tensor([pair[1].reference_end for pair in records])
        return ends

    def get_segment_ids(self, records):
        ids = [pair[0].query_name for pair in records]
        return ids

    def choose_chromosome(self, chromo_idx):
        if chromo_idx == -1:
            raise IndexError(f"found chromosome index is equal to {chromo_idx}")
        chosen_chromosome = self.processing_chromosomes[chromo_idx]
        return chosen_chromosome

    def process_records(self, idx):
        chromo_idx = self._find_first_greater(self.index, idx)
        self.chromosome = self.choose_chromosome(chromo_idx)

        if chromo_idx == 0:
            offset = idx
        else:
            offset = idx - self.index[chromo_idx - 1]

        chrom_lengths = self.get_chromosome_lengths()
        chrom_len = chrom_lengths[self.chromosome]

        effective_start = self.start
        effective_stop = self.stop if self.stop is not None else chrom_len

        self.window_start = effective_start + offset * self.step
        self.window_stop = min(self.window_start + self.window_size, effective_stop)

        records = self.fetch_records(self.chromosome, self.window_start, self.window_stop)
        paired_records = self._collect_paired_reads(records)
        return paired_records

    def get_window_start(self):
        return self.window_start

    def get_window_end(self):
        return self.window_stop

    def __len__(self):
        total_windows = 0
        chrom_lengths = self.get_chromosome_lengths()
        for chromo in self.processing_chromosomes:
            chrom_len = chrom_lengths[chromo]

            effective_start = self.start
            effective_stop = self.stop if self.stop is not None else chrom_len

            if effective_stop > effective_start + self.window_size:
                num_windows = (
                    effective_stop - effective_start - self.window_size
                ) // self.step + 1
            else:
                num_windows = 0

            total_windows += num_windows
        return total_windows


class BamRegionsDataset(NucdpositDataset):
    def __init__(self,
        alignment_file,
        bed_file,
        start=0,
        stop=None,
        step=4800,
        window_size=5000,
        except_chromo: list = None,
        accept_chromo: list = None,):
        super().__init__(alignment_file, start, stop, step, window_size, except_chromo, accept_chromo)
        self.bed_file = bed_file
        self.intervals = None

    @property
    def index(self):
        if self.intervals is not None:
            return self.intervals
            
        self.intervals = []
        bed = pybedtools.BedTool(self.bed_file)
        
        for interval in bed:
            chrom = interval.chrom
            if chrom not in self.processing_chromosomes:
                continue
            start = self.start if self.start != 0 else int(interval.start)
            end = self.stop if self.stop is not None else int(interval.end)
            
            if start >= end:
                continue
            current_start = start
            while current_start < end:
                current_end = min(current_start + self.window_size, end)
                self.intervals.append((chrom, current_start, current_end))
                current_start += self.step
        return self.intervals

    def __len__(self):
        return len(self.index) 

    def process_records(self, idx):
        interval = self.index[idx]
        self.chromosome, self.window_start, self.window_stop = interval
        records = self.fetch_records(self.chromosome, self.window_start, self.window_stop)
        paired_records = self._collect_paired_reads(records)
        return paired_records

    def get_segments_references(self, records):
        refname = records[0][0].reference_name
        return refname

    def get_segment_starts(self, records):
        starts = torch.tensor([pair[0].reference_start for pair in records])
        return starts

    def get_segment_ends(self, records):
        ends = torch.tensor([pair[1].reference_end for pair in records])
        return ends

    def get_segment_ids(self, records):
        ids = [pair[0].query_name for pair in records]
        return ids

    def get_window_start(self):
        return self.window_start

    def get_window_end(self):
        return self.window_stop

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        items['interval_idx'] = idx
        return items


class DatasetFactory():

    def is_bam_valid(self, filepath: str) -> bool:
        try:
            with pysam.AlignmentFile(filepath, 'rb', check_header=True) as bam:
                header = bam.header
                return True
        except (OSError, ValueError, pysam.SamtoolsError) as error:
            logger.error(f"BAMfile {filepath} is not valid: {error}")
            return False

    def is_bed_valid(self, filepath: str) -> bool:
        try:
            bed = pybedtools.BedTool(filepath)
            bed.to_dataframe()
        except Exception as error:
            logger.error(f"Bedfile {filepath} is not valid: {error}")
            return False
        return True
    
    def create_dataset(self, alignment_file, bed_file=None, *args, **kwargs):
        if bed_file:
            dataset = BamRegionsDataset(alignment_file, bed_file, *args, **kwargs)
        else:
            dataset = BamDataset(alignment_file, *args, **kwargs)
        return dataset

    def create_loader(self, alignment_file, bed_file=None, *args, **kwargs):
        dataset = self.create_dataset(alignment_file, bed_file, *args, **kwargs)
        loader = DataLoader(dataset, shuffle=False)
        return loader

        
        
    