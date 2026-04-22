from abc import ABC, abstractmethod
import pysam
from collections import defaultdict
import torch


class BaseIterator(ABC):

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @staticmethod
    def get_chromosomes(bam_path):
        file = pysam.AlignmentFile(bam_path, "rb")
        refs = file.references
        file.close()
        return refs


class BamChromosomeIterator(BaseIterator):
    def __init__(
        self, bam_path, chromosome, start=0, stop=None, step=5000, skip_empty=False
    ):
        self.bam_path = bam_path
        self.chromosome = chromosome
        self.__start = int(start)
        self.__stop = (
            int(stop)
            if stop
            else int(self.get_chromosome_lengths(bam_path)[chromosome])
        )
        self.__step = int(step)
        self.skip_empty = skip_empty

    @property
    def start(self):
        return self.__start

    @property
    def stop(self):
        return self.__stop

    @property
    def step(self):
        return self.__step

    def get_chromosome_lengths(self, bam_path):
        with pysam.AlignmentFile(bam_path) as bam:
            return dict(zip(bam.references, bam.lengths))

    def fetch_records(self, start, stop):
        with pysam.AlignmentFile(self.bam_path, "rb") as bamfile:
            records = list(
                bamfile.fetch(self.chromosome, start, stop)
            )  # Convert to list
        processed_records = self._process_records(records)
        return processed_records

    def _process_records(self, records):
        """Process records and extract paired reads"""
        paired_records = self._collect_paired_reads(records)

        if not paired_records:
            return {
                "start": torch.tensor([]).view(-1, 1),
                "end": torch.tensor([]).view(-1, 1),
                "id": [],
                "chromosome": self.chromosome,
            }

        return {
            "start": torch.tensor(
                list(map(lambda x: x[0].reference_start, paired_records))
            ).view(-1, 1),
            "end": torch.tensor(
                list(map(lambda x: x[1].reference_end, paired_records))
            ).view(-1, 1),
            "id": list(
                map(lambda x: x[0].query_name, paired_records)
            ),  # Fixed: qname -> query_name
            "chromosome": self.chromosome,
        }

    def _collect_paired_reads(self, records):
        """Collect valid paired reads from records"""
        read_pairs = defaultdict(dict)
        valid_pairs = []
        for read in records:
            name = read.query_name
            if read.is_read1:
                read_pairs[name]["R1"] = read
            elif read.is_read2:
                read_pairs[name]["R2"] = read
        for name, pair in read_pairs.items():
            if "R1" in pair and "R2" in pair:
                valid_pairs.append((pair["R1"], pair["R2"]))
        return valid_pairs

    def __iter__(self):
        self.cur_start = self.start
        self.cur_stop = min(
            self.cur_start + self.step, self.stop
        )  # Initialize properly
        return self

    def __next__(self):
        if self.cur_start >= self.stop:
            raise StopIteration

        records = self.fetch_records(self.cur_start, self.cur_stop)

        self.cur_start = self.cur_stop
        self.cur_stop = min(self.cur_stop + self.step, self.stop)

        if self.skip_empty and len(records["start"]) == 0:
            return self.__next__()
        return records

    def __len__(self):
        return (
            (self.stop - self.start) + self.step - 1
        ) // self.step  


class BedRegionsIterator(BaseIterator):
    def __init__(self, bed_path, bam_path, step=5000, skip_empty=False):
        self.bed_path = bed_path
        self.bam_path = bam_path
        self.__step = int(step)
        self.skip_empty = skip_empty

    @property
    def step(self):
        return self.__step

    def _iter_bed(self, bam_path, bed_path):
        with open(bed_path, "r") as bed_file:
            for i, line in enumerate(bed_file):
                if line.startswith(("#", "track", "browser")):
                    continue
                splitline = line.split()
                chromosome, start, end = splitline[0], splitline[1], splitline[2]
                bam_iterator = BamChromosomeIterator(
                    bam_path, chromosome, start=start, stop=end, step=self.step
                )
                for item in bam_iterator:
                    item['region'] = i
                    item['win_start'] = int(start)
                    item['win_end'] = int(end)
                    yield item

    def __len__(self):
        c = 0
        with open(self.bed_path) as file:
            for line in file:
                c += 1
        return c

    def __iter__(self):
        self._iterator = self._iter_bed(self.bam_path, self.bed_path)
        return self

    def __next__(self):
        for item in self._iterator:
            if self.skip_empty and len(item['end']) == 0:
                continue
            return item