import pysam
import torch
import random
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Tuple, Any, Generator


class BaseBamIterator(ABC):
    """Базовый класс для всех итераторов BAM файлов"""
    
    def __init__(
        self,
        pysam_path,
        chromosome,
        window_size,
        start=0,
        stop=None,
        device="cpu",
    ):
        self.al_file = pysam.AlignmentFile(pysam_path)
        self.chromosome = chromosome
        self.window_size = window_size
        self.__initial_start = start
        self.__stop = (
            stop
            if stop is not None
            else dict(zip(self.al_file.references, self.al_file.lengths))[chromosome]
        )
        self.device = device

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _fetch_records(self, start, stop):
        """Fetch records from BAM file for given interval"""
        if start >= self.__stop:
            return []
        return list(self.al_file.fetch(self.chromosome, start, stop))
    
    def _process_records(self, records):
        """Process records and extract paired reads"""
        paired_records = self._collect_paired_reads(records)
        return {
            "start": torch.tensor(
                list(map(lambda x: x[0].reference_start, paired_records))
            ).view(-1, 1),
            "end": torch.tensor(
                list(map(lambda x: x[1].reference_end, paired_records))
            ).view(-1, 1),
            "id": list(map(lambda x: x[0].qname, paired_records)),
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

    @property
    def initial_start(self):
        return self.__initial_start

    @property
    def stop(self):
        return self.__stop

    def __del__(self):
        if hasattr(self, 'al_file'):
            self.al_file.close()

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= len(self):
            raise StopIteration
        result = self.__getitem__(self._iter_index)
        self._iter_index += 1
        return result


class BamFileIterator(BaseBamIterator):
    """Итератор для последовательного обхода хромосомы с фиксированным шагом"""
    
    def __init__(
        self,
        pysam_path,
        chromosome,
        window_size=4000,
        start=0,
        stop=None,
        step=3500,
        device="cpu",
    ):
        super().__init__(pysam_path, chromosome, window_size, start, stop, device)
        self.step = step

    def __len__(self):
        return (self.stop - self.initial_start) // self.step

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        cur_start = self.initial_start + idx * self.step
        cur_stop = min(cur_start + self.window_size, self.stop)
        
        if cur_start >= self.stop:
            return {
                "start": torch.tensor([]).view(-1, 1),
                "end": torch.tensor([]).view(-1, 1),
                "id": [],
            }
        
        records = self._fetch_records(cur_start, cur_stop)
        return self._process_records(records)


class RandomBamFileIterator(BaseBamIterator):
    """Итератор для случайной выборки окон заданного размера"""
    
    def __init__(
        self,
        pysam_path,
        chromosome,
        nwindows,
        window_size=5000,
        start=0,
        stop=None,
        step=3500,
        device="cpu",
        seed=None
    ):
        super().__init__(pysam_path, chromosome, window_size, start, stop, device)
        self.nwindows = nwindows
        self.step = step
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
        
        # Pre-generate random windows
        self.random_windows = self._generate_random_windows()

    def __len__(self):
        return self.nwindows

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.nwindows:
            raise IndexError(f"Index {idx} out of range [0, {self.nwindows})")
        
        cur_start, cur_stop = self.random_windows[idx]
        
        if cur_start >= self.stop:
            return {
                "start": torch.tensor([]).view(-1, 1),
                "end": torch.tensor([]).view(-1, 1),
                "id": [],
            }
        
        records = self._fetch_records(cur_start, cur_stop)
        return self._process_records(records)

    def _generate_random_windows(self):
        """Generate n random windows within the chromosome range"""
        random_windows = []
        
        # Calculate maximum possible start position
        max_start = self.stop - self.window_size
        
        if max_start < self.initial_start:
            # Chromosome is shorter than window size
            return [(self.initial_start, self.stop)]
        
        # Generate random windows
        possible_starts = range(self.initial_start, max_start + 1)
        
        # If we need more windows than possible, use with replacement
        if self.nwindows > len(possible_starts):
            # Sample with replacement
            for _ in range(self.nwindows):
                cur_start = random.randint(self.initial_start, max_start)
                cur_stop = cur_start + self.window_size
                random_windows.append((cur_start, cur_stop))
        else:
            # Sample without replacement for uniqueness
            sampled_starts = random.sample(possible_starts, self.nwindows)
            for cur_start in sampled_starts:
                cur_stop = cur_start + self.window_size
                random_windows.append((cur_start, cur_stop))
        
        return random_windows


class WeightedRandomChromosomeIterator(BaseBamIterator):
    """Итератор для случайной выборки хромосом пропорционально количеству ридов"""
    
    def __init__(
        self,
        pysam_path: str,
        chromosome: Optional[Union[str, List[str], tuple]] = None,  
        window_size: int = 5000,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 3500,
        device: str = "cpu",
        min_reads_per_chromosome: int = 0,
        n_windows_per_chromosome: int = 1,
        n_windows_total: Optional[int] = None,
        random_seed: Optional[int] = None
    ):
        """
        Args:
            chromosome: список хромосом для итерации 
                       (если None - все хромосомы, если строка - одна хромосома)
            min_reads_per_chromosome: минимальное количество ридов на хромосоме
            n_windows_per_chromosome: количество окон на одну хромосому
            n_windows_total: общее количество окон
            random_seed: seed для воспроизводимости результатов
        """
        self.pysam_path = pysam_path
        # Устанавливаем seed для воспроизводимости
        if random_seed is not None:
            random.seed(random_seed)
        
        # Обработка параметра chromosome
        if chromosome is None:
            self.allowed_chromosomes = None
            temp_chromosome = None
        elif isinstance(chromosome, str):
            self.allowed_chromosomes = [chromosome]
            temp_chromosome = chromosome
        elif isinstance(chromosome, (list, tuple)):
            self.allowed_chromosomes = list(chromosome)
            temp_chromosome = self.allowed_chromosomes[0] if self.allowed_chromosomes else None
        else:
            raise TypeError(f"chromosome must be str, list, tuple or None, got {type(chromosome)}")
        
        # Получаем временную хромосому для инициализации родительского класса
        if temp_chromosome is None:
            with pysam.AlignmentFile(pysam_path, "rb") as tmp_file:
                temp_chromosome = tmp_file.references[0] if tmp_file.references else "chr1"
        
        super().__init__(pysam_path, temp_chromosome, window_size, start, stop, device)
        
        self.step = step
        self.n_windows_total = n_windows_total
        self.n_windows_per_chromosome = n_windows_per_chromosome
        self.min_reads_per_chromosome = min_reads_per_chromosome
        
        # Кэш для длин хромосом
        self._chromosome_lengths: Dict[str, int] = {}
        
        # Подсчет ридов по хромосомам
        self.chromosome_read_counts = self._count_reads_per_chromosome()
        if not self.chromosome_read_counts:
            raise ValueError(
                f"No chromosomes found matching criteria. "
                f"Allowed: {self.allowed_chromosomes}, "
                f"min_reads: {min_reads_per_chromosome}"
            )
        
        self.chromosomes = list(self.chromosome_read_counts.keys())
        self.weights = list(self.chromosome_read_counts.values())
        total_reads = sum(self.weights)
        self.probabilities = [w / total_reads for w in self.weights]
        
        # Предварительная генерация окон (только если указано общее количество)
        if n_windows_total is not None:
            self._windows = self._precompute_windows()
        else:
            self._windows = None
    
    def _count_reads_per_chromosome(self) -> Dict[str, int]:
        """Подсчет количества ридов на каждой хромосоме с использованием count()"""
        read_counts = {}
        with pysam.AlignmentFile(self.pysam_path, "rb") as bamfile:
            all_chromosomes = bamfile.references
            
            # Определяем, какие хромосомы проверять
            if self.allowed_chromosomes is not None:
                allowed_set = set(self.allowed_chromosomes)
                chromosomes_to_check = [chrom for chrom in all_chromosomes if chrom in allowed_set]
                missing_chroms = allowed_set - set(all_chromosomes)
                if missing_chroms:
                    print(f"Warning: Chromosomes not found in BAM file: {missing_chroms}")
            else:
                chromosomes_to_check = all_chromosomes
            
            # Подсчет ридов для каждой хромосомы
            for chrom in chromosomes_to_check:
                chrom_idx = bamfile.references.index(chrom)
                chrom_length = bamfile.lengths[chrom_idx]
                actual_stop = self.stop if self.stop is not None else chrom_length
                actual_start = min(self.initial_start, actual_stop)
                
                # Кэшируем длину хромосомы
                self._chromosome_lengths[chrom] = chrom_length
                
                # Подсчет ридов
                if actual_start == 0 and actual_stop >= chrom_length:
                    count = bamfile.count(contig=chrom, until_eof=False)
                else:
                    count = bamfile.count(contig=chrom, start=actual_start, stop=actual_stop, until_eof=False)
                
                if count >= self.min_reads_per_chromosome:
                    read_counts[chrom] = count
        
        return dict(sorted(read_counts.items()))
    
    def _get_random_chromosome(self) -> str:
        """Выбор случайной хромосомы согласно распределению"""
        if not self.chromosomes:
            raise RuntimeError("No chromosomes available for selection")
        return random.choices(self.chromosomes, weights=self.weights, k=1)[0]
    
    def _get_chromosome_stop(self, chromosome: str) -> int:
        """Получить длину хромосомы из кэша или BAM файла"""
        if chromosome not in self._chromosome_lengths:
            with pysam.AlignmentFile(self.pysam_path, "rb") as bamfile:
                chrom_idx = bamfile.references.index(chromosome)
                self._chromosome_lengths[chromosome] = bamfile.lengths[chrom_idx]
        return self._chromosome_lengths[chromosome]
    
    def _precompute_windows(self) -> List[Tuple[str, Any]]:
        """Предварительная генерация всех окон"""
        windows = []
        
        for _ in range(self.n_windows_total):
            chromosome = self._get_random_chromosome()
            
            random_iter = RandomBamFileIterator(
                self.pysam_path,
                chromosome,
                nwindows=self.n_windows_per_chromosome,
                window_size=self.window_size,
                start=self.initial_start,
                stop=self._get_chromosome_stop(chromosome),
                step=self.step,
                device=self.device,
            )
            
            # Добавляем все окна для этой хромосомы
            for i in range(len(random_iter)):
                item = random_iter[i]
                # Добавляем информацию о хромосоме в каждый элемент
                if isinstance(item, dict):
                    item['chromosome'] = chromosome
                windows.append(item)
        
        return windows
    
    def _generate_empty_window(self, chromosome: str) -> Tuple[str, Dict[str, Any]]:
        """Создает пустое окно для хромосомы без ридов"""
        return {
            "start": torch.tensor([]).view(-1, 1),
            "end": torch.tensor([]).view(-1, 1),
            "id": [],
            "chromosome": chromosome,
        }
    
    def __len__(self) -> int:
        """Возвращает длину итератора"""
        if self.n_windows_total is not None:
            return len(self._windows) if self._windows is not None else 0
        else:
            # Для бесконечного итератора возвращаем большое число
            # или можно вызвать исключение
            return 10**9  # Практически бесконечность
    
    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        """Получение окна по индексу"""
        # Режим с предварительной генерацией
        if self.n_windows_total is not None:
            if self._windows is None:
                raise RuntimeError("Windows not precomputed")
            if idx < 0 or idx >= len(self._windows):
                raise IndexError(f"Index {idx} out of range [0, {len(self._windows)})")
            return self._windows[idx]
        
        # Режим генерации на лету
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        
        chromosome = self._get_random_chromosome()
        random_iter = RandomBamFileIterator(
            self.pysam_path,
            chromosome,
            nwindows=self.n_windows_per_chromosome,
            window_size=self.window_size,
            start=self.initial_start,
            stop=self._get_chromosome_stop(chromosome),
            step=self.step,
            device=self.device,
        )
        
        # Если есть окна, возвращаем случайное
        if len(random_iter) > 0:
            window_idx = idx % len(random_iter)
            item = random_iter[window_idx]
            # Унифицируем формат возврата
            if isinstance(item, dict):
                item['chromosome'] = chromosome
            return item
        else:
            # Возвращаем пустое окно в том же формате
            return self._generate_empty_window(chromosome)
    
    def generate_windows(self) -> Generator[Tuple[str, Any], None, None]:
        """Генератор для итерации без предварительного вычисления всех окон"""
        if self.n_windows_total is None:
            # Бесконечная генерация
            while True:
                chromosome = self._get_random_chromosome()
                random_iter = RandomBamFileIterator(
                    self.pysam_path,
                    chromosome,
                    nwindows=self.n_windows_per_chromosome,
                    window_size=self.window_size,
                    start=self.initial_start,
                    stop=self._get_chromosome_stop(chromosome),
                    step=self.step,
                    device=self.device,
                )
                for i in range(len(random_iter)):
                    item = random_iter[i]
                    if isinstance(item, dict):
                        item['chromosome'] = chromosome
                    yield item
        else:
            # Генерация заданного количества окон
            for _ in range(self.n_windows_total):
                chromosome = self._get_random_chromosome()
                random_iter = RandomBamFileIterator(
                    self.pysam_path,
                    chromosome,
                    nwindows=self.n_windows_per_chromosome,
                    window_size=self.window_size,
                    start=self.initial_start,
                    stop=self._get_chromosome_stop(chromosome),
                    step=self.step,
                    device=self.device,
                )
                for i in range(len(random_iter)):
                    item = random_iter[i]
                    if isinstance(item, dict):
                        item['chromosome'] = chromosome
                    yield item
    
    def get_chromosome_probabilities(self) -> Dict[str, float]:
        """Возвращает вероятности выбора каждой хромосомы"""
        return dict(zip(self.chromosomes, self.probabilities))
    
    def get_chromosome_read_counts(self) -> Dict[str, int]:
        """Возвращает количество ридов на каждой хромосоме"""
        return self.chromosome_read_counts.copy()
    
    def get_total_reads(self) -> int:
        """Возвращает общее количество ридов"""
        return sum(self.weights)
    
    def get_allowed_chromosomes(self) -> Optional[List[str]]:
        """Возвращает список разрешенных хромосом"""
        return self.allowed_chromosomes.copy() if self.allowed_chromosomes else None


class BamLoader:
    """Загрузчик BAM файлов"""
    
    def __init__(self, pysam_path):
        self.pysam_path = pysam_path

    def get_chromosomes(self):
        file = pysam.AlignmentFile(self.pysam_path)
        refs = file.references
        file.close()
        return refs

    def iter_chromosome(
        self, 
        chromo: str, 
        window_size=5000, 
        start=0, 
        stop=None, 
        step=3500, 
        device='cpu'
    ):
        if chromo not in self.get_chromosomes():
            raise ValueError(f"chromosome {chromo} not in file")
        return BamFileIterator(
            self.pysam_path,
            chromo,
            window_size,
            start,
            stop,
            step,
            device
        )

    def iter_randomwindows(
        self, 
        chromo, 
        nwindows, 
        window_size=5000, 
        start=0, 
        stop=None, 
        step=3500, 
        device='cpu', 
        seed=None
    ):
        if chromo not in self.get_chromosomes():
            raise ValueError(f"chromosome {chromo} not in file")
        return RandomBamFileIterator(
            self.pysam_path,
            chromo,
            nwindows,
            window_size,
            start,
            stop,
            step,
            device,
            seed
        )

    def iter_random_chromosomes(
        self,
        chromosome=None,  
        window_size=5000,
        start=0,
        stop=None,
        step=3500,
        device="cpu",
        seed=None,
        min_reads_per_chromosome=0,
        n_windows_per_chromosome=1,
        n_windows_total=None):
        return WeightedRandomChromosomeIterator(
            self.pysam_path,
        chromosome,
         window_size,
          start,
           stop,
            step,
             device,
               min_reads_per_chromosome,
               n_windows_per_chromosome,
               n_windows_total
               )
