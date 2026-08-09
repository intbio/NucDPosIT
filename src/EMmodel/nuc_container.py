import numpy as np
import networkx as nx
import warnings


# from fiber_graph import FiberGraph


class NucleosomeContainer:
    pass


class Nucleosome:
    def __init__(self, dyad, right, left, count=1):
        self.dyad = self._validate_dyad(dyad)
        self.right, self.left = self._validate_cords(left, right)
        self.count = count

    def _validate_dyad(self, dyad):
        return dyad

    def _validate_cords(self, left, right):
        return left, right

    def __repr__(self):
        return f"Nucleosome(dyad={self.dyad}, left={self.left}, right={self.right})"

    def __mul__(self, other):
        return NucleosomeContainer.from_list(
            [Nucleosome(self.dyad, self.right, self.left) for i in range(other)]
        )

    def __add__(self, other):
        container = NucleosomeContainer()
        container.append(self)
        container.append(other)
        return container

    def __len__(self):
        return self.right - self.left + 1


class CanonicNucleosome(Nucleosome):
    def __init__(self, dyad, count=1):
        super().__init__(dyad, dyad - 73, dyad + 73, count=count)


class NucleosomeContainer:
    def __init__(self):
        self._nucleosomes = []
        self._by_position = {}

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self._nucleosomes)}"

    def __len__(self):
        return len(self.nucleosomes)

    @staticmethod
    def from_df(df, start="start", end="end", dyad="dyad", weights:list=None):
        weights = weights if weights is not None else np.ones(len(df))
        container = NucleosomeContainer()
        for i, row in df.iterrows():
            cur_start, cur_end, cur_dyad, cur_weight = (
                int(row[start]),
                int(row[end]),
                int(row[dyad]),
                weights[i]
            )
            new_nucleosome = Nucleosome(cur_dyad, cur_start, cur_end, cur_weight)
            container.append(new_nucleosome)
        return container

    @staticmethod
    def from_dyads(dyads: list, weights:list=None):
        weights = weights if weights is not None else np.ones_like(dyads)
        container = NucleosomeContainer()
        for i, dyad in enumerate(dyads):
            cur_weight = weights[i]
            new_nuc = CanonicNucleosome(dyad, count=cur_weight)
            container.append(new_nuc)
        return container

    def __getitem__(self, *args, **kwargs):
        return self._nucleosomes.__getitem__(*args, **kwargs)

    @staticmethod
    def from_list(nucleosomes: list):
        container = NucleosomeContainer()
        container.extend(*nucleosomes)
        return container

    def get_cords(self):
        cords = [(nuc.left, nuc.right) for nuc in self.iter_nucleosomes()]
        return np.array(cords)

    @property
    def nucleosomes(self):
        return self._nucleosomes

    def __add__(self, other):
        if isinstance(other, Nucleosome):
            self.append(other)
            return self
        if isinstance(other, NucleosomeContainer):
            self.extend(*other.nucleosomes)
            return self

    def append(self, nucleosome):
        if not isinstance(nucleosome, Nucleosome):
            raise TypeError(
                f"Only Nucleosome objects can be added. Got {nucleosome} og type {type(nucleosome)}"
            )
        self._nucleosomes.append(nucleosome)
        for pos in range(nucleosome.left, nucleosome.right + 1):
            if pos not in self._by_position:
                self._by_position[pos] = []
            self._by_position[pos].append(nucleosome)

    def extend(self, *args):
        for item in args:
            self.append(item)

    def iter_nucleosomes(self):
        return iter(self._nucleosomes)

    def get_at_position(self, position):
        """Получить все нуклеосомы, покрывающие позицию"""
        return self._by_position.get(position, [])

    def get_by_dyad(self, dyad):
        """Найти нуклеосому по позиции dyad"""
        for nuc in self._nucleosomes:
            if nuc.dyad == dyad:
                return nuc
        return None

    def get_in_range(self, start, end):
        """Получить все нуклеосомы, пересекающиеся с диапазоном"""
        result = []
        for nuc in self._nucleosomes:
            if nuc.dyad <= end and nuc.dyad >= start:
                result.append(nuc)
        return result

    def remove(self, nucleosome):
        """Удалить нуклеосому"""
        if nucleosome in self._nucleosomes:
            self._nucleosomes.remove(nucleosome)
            self._rebuild_index()

    def _rebuild_index(self):
        """Перестроить индекс позиций"""
        self._by_position = {}
        for nuc in self._nucleosomes:
            for pos in range(nuc.left, nuc.right + 1):
                if pos not in self._by_position:
                    self._by_position[pos] = []
                self._by_position[pos].append(nuc)
    
    def connectivity_graph(self, min_dist=140, max_dist=300):
        assert min_dist < max_dist
        G = nx.DiGraph()
        
        # Получаем все нуклеосомы и сортируем по координате
        all_nucleosomes = sorted(self.iter_nucleosomes(), key=lambda n: n.dyad)
        
        for i, cur_nuc in enumerate(all_nucleosomes):
            if cur_nuc.dyad not in G:
                G.add_node(cur_nuc.dyad)
            
            # Ищем соседей в диапазоне
            neighbors = self.get_in_range(cur_nuc.dyad + min_dist, cur_nuc.dyad + max_dist)     

            # if not neighbors:
            #     blacklist.add(cur_nuc.dyad)
                # if i + 1 < len(all_nucleosomes):
                #     nearest_right = self.get_in_range(cur_nuc.dyad + 1, cur_nuc.dyad + min_dist)
                #     if nearest_right:
                #         for close_right in nearest_right:
                #             if close_right.dyad not in G:
                #                 G.add_edge(close_right.dyad)
                #             G.add_edge(cur_nuc.dyad, close_right.dyad)
                #             warnings.warn(f"{cur_nuc} has no neighbors in range, connected to right {close_right}")
                #     else:
                #         warnings.warn(f"{cur_nuc} has no neighbors")
                # continue
            
            for neighbor in neighbors:
                if neighbor == cur_nuc:
                    continue
                if neighbor.dyad not in G:
                    G.add_node(neighbor.dyad)
                G.add_edge(cur_nuc.dyad, neighbor.dyad)
        roots = [node for node in G.nodes() if G.in_degree(node) == 0 ]
        leaves = [node for node in G.nodes() if G.out_degree(node) == 0]
        return G, roots, leaves

    def connectivity_cyclic_graph(self, total_len, min_dist=140, max_dist=250):
        G, roots, leaves = self.connectivity_graph(min_dist, max_dist)
        for root_pos in roots:
            for leaf_pos in leaves:
                parents = list(G.predecessors(leaf_pos))
                parents.append(leaf_pos)
                for link_node in parents:
                    cur_dist = total_len - link_node + root_pos
                    if cur_dist >= min_dist and cur_dist <= max_dist:
                        G.add_edge(link_node, root_pos)
        return G, roots, leaves
                    

        
        



