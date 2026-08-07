import copy
import networkx as nx
import numpy as np


class FiberGraph(nx.DiGraph):
    def __init__(self, dist: int=None, scoring_matrix=None):
        self.dist = dist
        self.scoring_matrix = scoring_matrix

    def check_parameters(self, func):
        def wrapper(*args, **kwargs):
            assert self.dist is not None, f"dist is None"
            assert self.scoring_matrix is not None, f"scoring_matrix is None"
            return func(*args, **kwargs)
        return wrapper
            

    def score_linkers(self, linkers):
        return self.scoring_matrix[*linkers]

    def find_paths_length(self, source):
        paths = []
        
        distances = nx.single_source_shortest_path_length(self, source, cutoff=self.dist)
        target_nodes = [node for node, dist in distances.items() 
                        if dist == dist and node != source]
        
        for target in target_nodes:
            all_paths = nx.all_simple_paths(self, source, target, cutoff=self.dist)
            for path in all_paths:
                if len(path) == self.dist + 1:
                    paths.append(path)
                    
        return paths

    def contract_graph(self, dist, scoring_matrix):
        newG = FiberGraph(self.dist, self.scoring_matrix)
        distance = self.dist
        node_hashes = {}
    
        for start_node in self.nodes():
            all_paths = self.find_paths_length(G, start_node)
            for path in all_paths:
                path_tuple = tuple(path)
                if path_tuple not in newG:
                    linkers = np.diff(path_tuple) - 145
                    if ((linkers >= 100) | (linkers < 0)).any():
                        weight = 0
                    else:
                        weight = self.score_linkers(linkers)
                    newG.add_node(path_tuple, weight=weight)
                
                # Индексируем пути по их префиксам и суффиксам
                prefix = path_tuple[:-1]  # первые 5 вершин
                suffix = path_tuple[1:]   # последние 5 вершин
                
                if prefix not in node_hashes:
                    node_hashes[prefix] = []
                node_hashes[prefix].append(path_tuple)
        
        # Теперь добавляем ребра между перекрывающимися путями
        for path_tuple in list(newG.nodes()):
            suffix = path_tuple[1:]  # путь без первой вершины
            if suffix in node_hashes:
                for next_path in node_hashes[suffix]:
                    if path_tuple != next_path:  # избегаем петель
                        newG.add_edge(path_tuple, next_path)
        
        isolated_nodes = list(nx.isolates(newG))
        newG.remove_nodes_from(isolated_nodes)
        return newG
