# generate nodel elimination results
import networkx as nx
import pandas as pd
from itertools import combinations
from collections import deque
import matplotlib.pyplot as plt
import os


def extract_graph(graph_id:int, edges_info:pd.DataFrame)->nx.Graph:
    # extract all pairwise nodes info from one graph
    egdes = edges_info[edges_info.graph_id == graph_id].iloc[:,1:].values.tolist()
    # adjust input format for networkx parsing
    egdes = [f'{i} {j}' for i,j in egdes]
    graph = nx.parse_adjlist(egdes)
    return graph

def extract_order(graph_id:int, nodes_info: pd.DataFrame) -> list:
    nodes = nodes_info[nodes_info.graph_id == graph_id].iloc[:,1:]
    nodes = nodes.sort_values(by=['node_label'])
    return nodes.node_id.values.tolist()


class EliminationEngine:
    def __init__(self, graph:nx.Graph, recordON=True) -> None:
        self.graph = graph
        self.graph_clone = graph.copy()
        self.fill_in_count = 0
        self.recordON = recordON
        self.fill_in_edges = []
        self.cur_fill_in = []

    def _add_fill_in(self, target_node):
        neighbors = self.graph.neighbors(target_node)
        potential_edges = combinations(neighbors,2)
        for edge in potential_edges:
            if edge not in self.graph.edges:
                self.graph.add_edge(edge[0], edge[1])
                self.fill_in_edges.append((edge[0], edge[1]))
                self.cur_fill_in.append((edge[0], edge[1]))
                if self.recordON:
                    self.fill_in_count += 1
                    self.graph.add_edge(edge[0], edge[1])
                              
    def eliminate(self, target_node):
        self._add_fill_in(target_node)
        self.graph.remove_node(target_node)


class EliminatorS(EliminationEngine):
    def __init__(self, graph_id, graph: nx.Graph, eliminate_seq:list, taskname, type,visualize=True, recordON=True) -> None:
        super().__init__(graph, recordON)
        self.nodes = deque(eliminate_seq)
        self.visualize = visualize
        # father id: previous graph id 
        self.father_id = graph_id-1
        # child id: the index of the eliminated graph from the father graph
        self.child_id = 0
        self.taskname = taskname
        self.type = type
    

    def _visualize(self, target_node, record=False):
        plt.figure()
        plt.title(f'Graph ID: {self.father_id}    Step:{self.child_id}\neliminating {target_node}')
        
        pos = nx.circular_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, font_color='white', font_size=14)
        # edge_labels = nx.get_edge_attributes(self.graph,'weight')
        # nx.draw_networkx_edge_labels(self.graph, pos, edge_labels = edge_labels,  font_color="tab:green")
        # node_attr = nx.get_node_attributes(self.graph, 'unvisited')
        # offset_pos = {key:pos[key]+0.08 for key in pos.keys()}
        # nx.draw_networkx_labels(self.graph, offset_pos, labels = node_attr, font_color="tab:red")

        # check the existance of the nodes
        nodes = list(self.graph.nodes)
        new_fill_in_list = []
        for node1, node2 in self.cur_fill_in:
            if node1 in nodes and node2 in nodes:
                new_fill_in_list.append((node1, node2))

        self.cur_fill_in = new_fill_in_list
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.cur_fill_in, edge_color='red',width=2.5)
        

        if record:
            plt.savefig(f'./results/{self.taskname}/{self.father_id}/{self.type}/{self.father_id}_{self.child_id}.png')

        # plt.show()
        



    
    def step(self, isRecord=True):
        if self.nodes:
            target_node = str(self.nodes.popleft())
            

            if self.visualize:
                # plt.figure()
                # plt.title(f'Graph ID: {self.father_id}{self.child_id}\neliminating {target_node}')
                # nx.draw(self.graph, with_labels=True)
                self._visualize(target_node, isRecord)
            self.eliminate(target_node)
            self.child_id += 1
        
        elif self.recordON:
            summary = self.summary()
            if isRecord:
                with open(f'./results/{self.taskname}/{self.father_id}/{self.type}/summary.txt', 'a+') as f:
                    f.write(summary)
        else:
            # print("finished")
            pass
      
    def _record(self,edge_save_path, graph_save_path, target_node):
        # save the edge info of the current elimination results
        cur_id = f'{self.father_id}_{self.child_id}'
        cur_graph = self.graph.copy()

        # reset the graph label, reset index starting from 0, detach from the father graph
        rename_label = {nodes: index for index, nodes in enumerate(cur_graph.nodes)}
        nx.relabel_nodes(cur_graph, rename_label,copy=False)

        # save edge_info
        edges = list(cur_graph.edges)
        raw = []
        for source_id, destination_id in edges:
            raw.append({"graph_id":cur_id, 'source_node_id':source_id, 'destination_node_id': destination_id})
        edges_pd = pd.DataFrame(raw)
        if not os.path.exists(edge_save_path):
            edges_pd.to_csv(edge_save_path,mode='w+',header=True, index=False)
        else:
            edges_pd.to_csv(edge_save_path, mode='a+',header=False, index=False)

        # save node info
        raw = {'graph_id':cur_id, 'graph_label':rename_label[str(target_node)]}
        graph_df = pd.DataFrame([raw])
        if not os.path.exists(graph_save_path):
            graph_df.to_csv(graph_save_path,mode='w+',header=True, index=False)
        else:
            graph_df.to_csv(graph_save_path, mode='a+',header=False, index=False)


    def auto_step(self,  record=False):
        while self.nodes:
            target_node = str(self.nodes.popleft())
            # self._record(edge_save_path, graph_save_path, target_node)
            self.child_id += 1
            if self.visualize:
                # plt.figure()
                # plt.title(f'Graph ID: {self.father_id}{self.child_id}\neliminating {target_node}')
                # nx.draw(self.graph, with_labels=True)
                self._visualize(target_node, record)
                
            self.eliminate(target_node)
        if self.recordON:
            self.summary()
        with open(f'./results/{self.taskname}/{self.father_id}/{self.type}/summary.txt', 'a+') as f:
                f.write(self.summary())
    
    def summary(self):
        # print(f'Global Graph ID: {self.father_id}\nInput Node Size: {len(self.graph_clone.nodes)}\nTotal fill-in: {self.fill_in_count}')

        # draw the overall fill-in graph
        pos = nx.circular_layout(self.graph_clone)
        plt.figure()
        nx.draw(self.graph_clone, pos, with_labels=True,  font_color='white', font_size=14)
        nx.draw_networkx_edges(self.graph_clone, pos, edgelist=self.fill_in_edges, edge_color='red', width=3)
        plt.savefig(f'./results/{self.taskname}/{self.father_id}/{self.type}/{self.father_id}_overall.png')

        # plt.show()
        


        return f'Global Graph ID: {self.father_id}\nInput Node Size: {len(self.graph_clone.nodes)}\nTotal fill-in: {self.fill_in_count}'