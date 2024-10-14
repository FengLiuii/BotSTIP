import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from torch_geometric.nn import HypergraphConv  
from torch_geometric.data import Data
from tqdm import tqdm
import community as community_louvain
import networkx as nx
from networkx.algorithms.community import girvan_newman,asyn_fluidc

class HyperGraphStructuralLayer_sample(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        
        self.layer1 = HypergraphConv(hidden_dim, n_heads* hidden_dim // n_heads, dropout=dropout)
        self.layer2 = HypergraphConv(hidden_dim, n_heads*hidden_dim // n_heads, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
       
        out1 = self.layer1(x, edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()



class HyperGraphStructuralLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        
      
        self.layer1 = HypergraphConv(hidden_dim, n_heads* hidden_dim // n_heads, dropout=dropout)
        self.layer2 = HypergraphConv(hidden_dim, n_heads*hidden_dim // n_heads, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        
        hyper_edge_index = self.build_hypergraph_from_graph(data, 3)
        
        hyper_edge_index = hyper_edge_index.to(x.device)
        out1 = self.layer1(x, hyper_edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, hyper_edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

                    
    def build_hypergraph_from_graph(self, data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data.x.cpu().detach().numpy())

        hyperedges = []
        for community in range(n_clusters):
            community_nodes = (labels == community).nonzero()  
            if community_nodes[0].size > 0:
                hyperedges.append(torch.tensor(community_nodes[0], dtype=torch.long))
        
        edge_list = []
        for edge in hyperedges:
            
            for i in range(len(edge)):
                for j in range(i + 1, len(edge)):
                    edge_list.append([edge[i], edge[j]])

       
        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()
        return hyperedges
    



class HyperGraphStructuralLayer_Knn(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer_Knn, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Using HypergraphConv
        self.layer1 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.layer2 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        
        hyper_edge_index = self.build_hypergraph_from_graph(data, k=5)  
        hyper_edge_index = hyper_edge_index.to(x.device)
        out1 = self.layer1(x, hyper_edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, hyper_edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def build_hypergraph_from_graph(self, data, k):
        
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data.x.cpu().detach().numpy())
        distances, indices = knn.kneighbors(data.x.cpu().detach().numpy())

        edge_list = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # Skip self-loop (first neighbor is itself)
                edge_list.append([i, neighbor])

        #
        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()
        return hyperedges




class HyperGraphStructuralLayer_Knn_and_Kmeans(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer_Knn_and_Kmeans, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

       
        self.layer1 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.layer2 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        
        hyper_edge_index = self.build_hypergraph_from_graph(data, n_clusters=5, k=5)  
        hyper_edge_index = hyper_edge_index.to(x.device)
        out1 = self.layer1(x, hyper_edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, hyper_edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def build_hypergraph_from_graph(self, data, n_clusters, k):
        # Step 1: Apply KMeans to get clusters
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data.x.cpu().detach().numpy())

        edge_list = []
        
        # Step 2: Apply KNN within each cluster
        for cluster_id in range(n_clusters):
            # Get nodes belonging to the current cluster
            cluster_nodes = (labels == cluster_id).nonzero()[0]

            if len(cluster_nodes) > 1:  # Only apply KNN if there are enough nodes
                knn = NearestNeighbors(n_neighbors=min(k, len(cluster_nodes)), algorithm='auto')
                node_features = data.x[cluster_nodes].cpu().detach().numpy()
                knn.fit(node_features)
                
                distances, indices = knn.kneighbors(node_features)
                
                # Build edges between nodes in this cluster based on KNN
                for i, neighbors in enumerate(indices):
                    for neighbor in neighbors[1:]:  # Skip self-loop
                        edge_list.append([cluster_nodes[i].item(), cluster_nodes[neighbor].item()])

        # Convert the list of edges to a tensor
        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()
        return hyperedges





class HyperGraphStructuralLayer_Louvain_and_knn(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer_Louvain_and_knn, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

      
        # Using HGCNetwork with attention mechanism
        self.layer1 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout, use_attention = True, attention_mode = "node")   # "node","EDGE"
        self.layer2 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout, use_attention = True, attention_mode = "node")
        self.init_weights()
        self.layer1 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout, use_attention = True, attention_mode = "node")
        self.layer2 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout, use_attention = True, attention_mode = "node")
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        # Use Louvain for community detection and KNN within each community
        hyper_edge_index = self.build_hypergraph_from_graph(data, k=5)  
        hyper_edge_index = hyper_edge_index.to(x.device)
        out1 = self.layer1(x, hyper_edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, hyper_edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def build_hypergraph_from_graph(self, data, k):
        
        G = nx.Graph()
        edge_list = data.edge_index.t().cpu().numpy().tolist()  
        if len(edge_list) == 0:
            
            num_nodes = data.x.shape[0]
            if num_nodes > 1:
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_list.append([i, j])  
            elif num_nodes == 1:
                edge_list.append([0, 0])  
        G.add_edges_from(edge_list)

        
        partition = community_louvain.best_partition(G)

        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        edge_list = []

       
        for community_nodes in communities.values():
            if len(community_nodes) > 1:  
                knn = NearestNeighbors(n_neighbors=min(k, len(community_nodes)), algorithm='auto')
                node_features = data.x[community_nodes].cpu().detach().numpy()
                knn.fit(node_features)
                
                distances, indices = knn.kneighbors(node_features)
                
                
                for i, neighbors in enumerate(indices):
                    for neighbor in neighbors[1:]:  
                        edge_list.append([community_nodes[i], community_nodes[neighbor]])

       
        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()
        return hyperedges


class HyperGraphStructuralLayer_Louvain(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer_Louvain, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

        
        self.layer1 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.layer2 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        
        hyper_edge_index = self.build_hypergraph_from_graph(data)
        hyper_edge_index = hyper_edge_index.to(x.device)
        out1 = self.layer1(x, hyper_edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, hyper_edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def build_hypergraph_from_graph(self, data):
        
        G = nx.Graph()
        edge_list = data.edge_index.t().cpu().numpy().tolist()  
        if len(edge_list) == 0:
            
            num_nodes = data.x.shape[0]
            if num_nodes > 1:
                
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_list.append([i, j])  
            elif num_nodes == 1:
                edge_list.append([0, 0])  

        G.add_edges_from(edge_list)
       
        partition = community_louvain.best_partition(G)
      
        if not partition:
            
            partition = {node: 0 for node in G.nodes()}
            print(partition)

        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        edge_list = []

        
        for community_id, community_nodes in communities.items():
            
            if len(community_nodes) > 1:  
                for i in range(len(community_nodes)):
                    for j in range(i + 1, len(community_nodes)):
                        edge_list.append([community_nodes[i], community_nodes[j]])
            elif len(community_nodes) == 1:
                pass
            else:
              edge_list.append([community_nodes[0], community_nodes[0]])

        
        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()
        return hyperedges
    




    
class HyperGraphStructuralLayer_GN(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer_GN, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

     
        self.layer1 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.layer2 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        hyper_edge_index = self.build_hypergraph_from_graph(data)
        hyper_edge_index = hyper_edge_index.to(x.device)
        out1 = self.layer1(x, hyper_edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, hyper_edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

  
    def build_hypergraph_from_graph(self, data):
        G = nx.Graph()
        edge_list = data.edge_index.t().cpu().numpy().tolist()
        
        
        if len(edge_list) == 0:
            num_nodes = data.x.shape[0]
            if num_nodes > 1:
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_list.append([i, j])  
            elif num_nodes == 1:
                edge_list.append([0, 0])  
        G.add_edges_from(edge_list)
        
        try:
            communities_generator = girvan_newman(G)
            top_level_communities = next(communities_generator)
            partition = {node: idx for idx, community in enumerate(top_level_communities) for node in community}
        except Exception as e:
            
            partition = {node: 0 for node in G.nodes()}  
        
        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        edge_list = []
        
        for community_id, community_nodes in communities.items():
            if len(community_nodes) > 1:
                for i in range(len(community_nodes)):
                    for j in range(i + 1, len(community_nodes)):
                        edge_list.append([community_nodes[i], community_nodes[j]])

        if len(edge_list) == 0:
            
            num_nodes = data.x.shape[0]
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    edge_list.append([i, j])

        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()

        return hyperedges



class HyperGraphStructuralLayer_AFC(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGraphStructuralLayer_AFC, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Using HypergraphConv
        self.layer1 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.layer2 = HypergraphConv(hidden_dim, n_heads * hidden_dim // n_heads, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        #  Asynchronous Fluid Communities 
        hyper_edge_index = self.build_hypergraph_from_graph(data)
        hyper_edge_index = hyper_edge_index.to(x.device)
        out1 = self.layer1(x, hyper_edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, hyper_edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def build_hypergraph_from_graph(self, data):
        G = nx.Graph()
        edge_list = data.edge_index.t().cpu().numpy().tolist()
        
        if len(edge_list) == 0:
            num_nodes = data.x.shape[0]
            if num_nodes > 1:
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_list.append([i, j])  
            elif num_nodes == 1:
                edge_list.append([0, 0])  
        G.add_edges_from(edge_list)

     
        try:
            k = min(3, G.number_of_nodes())  # 
            communities = nx.algorithms.community.asyn_fluidc(G, k=k)
            partition = {node: idx for idx, community in enumerate(communities) for node in community}
        except Exception as e:
            partition = {node: 0 for node in G.nodes()}  
        
        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        edge_list = []
        
        for community_id, community_nodes in communities.items():
            if len(community_nodes) > 1:
                for i in range(len(community_nodes)):
                    for j in range(i + 1, len(community_nodes)):
                        edge_list.append([community_nodes[i], community_nodes[j]])

        if len(edge_list) == 0:
            num_nodes = data.x.shape[0]
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    edge_list.append([i, j])

        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()

        return hyperedges
