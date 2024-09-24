
from torch_geometric.utils import from_networkx, to_dense_adj
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import numpy as np 
import torch 



graph_type_options = {
        "fully_connected": {
            "params": None
        },
        "fully_connected_w_self": {
            "params": None
        },

        "fully_connected_no_sens2sup": {
            "params": None
        }, 

        "barabasi": {
            "params": {
                "num_edges_to_attach": 5  # Example parameter for Barabasi graph
            }
        },
        "stochastic_block": {
            "params": {
                "num_communities": 5,      # Number of communities
                "community_size": 10,       # Size of each community
                "p_intra": 0.1,             # Probability of edges within the same community
                "p_inter": 0.1,             # Probability of edges between different communities
                "remove_sens_2_sens": True, 
                "remove_sens_2_sup": True
                }
        },
        
    

        "stochastic_block_w_supvision_clusters": {
            "params": {
                "num_communities": 50,      # Number of communities
                "community_size": 30,       # Size of each community
                "p_intra": 0.5,             # Probability of edges within the same community
                "p_inter": 0.1,             # Probability of edges between different communities
                "remove_sens_2_sens": True, 
                "remove_sens_2_sup": True
                }
        },
        

    }








class GraphBuilder:

    def __init__(self, graph_type_options, internal_nodes, supervised_learning, graph_type):

        
        # allowed_topologies = ["fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block", "fully_connected_no_sens2sup"]
        allowed_topologies = list(graph_type_options.keys())
        
        assert graph_type["name"].lower() in allowed_topologies, "Graph structure unkown"
        self.graph_type = graph_type

        # check if graph tpye aligns with the needed graph_params 
        # TODO 

        # init base, but can change depending on the graph type.
        print("--------Init base indices for sensory, internal, supervision nodes--------")
        
        self.SENSORY_NODES = 784
        self.num_sensor_nodes = range(self.SENSORY_NODES)
        self.NUM_INTERNAL_NODES = internal_nodes
        self.num_internal_nodes = range(self.SENSORY_NODES, self.SENSORY_NODES + self.NUM_INTERNAL_NODES)
        self.INTERNAL = self.num_internal_nodes

        self.supervised_learning = supervised_learning
        
        # Define total number of nodes depending on supervised or unsupervised learning
        if self.supervised_learning:
            self.num_all_nodes = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES + 10)  # Adding 10 label nodes
            self.supervision_indices = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES, self.SENSORY_NODES + self.NUM_INTERNAL_NODES + 10)
        else:
            self.num_all_nodes = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES)

        # Initialize the number of vertices and indices
        print("--------Updating base indices for sensory, internal, supervision nodes--------")
        self.num_vertices = len(self.num_sensor_nodes) + len(self.num_internal_nodes)
        if self.supervised_learning:
            self.num_vertices += len(self.supervision_indices)  # Add supervision nodes to the vertex count
        self.sensory_indices = list(range(self.SENSORY_NODES))  # Assuming SENSORY_NODES is the number of sensory nodes
        self.internal_indices = list(self.num_internal_nodes)

        self.graph_params = graph_type["params"]

        print(self.NUM_INTERNAL_NODES)
        print(self.num_internal_nodes)

        self.create_graph()


    def get_data():

        return 

    def create_graph(self):

        self.edge_index = []

        if self.graph_type["name"] == "fully_connected":
            self.fully_connected(self_connection=False)
        elif self.graph_type["name"] == "fully_connected_w_self":
            self.fully_connected(self_connection=True)
        elif self.graph_type["name"] == "barabasi":
            self.barabasi()
        elif self.graph_type["name"] == "stochastic_block":
            self.stochastic_block()

        elif self.graph_type["name"] == "stochastic_block_w_supvision_clusters":
            self.stochastic_block_w_supervision_clusters()

        elif self.graph_type["name"] == "fully_connected_no_sens2sup":
            self.fully_connected_no_sens2sup()
        
        # 



        # Convert edge_index to tensor only if it's not already a tensor
        if not isinstance(self.edge_index, torch.Tensor):
            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)

        # Transpose if edge_index was manually created
        if self.graph_type["name"] in ["fully_connected", "fully_connected_w_self", "fully_connected_no_sens2sup", "barabasi"]:
            self.edge_index = self.edge_index.t().contiguous()
            self.edge_index_tensor = self.edge_index.t().contiguous()

        assert self.edge_index.shape[0] == 2
        print("self.edge_index", self.edge_index.shape)

    def fully_connected_no_sens2sup(self):

        print("Creating fully connected graph without sensory to internal (and otherwayaround)")
        # Sensory to sensory (both ways)
        for i in self.num_sensor_nodes:
            for j in self.num_sensor_nodes:
                if i != j:
                    self.edge_index.append([i, j])

        # Sensory to internal (both ways )
        for i in self.num_sensor_nodes:
            for j in self.num_internal_nodes:
                self.edge_index.append([i, j])
                self.edge_index.append([j, i])

        # Internal to internal (both directions)
        for i in self.num_internal_nodes:
            for j in self.num_internal_nodes:
                if i != j:
                    self.edge_index.append([i, j])

        if self.supervised_learning:
            # Internal to label (both directions)
            label_nodes = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES, self.SENSORY_NODES + self.NUM_INTERNAL_NODES + 10)
            for i in self.num_internal_nodes:
                for j in label_nodes:
                    self.edge_index.append([i, j])
                    self.edge_index.append([j, i])
            


    def fully_connected(self, self_connection):
        if self_connection:
            print("Creating fully connected directed graph with self connections")
            self.edge_index += [[i, j] for i in self.num_all_nodes for j in self.num_all_nodes]
        else:
            print("Creating fully connected directed graph without self connections")
            self.edge_index += [[i, j] for i in self.num_all_nodes for j in self.num_all_nodes if i != j]
        
    def barabasi(self):
        num_nodes = len(self.num_all_nodes)
        m = self.graph_params.get("m", 5)  # Number of edges to attach from a new node to existing nodes
        G = nx.barabasi_albert_graph(num_nodes, m)
        self.edge_index = [[u, v] for u, v in G.edges()]
        print(f"Creating Barabási-Albert graph with {num_nodes} nodes and {m} edges to attach per new node")

    
    def stochastic_block(self):

        # Given code parameters; else take default 40, 50
        num_communities = self.graph_params.get("num_communities", 40)  # Example block sizes
        community_size = self.graph_params.get("community_size",   50)

        print(num_communities, community_size,  len(self.num_internal_nodes) )
        assert (num_communities * community_size) == len(self.num_internal_nodes), "must be equal"
        # Sizes of communitieser
        sizes = [community_size for _ in  range(num_communities)]
        # SENSORY_NODES = range(0, range(self.SENSORY_NODES))
        self.INTERNAL = range(self.SENSORY_NODES, (self.SENSORY_NODES+sum(sizes)))
        self.num_internal_nodes = sum(self.INTERNAL)
        # SUPERVISED_NODES = range(self.SENSORY_NODES+sum(sizes), sum(sizes)+10)


        sizes.insert(0, self.SENSORY_NODES)
        sizes.append(10)
        community_sizes = sizes

        num_communities = len(community_sizes)
        p_intra = 0.5  # Probability of edges within the same community
        p_inter = 0.1  # Probability of edges between different communities

        # Create the stochastic block model graph
        p = np.full((num_communities, num_communities), p_inter)
        np.fill_diagonal(p, p_intra)

        # removing sensory to sensory connection
        if self.graph_params.get("remove_sens_2_sens", True):
            p[0, 0] = 0
        
        # adding connections from community to community  
        for i in range(1, num_communities):
            p[0, i] = 0.1
            p[i, 0] = 0.1

        if self.graph_params.get("remove_sens_2_sup", True):
            # remove sensory to supervised
            p[0, -1] = 0 
            p[-1, 0] = 0 

        G = nx.stochastic_block_model(sizes, p, directed=True)
     
        # Convert the graph to an adjacency matrix
        # adj_matrix = nx.adjacency_matrix(G).todense()

        # Convert the NetworkX graph to PyTorch Geometric format
        data = from_networkx(G)

        # Extract the edge_index tensor
        self.edge_index = data.edge_index

        print("Creating Stochastic Block Model graph")

    def stochastic_block_w_supervision_clusters(self):

        # Parameters
        num_communities = self.graph_params.get("num_communities", 3)  # 3 internal clusters
        community_size = self.graph_params.get("community_size", 5)  # Each internal cluster has 5 nodes
        supervision_cluster_size = 10  # 10 supervision clusters

        # Ensure that the total number of internal nodes is correct
        assert (num_communities * community_size) == len(self.num_internal_nodes), "Internal node count must match."

        # Initialize ranges
        SENSORY_NODES = self.SENSORY_NODES
        self.INTERNAL = range(self.SENSORY_NODES, self.SENSORY_NODES + num_communities * community_size)
        self.num_internal_nodes = sum(self.INTERNAL)

        # Sizes for internal clusters
        internal_community_sizes = [community_size for _ in range(num_communities)]

        # Indices for supervision clusters, 1 supervision node + 9 internal nodes per cluster
        supervision_internal_nodes_per_cluster = 9
        self.SUPERVISION = []  # Supervision nodes
        supervision_internal_indices = []  # Internal nodes from the supervision clusters

        supervision_start_idx = self.SENSORY_NODES + num_communities * community_size
        for i in range(supervision_cluster_size):
            supervision_node_idx = supervision_start_idx + i * (supervision_internal_nodes_per_cluster + 1)
            self.SUPERVISION.append(supervision_node_idx)  # First node is the supervision node

            # The remaining 9 nodes in the supervision cluster are internal nodes
            supervision_internal_indices.extend(range(supervision_node_idx + 1, supervision_node_idx + 10))

        # Combine all internal nodes (from internal clusters + supervision clusters)
        self.internal_indices = list(range(self.SENSORY_NODES, self.SENSORY_NODES + num_communities * community_size)) + supervision_internal_indices

        # Set up intra-cluster and inter-cluster probabilities
        p_intra = self.graph_params.get("p_intra", 0.5)
        p_inter = self.graph_params.get("p_inter", 0.1)

        # Total communities: internal + supervision
        total_community_sizes = internal_community_sizes + [supervision_internal_nodes_per_cluster + 1] * supervision_cluster_size

        # Create block probability matrix
        p = np.full((len(total_community_sizes), len(total_community_sizes)), p_inter)
        np.fill_diagonal(p, p_intra)

        # removing sensory to sensory connection
        if self.graph_params.get("remove_sens_2_sens", True):
            p[0, 0] = 0
        
        # Adding connections from community to community
        for i in range(1, len(total_community_sizes)):
            p[0, i] = 0.1
            p[i, 0] = 0.1

        # Remove sensory to supervised connections if specified
        if self.graph_params.get("remove_sens_2_sup", True):
            # Sensory community is at index 0, and supervision nodes are at the end
            p[0, -1] = 0
            p[-1, 0] = 0

        # Build stochastic block model
        G = nx.stochastic_block_model(total_community_sizes, p, directed=True)

        # Convert the graph to PyTorch Geometric format
        data = from_networkx(G)

        # Update edge_index tensor
        self.edge_index = data.edge_index

        # Update the self.indices to reflect the new structure
        print("--------Updating base indices for sensory, internal, supervision nodes--------")
        self.num_vertices = len(self.sensory_indices) + len(self.internal_indices) + len(self.supervision_indices)
        self.sensory_indices = list(range(SENSORY_NODES))

        self.supervision_indices = self.SUPERVISION
        self.internal_indices = list(self.INTERNAL) + supervision_internal_indices


        print("Created Stochastic Block Model graph with supervision clusters")
