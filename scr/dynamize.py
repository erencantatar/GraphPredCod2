from dataset import CustomGraphDataset


class Dynamical_graph():

    """ 
    Dynamical graph class
    -----------------------

    Starting with adding random nodes to the graph during training
    """

    def __init__(self, model, graph_dataset: CustomGraphDataset) -> None:
        self.model = model  # The model that uses the graph
        self.graph_dataset = graph_dataset  # The dataset containing the graph
        
        ### all graphs 
        self.policy_nodes = {"add": "randomly", "remove": "randomly"}
        self.policy_edges = {"add": "randomly", "remove": "randomly"}
        
        #####   stoic. block graphs 
        self.policy_clusters = {"add": "randomly", "remove": "randomly"}


    def policy(self, policy_nodes):
        pass 

    
    def info(self):

        self.graph_dataset.info()

    

    def add_nodes(self, print_info=False):
        """Dynamically add a node based on the current policy."""
        tmp_edges = self.graph_dataset.edge_index
        tmp_nodes = len(self.graph_dataset.internal_indices)
        
        if self.policy_nodes["add"] == "randomly":
            # Add node to the graph dataset and connect it randomly to other nodes
            self.graph_dataset.add_internal_node(num_new_nodes=150)
        elif self.policy_nodes["add"] == "performance_based":
            # Example: Add node based on a certain performance metric (to be implemented)
            pass

        new_edge_index = self.graph_dataset.edge_index

        if print_info:
            print("added new nodes", len(self.graph_dataset.internal_indices) - (tmp_nodes))
            print("added new edges", new_edge_index.shape[1] - tmp_edges.shape[1])

        # Notify the model to update its internal graph structure
        new_internal_indices = self.graph_dataset.internal_indices

        self.model.update_graph(new_edge_index, new_internal_indices)



