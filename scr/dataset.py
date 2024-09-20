
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import random 

from graphbuilder import GraphBuilder

class CustomGraphDataset(Dataset):
    def __init__(self, graph_params, mnist_dataset, supervised_learning, numbers_list,  
                 same_digit=False,
                 add_noise=False, noise_intensity=0.1, N=2,
                 edge_index=None, supervision_label_val=1, indices=None):
        
        self.mnist_dataset = mnist_dataset

        self.graph_params = graph_params

        # self.graph_structure = graph_structure
        self.NUM_INTERNAL_NODES = self.graph_params["internal_nodes"]
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.same_digit = same_digit
        self.supervised_learning = supervised_learning
        self.N = N.lower() if type(N)==str else N
        self.supervision_label_val = supervision_label_val 

        if self.supervised_learning:
            print("Supervised learning")
        else:
            print("Un supervised learning")
        
        print(f"Taking first n={self.N} digits from each class")

        if numbers_list:
            self.numbers_list = numbers_list
        else:
            self.numbers_list = list(range(10))

        # Instead of storing just the first occurrence, store all occurrences of each digit
        self.indices = {int(digit): [] for digit in self.numbers_list}

        # Populate the indices dictionary with all occurrences of each digit
        for idx, (image, label) in enumerate(self.mnist_dataset):
            if int(label) in self.numbers_list:
                self.indices[int(label)].append(idx)

        self.edge_index = edge_index
        # ------------------- Create the graph structure -------------------


        if self.edge_index is not None:
            self.edge_index = self.edge_index
            self.edge_index_tensor = self.edge_index

            # REMEMBER WHAT 'indices' is for!!!!?
            assert indices
            self.num_vertices, self.sensory_indices, self.internal_indices, self.supervision_indices = indices

        else:
                        
            # Call to create initial graph
            self.create_graph()


        print("-----Done-----")
        print(self.num_vertices)
        print(self.sensory_indices)
        print(self.internal_indices)
        print(self.supervision_indices)
        print("-----Done-----")


    def create_graph(self):
        from graphbuilder import graph_type_options
        """Create the initial graph with the provided graph parameters."""
        graph_builder = GraphBuilder(graph_type_options, **self.graph_params)
        self.edge_index = graph_builder.edge_index
        self.edge_index_tensor = graph_builder.edge_index


        self.num_vertices = graph_builder.num_vertices
        self.sensory_indices = graph_builder.sensory_indices
        self.internal_indices = graph_builder.internal_indices
        self.supervision_indices = graph_builder.supervision_indices
    
        
    def __len__(self):
        # TODO check this --> maybe len(self.indices)
        # return len(self.mnist_dataset)
        return sum(len(indices) for indices in self.indices.values())


    def add_internal_node(self, num_new_nodes=1):
        assert self.edge_index is not None, "Can't add to a non-existing graph"
        
        new_node_start_idx = self.num_vertices  # The index at which new nodes will start
        for i in range(num_new_nodes):
            new_node_idx = new_node_start_idx + i
            self.internal_indices.append(new_node_idx)  # Add the new internal node

            # Randomly select 4 existing internal nodes to connect to the new node
            existing_nodes = random.sample(self.internal_indices, k=min(4, len(self.internal_indices)))  # Pick up to 4 nodes

            for existing_node in existing_nodes:
                # Add edge from new_node -> existing_node
                self.edge_index = torch.cat(
                    [self.edge_index, torch.tensor([[new_node_idx], [existing_node]], dtype=torch.long)], dim=1
                )
                # Add edge from existing_node -> new_node
                self.edge_index = torch.cat(
                    [self.edge_index, torch.tensor([[existing_node], [new_node_idx]], dtype=torch.long)], dim=1
                )

        # Update the number of vertices after adding the new nodes
        self.num_vertices += num_new_nodes


    def __getitem__(self, idx):
        
        # if self.same_digit:
        #     # fix idx to be the same digit
        #     selected_idx = 9

        # else:
        #     digit = np.random.choice(self.numbers_list)
        
        #     # Randomly select an index from the list of indices for the selected digit
        #     digit_indices = self.indices[digit]

        #     if self.N == "all":
        #         self.N = len(digit_indices)
        #     digit_indices = digit_indices[0:self.N]
        #     selected_idx = np.random.choice(digit_indices)

        flat_indices = [(digit, idx_in_digit) for digit, digit_indices in self.indices.items() for idx_in_digit in digit_indices]

        # Get the correct digit and its associated sample index
        digit, selected_idx = flat_indices[idx]

        if idx == 0:  # Only print for the first batch to avoid repetitive output

            print("Selected idx: ", selected_idx)

        # Get the image and label from the dataset using the randomly selected index
        image, label = self.mnist_dataset[selected_idx]
        clean_image = image.clone()  # Store the clean image

        # Optionally add noise to the image
        if self.add_noise:
            noise = torch.randn(image.size()) * self.noise_intensity
            image = image + noise
            # Ensure the noisy image is still within valid range
            image = torch.clamp(image, 0, 1)
        
        # Initialize sensory nodes with image values
        # if self.supervised_learning:
        #     # the sensory nodes are fixed to a 1-hot vector with the labels
        #     # we want outgoing edges from the label_nodes (1 hot of 10 classes) to the pixel nodes

        #     # Create a one-hot encoded vector of the label
        #     label_vector = torch.zeros(10)  # Assuming 10 classes for MNIST
        #     label_vector[label] = self.supervision_label_val

        #     # Concatenate the label vector with the image vector
        #     # concat the image, internal nodes (zeros) and label nodes

        #     values = torch.cat((image.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1), label_vector.view(-1, 1)), dim=0)    
        #     # values = torch.cat((one_hot.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1), label_vector.view(-1, 1)), dim=0)

        #     # x = torch.cat((image.view(-1, 1), label_vector.view(-1, 1)), dim=0)
        # else:
        #     # Flatten image to use as part of the node features
        #     # NO: concat the image, internal nodes (zeros) and label nodes
        #     # YES: concat the image
        #     values = torch.cat((image.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1)), dim=0)
        #     # values = torch.cat((one_hot.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1)), dim=0)
        #     # x = image.view(-1, 1)

        ##### (new) Initialize the values tensor with the number of vertices
        values = torch.zeros(self.num_vertices, 1)

        ## Assign values to sensory nodes based on image
        image_flattened = image.view(-1, 1)
        for sensory_idx in self.sensory_indices:
            values[sensory_idx] = image_flattened[sensory_idx]

        values[:784] = image.view(-1, 1)

        # Initialize internal nodes with zeros
        for internal_idx in self.internal_indices:
            values[internal_idx] = 0  # Or some other initial value

        if self.supervised_learning:
            # Create a one-hot encoded vector of the label
            label_vector = torch.zeros(10)  # Assuming 10 classes for MNIST
            label_vector[label] = self.supervision_label_val

            # Assign values to the supervision nodes
            for i, supervision_idx in enumerate(self.supervision_indices):
                values[supervision_idx] = label_vector[i]

            if idx == 0:  # Only print for the first batch to avoid repetitive output
                print("Adding label", label, label_vector)

        print("Done for idx", idx)
        # Node features: value, prediction, and error for each node
        errors = torch.zeros_like(values)
        predictions = torch.zeros_like(values)

        # Combine attributes into a feature matrix
        features = torch.stack((values, errors, predictions), dim=1)

        # assign the weights to the edges to be 1 
        # print("edge_index_tensor shape: ", self.edge_index_tensor.shape)
        edge_attr = torch.ones(self.edge_index_tensor.size(1))

        self.edge_attr = edge_attr

        return Data(x=features, edge_index=self.edge_index_tensor, y=label, edge_attr=edge_attr), clean_image.squeeze(0)
        # return Data(x=features, edge_index=self.edge_index_tensor, y=label), clean_image.squeeze(0)

