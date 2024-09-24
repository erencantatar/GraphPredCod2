import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

def plot_adj_matrix(edge_index, model_dir, node_types=None):
    """
    Function to plot the adjacency matrix or node types (sensory, internal, supervision) as colored pixels.
    
    Parameters:
    edge_index (PyG edge_index): Edge index of the graph.
    model_dir (str): Directory to save the plot.
    node_types (tuple): Tuple containing sensory, internal, and supervision indices for overlay. 
                        If None, just plots the adjacency matrix.
    """
    # Convert edge_index to adjacency matrix
    adj_matrix_pyg = to_dense_adj(edge_index)[0].numpy()
    adj_matrix_size = adj_matrix_pyg.shape[0]

    if node_types:
        # Create a white empty grid for node types
        fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size as needed
        grid = np.ones((adj_matrix_size, adj_matrix_size))  # White background
        ax.imshow(grid, cmap='gray', vmin=0, vmax=1, origin='upper')  # Display empty white grid with top-left origin

        # Extract sensory, internal, and supervision indices from node_types tuple
        sensory_indices, internal_indices, supervision_indices = node_types

        # Plot each node type as a single point/pixel
        if sensory_indices is not None:
            sensory_indices = np.array(sensory_indices)
            ax.scatter(sensory_indices, sensory_indices, color='red', label='Sensory Nodes', s=10)

        if internal_indices is not None:
            internal_indices = np.array(internal_indices)
            ax.scatter(internal_indices, internal_indices, color='blue', label='Internal Nodes', s=10)

        if supervision_indices is not None:
            supervision_indices = np.array(supervision_indices)
            ax.scatter(supervision_indices, supervision_indices, color='green', label='Supervision Nodes', s=10)

        # Set title and legend
        ax.set_title("Node Types: Sensory, Internal, and Supervision")
        ax.legend(loc='upper right')

        # Ensure the axis limits match the adjacency matrix size
        ax.set_xlim(0, adj_matrix_size - 1)
        ax.set_ylim(adj_matrix_size - 1, 0)  # Reverse the y-axis to match top-left origin

        # Save the figure with node types
        plt.tight_layout()
        if model_dir:
            fig.savefig(f'{model_dir}/node_types_grid.png')

            # Close the figure after saving
            plt.close(fig)
        else:
            plt.show()
    else:
        # Create figure and axis for the adjacency matrix
        fig, ax = plt.subplots(figsize=(30, 18))

        # Plot the adjacency matrix with top-left origin
        cax = ax.imshow(adj_matrix_pyg, cmap='viridis', origin='upper')
        ax.set_title("Adjacency Matrix")
        fig.colorbar(cax, ax=ax)

        # Ensure the axis limits match the adjacency matrix size
        ax.set_xlim(0, adj_matrix_size - 1)
        ax.set_ylim(adj_matrix_size - 1, 0)  # Reverse the y-axis to match top-left origin

        # Save the figure for the adjacency matrix
        plt.tight_layout()

        if model_dir:
            fig.savefig(f'{model_dir}/adj_matrix.png')

            # Close the figure after saving
            plt.close(fig)
        else:
            plt.show()  


def plot_connection_strength_dist(W):

    incomming = W[-10:, ].flatten()
    outgoing = W[:, -10:].flatten()
    plt.hist(incomming, alpha=0.5, bins=20, label="")
    plt.hist(outgoing, alpha=0.5, bins=20, label="")
    plt.show()



    incomming = W[-10:, 0:784 ].flatten()
    outgoing = W[0:784 :, -10:].flatten()
    plt.hist(incomming, alpha=0.5, bins=20)
    plt.hist(outgoing, alpha=0.5, bins=20)
    plt.show()



def plot_energy_during_training(internal_energy, sensory_energy, history, model_dir=None, epoch="end"):



    # Assuming model and history are already defined and contain the required data

    # Create a subplot mosaic
    fig, ax = plt.subplot_mosaic([["A"], ["B"]], figsize=(12, 10))

    # Plot the first set of energy values with two y-axes
    ax["A"].plot(internal_energy, label="Internal Energy", color='blue')
    ax["A"].set_xlabel("Iterations")
    ax["A"].set_ylabel("Internal Energy", color='blue')
    ax["A"].tick_params(axis='y', labelcolor='blue')
    ax["A"].set_title("Energy over whole training")

    ax2 = ax["A"].twinx()
    ax2.plot(sensory_energy, label="Sensory Energy", color='orange')
    ax2.set_ylabel("Sensory Energy", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add legends to both y-axes
    ax["A"].legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Plot energy per epoch with two y-axes
    ax["B"].plot(history["internal_energy_per_epoch"], label="Internal Energy", color='blue')
    ax["B"].set_xlabel("Epochs")
    ax["B"].set_ylabel("Mean Internal Energy", color='blue')
    ax["B"].tick_params(axis='y', labelcolor='blue')
    ax["B"].set_title("Energy per Epoch")

    ax2_B = ax["B"].twinx()
    ax2_B.plot(history["sensory_energy_per_epoch"], label="Sensory Energy", color='orange')
    ax2_B.set_ylabel("Sensory Energy", color='orange')
    ax2_B.tick_params(axis='y', labelcolor='orange')

    # Add legends to both y-axes
    ax["B"].legend(loc='upper left')
    ax2_B.legend(loc='upper right')

    # Display the plots
    plt.tight_layout()

    if model_dir:
        fig.savefig(f'{model_dir}/energy/energy_{epoch}.png')
    else:
        plt.show()
    
    plt.close(fig)







def plot_effective_train_lr(model, model_dir):

    # Example: Plotting the effective learning rate
    import matplotlib.pyplot as plt
    plt.plot(model.pc_conv1.effective_learning["v_mean"], label='Mean Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["v_max"], label='Max Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["v_min"], label='Min Effective Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Effective Learning Rate')
    plt.legend()
    # plt.show()


    # Example: Plotting the effective learning rate
    import matplotlib.pyplot as plt
    plt.plot(model.pc_conv1.effective_learning["w_mean"], label='Mean Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["w_max"], label='Max Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["w_min"], label='Min Effective Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Effective Learning Rate')
    plt.legend()
    
    # plt.show()



import matplotlib.pyplot as plt
import torch
import scipy.sparse as sp


import os
import matplotlib.pyplot as plt
import scipy.sparse as sp

def plot_model_weights(model, GRAPH_TYPE=None, model_dir=None):
    # Ensure the output directory exists
    # os.makedirs(model_dir, exist_ok=True)

    # Extract the edge indices and weights
    edge_index = model.pc_conv1.edge_index_single_graph.cpu().numpy()
    weights = model.pc_conv1.weights.cpu().detach().numpy()

    # Create a sparse matrix using the edge indices and weights
    W_sparse = sp.coo_matrix((weights, (edge_index[0], edge_index[1])), shape=(model.pc_conv1.num_vertices, model.pc_conv1.num_vertices))

    # Convert to dense for detailed visualization (if the graph is not too large)
    W = W_sparse.toarray()

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    if GRAPH_TYPE:
        fig.suptitle(f'Visualization of Weight Matrix of {GRAPH_TYPE}', fontsize=20)
    else:
        fig.suptitle(f'Visualization of Weight Matrix', fontsize=20)

    # Subplot 1: Sparsity pattern
    axes[0, 0].spy(W_sparse, markersize=1)
    axes[0, 0].set_title('Sparsity Pattern of the Weight Matrix')

    # Subplot 2: Full weight matrix
    im1 = axes[0, 1].imshow(W, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=axes[0, 1], label='Weight value')
    axes[0, 1].set_title('Full Weight Matrix')

    # Subplot 3: Weights > 0.001
    im2 = axes[1, 0].imshow(W > 0.001, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=axes[1, 0], label='Weight value')
    axes[1, 0].set_title('Weights > 0.001')

    # Subplot 4: Weights > 0.0001
    im3 = axes[1, 1].imshow(W > 0.0001, cmap='viridis', aspect='auto')
    fig.colorbar(im3, ax=axes[1, 1], label='Weight value')
    axes[1, 1].set_title('Weights > 0.0001')

    # Subplot 5: Weights > 0.00001
    im4 = axes[2, 0].imshow(W > 0.00001, cmap='viridis', aspect='auto')
    fig.colorbar(im4, ax=axes[2, 0], label='Weight value')
    axes[2, 0].set_title('Weights > 0.00001')

    # Subplot 6: Weights > 0.000001
    im5 = axes[2, 1].imshow(W > 0.000001, cmap='viridis', aspect='auto')
    fig.colorbar(im5, ax=axes[2, 1], label='Weight value')
    axes[2, 1].set_title('Weights > 0.000001')

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit the title

    if model_dir:
        plt.savefig(model_dir)
        plt.close(fig)

        print(f'Figure saved to {model_dir}')
    else:
        plt.show()