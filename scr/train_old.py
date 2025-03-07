import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch_geometric.data import Data

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch.nn.init as init
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from icecream import ic


from torch_scatter import scatter_mean
from torch_geometric.data import Data
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import os


######################################################################################################### 
####                                            Arguments                                           #####
######################################################################################################### 

# import wandb 

# make change here

import os
import argparse
from helper.args import true_with_float, valid_str_or_float, valid_int_or_all, valid_int_list, str2bool, validate_weight_init
from helper.activation_func import activation_functions
from graphbuilder import graph_type_options

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Train a model with specified parameters.')

# Training mode 
parser.add_argument('--mode', choices=['training', 'experimenting'], required=True,  help="Mode for training the model or testing new features.")

# -----dataset----- 
#data_dir default --data_dir default to $TMPDIR
parser.add_argument('--data_dir', type=str, default='../data', help='Path to the directory to store the dataset. Use $TMPDIR for scratch. or use ../data ')

parser.add_argument('--dataset_transform', nargs='*', default=[], help='List of transformations to apply.', choices=['normalize_min1_plus1', 'normalize_mnist_mean_std', 'random_rotation', 'none'])
parser.add_argument('--numbers_list', type=valid_int_list, default=[0, 1, 3, 4, 5, 6, 7], help="A comma-separated list of integers describing which distinct classes we take during training alone")
parser.add_argument('--N', type=valid_int_or_all, default=20, help="Number of distinct trainig images per class; greater than 0 or the string 'all' for all instances o.")

parser.add_argument('--supervision_label_val', default=10, type=int, required=True, help='An integer value.')


# -----graph-----  
parser.add_argument('--num_internal_nodes', type=int, default=1500, help='Number of internal nodes.')
parser.add_argument('--graph_type', type=str, default="fully_connected", help='Type of Graph', choices=list(graph_type_options.keys()))
parser.add_argument('--remove_sens_2_sens', type=str2bool, required=True, help='Whether to remove sensory-to-sensory connections.')
parser.add_argument('--remove_sens_2_sup', type=str2bool, required=True, help='Whether to remove sensory-to-supervised connections.')

# --MessagePassing--
parser.add_argument('--normalize_msg', choices=['True', 'False'], required=True,  help='Normalize message passing, expected True or False')

# -----model-----  
parser.add_argument('--model_type', type=str, default="PC", help='Predictive Model type: [PC,IPC] ', choices=["PC", "IPC"])
# parser.add_argument("--weight_init", type=str, default="fixed 0.001", help="Initialization method and params for weights")
parser.add_argument("--weight_init", type=validate_weight_init, default="fixed 0.001", help="Initialization method and params for weights")

parser.add_argument('--use_bias',  choices=['True', 'False'], required=True, help="....")
parser.add_argument("--bias_init", type=str, default="", required=False, help="ege. fixed 0.0 Initialization method and params for biases")

parser.add_argument('--T', type=int, default=40, help='Number of iterations for gradient descent.')
parser.add_argument('--lr_values', type=float, default=0.001, help='Learning rate values (alpha).')
parser.add_argument('--lr_weights', type=float, default=0.01, help='Learning rate weights (gamma).')
parser.add_argument('--activation_func', default="swish", type=str, choices=list(activation_functions.keys()), required=True, help='Choose an activation function: tanh, relu, leaky_relu, linear, sigmoid, hard_tanh, swish')


# -----training----- 
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--optimizer', type=true_with_float, default=False,
                    help="Either False or, if set to True, requires a float value.")

# ---after training-----
parser.add_argument('--set_abs_small_w_2_zero',  choices=['True', 'False'], required=True, help="....")

# logging 
import wandb
parser.add_argument('--use_wandb', type=str, default="disabled", help='Wandb mode.', choices=['shared', 'online', 'run', 'dryrun', 'offline', 'disabled'])
parser.add_argument('--tags', type=str, default="", help="Comma-separated tags to add to wandb logging (e.g., 'experiment,PC,test')")



args = parser.parse_args()

# Using argparse values
torch.manual_seed(args.seed)

generator_seed = torch.Generator()
generator_seed.manual_seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f"Seed used", args.seed)
if torch.cuda.is_available():
    print("Device name: ", torch.cuda.get_device_name(0))

# Make True of False bool
args.normalize_msg = args.normalize_msg == 'True'
args.use_bias = args.use_bias == 'True'
args.set_abs_small_w_2_zero = args.set_abs_small_w_2_zero == 'True'

tags_list = args.tags.split(",") if args.tags else []

import torchvision.transforms as transforms
import numpy as np


transform_list = [
    transforms.ToTensor()
]

if args.dataset_transform:

    if "normalize_min1_plus1" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    if "normalize_mnist_mean_std" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    if "random_rotation" in args.dataset_transform:
        transform_list.append(transforms.RandomRotation(degrees=20))
    

# Create the transform
transform = transforms.Compose(transform_list)

mnist_trainset = torchvision.datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
mnist_testset  = torchvision.datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




######################################################################################################### 
####                                            Dataset                                             #####
######################################################################################################### 


## Subset of the dataset (for faster development)
# subset_size = 100  # Number of samples to use from the training set
# indices = list(range(len(mnist_trainset)))
# random.shuffle(indices)
# subset_indices = indices[:subset_size]

# mnist_train_subset = torch.utils.data.Subset(mnist_trainset, subset_indices)
# print("USSSSSING SUBSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET")

# CustomGraphDataset params
dataset_params = {
    "mnist_dataset":            mnist_trainset,
    # "mnist_dataset":            mnist_train_subset,
    "supervised_learning":      True,
    "numbers_list":             args.numbers_list,
    "same_digit":               False,
    "add_noise":                False,
    "noise_intensity":          0.0,
    "N":                        args.N,     # taking the first n instances of each digit or use "all"

    "edge_index":               None,
    "supervision_label_val":    args.supervision_label_val,         # Strength of label signal within the graph. MNIST ~0-1, label_vector[label] = self.supervision_label_val
} 

print("------------------Importing Graph Params ---------------- ")
from graphbuilder import graph_type_options

# Define the graph type
# Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block", "fully_connected_no_sens2sup"
graph_params = {
    "internal_nodes": args.num_internal_nodes,  # Number of internal nodes
    "supervised_learning": True,  # Whether the task involves supervised learning
    "graph_type": {    
        "name": args.graph_type, # Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block"
        "params": graph_type_options[args.graph_type]["params"], 
        # "params_general": {
        #     "remove_sens_2_sens": args.remove_sens_2_sens,  # Set from command line
        #     "remove_sens_2_sup": args.remove_sens_2_sup,    # Set from command line
        #     },
        },  
    "seed": args.seed,   
}




# add graph specific info: 
print("zzz", args.remove_sens_2_sens, args.remove_sens_2_sup)
graph_params["graph_type"]["params"]["remove_sens_2_sens"] = args.remove_sens_2_sens  
graph_params["graph_type"]["params"]["remove_sens_2_sup"]  = args.remove_sens_2_sup 

print("graph_params 1 ", graph_params)

if graph_params["graph_type"]["name"] == "stochastic_block":
    
    # override internal nodes if doing clustering
    graph_params["internal_nodes"] = (graph_params["graph_type"]["params"]["num_communities"] * graph_params["graph_type"]["params"]["community_size"])

if graph_params["graph_type"]["name"] == "stochastic_block_hierarchy":
    raise ValueError("Not implemented yet")


if graph_params["graph_type"]["name"] in ["custom_two_branch","two_branch_graph"]:
    # Configure internal nodes for two_branch_graph
    # This assumes two branches with specified configurations
    branch1_layers, branch1_clusters_per_layer, branch1_nodes_per_cluster = graph_params["graph_type"]["params"]["branch1_config"]
    branch2_layers, branch2_clusters_per_layer, branch2_nodes_per_cluster = graph_params["graph_type"]["params"]["branch2_config"]
    
    # Calculate total internal nodes for both branches
    # Branch 1
    branch1_internal_nodes = branch1_layers * branch1_clusters_per_layer * branch1_nodes_per_cluster
    # Branch 2 (Reversed order)
    branch2_internal_nodes = branch2_layers * branch2_clusters_per_layer * branch2_nodes_per_cluster
    
    # The total number of internal nodes will be the sum of both branches
    graph_params["internal_nodes"] = branch1_internal_nodes + branch2_internal_nodes


from dataset import CustomGraphDataset

# Initialize the GraphBuilder
custom_dataset_train = CustomGraphDataset(graph_params, **dataset_params)

dataset_params["batch_size"] = args.batch_size
dataset_params["NUM_INTERNAL_NODES"] = graph_params["internal_nodes"]
# dataset_params["NUM_INTERNAL_NODES"] = (custom_dataset_train.NUM_INTERNAL_NODES)

print("Device \t\t\t:", device)
print("SUPERVISED on/off \t", dataset_params["supervised_learning"])


from helper.plot import plot_adj_matrix

single_graph = custom_dataset_train.edge_index


print("--------------Init DataLoader --------------------")
train_loader = DataLoader(custom_dataset_train, 
                          batch_size=dataset_params["batch_size"], 
                          shuffle=True, 
                          generator=generator_seed,
                          num_workers=1,
                          pin_memory=True,
                          )


NUM_SENSORY = 28*28  # 10

## TODO: FIX HOW TO DO THIS 
#### ---------------------------------------------------------------------------------------------------------------
# sensory_indices    = range(NUM_SENSORY)
# internal_indices   = range(NUM_SENSORY, NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"])
# num_vertices = NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"]  # Number of nodes in the graph
# supervision_indices = None

# if dataset_params["supervised_learning"]:
#     label_indices     = range(NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"], NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"] + 10)
#     supervision_indices = label_indices
#     num_vertices += 10

# print("sensory_indices\t\t:", len(sensory_indices), sensory_indices[0], "...", sensory_indices[-1])
# print("internal_indices\t:", len(internal_indices), internal_indices[0], "...", internal_indices[-1])
# print("num_vertices \t\t:", num_vertices)

# if dataset_params["supervised_learning"]:
#   assert num_vertices == len(sensory_indices) + len(internal_indices) + 10, "Number of vertices must match the sum of sensory and internal indices + labels"
# else:
#   assert num_vertices == len(sensory_indices) + len(internal_indices), "Number of vertices must match the sum of sensory and internal indices"
#### ---------------------------------------------------------------------------------------------------------------

num_vertices = custom_dataset_train.num_vertices
sensory_indices = custom_dataset_train.sensory_indices
internal_indices = custom_dataset_train.internal_indices
supervision_indices = custom_dataset_train.supervision_indices



print(train_loader.batch_size)
for batch, clean_image in train_loader:
    
    values, errors, predictions = batch.x[:, 0], batch.x[:, 1], batch.x[:, 2]
  
    x, edge_index, y, edge_weight = batch.x, batch.edge_index, batch.y, batch.edge_attr
    print("edge_index", edge_index.shape)

    print(batch.x[:, 0].shape)
    print(custom_dataset_train.edge_index.shape)
    

    full_batch = edge_index

    if graph_params["graph_type"]["name"] == "two_branch_graph":

        number_of_internal_nodes = branch1_internal_nodes + branch2_internal_nodes

        number_of_nodes = 784 + number_of_internal_nodes + 10

        assert batch.x[:, 0] == number_of_nodes, f"Number of nodes in the graph must be {number_of_nodes} but is {batch.x[:, 0]}"

    break




######################################################################################################### 
####                                            VALIDATION                                          #####
######################################################################################################### 
 
# from helper.validation import validate_messagePassing
# validate_messagePassing()


from helper.validation import compare_class_args

from models.PC import PCGNN, PCGraphConv
from models.IPC import IPCGNN, IPCGraphConv

# Usage example: comparing IPCGraphConv and PCGraphConv
# compare_class_args(IPCGraphConv, PCGraphConv)


######################################################################################################### 
####                                            FIND OPTIMAL LR                                     #####
######################################################################################################### 
""" 
SKIPPING FOR NOW, see local  
"""





######################################################################################################### 
####                                              Model  (setup)                                    #####
######################################################################################################### 

# lr_gamma, lr_alpha =  (0.1 ,  0.0001)
# lr_gamma, lr_alpha =  (0.1, 0.00001)


model_params = {
    
    'num_vertices': num_vertices,
    'sensory_indices': (sensory_indices), 
    'internal_indices': (internal_indices), 
    "supervised_learning": (supervision_indices),
    "use_bias": args.use_bias,

    "normalize_msg": args.normalize_msg,

    "lr_params": (args.lr_values, args.lr_weights),
    #   (args.lr_gamma, args.lr_alpha), 
    "T": args.T,
    "graph_structure": custom_dataset_train.edge_index_tensor, 
    "batch_size": train_loader.batch_size, 
    "edge_type":  custom_dataset_train.edge_type,

    "use_learning_optimizer": args.optimizer if not args.optimizer  else [args.optimizer],    # False or [0], [(weight_decay=)]
    
    # "weight_init": "uniform",   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'
    "weight_init": args.weight_init,   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'
    "activation": args.activation_func,  
    "clamping": None , # (0, torch.inf) or 'None' 
 }

# 

learning_params = model_params.copy()
learning_params['sensory_indices'] = list(learning_params['sensory_indices'])
learning_params['internal_indices'] = list(learning_params['internal_indices'])
learning_params['supervised_learning'] = list(learning_params['supervised_learning'])
# learning_params['transform'] = transform.to_dict()["transform"]
learning_params['dataset_transform'] = args.dataset_transform

learning_params['graph_structure'] = (learning_params['graph_structure']).cpu().numpy().tolist()
optimizer_str = str(args.optimizer) if isinstance(args.optimizer, float) else str(args.optimizer)

# model_params_short = f"num_iternode_{args.num_internal_nodes}_T_{args.T}_lr_w_{args.lr_weights}_lr_val_{args.lr_values}_Bsize_{train_loader.batch_size}"
model_params_short = f"{args.model_type}_{args.graph_type}_T_{args.T}_lr_w_{args.lr_weights}_lr_val_{args.lr_values}"
print(len(model_params_short), model_params_short)
model_params_name = (
    f"{args.model_type}_"
    f"nodes_{graph_params['internal_nodes']}_" 
    f"T_{args.T}_"
    f"lr_vals_{args.lr_values}_"
    f"lr_wts_{args.lr_weights}_"
    f"bs_{args.batch_size}_"
    f"act_{args.activation_func}_"
    f"init_{args.weight_init}_"
    f"graph_{args.graph_type}_"
    f"sup_{args.supervision_label_val}_"
    f"norm_{args.normalize_msg}_"
    f"nums_{'_'.join(map(str, args.numbers_list))}_"
    f"N_{args.N}_"
    f"ep_{args.epochs}_"
    f"opt_{optimizer_str}_"
    f"trans_{'_'.join(args.dataset_transform) if args.dataset_transform else 'none'}"
)

model_params_name_full = (
    f"model_{args.model_type}_"
    f"num_internal_nodes_{graph_params['internal_nodes']}_"
    f"T_{args.T}_"
    f"lr_values_{args.lr_values}_"
    f"lr_weights_{args.lr_weights}_"
    f"batch_size_{args.batch_size}_"
    f"activation_{args.activation_func}_"
    f"weight_init_{args.weight_init}_"
    f"graph_type_{args.graph_type}_"
    f"supervision_val_{args.supervision_label_val}_"
    f"normalize_msg_{args.normalize_msg}_"
    f"numbers_list_{'_'.join(map(str, args.numbers_list))}_"
    f"N_{args.N}_"
    f"epochs_{args.epochs}_"
    f"optimizer_{optimizer_str}_"
    f"dataset_transform_{'_'.join(args.dataset_transform) if args.dataset_transform else 'none'}"
)

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    
    raise TypeError('Unknown type:', type(obj))

# combi of learning params and dataset params
params_dict = {**dataset_params, **learning_params}


import json

from datetime import datetime




save_model_params = False

GRAPH_TYPE = graph_params["graph_type"]["name"]    #"fully_connected"
# GRAPH_TYPE = "test"    #"fully_connected"

# Fetch the graph parameters based on the selected graph type

date_hour = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
path = ""
# Initialize base path depending on mode (training or experimenting)
if args.mode == "experimenting":
    path += f"trained_models_experimenting/"
elif args.mode == "training":
    path += f"trained_models/"
else:
    raise ValueError("Invalid mode")

path += f"{args.model_type.lower()}/{graph_params['graph_type']['name']}/"
  
# Modify the path based on the graph configuration (removing sens2sens or sens2sup)
if graph_params["graph_type"]["params"]["remove_sens_2_sens"] and graph_params["graph_type"]["params"]["remove_sens_2_sup"]:
    graph_type_ = "_no_sens2sens_no_sens2sup"
elif graph_params["graph_type"]["params"]["remove_sens_2_sens"]:
    graph_type_ = "_no_sens2sens"
elif graph_params["graph_type"]["params"]["remove_sens_2_sup"]:
    graph_type_ = "_no_sens2sup"
else:
    graph_type_ = "_normal"  # If neither are removed, label the folder as 'normal'

path += graph_type_
# Append graph type, model parameters, and timestamp to the path
path += f"/{model_params_name}_{date_hour}/"
model_dir = path

# Define the directory path
print("Saving model, params, graph_structure to :", model_dir)

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# For saving, validation, re-creation 
os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(model_dir, "parameter_info"), exist_ok=True)

# For testing the model
os.makedirs(os.path.join(model_dir, "eval"), exist_ok=True)
os.makedirs(os.path.join(model_dir, "eval/generation"), exist_ok=True)
# os.makedirs(os.path.join(model_dir, "reconstruction"), exist_ok=True)
os.makedirs(os.path.join(model_dir, "eval/classification"), exist_ok=True)
os.makedirs(os.path.join(model_dir, "eval/denoise"), exist_ok=True)
os.makedirs(os.path.join(model_dir, "eval/occlusion"), exist_ok=True)

# Monitor during training
os.makedirs(os.path.join(model_dir, "energy"), exist_ok=True)

plot_adj_matrix(single_graph, model_dir, 
                node_types=(sensory_indices, internal_indices, supervision_indices))
plot_adj_matrix(full_batch, model_dir, node_types=None)


config_dict = vars(args)  # Convert argparse Namespace to dictionary

config_dict.update({
    'graph_type_conn': graph_type_,  # Adding model type manually
    'remove_sens_2_sens_': args.remove_sens_2_sens,  # Custom boolean flag for data augmentation
    'remove_sens_2_sup_': args.remove_sens_2_sup,  # Custom boolean flag for data augmentation
    "checkpoint_dir": model_dir,  # Track where model checkpoints are saved
    **graph_params["graph_type"]["params"],  # Merge dynamic graph parameters
})

run = wandb.init(
    mode=args.use_wandb,
    # entity="Erencan Tatar", 
    project=f"PredCod",
    # name=f"T_{args.T}_lr_value_{args.lr_values}_lr_weights_{args.lr_weights}_",
    name=f"{model_params_short}_{date_hour}",
    # id=f"{model_params_short}_{date_hour}",
    tags=tags_list,  # Add tags list here

    dir=model_dir,
    # tags=["param_search", str(model_params["weight_init"]), model_params["activation"],  *learning_params['dataset_transform']], 
    # Track hyperparameters and run metadata
    config=config_dict,  # Pass the updated config dictionary to wandb.init
)

# Contains graph edge matrix and other parameters so quite big to open.
if save_model_params:
    # Save the dictionary to a text file
    with open(model_dir + "parameter_info/params_full.txt", 'w') as file:
        json.dump(params_dict, file, default=default)
    print('Done')

# Store the exact command-line arguments in a text file
import sys
command = ' '.join(sys.argv)
with open(model_dir +'parameter_info/command_log.txt', 'w') as f:
    f.write(command)

with open('trained_models/current_running.txt', 'w') as f:
    f.write(model_dir)

# Save the (small) dictionary to a text file
params_dict_small = {}
keys_to_copy = ['supervised_learning', 'numbers_list', 'NUM_INTERNAL_NODES',  
                'N',  'batch_size','use_learning_optimizer', 'weight_init', 'activation', ]
# copy value from params_dict to params_dict_small
for key in keys_to_copy:
    params_dict_small[key] = params_dict[key]

if "dataset_transform" in params_dict:
    params_dict_small["dataset_transform"] = params_dict["dataset_transform"]


if save_model_params:
    # Save the dictionary to a text file
    with open(model_dir + "parameter_info/params_small.txt", 'w') as file:
        json.dump(params_dict_small, file, default=default)
    print('Done')

print(f"Using batch size of \t: {train_loader.batch_size}")
print("Device \t\t\t:",          device)
print("Model type", args.model_type.lower())



if args.model_type.lower() == "pc":
        

    model = PCGNN(**model_params,   
        log_tensorboard=False,
        wandb_logger=run if args.use_wandb in ['online', 'run'] else None,
        debug=False, device=device)

    print("-----------Loading PC model-----------")

if args.model_type.lower() == "ipc":
        

    model = IPCGNN(**model_params,   
        log_tensorboard=False,
        wandb_logger=run if args.use_wandb in ['online', 'run'] else None,
        debug=False, device=device)
    print("-----------Loading IPC model-----------")

# Magic
wandb.watch(model, 
            log="all",   # (str) One of "gradients", "parameters", "all", or None
            log_freq=10)


from helper.plot import plot_model_weights, plot_energy_graphs


save_path = os.path.join(model_dir, 'parameter_info/weight_matrix_visualization_epoch0.png')
# plot_model_weights(model, save_path)
plot_model_weights(model, GRAPH_TYPE, model_dir=save_path)



######################################################################################################### 
####                                              Model  (training)                                 #####
######################################################################################################### 

model.pc_conv1.set_mode("training")
import time 
torch.cuda.empty_cache()

print(model)      

model = model.to(device)
# assert train_loader.batch_size == 1
print(len(train_loader))
print("Starting training")


# Initialize early stopping and history
earlystop = False
history = {
    "internal_energy_per_epoch": [],
    "sensory_energy_per_epoch": [],
}

wandb.watch(model.pc_conv1, log="all", log_freq=10)
# wandb.watch(self.pc_conv1, log="all", log_freq=10)



reduce_lr_weights = True
print("Using reduce_lr_weights: ", reduce_lr_weights)

model.train()

# Define the early stopping threshold and OOM warning
threshold_earlystop = 0.05
max_energy_threshold = 1e6

start_time = time.time()

training_labels = [] 
from collections import Counter

for epoch in range(args.epochs):
    total_loss = 0
    last_loss = 1e10

    if earlystop:
        break

    for idx, (batch, clean) in enumerate(train_loader):
        torch.cuda.empty_cache()
        # training_labels.append(int(batch.y.item()))

        for label in batch.y:
            training_labels.append(int(label.item()))

        try:
            print("Label:", batch.y, "Input Shape:", batch.x.shape)
            model.train()

            batch = batch.to(device)
            history_epoch = model.learning(batch)

            # Append energy values to history
            history["internal_energy_per_epoch"].append(history_epoch["internal_energy_mean"])
            history["sensory_energy_per_epoch"].append(history_epoch["sensory_energy_mean"])

            # Log energy values for this batch/epoch to wandb
            wandb.log({
                "epoch": epoch,
                "internal_energy_mean": history_epoch["internal_energy_mean"],
                "sensory_energy_mean": history_epoch["sensory_energy_mean"]
            })

            model.pc_conv1.restart_activity()

            print(f"------------------ Epoch {epoch}: Batch {idx} ------------------")

            # if internal_energy_mean or sensory_energy_mean is nan or inf, break
            if not np.isfinite(history_epoch["internal_energy_mean"]) or not np.isfinite(history_epoch["sensory_energy_mean"]):
                print("Energy is not finite, stopping training")
                earlystop = True
                break

            # Early stopping based on loss change
            if abs(last_loss - history_epoch["internal_energy_mean"]) < threshold_earlystop:
                earlystop = True
                print(f"EARLY STOPPED at epoch {epoch}")
                print(f"Last Loss: {last_loss}, Current Loss: {history_epoch['internal_energy_mean']}")
                break

            # Early stopping based on high energy
            if history_epoch["internal_energy_mean"] > max_energy_threshold:
                print("energy :", history_epoch["internal_energy_mean"])
                print("Energy too high, stopping training")
                earlystop = True
                break

            if idx >= 20:
                print("Epoch checkpoint reached, saving model...")

                # model_filename = f"model_state_dict_{epoch}.pth"
                # model_path = os.path.join(model_dir, model_filename)
                # torch.save(model.state_dict(), model_path)


                from helper.plot import plot_energy_during_training

                plot_energy_during_training(model.pc_conv1.energy_vals["internal_energy"][:], 
                                            model.pc_conv1.energy_vals["sensory_energy"][:],
                                            history, 
                                            model_dir=model_dir,
                                            epoch=epoch)
                

                # if reduce_lr_weights:
                #     print("Reducing learning rate for weights")
                #     model.pc_conv1.lr_weights = model.pc_conv1.lr_weights / 2
                #     print("using lr_weights: ", model.pc_conv1.lr_weights)

                  # Plot energy per epoch with two y-axes

                # for value in history["internal_energy_per_epoch"]:
                #     wandb.log({"internal_energy_per_epoch": value})

                # for value in history["sensory_energy_per_epoch"]:
                #     wandb.log({"sensory_energy_per_epoch": value})

                break 


        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('WARNING: CUDA ran out of memory, skipping batch...')
                torch.cuda.empty_cache()
                continue
            else:
                torch.cuda.empty_cache()
                raise e

    print(f"Epoch {epoch} / {args.epochs} completed")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds for {args.epochs} epochs")

############ TRAINING DONE ####################

save_path = os.path.join(model_dir, 'parameter_info/weight_matrix_visualization_epoch_End.png')
plot_model_weights(model, GRAPH_TYPE, model_dir=save_path, save_wandb=True)


if args.set_abs_small_w_2_zero:


    w_copy = model.pc_conv1.weights.clone()

    # Define the threshold
    threshold = 0.0001

    print("Thresholding weights with absolute values below", threshold)

    # Apply the threshold: set weights with absolute values below the threshold to zero
    new_w = torch.where(torch.abs(w_copy) < threshold, torch.tensor(0.0, device=w_copy.device), w_copy)

    # Assign the thresholded weights back to the model
    model.pc_conv1.weights.data = new_w

    save_path = os.path.join(model_dir, 'parameter_info/weight_matrix_visualization_epoch_End_remove_small_abs_weights.png')

    plot_model_weights(model, GRAPH_TYPE, model_dir=save_path, save_wandb=True)
else:
    print("Skipping thresholding of small weights")
# if remove_internal_edges:
#     pass 


plot_energy_graphs(model.pc_conv1.energy_vals, model_dir=model_dir, window_size=model.pc_conv1.T)


# Append to the appropriate file based on whether the training crashed or completed successfully
if earlystop:
    print("Stopping program-------")

    # Log in WandB that the run crashed
    wandb.log({"crashed": True})

    # Finish WandB logging first
    try:
        wandb.finish()
    except Exception as e:
        print(f"WandB logging failed: {e}")

    # Now remove the folder to save storage
    import shutil
    shutil.rmtree(model_dir)
    
    print("----------------------------Removed run folder--------------------------- ")
    print(model_dir)
    
    exit()
else:
    wandb.log({"crashed": False})

    element_counts = Counter(training_labels)

    # Log a bar plot to WandB
    wandb.log({"element_counts_bar": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in element_counts.items()], columns=["Label", "Count"]),
        "Label",
        "Count",
        title="Element Counts"
    )})

# If training completed successfully, log to the finished runs file
with open('trained_models/finished_training.txt', 'a') as file:
    file.write(f"{model_dir}\n")

save_path = os.path.join(model_dir, 'parameter_info')
model.save_weights(path=save_path, overwrite=False)

wandb.log({"run_complete": True})
wandb.log({"model_dir": model_dir})

# Save model weights 
######################################################################################################### 
####                                            Evaluation (setup)                                  #####
######################################################################################################### 
 
# device = torch.device('cpu')
from eval_tasks import classification, denoise, occlusion, generation #, reconstruction

num_wandb_img_log = len(custom_dataset_train.numbers_list)   # Number of images to log to wandb
model.pc_conv1.batchsize = 1

### Make dataloader for testing where we take all the digits of the number_list we trained on ###
dataset_params_testing = dataset_params.copy()

if "batch_size" in dataset_params_testing.keys():
    # remove keys 
    del dataset_params_testing["batch_size"]

if "NUM_INTERNAL_NODES" in dataset_params_testing.keys():
    # remove keys 
    del dataset_params_testing["NUM_INTERNAL_NODES"]

dataset_params_testing["edge_index"] = custom_dataset_train.edge_index

dataset_params_testing["mnist_dataset"] = mnist_testset
dataset_params_testing["N"] = "all"
dataset_params_testing["supervision_label_val"] = dataset_params["supervision_label_val"]

# --------------------------------------------
model.trace(values=True, errors=True)
model.pc_conv1.trace_activity_values = True 
model.pc_conv1.trace_activity_preds = True 
model.pc_conv1.batchsize = 1


for key in dataset_params_testing:
    print(key, ":\t ", dataset_params_testing[key])


# CustomGraphDataset params
custom_dataset_test = CustomGraphDataset(graph_params, **dataset_params_testing, 
                                        indices=(num_vertices, sensory_indices, internal_indices, supervision_indices)
                                        )
# dataset_params_testing["batch_size"] = 2

test_loader = DataLoader(custom_dataset_test, batch_size=1, shuffle=True, generator=generator_seed)

from helper.eval import get_clean_images_by_label

clean_images = get_clean_images_by_label(mnist_trainset, num_images=10)

######################################################################################################### 
####                                            Evaluation (tasks)                                  #####
######################################################################################################### 
 
import os
import logging

from helper.log import write_eval_log


test_params = {
    "model_dir": model_dir,
    "T": 300,
    "supervised_learning":True, 
    "num_samples": 5,
    "add_sens_noise": False,
    "num_wandb_img_log": num_wandb_img_log,
}

# model.pc_conv1.lr_values = 0.1 
# model.pc_conv1.lr_values = model_params["lr_params"][0]

MSE_values_occ = occlusion(test_loader, model, test_params)

test_params["add_sens_noise"] = True
MSE_values_occ_noise = occlusion(test_loader, model, test_params)

# After occlusion
eval_data_occlusion = {
    "occlusion": {
        "MSE_values_occ_noise": MSE_values_occ_noise,
        "MSE_values_occ": MSE_values_occ
    }
}
write_eval_log(eval_data_occlusion, model_dir)
######################################################################################################### 

model.pc_conv1.batchsize = 1
test_params = {
    "model_dir": model_dir,
    "T":300,
    "supervised_learning":False, 
    "num_samples": 15,
    "num_wandb_img_log": num_wandb_img_log,
}

# model.pc_conv1.lr_values = 0.1
# model.pc_conv1.lr_values = model_params["lr_params"][0]

y_true, y_pred, accuracy_mean = classification(test_loader, model, test_params)
# Log all the evaluation metrics to wandb
# After classification
eval_data_classification = {
    "classification": {
        "accuracy_mean": accuracy_mean,
        "y_true": y_true,
        "y_pred": y_pred
    }
}
write_eval_log(eval_data_classification, model_dir)


######################################################################################################### 

test_params = {
    "model_dir": model_dir,
    "T": 300,
    "supervised_learning":True, 
    "num_samples": 6,
    "num_wandb_img_log": num_wandb_img_log,
}

# model.pc_conv1.lr_values = 0.1
# model.pc_conv1.lr_values = model_params["lr_params"][0]

MSE_values_denoise_sup = denoise(test_loader, model, test_params)

test_params["supervised_learning"] = False
MSE_values_denoise = denoise(test_loader, model, test_params)

eval_data_denoise = {
    "denoise": {
        "MSE_values_denoise_sup": MSE_values_denoise_sup,
        "MSE_values_denoise": MSE_values_denoise
    }
}
write_eval_log(eval_data_denoise, model_dir)
# MSE_values = denoise(test_loader, model, supervised_learning=True)
# print("MSE_values", MSE_values)
######################################################################################################### 



test_params = {
    "model_dir": model_dir,
    "T": 300,
    "supervised_learning":True, 
    "num_samples": 15,
    "num_wandb_img_log": num_wandb_img_log,
}

# model.pc_conv1.lr_values = 0.1
# model.pc_conv1.lr_values = model_params["lr_params"][0]

model.pc_conv1.trace_activity_values = True 
model.pc_conv1.trace_activity_preds = True 

avg_SSIM_mean, avg_SSIM_max, avg_MSE_mean, avg_MSE_max = generation(test_loader, model, test_params, clean_images, verbose=0)

# After generation
eval_data_generation = {
    "Generation": {
        "avg_SSIM_mean": avg_SSIM_mean,
        "avg_SSIM_max": avg_SSIM_max,
        "avg_MSE_mean": avg_MSE_mean,
        "avg_MSE_max": avg_MSE_max
    }
}
write_eval_log(eval_data_generation, model_dir)

# MSE_values = denoise(test_loader, model, supervised_learning=True)
# print("MSE_values", MSE_values)


print("accuracy_mean", accuracy_mean)

print("model_dir", model_dir)

# write a text file with these 

# Log all the evaluation metrics to wandb
# wandb.log({
#     "Mean_MSE_values_denoise_sup": np.mean(MSE_values_denoise_sup),
#     "Mean_MSE_values_denoise": np.mean(MSE_values_denoise),
#     "Mean_MSE_values_occ_noise": np.mean(MSE_values_occ_noise),
#     "Mean_MSE_values_occ": np.mean(MSE_values_occ),
#     "accuracy_mean": accuracy_mean,
#     "y_true": y_true,
#     "y_pred": y_pred,
#     "avg_SSIM_mean": avg_SSIM_mean,
#     "avg_SSIM_max": avg_SSIM_max,
#     "avg_MSE_mean": avg_MSE_mean,
#     "avg_MSE_max": avg_MSE_max
# })


from datetime import datetime
# Get the current date and time
current_datetime = datetime.now()
# Print the current date and time
print("Current date and time:", current_datetime)

# wandb.log({"energy_sensory": energy["sensory_energy"]})


wandb.finish()
print(f"Training completed in {end_time - start_time:.2f} seconds for {args.epochs} epochs")
