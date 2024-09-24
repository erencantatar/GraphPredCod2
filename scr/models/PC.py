import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import math
from torch_geometric.utils import to_dense_adj, degree
from models.MessagePassing import PredictionMessagePassing, ValueMessagePassing
from helper.activation_func import set_activation

import os


class PCGraphConv(torch.nn.Module): 
    def __init__(self, num_vertices, sensory_indices, internal_indices, 
                 learning_rate, T, graph_structure,
                 batch_size, use_learning_optimizer, weight_init, clamping,
                 supervised_learning=False, normalize_msg=False, debug=False, activation=None, 
                 log_tensorboard=True, wandb_logger=None, device="cpu"):
        super(PCGraphConv, self).__init__()  # 'add' aggregation
        self.num_vertices = num_vertices
        
        
        # these are fixed and inmutable, such that we can copy 
        self.sensory_indices_single_graph = sensory_indices
        self.internal_indices_single_graph = internal_indices
        self.supervised_labels_single_graph = supervised_learning 
        

        # init, but these are going to updated depending on batchsize
        self.sensory_indices = self.sensory_indices_single_graph
        self.internal_indices = self.internal_indices_single_graph
        self.supervised_labels = self.supervised_labels_single_graph 
        
        self.lr_values , self.lr_weights = learning_rate  

        self.T = T  # Number of iterations for gradient descent

        self.debug = debug
        self.edge_index_single_graph = graph_structure  # a geometric graph structure
        self.mode = ""

        self.task = None
        self.device = device
        self.weight_init = weight_init
        self.internal_clock = 0
        self.clamping = clamping
        self.wandb_logger = wandb_logger

        self.trace_activity_values, self.trace_activity_errors = False, False  
        self.trace = {
            "values": [], 
            "errors": [],
         }

        self.energy_vals = {
            "internal_energy": [],
            "sensory_energy": [],
            "supervised_energy": [], 
            "internal_energy_testing": [],
            "sensory_energy_testing": [],
            "supervised_energy_testing": []
        }


        self.use_bias = False 
        self.gradients_minus_1 = 1 # or -1 
        # self.gradients_minus_1 = -1 # or -1 #NEVER 

        print("------------------------------------")
        print(f"gradients_minus_1: x and w.grad += {self.gradients_minus_1} * grad")
        print("------------------------------------")

        self.values_at_t = []
        
        self.TODO = """ 
                    - think about removing self loops
                    - requires_grad=False for values and errors
                    - understand aggr_out_i, aggr_out is the message passed to the update function
                    """
        self.log = {"Zero_pred" : [], }   # Zero prediction error encountered for node {vertex}

        # if self.debug:
        #     ic.enable()
        # else: 
        #     ic.disable()
        # assert num_vertices == len(sensory_indices) + len(internal_indices), "Number of vertices must match the sum of sensory and internal indices"

        self.batchsize = batch_size
        # TODO use torch.nn.Parameter instead of torch.zeros
        
        self.use_optimzers = use_learning_optimizer

        self.grad_accum_method = "mean" 
        assert self.grad_accum_method in ["sum", "mean"]
        
        # Initialize weights with uniform distribution in the range (-k, k)

        # ------------- init weights -------------------------- 

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        import math 
        k = 1 / math.sqrt(num_vertices)

        # USING BATCH SIZE, we want the same edge weights at each subgraph of the batch
        self.weights = torch.nn.Parameter(torch.zeros(self.edge_index_single_graph.size(1), device=self.device))
        # init.uniform_(self.weights.data, -k, k)
        
        if type(self.weight_init) == float:
            # VAL = 0.001
            VAL = weight_init
            print("VAL VS K", VAL, k)
            self.weights.data = torch.full_like(self.weights.data, VAL)

        if self.weight_init == "uniform":
            nn.init.uniform_(self.weights, -k, k)

        self.w_t_min_1 = self.weights.clone().detach()


        # https://chatgpt.com/c/0f9c0802-c81b-40df-8870-3cea4d2fc9b7

        
        # single_graph_weights = torch.nn.Parameter(torch.zeros(edge_index.size(1) // self.batchsize,  device=self.device), requires_grad=False)
        # init.uniform_(single_graph_weights, -k, k)
        # # Repeat weights across the batch
        # self.weights = single_graph_weights.repeat(self.batchsize)

        # single_graph_weights = torch.nn.Parameter(torch.zeros(edge_index.size(1) // self.batchsize, device=self.device))
        # init.uniform_(single_graph_weights, -k, k)
        # # Repeat weights across the batch
        # repeated_weights = single_graph_weights.repeat(self.batchsize)
        # # Convert repeated weights back into a Parameter
        # self.weights = nn.Parameter(repeated_weights)

        # self.weights = torch.nn.Parameter(torch.ones(self.num_vertices, self.num_vertices))
        # self.initialize_weights(init_method="uniform")


        self.values_dummy = torch.nn.Parameter(torch.zeros(self.batchsize * self.num_vertices, device=self.device), requires_grad=True) # requires_grad=False)                
        self.values = None
        self.errors = None
        self.predictions = None  


        if self.use_optimzers:
            weight_decay = self.use_optimzers[0]

            print("------------Using optimizers for values/weights updating ------------")
            # self.optimizer_weights = torch.optim.Adam([self.weights], lr=self.lr_weights, weight_decay=weight_decay) #weight_decay=1e-2)        
            self.optimizer_weights = torch.optim.SGD([self.weights], lr=self.lr_weights)

            # self.optimizer_weights = torch.optim.SGD([self.weights], lr=self.gamma) #weight_decay=1e-2)

            self.optimizer_values = torch.optim.SGD([self.values_dummy], lr=self.lr_values)
            
            self.weights.grad = torch.zeros_like(self.weights)
            self.values_dummy.grad = torch.zeros_like(self.values_dummy)
            
        
        self.effective_learning = {}
        self.effective_learning["w_mean"] = []
        self.effective_learning["w_max"] = []
        self.effective_learning["w_min"] = []


        self.effective_learning["v_mean"] = []
        self.effective_learning["v_max"] = []
        self.effective_learning["v_min"] = []

        # 2. during training set batch.e

        # k = 1.0 / num_vertices ** 0.5
        # init.uniform_(self.weights, -k, k)
            
        # using graph_structure to initialize mask Data(x=x, edge_index=edge_index, y=label)
        # self.mask = self.initialize_mask(graph_structure)
        
        print("normalize_msg", normalize_msg)
        if normalize_msg:

            edge_index_batch = torch.cat([self.edge_index_single_graph for _ in range(batch_size)], dim=1)

            # Compute normalization for the entire batch edge_index, num_nodes, device):
            self.norm = self.compute_normalization(edge_index_batch, self.num_vertices * batch_size, device)

            print(self.edge_index_single_graph.shape)
            print(self.norm.shape)
            
            assert self.norm.shape[0] == self.edge_index_single_graph.shape[1], "Norm shape must match the number of edges"
            # self.norm_single_batch = self.compute_normalization(self.edge_index_single_graph, self.num_vertices, self.device)
            print("-----compute_normalization-----")
        else:
            self.norm_single_batch = torch.ones(self.edge_index_single_graph.size(1))
            # self.norm = self.norm_single_batch.repeat(1, self.batchsize).to(self.device)
            
            
            self.norm = torch.tensor(1)

        self.norm = self.norm.cpu()

        # Apply mask to weights
        # self.weights.data *= self.mask

        self.prediction_mp  = PredictionMessagePassing(activation=activation)
        self.values_mp      = ValueMessagePassing(activation=activation)

        self.set_phase('initialize') # or 'weight_update'
         
        if activation:
            self.f, self.f_prime = set_activation(activation)
            set_activation(activation)
            tmp = f'Activation func set to {activation}'
            self.set_phase(tmp) # or 'weight_update'
            # for now MUST SET Activation function before calling self.prediction_msg_passing
        assert activation, "Activation function not set"

        self.t = ""

        # if activation == "tanh":
        #     print("Since using tanh, using xavier_normal_")
        #     init.xavier_normal_(self.weights)   # or xavier_normal_ / xavier_uniform_
        
        # elif activation == "relu":
        #     # kaiming_uniform_ or kaiming_normal_
        #     print("Since using relu, using kaiming_normal_")
        #     init.kaiming_normal_(self.weights)   # or xavier_normal_ / xavier_uniform_
        # else:
        #     print("Std random weight initialization --> xavier_uniform_")
        #     init.xavier_uniform_(self.weights)   # or xavier_normal_ / xavier_uniform_


    # def initialize_mask(self, edge_index):
    #     """Initialize the mask using the graph_structure's edge_index."""
    #     self.set_phase("Initialize the mask using the graph_structure's edge_index")

    #     # mask = torch.zeros(self.num_vertices, self.num_vertices, dtype=torch.float32, device=self.weights.device)
    #     self.edge_index = edge_index
    #     # print("inint mask", edge_index.shape)

    #     # # Set mask to 1 for edges that exist according to edge_index
    #     # mask[edge_index[0], edge_index[1]] = 1

    #     from torch_geometric.utils import to_dense_adj

    #     # get structure
    #     mask = to_dense_adj(edge_index).squeeze(0) 
        
    #     if self.include_self_connections:
    #         print("Including self connections")
    #         mask = mask.fill_diagonal_(1)
    #     else:
    #         # Set the diagonal to zero to avoid self-connections
    #         mask = mask.fill_diagonal_(0)

    #     mask = mask.to(self.weights.device)
    #     return mask
    
        from torch_geometric.utils import degree

        self.gpu_cntr = 0 
        self.print_GPU = False 

    


    def helper_GPU(self,on=False):
        if on:
            current_memory_allocated = torch.cuda.memory_allocated()
            current_memory_reserved = torch.cuda.memory_reserved()
            print(f"{self.gpu_cntr} current_memory_allocated", current_memory_allocated)
            print(f"{self.gpu_cntr} current_memory_reserved", current_memory_reserved)
            
            self.gpu_cntr += 1 

    def compute_normalization(self, edge_index, num_nodes, device):
        # Calculate degree for normalization
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.int16)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Using sparse matrix for normalization to reduce memory consumption
        # norm = torch.sparse_coo_tensor(edge_index, deg_inv_sqrt[row] * deg_inv_sqrt[col], size=(num_nodes, num_nodes)).to(device)

        return norm


    def initialize_weights(self, init_method):
        if init_method == "uniform":
            init.uniform_(self.weights, -0.1, 0.1)
        elif init_method == "normal":
            init.normal_(self.weights, mean=0, std=0.1)
        elif init_method == "xavier_uniform":
            init.xavier_uniform_(self.weights)
        elif init_method == "xavier_normal":
            init.xavier_normal_(self.weights)
        elif init_method == "kaiming_uniform":
            init.kaiming_uniform_(self.weights, nonlinearity='relu')
        elif init_method == "kaiming_normal":
            init.kaiming_normal_(self.weights, nonlinearity='relu')
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
    # def update(self, aggr_out, x):
    def update_values(self, data):

        # Only iterate over internal indices for updating values
        # for i in self.internal_indices:

        """ 
        Query by initialization: Again, every value node is randomly initialized, but the value nodes of
        specific nodes are initialized (for t = 0 only), but not fixed (for all t), to some desired value. This
        differs from the previous query, as here every value node is unconstrained, and hence free to change
        during inference. The sensory vertices will then converge to the minimum found by gradient descent,
        when provided with that specific initialization. 
        """ 
        self.get_graph()

        # num_nodes, (features)
        weights_batched_graph = self.weights.repeat(1, self.batchsize).to(self.device)
    
        self.helper_GPU(self.print_GPU)


        with torch.no_grad():

            delta_x = self.values_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), weights_batched_graph, norm=self.norm.to(self.device)).squeeze()
        
            delta_x = delta_x.detach()
        

    
        # delta_x = self.values_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.weights)

        # self.values.data[self.nodes_2_update, :] += delta_x[self.nodes_2_update, :]
        # data.x[self.nodes_2_update, 0] += delta_x[self.nodes_2_update, :]


        # if self.use_optimzers:
        #     self.optimizer_values.zero_grad()
        #     if self.values.grad is None:
        #         self.values.grad = torch.zeros_like(self.values)
        #     else:
        #         self.values.grad.zero_()  # Reset the gradients to zero
                
        #     self.values_dummy.grad[self.nodes_2_update] = -delta_x[self.nodes_2_update]
        #     self.optimizer_values.step()
        # else:
        #     self.values_dummy.data[self.nodes_2_update] += self.gamma * delta_x[self.nodes_2_update]


        # self.optimizer_values.zero_grad()
        # if self.values_dummy.grad is None:
        #     self.values_dummy.grad = torch.zeros_like(self.values_dummy)
        # else:
        #     self.values_dummy.grad.zero_()  # Reset the gradients to zero

        # print("a", self.values_dummy.grad.shape)
        # print("b", delta_x.view(self.batchsize, self.num_vertices).shape)    

        # self.data.x[self.nodes_2_update, 0] = self.values_dummy.data.flatten()  # Detach to avoid retaining the computation graph
    


        #     self.values_dummy.grad[self.nodes_2_update] = delta_x.view(self.batchsize, self.num_vertices)[self.nodes_2_update]

        # print(self.values_dummy.grad[:, self.nodes_2_update].shape)
        # print(delta_x.view(self.batchsize, self.num_vertices)[:, self.nodes_2_update].shape)

        # self.values_dummy.grad[self.nodes_2_update] = delta_x[self.nodes_2_update]

                                                                                                                                                                                          
        # torch.Size([2, 1500])
        # torch.Size([2, 1500])
        # torch.Size([1500, 1]) 

        # print(self.values_dummy.grad[:, self.nodes_2_update].shape)
        # print(self.values_dummy.data[:, self.nodes_2_update].shape)
        # print(self.data.x[self.nodes_2_update, 0].shape)

        # self.data.x[self.nodes_2_update, 0] = self.values_dummy.data[:, self.nodes_2_update].detach()  # Detach to avoid retaining the computation graph
        ## GOOD ONE #### 
        
        if self.trace_activity_values:
            self.trace["values"].append(self.data.x[:,0].cpu().detach())

        # https://chatgpt.com/share/54c649d0-e7de-48be-9c00-442bef5a24b8
        # This confirms that the optimizer internally performs the subtraction of the gradient (grad), which is why you should assign theta.grad = grad rather than theta.grad = -grad. If you set theta.grad = -grad, it would result in adding the gradient to the weights, which would maximize the loss instead of minimizing it.

        if self.use_optimzers:
            self.optimizer_values.zero_grad()
            if self.values_dummy.grad is None:
                self.values_dummy.grad = torch.zeros_like(self.values_dummy)
            else:
                self.values_dummy.grad.zero_()  # Reset the gradients to zero
            
            # print("ai ai ")
            self.values_dummy.grad[self.nodes_2_update] = delta_x[self.nodes_2_update]
            self.optimizer_values.step()

            # print(self.data.x[self.nodes_2_update, 0].shape)
            # print(self.values_dummy.data[self.nodes_2_update].shape)
            self.data.x[self.nodes_2_update, 0] = self.values_dummy.data[self.nodes_2_update].unsqueeze(-1).detach()  # Detach to avoid retaining the computation graph
            # self.values[self.nodes_2_update] = self.values_dummy.data
    
        else:
            # self.values_dummy.data[self.nodes_2_update] += self.gamma * delta_x[self.nodes_2_update].detach() 
            self.data.x[self.nodes_2_update, 0] += self.gradients_minus_1 * self.lr_values * delta_x[self.nodes_2_update].unsqueeze(-1).detach()  # Detach to avoid retaining the computation graph
        





            # self.values[self.nodes_2_update] += delta_x[self.nodes_2_update].detach()
     
        # if self.values_dummy.grad is None:
        #     self.values_dummy.grad = torch.zeros_like(self.values_dummy)
        # else:
        #     self.values_dummy.grad.zero_()  # Reset the gradients to zero
            
        # print("-----------------------")
        # print("1", self.values_dummy.grad[self.nodes_2_update].shape)
        # print("2", delta_x[self.nodes_2_update].shape) # KAN NIET
        # print("2", delta_x.shape)   
        
        # print("-----------------------")

        # self.data.x[self.nodes_2_update, 0] += self.lr_values * delta_x[self.nodes_2_update, :].detach()  # Detach to avoid retaining the computation graph
     
        # if self.use_optimzers:
        #     self.optimizer_values.zero_grad()
        #     if self.values_dummy.grad is None:
        #         self.values_dummy.grad = torch.zeros_like(self.values)
        #     else:
        #         self.values_dummy.grad.zero_()  # Reset the gradients to zero
                
        #     self.values_dummy.grad[self.nodes_2_update] = delta_x[self.nodes_2_update]
        #     self.optimizer_values.step()

        #     self.data.x[self.nodes_2_update, 0] = self.values_dummy.data  # Detach to avoid retaining the computation graph
        #     # self.values[self.nodes_2_update] = self.values_dummy.data
    
        # else:
        #     # self.values_dummy.data[self.nodes_2_update] += self.gamma * delta_x[self.nodes_2_update].detach() 

        #     self.data.x[self.nodes_2_update, 0] += self.lr_values * delta_x[self.nodes_2_update, :].detach()  # Detach to avoid retaining the computation graph
        #     # self.values[self.nodes_2_update] += delta_x[self.nodes_2_update].detach()
     
        # old 
        # self.data.x[self.nodes_2_update, 0] += self.lr_values * delta_x[self.nodes_2_update, :].detach()  # Detach to avoid retaining the computation graph
        


        # Calculate the effective learning rate
        effective_lr = self.lr_values * delta_x
        self.effective_learning["v_mean"].append(effective_lr.mean().item())
        self.effective_learning["v_max"].append(effective_lr.max().item())
        self.effective_learning["v_min"].append(effective_lr.min().item())



    def get_predictions(self, data):
        self.get_graph()

        # with a single batch of n items the weights are shared/the same (self.weights.to(self.device))
        weights_batched_graph = self.weights.repeat(1, self.batchsize).to(self.device)

        

        with torch.no_grad():

            self.predictions = self.prediction_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), weights_batched_graph, norm=self.norm.to(self.device))
            # self.predictions = self.prediction_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.weights)

            self.predictions = self.predictions.detach()

        return self.predictions


    def get_graph(self):
        """ 
        Don't need to reset preds/errors/values because we already set them in the dataloader to be zero's
        """
        self.values, self.errors, self.predictions = self.data.x[:, 0], self.data.x[:, 1], self.data.x[:, 2]

        return self.values, self.errors, self.predictions

    def energy(self, data):
        """
        Compute the total energy of the network, defined as:
        E_t = 1/2 * ∑_i (ε_i,t)**2,
        where ε_i,t is the error at vertex i at time t.

        For batching         
        """
        self.get_graph()

        self.helper_GPU(self.print_GPU)


        self.predictions = self.get_predictions(self.data)

     
        
        self.data.x[:, 2] = self.predictions.detach()
        
        # print("predictions shape", self.predictions.shape)

        # self.errors = (self.values.to(self.device) - self.predictions.to(self.device)).squeeze(-1) 
        self.errors = (self.values.to(self.device) - self.predictions.to(self.device)).squeeze(-1).detach()  # Detach to avoid retaining the computation graph
        self.data.x[:, 1] = self.errors.unsqueeze(-1).detach()
        # data.x[self.nodes_2_update, 1] = errors[self.nodes_2_update, :]

       
        energy = {
                "internal_energy": [],
                "supervised_energy": [],
                "sensory_energy":  [],
        }

        energy['internal_energy'] = 0.5 * (self.errors[self.internal_indices] ** 2).sum().item()
        energy['sensory_energy']  = 0.5 * (self.errors[self.sensory_indices] ** 2).sum().item()
        energy['supervised_energy']  = 0.5 * (self.errors[self.supervised_labels] ** 2).sum().item()
        
        if self.mode == "training":

            self.energy_vals["internal_energy"].append(energy["internal_energy"])
            self.energy_vals["sensory_energy"].append(energy["sensory_energy"])
            self.energy_vals["supervised_energy"].append(energy["supervised_energy"])

            if self.wandb_logger:
                self.wandb_logger.log({"energy_internal": energy["internal_energy"]})
                self.wandb_logger.log({"energy_sensory": energy["sensory_energy"]})


        else:

            if self.wandb_logger:
                self.wandb_logger.log({"energy_internal_testing": energy["internal_energy"]})
                self.wandb_logger.log({"energy_sensory_testing": energy["sensory_energy"]})


        return energy


    def set_sensory_nodes(self):
        """ 
        When presented with a training point s̄ taken from a training set, the value nodes of
        the sensory vertices are fixed to be equal to the entries of s̄ for the whole duration of the training
        process, i.e., for every t. 
        """

        print("------------Setting sensory nodes--------------")
        # disable gradient computation, since we are not updating the values of the sensory nodes
        print("all information already in the graph object")
        pass 
        # x: (batch X nodes) X features (pixel values)
        # assert x.shape == [self.num_vertices, 1], f"Shape of x is {x.shape} and num_vertices is {self.num_vertices}"

    def restart_activity(self):

        self.set_phase("Restarting activity (pred/errors/values)")

        # Initialize tensors to zeros without creating new variables where not needed
        with torch.no_grad():
            self.values = torch.zeros(self.data.size(0), device=self.device) if self.values is None else self.values.zero_()
            self.errors = torch.zeros(self.data.size(0), device=self.device) if self.errors is None else self.errors.zero_()
            self.predictions = torch.zeros(self.data.size(0), device=self.device) if self.predictions is None else self.predictions.zero_()

            # Use in-place operation for values_dummy to reduce memory overhead
            self.values_dummy.data.zero_()  # Zero out values_dummy without creating a new tensor

        # Reset optimizer gradients if needed
        if self.use_optimzers:
            self.optimizer_values.zero_grad()
            self.optimizer_weights.zero_grad()

    def inference(self, data, restart_activity=True):

        """      During the inference phase, the weights are fixed, and the value nodes are continuously updated via gradient descent for T iterations, where T is
        a hyperparameter of the model. The update rule is the following (inference) (Eq. 3)
        
        First, get the aggregated messages for each node
        This could involve a separate function or mechanism to perform the message aggregation
        For simplicity, let's assume you have a function get_aggregated_messages that does this

        Update values as per Eq. (3)
        self.set_sensory_nodes(data.x)

        if restart_activity:
            self.restart_activity()
        """

        assert self.mode in ['training', 'testing', 'classification'], "Mode not set, (training or testing / classification )"


        # self.edge_weights = self.extract_edge_weights(edge_index=self.edge_index, weights=self.weights, mask=self.mask)
        self.data = data
        # self.values, _pred_ , self.errors, = data.x[:, 0], data.x[:, 1], data.x[:, 2]

        self.helper_GPU(self.print_GPU)


        self.get_graph()

        self.helper_GPU(self.print_GPU)



        energy = self.energy(self.data)

        from tqdm import tqdm

        t_bar = tqdm(range(self.T), leave=False)

        # t_bar.set_description(f"Total energy at time 0 {energy} Per avg. vertex {energy['internal_energy'] / len(self.internal_indices)}")
        t_bar.set_description(f"Total energy at time 0 {energy}")
        # print(f"Total energy at time 0", energy, "Per avg. vertex", energy["internal_energy"] / len(self.internal_indices))

        for t in t_bar:
            
            # aggr_out = self.forward(data)
            self.t = t 

            self.update_values(self.data)
            
            energy = self.energy(self.data)
            t_bar.set_description(f"Total energy at time {t+1} / {self.T} {energy},")
            
        if self.mode == "train":
            self.restart_activity()

        return True

    def set_mode(self, mode, task=None):
        """
        Setting which nodes to update their values based on the mode (training or testing) and task (classification, generation, reconstruction, etc.)
        """

        self.mode = mode
        self.task = task

        if self.task is None:
            assert self.mode == "training", "Task must be set for testing mode"
        assert self.mode in ['training', 'testing', 'classification'], "Mode not set, (training or testing / classification )"

        # Base indices (for a single graph instance)
        base_sensory_indices = list(self.sensory_indices_single_graph)
        base_internal_indices = list(self.internal_indices_single_graph)
        base_supervised_labels = list(self.supervised_labels_single_graph) if self.supervised_labels else []

        # Extended indices to account for the batch size
        self.sensory_indices_batch = []
        self.internal_indices_batch = []
        self.supervised_labels_batch = []

        for i in range(self.batchsize):
            self.sensory_indices_batch.extend([index + i * self.num_vertices for index in base_sensory_indices])
            self.internal_indices_batch.extend([index + i * self.num_vertices for index in base_internal_indices])
            if base_supervised_labels:
                self.supervised_labels_batch.extend([index + i * self.num_vertices for index in base_supervised_labels])

        print("vertix", self.num_vertices)
        print("before after", len(base_sensory_indices), len(self.sensory_indices_batch))
        print("before after", len(base_internal_indices), len(self.internal_indices_batch))
        print("before after", len(base_supervised_labels), len(self.supervised_labels_batch))

        self.sensory_indices   = self.sensory_indices_batch
        self.internal_indices  = self.internal_indices_batch
        self.supervised_labels = self.supervised_labels_batch

        if self.mode == "training":
            # Update only the internal nodes during training
            self.nodes_2_update_base = self.internal_indices_batch

        elif self.mode == "testing":
            assert self.task in ["classification", "generation", "reconstruction", "denoising", "Associative_Memories"], \
                "Task not set, (generation, reconstruction, denoising, Associative_Memories)"

            if self.task == "classification":
                # Update both the internal and supervised nodes during classification
                self.nodes_2_update_base = self.internal_indices_batch + self.supervised_labels_batch

            elif self.task in ["generation", "reconstruction", "denoising"]:
                # Update both the internal and sensory nodes during these tasks
                self.nodes_2_update_base = self.internal_indices_batch + self.sensory_indices_batch

        # Ensure nodes_2_update are expanded to include all batch items
        self.nodes_2_update = self.nodes_2_update_base

        assert self.nodes_2_update, "No nodes selected for updating"

        print(f"-------------mode {self.mode}--------------")
        print(f"-------------task {self.task}--------------")


        
    def set_phase(self, phase):
        self.phase = phase
        print(f"-------------{self.phase}--------------")
        
    def learning(self, data):

        self.helper_GPU(self.print_GPU)


        self.data = data

        # random inint value of internal nodes
        data.x[:, 0][self.internal_indices] = torch.rand(data.x[:, 0][self.internal_indices].shape).to(self.device)

        # random inint errors of internal nodes
        # data.x[:, 1][self.internal_indices] = torch.rand(data.x[:, 1][self.internal_indices].shape).to(self.device)



        x, self.edge_index = self.data.x, self.data.edge_index


        
        # edge_index: has shape [2, E * batch] where E is the number of edges, but in each batch the edges are the same

        # 1. fix value nodes of sensory vertices to be 
        # self.restart_activity()
        # self.set_sensory_nodes()
        
        self.helper_GPU(self.print_GPU)


        ## 2. Then, the total energy of Eq. (2) is minimized in two phases: inference and weight update. 
        ## INFERENCE: This process of iteratively updating the value nodes distributes the output error throughout the PC graph. 
        self.set_phase('inference')
        self.inference(self.data)
        self.set_phase('inference done')

        ## WEIGHT UPDATE 
        self.set_phase('weight_update')
        self.weight_update(self.data)
        self.set_phase('weight_update done')

        energy = self.energy(self.data)
        
        self.helper_GPU(self.print_GPU)

    def weight_update(self, data):

        self.get_graph()                
        # self.values, self.errors, self.predictions, = self.data.x[:, 0], self.data.x[:, 1], self.data.x[:, 2]
        
        errors = self.errors.squeeze().detach() 
        f_x    = self.f(self.values).squeeze().detach()  #* self.mask  # * self.mask

        print(errors.shape, f_x.shape)

        print("Errors / max, mean", errors.max(), errors.mean())
        print("f_x / max, mean", f_x.max(), f_x.mean())

        # self.delta_w = self.alpha * torch.einsum('i,j->ij', self.f_x_j_T, self.error_i_T)  * self.mask  # * self.mask

        print(errors.shape, f_x.shape)
        # this gets the delta_w for all possible edges (even non-existing edge in the graph) (assumes fully connected)
        # self.delta_w = torch.einsum('i,j->ij', errors, f_x )    #.view_as(self.weights)  #* self.mask  # * self.mask

        # Gather the indices of the source and target nodes for each edge
        source_nodes = self.edge_index[0]   # all i's 
        target_nodes = self.edge_index[1]   # all j's 

        # Gather the corresponding errors and f_x values for each edge
        # print("source_nodes shape", source_nodes.shape)
        # print("errors shape", errors.shape)
        # print("f_x shape", f_x.shape)
        source_errors = errors[source_nodes].detach()    # get all e_i's 
        target_fx = f_x[target_nodes].detach()           # get all f_x_j's 

        # print("TEST WEIGHTS", errors.shape, source_nodes.shape)
        # Calculate delta_w in a vectorized manner
        delta_w_batch = source_errors * target_fx
        # print("TEST delta_w_batch", delta_w_batch.shape)

        delta_w = delta_w_batch.reshape(self.batchsize, self.edge_index_single_graph.size(1)) 
        # print("self.delta_w shape", delta_w.shape)

        if self.grad_accum_method == "sum":
            delta_w = delta_w.sum(0).detach()
        if self.grad_accum_method == "mean":
            delta_w = delta_w.mean(0).detach()
        
        # print("self.delta_w shape", delta_w.shape)

        if self.use_optimzers:
            
            # self.optimizer_weights.zero_grad()             self.optimizer_values.grad = torch.zeros_like(self.values.grad)  # .zero_grad()
            # self.weights.grad = torch.zeros_like(self.weights.grad)  # .zero_grad()
            self.optimizer_weights.zero_grad()
            self.weights.grad = delta_w
            self.optimizer_weights.step()


            # self.optimizer_weights.zero_grad()
            # # Accumulate gradients
            # self.weights.backward(delta_w)
            # # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.weights, max_norm=1.0)
            # # Update weights
            # self.optimizer_weights.step()

        else:
            print(self.lr_weights, delta_w.shape)
            print(self.weights.data.shape)

            self.weights.data += self.gradients_minus_1 * (self.lr_weights * delta_w)
            # self.weights.data += self.lr_weights * self.delta_w
        
            # self.weights.data = (1 - self.damping_factor) * self.w_t_min_1 + self.damping_factor * (self.weights.data)
        
            self.w_t_min_1 = self.weights.data.clone()
        
        # Calculate the effective learning rate
        effective_lr = self.lr_weights * delta_w


        self.effective_learning["w_mean"].append(effective_lr.mean().item())
        self.effective_learning["w_max"].append(effective_lr.max().item())
        self.effective_learning["w_min"].append(effective_lr.min().item())

        print("leeen", len(self.effective_learning["w_mean"]))

        ## clamp weights to be above zero
        print("--------------------CLAMP THE WEIGHTS TO BE ABOVE ZERO-----------")
        # self.weights.data = torch.clamp(self.weights.data, min=0)

        print("----------------NO CLAMPING----------------")

    


class PCGNN(torch.nn.Module):
    def __init__(self, num_vertices, sensory_indices, internal_indices, 
                 lr_params, T, graph_structure, 
                 batch_size, 
                 use_learning_optimizer=False, weight_init="xavier", clamping=None, supervised_learning=False, 
                 normalize_msg=False, 
                 debug=False, activation=None, log_tensorboard=True, wandb_logger=None, device='cpu'):
        super(PCGNN, self).__init__()
        
        """ TODO: in_channels, hidden_channels, out_channels, """
        # INSIDE LAYERS CAN HAVE PREDCODING - intra-layer 
        self.pc_conv1 = PCGraphConv(num_vertices, sensory_indices, internal_indices, 
                                    lr_params, T, graph_structure, 
                                    batch_size, use_learning_optimizer, weight_init, clamping, supervised_learning, 
                                    normalize_msg, 
                                    debug, activation, log_tensorboard, wandb_logger, device)

        self.original_weights = None  # Placeholder for storing the original weights


    def log():
        pass
    
    def learning(self, batch):       
        
        self.pc_conv1.mode = "training"
        self.pc_conv1.learning(batch)
        
        history = {
            "internal_energy_mean": np.mean(self.pc_conv1.energy_vals["internal_energy"]),
            "sensory_energy_mean": np.mean(self.pc_conv1.energy_vals["sensory_energy"]),
            }
        
        return history

    def update_graph(self, new_edge_index, new_internal_indices, mode="training", task=None):
        """
        Update the graph structure (edge_index), internal indices, and associated parameters
        like edge weights and batch-specific indices, given the addition/removal of nodes.
        
        Parameters:
        - new_edge_index (torch.Tensor): The updated edge index defining the new graph structure.
        - new_internal_indices (list or torch.Tensor): The updated list of internal node indices.
        - mode (str): The mode for the graph ('training' or 'testing').
        - task (str): The task to perform (e.g., 'classification', 'generation', etc.).
        
        TODO = 
        - Extend the weights 
        - Remove the weights (Later)
        - testing edge_index shape  
        - ... think about it 
        """

        # Step 1: Update internal indices
        self.pc_conv1.internal_indices_single_graph = new_internal_indices
        self.pc_conv1.num_vertices = len(self.pc_conv1.sensory_indices_single_graph) + len(self.pc_conv1.internal_indices_single_graph)

        # If in supervised mode, add the number of supervision nodes to num_vertices
        if self.pc_conv1.supervised_labels_single_graph:
            self.pc_conv1.num_vertices += len(self.pc_conv1.supervised_labels_single_graph)

        # Step 2: Extend or shrink the weights to match the new edge index size
        old_num_edges = self.pc_conv1.edge_index_single_graph.size(1)
        new_num_edges = new_edge_index.size(1)

        if new_num_edges > old_num_edges:
            # Add new weights for the added edges
            additional_weights = torch.zeros(new_num_edges - old_num_edges, device=self.pc_conv1.device)
                    
            self.pc_conv1.weights = torch.nn.Parameter(torch.cat([self.pc_conv1.weights, additional_weights], dim=0))

        elif new_num_edges < old_num_edges:
            raise NotImplementedError
        #     # Truncate the weights to match the new number of edges
        #     self.pc_conv1.weights = self.pc_conv1.weights[:new_num_edges]

        # Step 3: Update the internal graph structure (edge_index)
        self.pc_conv1.edge_index_single_graph = new_edge_index

        # Step 4: Recompute normalization if necessary
        if hasattr(self.pc_conv1, 'normalize_msg') and self.pc_conv1.normalize_msg:
            # Recompute the normalization for the new graph
            self.pc_conv1.norm = self.pc_conv1.compute_normalization(new_edge_index, self.pc_conv1.num_vertices, self.pc_conv1.device)

        # Step 5: Reuse set_mode to recalculate batch-specific indices
        self.pc_conv1.set_mode(mode=mode, task=task)

        # Step 6: Resize tensors that depend on num_vertices (e.g., values_dummy)
        self.pc_conv1.values_dummy = torch.nn.Parameter(torch.zeros(self.pc_conv1.batchsize * self.pc_conv1.num_vertices, device=self.pc_conv1.device), requires_grad=True)

        print(f"Updated graph structure with {new_num_edges} edges and {self.pc_conv1.num_vertices} vertices.")


    def trace(self, values=False, errors=False):
        
        self.pc_conv1.trace = {
            "values": [], 
            "errors": [],
         }
    
        if values:
            self.pc_conv1.trace_activity_values = True 
            
        if errors:
            self.pc_conv1.trace_activity_errors = True  


    def Disable_connection(self, from_indices, to_indices):
        """
        Temporarily disable connections between specified nodes by setting their weights to zero.

        Parameters:
        - from_indices: list of node indices from which connections originate.
        - to_indices: list of node indices to which connections lead.
        """
        if self.original_weights is None:
            # Make a copy of the original weights the first time a connection is disabled
            self.original_weights = self.pc_conv1.weights.clone()

        masks = []
        for from_idx in from_indices:
            for to_idx in to_indices:
                # Find the corresponding edge in the graph
                edge_mask = (self.pc_conv1.edge_index_single_graph[0] == from_idx) & \
                            (self.pc_conv1.edge_index_single_graph[1] == to_idx)
                # Temporarily set the weights of these edges to zero
                masks.append(edge_mask)
                self.pc_conv1.weights.data[edge_mask] = 0
        return masks

    def enable_all_connections(self):
        """
        Restore the original weights for all connections that were disabled.
        """
        if self.original_weights is not None:
            self.pc_conv1.weights.data = self.original_weights
            self.original_weights = None  # Clear the backup after restoration

    def retrieve_connection_strength(self, from_group, to_group):
        """
        Retrieve the connection strengths (weights) between two specific groups of nodes.

        Parameters:
        - from_group: list of node indices from which connections originate (e.g., sensory_indices).
        - to_group: list of node indices to which connections lead (e.g., supervision_label_indices).

        Returns:
        - connection_strengths: A dictionary with keys as tuples (from_idx, to_idx) and values as the corresponding weights.
        """
        connection_strengths = {}
        
        for from_idx in from_group:
            for to_idx in to_group:
                # Find the corresponding edge in the graph
                edge_mask = (self.pc_conv1.edge_index_single_graph[0] == from_idx) & \
                            (self.pc_conv1.edge_index_single_graph[1] == to_idx)
                # Retrieve the connection weight for this edge
                connection_weights = self.pc_conv1.weights[edge_mask]
                
                if connection_weights.numel() > 0:  # If there is a connection
                    connection_strengths[(from_idx, to_idx)] = connection_weights.item()

        return connection_strengths


    def load_weights(self, W, graph, b=None):

        print("Settng weights of self.pc_conv1")
        self.pc_conv1.weights = W 
        self.pc_conv1.edge_index = graph 

        if self.pc_conv1.use_bias:
            self.pc_conv1.bias = b  

        self.pc_conv1.values = torch.zeros(self.num_vertices,self.batch_size,device=self.device) # requires_grad=False)                

    
    def save_weights(self, path):
        

        # make dir if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        W = self.pc_conv1.weights
        graph = self.pc_conv1.edge_index

        # save to '"trained_models/weights.pt"' 
        torch.save(W, f"{path}/weights.pt")
        torch.save(graph, f"{path}/graph.pt")

        if self.pc_conv1.use_bias:
            b = self.pc_conv1.biases 
            torch.save(b, f"{path}/bias.pt")


    def query(self, method, data=None):
        
        print("Random init values of all internal nodes")

        data.x[:, 0][self.pc_conv1.internal_indices] = torch.rand(data.x[:, 0][self.pc_conv1.internal_indices].shape).to(self.pc_conv1.device)


        self.pc_conv1.energy_vals["internal_energy_testing"] = []
        self.pc_conv1.energy_vals["sensory_energy_testing"] = []

        assert self.pc_conv1.mode == "testing"
        assert data is not None, "We must get the labels (conditioning) or the image data (initialization)"

        self.pc_conv1.set_phase('---reconstruction---')
        print("TASK IS", self.pc_conv1.task)

        if method == "query_by_initialization":
            if self.pc_conv1.task == "denoising":
                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='initialization')
                self.pc_conv1.inference()
            elif self.pc_conv1.task == "reconstruction":
                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='initialization')
                self.pc_conv1.inference()
            elif self.pc_conv1.task == "occlusion":
                
                # x.view(-1)[i]
                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='initialization')
                self.pc_conv1.inference()
            
                print("TODO")
            else:
                raise Exception(f"Unknown task given {method}")
        
        elif method == "query_by_conditioning":
            assert len(self.pc_conv1.supervised_labels) > 0, "Can't do conditioning on labels without labels"

            '''            
            While each value node is randomly re-initialized, the value nodes of
            specific vertices are fixed to some desired value, and hence not allowed to change during the energy
            minimization process. The unconstrained sensory vertices will then converge to the minimum of the
            energy given the fixed vertices, thus computing the conditional expectation of the latent vertices given
            the observed stimulus.
            '''

            if self.pc_conv1.task == "classification":
                # classification, where internal nodes are fixed to the pixels of an image, and the sensory nodes are
                # ... fixed to a 1-hot vector with the labels

                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='conditioning')

            else:
                # conditioning model on the label 
                one_hot = torch.zeros(10)
                one_hot[data.y] = 1
                # one_hot = one_hot.view(-1, 1)
                one_hot = one_hot.to(self.device)
                self.pc_conv1.values.data[self.pc_conv1.supervised_labels] = one_hot
                
                if self.pc_conv1.task == "generation":
                    # here the single value node encoding the class information is fixed, 
                    # .. and the value nodes of the sensory nodes converge to an image of that clas
                    print()
                elif self.pc_conv1.task == "reconstruction":
                    
                    self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='conditioning')

                    # such as image completion, where a fraction of the sensory nodes are fixed to # the available pixels of an image, 
                else:
                    raise Exception(f"unkown task given {method}")
        elif method == "pass":
            print("Pass")
        else:
            raise Exception(f"unkown method: {method}")

        
        self.inference(data)
        
        print("QUery by condition or query by init")

        return self.pc_conv1.get_graph()
        
   
    
    def inference(self, data):
        self.pc_conv1.inference(data)
        print("Inference completed.")
        return True
 