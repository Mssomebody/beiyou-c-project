"""
FedProx Server for Federated Learning
Aggregates client updates using FedAvg or weighted average
"""

import torch
import copy
import numpy as np
from collections import OrderedDict


class FedProxServer:
    """
    FedProx Server for aggregating client models
    """
    
    def __init__(self, global_model, clients, device, aggregation='fedavg'):
        """
        Args:
            global_model: Global model (initial)
            clients: Dictionary of clients {node_id: {'client': client, ...}}
            device: Device
            aggregation: 'fedavg' or 'weighted'
        """
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.aggregation = aggregation
        self.round_losses = []
        
    def aggregate(self, client_weights, client_sizes):
        """
        Aggregate client weights
        
        Args:
            client_weights: List of client model state_dicts
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated global model weights
        """
        total_samples = sum(client_sizes)
        
        if self.aggregation == 'fedavg':
            # Simple average (all clients equal weight)
            avg_weights = OrderedDict()
            
            for key in client_weights[0].keys():
                avg_weights[key] = torch.zeros_like(client_weights[0][key])
                for w in client_weights:
                    avg_weights[key] += w[key]
                avg_weights[key] /= len(client_weights)
                
        else:  # weighted by dataset size
            avg_weights = OrderedDict()
            
            for key in client_weights[0].keys():
                avg_weights[key] = torch.zeros_like(client_weights[0][key])
                for w, size in zip(client_weights, client_sizes):
                    avg_weights[key] += w[key] * (size / total_samples)
        
        return avg_weights
    
    def federated_round(self, local_epochs=5, lr=0.001):
        """
        Perform one round of federated training
        
        Args:
            local_epochs: Number of local epochs per client
            lr: Learning rate
            
        Returns:
            Round loss (average of client losses)
        """
        client_updates = []
        client_sizes = []
        client_losses = []
        
        # Get global model weights
        global_weights = copy.deepcopy(self.global_model.state_dict())
        
        # Train each client locally
        for node_id, client_info in self.clients.items():
            client = client_info['client']
            
            # Get dataset size for weighting
            dataset_size = len(client.dataloader.dataset)
            client_sizes.append(dataset_size)
            
            # Local training
            updated_weights, losses = client.local_train(
                global_model=self.global_model,
                local_epochs=local_epochs,
                lr=lr
            )
            
            client_updates.append(updated_weights)
            client_losses.append(losses[-1])  # Last epoch loss
            
            print(f"    Client {node_id}: loss = {losses[-1]:.6f}, size = {dataset_size}")
        
        # Aggregate updates
        avg_weights = self.aggregate(client_updates, client_sizes)
        
        # Update global model
        self.global_model.load_state_dict(avg_weights)
        
        # Calculate average round loss
        round_loss = np.mean(client_losses)
        self.round_losses.append(round_loss)
        
        return round_loss, client_losses
    
    def evaluate_global(self, test_loaders):
        """
        Evaluate global model on all clients' test sets
        
        Args:
            test_loaders: Dictionary of test loaders {node_id: dataloader}
            
        Returns:
            Average test loss across clients
        """
        self.global_model.eval()
        criterion = torch.nn.MSELoss()
        
        total_loss = 0.0
        total_samples = 0
        
        for node_id, client_info in self.clients.items():
            test_loader = test_loaders[node_id]
            client_loss = client_info['client'].evaluate(test_loader)
            
            dataset_size = len(test_loader.dataset)
            total_loss += client_loss * dataset_size
            total_samples += dataset_size
        
        return total_loss / total_samples
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'round_losses': self.round_losses
        }, filepath)
        print(f"✅ Checkpoint saved: {filepath}")