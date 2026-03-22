"""
FedProx Client for Federated Learning
Each client trains locally with proximal term
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from torch.utils.data import DataLoader


class FedProxClient:
    """
    FedProx Client with local training
    Adds proximal term to loss: mu/2 * ||w - w_global||^2
    """
    
    def __init__(self, client_id, model, dataloader, device, mu=0.01):
        """
        Args:
            client_id: Client identifier
            model: Local model (same architecture as global)
            dataloader: DataLoader for this client's data
            device: 'cuda' or 'cpu'
            mu: Proximal term coefficient (0 = FedAvg)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.mu = mu
        self.criterion = nn.MSELoss()
        
    def local_train(self, global_model, local_epochs=5, lr=0.001):
        """
        Train locally with proximal term
        
        Args:
            global_model: Global model weights (for proximal term)
            local_epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            Updated model weights
        """
        # Copy global model for proximal term
        global_weights = copy.deepcopy(global_model.state_dict())
        
        # Local optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Local training
        self.model.train()
        losses = []
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for x, y in self.dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(x)
                
                # Standard loss
                loss = self.criterion(output, y)
                
                # Proximal term: mu/2 * ||w - w_global||^2
                if self.mu > 0:
                    proximal_loss = 0.0
                    for param, global_param in zip(self.model.parameters(), global_weights.values()):
                        proximal_loss += torch.sum((param - global_param) ** 2)
                    loss += (self.mu / 2) * proximal_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
        
        # Return updated weights
        return copy.deepcopy(self.model.state_dict()), losses
    
    def evaluate(self, dataloader=None):
        """
        Evaluate client on local data
        
        Args:
            dataloader: Optional separate dataloader (e.g., test set)
            
        Returns:
            Loss value
        """
        if dataloader is None:
            dataloader = self.dataloader
            
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        
        return total_loss / total_samples


def create_clients(node_ids, train_loaders, val_loaders, test_loaders, 
                   model_class, model_params, device, mu=0.01):
    """
    Create FedProx clients for multiple nodes
    
    Args:
        node_ids: List of node IDs
        train_loaders: Dictionary of train loaders {node_id: dataloader}
        val_loaders: Dictionary of validation loaders
        test_loaders: Dictionary of test loaders
        model_class: Model class
        model_params: Model parameters dict
        device: Device
        mu: Proximal coefficient
        
    Returns:
        Dictionary of clients
    """
    clients = {}
    
    for node_id in node_ids:
        # Create model for this client
        model = model_class(**model_params).to(device)
        
        # Create client
        client = FedProxClient(
            client_id=node_id,
            model=model,
            dataloader=train_loaders[node_id],
            device=device,
            mu=mu
        )
        
        clients[node_id] = {
            'client': client,
            'val_loader': val_loaders[node_id],
            'test_loader': test_loaders[node_id]
        }
    
    return clients