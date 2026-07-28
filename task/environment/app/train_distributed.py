import os
import resource
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
)
from utils.checkpointing import save_checkpoint

def setup():
    dist.init_process_group("gloo")

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = []
        in_dim = 10
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.targets = torch.randn(size, 1)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

def get_peak_rss_mb():
    # Track peak RSS in MB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    # Fixed seeds
    torch.manual_seed(42)
    # torch.use_deterministic_algorithms(True)
    
    setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cpu")
    
    # Record world size for the verifier
    if rank == 0:
        with open("world_size.log", "w") as f:
            f.write(str(world_size))
            
    # Manually split dataset unevenly.
    # Do NOT use DistributedSampler's drop_last=False padding anywhere.
    # The uneven per-rank data split requires structural handling.
    full_dataset = DummyDataset(100)
    if rank == 0:
        my_data = Subset(full_dataset, range(0, 55))
    else:
        my_data = Subset(full_dataset, range(55, 100))
        
    dataloader = DataLoader(my_data, batch_size=5)
    
    model = SimpleModel().to(device)
    
    # Apply activation checkpointing
    check_fn = lambda submodule: isinstance(submodule, nn.Linear)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn
    )
    
    # Wrap FSDP
    model = FSDP(model, device_id=device, use_orig_params=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    accumulation_steps = 2
    
    epochs = 2
    total_samples = 0
    
    for epoch in range(epochs):
        model.train()
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total_samples += inputs.size(0)
            
            is_accumulating = (step + 1) % accumulation_steps != 0
            
            # BUG 2: Incorrect no_sync usage combined with activation checkpointing & accumulation.
            # In FSDP, combining no_sync with activation checkpointing can cause accumulated
            # gradients to be dropped/overwritten. 
            if is_accumulating:
                with model.no_sync():
                    outputs = model(inputs)
                    loss = nn.functional.mse_loss(outputs, targets)
                    loss.backward()
            else:
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
        # Handle remaining gradients (if dataset length not divisible by accumulation_steps)
        if (step + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Save checkpoint (BUG 1: Deadlock inside here)
        save_checkpoint(model, optimizer, epoch, rank == 0)
        
    print(f"Rank {rank} processed {total_samples} samples.")
    
    # Write Peak RSS to file for verifier
    with open(f"rss_rank_{rank}.log", "w") as f:
        f.write(str(get_peak_rss_mb()))
        
    cleanup()

if __name__ == "__main__":
    main()
