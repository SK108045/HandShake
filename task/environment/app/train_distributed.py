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
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.targets = torch.randn(size, 2)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

def get_peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cpu")
    
    if rank == 0:
        with open("world_size.log", "w") as f:
            f.write(str(world_size))
            
    full_dataset = DummyDataset(100)
    if rank == 0:
        my_data = Subset(full_dataset, range(0, 55))
    else:
        my_data = Subset(full_dataset, range(55, 100))
        
    dataloader = DataLoader(my_data, batch_size=5)
    
    model = SimpleModel().to(device)
    
    check_fn = lambda submodule: isinstance(submodule, nn.Linear)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn
    )
    
    model = FSDP(model, device_id=device, use_orig_params=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    accumulation_steps = 2
    epochs = 2
    total_samples = 0
    
    max_batches = torch.tensor([len(dataloader)], dtype=torch.long, device=device)
    dist.all_reduce(max_batches, op=dist.ReduceOp.MAX)
    max_batches = max_batches.item()
    
    for epoch in range(epochs):
        model.train()
        dataloader_iter = iter(dataloader)
        
        for step in range(max_batches):
            is_active = True
            try:
                inputs, targets = next(dataloader_iter)
                inputs, targets = inputs.to(device), targets.to(device)
                total_samples += inputs.size(0)
            except StopIteration:
                is_active = False
                inputs = torch.zeros(5, 10, device=device)
                targets = torch.zeros(5, 2, device=device)
                
            is_last_batch = (step + 1) == max_batches
            is_accumulating = (step + 1) % accumulation_steps != 0 and not is_last_batch
            
            if not is_accumulating:
                with model.no_sync():
                    outputs = model(inputs)
                    loss = nn.functional.mse_loss(outputs, targets)
                    if not is_active: loss = loss * 0.0
                    loss.backward()
            else:
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                if not is_active: loss = loss * 0.0
                loss.backward()
                
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                
        save_checkpoint(model, optimizer, epoch, rank == 0)
        
    print(f"Rank {rank} processed {total_samples} samples.")
    with open(f"rank_{rank}_samples.txt", "w") as f:
        f.write(str(total_samples))
    
    with open(f"rss_rank_{rank}.log", "w") as f:
        f.write(str(get_peak_rss_mb()))
        
    cleanup()

if __name__ == "__main__":
    main()
