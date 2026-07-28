import os
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms.join import Join
import torch.distributed as dist

def setup():
    dist.init_process_group("gloo")

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
    def forward(self, x):
        return self.net(x)

def main():
    setup()
    rank = int(os.environ["RANK"])
    
    device = torch.device("cpu")
    model = SimpleModel().to(device)
    model = FSDP(model, device_id=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Uneven inputs: Rank 0 gets 5 batches, Rank 1 gets 3 batches
    num_batches = 5 if rank == 0 else 3
    
    print(f"Rank {rank} starting with {num_batches} batches.")
    
    try:
        with Join([model]):
            for i in range(num_batches):
                inputs = torch.randn(2, 10)
                outputs = model(inputs)
                loss = outputs.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"Rank {rank} completed batch {i+1}")
                
        print(f"Rank {rank} finished Join context successfully.")
    except Exception as e:
        print(f"Rank {rank} failed with exception: {e}")
        
    cleanup()

if __name__ == "__main__":
    main()
