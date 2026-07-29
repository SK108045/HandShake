import os
import torch
import torch.distributed as dist

def save_checkpoint(model, optimizer, epoch, is_rank_0):
    """Saves checkpoint. Includes a barrier to ensure all ranks wait for I/O."""
    # In FSDP, state_dict() is a collective operation (it gathers shards from all ranks).
    # ALL ranks must call it, not just rank 0!
    state_dict = model.state_dict()
    
    if is_rank_0:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({'model_state': state_dict}, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
        print(f"Rank 0 saved checkpoint for epoch {epoch}")
        
    # FIX 1: Barrier must be outside the rank 0 check so all ranks hit it!
    dist.barrier()
