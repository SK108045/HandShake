import os
import torch
import torch.distributed as dist

def save_checkpoint(model, optimizer, epoch, is_rank_0):
    """Saves checkpoint. Includes a barrier to ensure all ranks wait for I/O."""
    if is_rank_0:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Using FULL_STATE_DICT requires gathering on rank 0
        state_dict = model.state_dict()
        torch.save({'model_state': state_dict}, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
        print(f"Rank 0 saved checkpoint for epoch {epoch}")
        
        # BUG 1: Deadlock! Barrier inside rank 0 check
        dist.barrier()
