You are tasked with fixing a PyTorch Fully Sharded Data Parallel (FSDP) training script that is plagued by three distinct bugs. The script is located at `/app/train_distributed.py`.

The script attempts to train a simple Multi-Layer Perceptron (MLP) on a distributed setup with 2 ranks, accumulating gradients, saving FSDP checkpoints, and using Activation Checkpointing (AC). However, it fails or diverges due to the following interconnected issues:

1. **Checkpointing Deadlock:** The FSDP state dict collection deadlocks because of improper collective barrier logic inside the checkpointing function.
2. **Silent Loss Divergence:** The script accumulates gradients using the `no_sync` context manager but does so incorrectly. Gradients are being completely dropped instead of accumulated on accumulation steps, causing the loss to silently diverge and converge to the wrong weights mathematically.
3. **Straggler Deadlock:** The dataset is unevenly split between the two ranks (110 samples vs 90 samples). Because PyTorch collective operations (like FSDP forwards and backwards) require all ranks to participate, Rank 1 finishes its batches early and exits, while Rank 0 hangs indefinitely waiting for Rank 1 to join the final batch's collective operations.

Your goal is to fix `/app/train_distributed.py` so that it trains cleanly without crashing, deadlocking, or corrupting the mathematical correctness of the gradients.

### Requirements:
- You must run the training script using: `cd /app && torchrun --nproc_per_node=2 train_distributed.py`.
- **CRITICAL:** The data loader iteration must remain genuine. You must solve the straggler deadlock by forcing the shorter rank to "shadow" the longer rank's collective operations with dummy inputs that do not affect the gradients.
- Your final output must generate the checkpoint at `/app/checkpoints/checkpoint_epoch_1.pt`.
- Your final output must successfully write the sample count files `/app/rank_0_samples.txt` and `/app/rank_1_samples.txt`, which must contain exactly `110` and `90` respectively.
