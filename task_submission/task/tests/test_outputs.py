import os
import torch
import pytest

BASELINE_PATH = "/tests/baseline_checkpoints/checkpoint_epoch_1.pt"
AGENT_PATH = "/app/checkpoints/checkpoint_epoch_1.pt"
RANK_0_SAMPLES = "/app/rank_0_samples.txt"
RANK_1_SAMPLES = "/app/rank_1_samples.txt"

def test_samples_processed_exactly():
    """Verify that the agent did not cheat by repeating data (must process exactly 110 and 90)."""
    assert os.path.exists(RANK_0_SAMPLES), f"Missing {RANK_0_SAMPLES}"
    assert os.path.exists(RANK_1_SAMPLES), f"Missing {RANK_1_SAMPLES}"
    
    with open(RANK_0_SAMPLES, "r") as f:
        rank_0 = int(f.read().strip())
    with open(RANK_1_SAMPLES, "r") as f:
        rank_1 = int(f.read().strip())
        
    assert rank_0 == 110, f"Rank 0 processed {rank_0} samples, expected exactly 110."
    assert rank_1 == 90, f"Rank 1 processed {rank_1} samples, expected exactly 90."

def test_model_weights_match_baseline():
    """Verify that the model weights match the baseline to prove Bug 2 (silent divergence) was fixed."""
    assert os.path.exists(BASELINE_PATH), f"Missing baseline {BASELINE_PATH}"
    assert os.path.exists(AGENT_PATH), f"Missing agent output {AGENT_PATH}"
    
    baseline = torch.load(BASELINE_PATH, map_location='cpu', weights_only=False)['model_state']
    agent = torch.load(AGENT_PATH, map_location='cpu', weights_only=False)['model_state']
    
    max_diff = 0.0
    for k in baseline.keys():
        assert k in agent, f"Missing key {k} in agent checkpoint"
        diff = (baseline[k] - agent[k]).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            
    assert max_diff <= 1e-6, f"Model diverged! Max diff: {max_diff}"
