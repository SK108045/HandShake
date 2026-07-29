import torch
import sys

def diff_state_dicts(path1, path2):
    sd1 = torch.load(path1, map_location='cpu', weights_only=False)['model_state']
    sd2 = torch.load(path2, map_location='cpu', weights_only=False)['model_state']
    
    max_diff = 0.0
    for k in sd1.keys():
        if k in sd2:
            diff = (sd1[k] - sd2[k]).abs().max().item()
            if diff > max_diff:
                max_diff = diff
            if diff > 0:
                print(f"{k}: max diff = {diff}")
        else:
            print(f"Key {k} missing in {path2}")
    
    for k in sd2.keys():
        if k not in sd1:
            print(f"Key {k} missing in {path1}")
            
    print(f"Max difference across all keys: {max_diff}")
    return max_diff

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python diff.py path1 path2")
        sys.exit(1)
    
    max_diff = diff_state_dicts(sys.argv[1], sys.argv[2])
    if max_diff > 1e-6:
        print("DIVERGENCE > 1e-6 DETECTED!")
        sys.exit(1)
    else:
        print("Checkpoints match within tolerance.")
        sys.exit(0)
