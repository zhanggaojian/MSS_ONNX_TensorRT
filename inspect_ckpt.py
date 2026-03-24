import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "./MelBandRoformer.ckpt"
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

if "state" in state_dict:
    state_dict = state_dict["state"]
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

layer_indices = set()
for k in state_dict:
    if k.startswith("layers."):
        idx = int(k.split(".")[1])
        layer_indices.add(idx)

max_layer = max(layer_indices)
depth = (max_layer + 1) // 2

for k, v in sorted(state_dict.items()):
    if "layers.0.0.layers.0.0.norm.gamma" in k:
        dim = v.shape[0]
        print(f"dim: {dim}")
    if "layers.0.0.layers.0.0.to_qkv.weight" in k:
        total_qkv = v.shape[0]
        dim_in = v.shape[1]
        heads_x_dim_head_x3 = total_qkv
        print(f"to_qkv shape: {v.shape} => dim_in={dim_in}, qkv_total={total_qkv}")
    if "layers.0.0.layers.0.0.to_gates.weight" in k:
        heads = v.shape[0]
        print(f"heads: {heads}")
    if "layers.0.0.layers.0.0.to_out.0.weight" in k:
        print(f"to_out shape: {v.shape}")
    if "mask_estimators.0.layers.0.1.weight" in k:
        print(f"mask_estimator first layer shape: {v.shape}")

print(f"\nmax layer index: {max_layer}")
print(f"depth (num layer pairs): {depth}")

time_layers = set()
freq_layers = set()
for k in state_dict:
    if k.startswith("layers."):
        parts = k.split(".")
        pair_idx = int(parts[1])
        sub_idx = int(parts[2])
        if sub_idx == 0:
            if "layers" in ".".join(parts[3:]):
                sub_parts = parts[3:]
                if sub_parts[0] == "layers":
                    time_depth_idx = int(sub_parts[1])
                    time_layers.add(time_depth_idx)
        elif sub_idx == 1:
            if "layers" in ".".join(parts[3:]):
                sub_parts = parts[3:]
                if sub_parts[0] == "layers":
                    freq_depth_idx = int(sub_parts[1])
                    freq_layers.add(freq_depth_idx)

print(f"time_transformer_depth: {max(time_layers) + 1 if time_layers else 0}")
print(f"freq_transformer_depth: {max(freq_layers) + 1 if freq_layers else 0}")

mask_est_layers = set()
for k in state_dict:
    if k.startswith("mask_estimators."):
        parts = k.split(".")
        if parts[2] == "layers":
            mask_est_layers.add(int(parts[3]))
print(f"mask_estimator_depth: {max(mask_est_layers) + 1 if mask_est_layers else 0}")

num_stems_keys = [k for k in state_dict if k.startswith("mask_estimators.")]
stem_indices = set()
for k in num_stems_keys:
    stem_indices.add(int(k.split(".")[1]))
print(f"num_stems: {max(stem_indices) + 1 if stem_indices else 0}")

for k, v in sorted(state_dict.items()):
    if "layers.0.0.layers.0.0.rotary_embed.freqs" in k:
        dim_head = v.shape[0] * 2
        print(f"dim_head: {dim_head}")
        break

ff_keys = [k for k in state_dict if "layers.0.0.layers.0.1.net.1.weight" in k]
for k in ff_keys:
    v = state_dict[k]
    expansion = v.shape[0] // (v.shape[1] * 2) if v.shape[1] > 0 else 0
    print(f"ff linear shape: {v.shape}, mlp_expansion_factor ~= {v.shape[0] / v.shape[1]:.1f}")
