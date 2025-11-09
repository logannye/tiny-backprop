# Vision Model Experiments

This directory contains experiments for **CNNs, ResNets, and Vision Transformers (ViTs)**.

The goal: understand how height-compressed backprop behaves on models with:

- skip connections,
- multi-branch residual blocks,
- patch embeddings and attention (for ViTs),
- spatial dimensions that influence activation size.

---

## Why vision models?

From first principles:

1. **ResNets & CNNs**
   - Deep, structured stacks of conv/BN/ReLU with residual edges.
   - The computation graph:
     - is still mostly layered in depth,
     - has bounded-width skip connections.
   - This makes them **good candidates for height-compression**:
     - frontier width is tied to a few feature maps, not the entire history.

2. **Vision Transformers**
   - Architecturally similar to text Transformers:
     - token/patch embeddings,
     - MHSA, MLP blocks, residuals.
   - Graph is again a DAG with strong sequential structure over layers.

3. **Practical relevance**
   - Vision backbones are ubiquitous (classification, detection, diffusion backbones).
   - Many users hit memory ceilings when scaling resolution or depth.

---

## Files

### `vit_large.py`

- Applies tiny-backprop to a ViT-like model.
- Compares:
  - naive autograd,
  - any existing checkpointing baseline,
  - tiny-backprop schedule.
- Focus:
  - impact of depth and hidden size on achievable memory savings.

### `resnet_deep.py`

- Uses a very deep ResNet (e.g. 152+ layers, or synthetic deeper variants).
- Evaluates:
  - how residual skip structure affects graph frontier,
  - how well the planner compresses memory without over-recompute.

---

## How to think about these experiments

These scripts:

- Start from the **graph structure**:
  - long depth,
  - finite number of concurrent live paths,
  - relatively easy-to-recompute interior blocks.

- Then test whether tiny-backprop can:
  - Identify an efficient decomposition,
  - Lower peak memory while preserving:
    - correctness of outputs and gradients,
    - acceptable training throughput.

They serve as:
- validation that height-compression is not “Transformer-only”,
- a bridge for deploying tiny-backprop in standard vision stacks.

If you’re a practitioner:
- This directory shows whether you can plug tiny-backprop into your existing
vision models to get **larger batches, higher resolution, or deeper nets**
on fixed hardware.
