# Diffusion & U-Net Experiments

This directory focuses on **diffusion models**, especially **U-Net-based
architectures** used in image, video, and generative modeling.

Diffusion models are a crucial stress-test:
- They have **complex, loopy-looking DAGs**:
  - downsampling and upsampling paths,
  - skip connections between encoder and decoder stages,
  - time-conditioning branches.
- They run **many timesteps**, often with repeated application of the same U-Net.

The goal: show how tiny-backprop handles **branchy, symmetric architectures**
and repeated application patterns.

---

## Why diffusion models matter (structurally)

From first principles:

1. **Single U-Net forward/backward**
   - The per-step U-Net is a DAG with:
     - encoder path → bottleneck → decoder path,
     - many skip connections that are still locally structured.
   - Despite the branching, there is:
     - a dominant depth dimension,
     - controlled frontier width.
   - This suggests the architecture is **height-compressible** in practice.

2. **Repeated application in diffusion**
   - Training often unrolls the U-Net over different timesteps.
   - Even when each step is independent in the loss, the cost profile is similar:
     - activations inside each call dominate memory.
   - Height-compressed backprop can:
     - reduce memory per step,
     - or enable larger U-Nets / higher resolutions for the same hardware.

3. **High commercial + research value**
   - Foundation image/video models are diffusion-based.
   - Improving memory usage here directly enables larger, higher-fidelity models.

---

## Files

### `unet_mem_bench.py`

- Benchmarks a U-Net (or family of them) with:
  - naive training,
  - existing checkpointing strategies,
  - tiny-backprop schedules.
- Measures:
  - peak memory,
  - recompute overhead,
  - gradient correctness.

### `high_res.py`

- Targets **high-resolution** scenarios (e.g. 1024², 2048²).
- Uses tiny-backprop to:
  - explore whether high-res training is feasible on commodity GPUs,
  - quantify savings when activations dominate due to large feature maps.

---

## How these experiments fit

Diffusion/U-Net experiments validate that:

- The height-compressed scheduling logic survives:
  - non-trivial branching,
  - cross-stage skips,
  - realistic generative architectures.
- We can still:
  - derive a structured checkpoint plan,
  - get meaningful memory savings,
  - maintain stable training.

For engineers:
- This directory answers:
  > “If I’m training a big diffusion model, can tiny-backprop give me
  > more resolution or larger U-Nets without exploding memory?”

That’s exactly what these experiments are designed to show.
