# NVIDIA LocateAnything batch runtime

This directory vendors the optional `batch_utils` and `kernel_utils` inference
runtime distributed with `nvidia/LocateAnything-3B` at Hugging Face revision
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

Local changes are limited to explicitly selecting the configured SDPA backend,
using BF16 through the current Transformers `dtype` argument, disabling the
unavailable fast image processor, and preserving this pipeline's existing
prompt wording. The model and runtime remain subject to NVIDIA's upstream
license terms.
