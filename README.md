## Work in progress

This is my implementation of GPT-2 from scratch with PyTorch, for learning purposes, following Andrej Karpathy's GPT-2 walkthrough

### Currently implemented:

- GPT2 architecture
    - Learned token and positional embeddings
    - Causal multi-head self-attention
    - Pre-LN blocks with residual connections and GELU MLPs
    - Final layer normalization and vocabulary projection
    - Weight tying between token embeddings and the output projection
    - GPT-2 weight initialization with smaller residual output projections

- Faster training by enabling TF32 instead of FP32



### Remarks:

This is an implementation for learning purposes and is not meant to be optimized 
I use PyTorch's default GELU instead of GPT-2's tanh approximation, so the activation does not exactly match the original model.
