## Work in progress

This is my implementation of GPT-2 from scratch with PyTorch, for learning purposes, following Andrej Karpathy's GPT-2 walkthrough

### Currently implemented:

- GPT2 architecture
    - Learned token and positional embeddings
    - Causal multi-head self-attention
    - Pre-LN blocks with residual connections and GELU MLPs
    - Final layer normalization and vocabulary projection
- Data loading and tokenization
- Training of model on tinyshakespeare dataset
- Basic next-token prediction



### Remarks:

This is an implementation for learning purposes and is not meant to be optimized 
I also went with the default implementation of GELU instead of the tanh approximation since there is not reason to use the approximation anymore (default was fixed by Pytorch)
