# PagedAttention Learning Plan

A structured guide to understanding PagedAttention from concept to implementation in vLLM.

---

## Phase 1: Understanding the Concept (External Resources)

Before diving into code, understand the motivation and design:

1. **Read the vLLM Blog Post**: [PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
   - Understand the memory fragmentation problem
   - Learn how block tables work (analogy to OS virtual memory)

2. **Read the Paper**: ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180)
   - Focus on Section 3 (PagedAttention) and Section 4 (vLLM Architecture)

---

## Phase 2: High-Level Architecture (Python Interface)

### Step 1: Entry Point & Configuration
**Goal**: Understand how PagedAttention is configured and selected

- [`vllm/config/cache.py`](file:///home/paperspace/projects/vllm/vllm/config.py) - `CacheConfig` class
  - Look for `block_size`, `cache_dtype` parameters
  
- [`vllm/attention/selector.py`](file:///home/paperspace/projects/vllm/vllm/attention/selector.py)
  - `get_attn_backend()` - How vLLM selects the attention backend
  - See how it chooses between different implementations

### Step 2: Attention Layer Interface
**Goal**: See how models use attention

- [`vllm/attention/layer.py`](file:///home/paperspace/projects/vllm/vllm/attention/layer.py)
  - `Attention.__init__()` (lines 223-359) - Initialization
  - `Attention.forward()` (lines 360-434) - How attention is called
  - Notice the `kv_cache` attribute and how it's used

---

## Phase 3: Memory Management & Block Tables

### Step 3: Understanding Block Management
**Goal**: Learn how vLLM manages memory blocks

- [`vllm/v1/core/kv_cache_manager.py`](file:///home/paperspace/projects/vllm/vllm/v1/core/kv_cache_manager.py)
  - `KVCacheManager` class
  - `allocate_slots()` - How blocks are allocated
  - `free_slots()` - How blocks are freed

### Step 4: Block Tables
**Goal**: Understand the mapping from logical to physical blocks

- Search for `block_tables` usage in the codebase
- [`vllm/attention/backends/abstract.py`](file:///home/paperspace/projects/vllm/vllm/attention/backends/abstract.py)
  - Look at `AttentionMetadata` classes
  - See how `block_tables` is structured

---

## Phase 4: PagedAttention Operations (Python Layer)

### Step 5: PagedAttention Python Interface
**Goal**: Understand the Python wrapper for kernels

- [`vllm/attention/ops/paged_attn.py`](file:///home/paperspace/projects/vllm/vllm/attention/ops/paged_attn.py)
  - `PagedAttention.get_kv_cache_shape()` (lines 47-54) - KV cache layout
  - `PagedAttention.split_kv_cache()` (lines 57-69) - How K/V are separated
  - `PagedAttention.write_to_paged_cache()` (lines 72-91) - Writing new tokens to cache
  - `PagedAttention.forward_decode()` (lines 94-199) - **MOST IMPORTANT**: Decode phase logic
    - See the V1 vs V2 selection heuristic (lines 134-136)
  - `PagedAttention.forward_prefix()` (lines 202-239) - Prefill phase

### Step 6: Prefill Attention (Triton Kernels)
**Goal**: Understand how the prompt is processed initially

- [`vllm/attention/ops/prefix_prefill.py`](file:///home/paperspace/projects/vllm/vllm/attention/ops/prefix_prefill.py)
  - `_fwd_kernel()` (lines 36-335) - Main Triton kernel for prefill
  - `context_attention_fwd()` (lines 618-815) - Entry point
  - Notice how it processes context (cached KV) and new queries separately

---

## Phase 5: CUDA Kernel Implementation (Deep Dive)

### Step 7: Kernel Entry Points
**Goal**: See how kernels are launched

- [`csrc/attention/paged_attention_v1.cu`](file:///home/paperspace/projects/vllm/csrc/attention/paged_attention_v1.cu)
  - `paged_attention_v1()` (lines 160-182) - Entry point
  - `paged_attention_v1_launcher()` (lines 46-125) - Launcher logic
  - Notice the macro `LAUNCH_PAGED_ATTENTION_V1` and grid/block configuration

- [`csrc/attention/paged_attention_v2.cu`](file:///home/paperspace/projects/vllm/csrc/attention/paged_attention_v2.cu)
  - Similar structure but with partitioning logic

### Step 8: Core Kernel Logic
**Goal**: Understand the actual attention computation

- [`csrc/attention/attention_kernels.cuh`](file:///home/paperspace/projects/vllm/csrc/attention/attention_kernels.cuh)
  - `paged_attention_kernel()` (lines 85-491) - **THIS IS THE CORE**
  
  **Study in this order**:
  1. **Lines 106-131**: Partition and block range calculation
  2. **Lines 168-183**: Loading query into shared memory
  3. **Lines 198-303**: **First loop** - Computing Q·K for each block
     - Lines 202: Block table lookup (`block_table[block_idx]`)
     - Lines 255-285: Loading K from paged cache
     - Lines 287-292: Computing dot product `qk = scale * q·k`
     - Lines 294-301: Storing logits and finding max
  4. **Lines 305-340**: Softmax computation
     - Lines 329-333: Computing exp and sum
     - Lines 337-340: Normalizing with softmax
  5. **Lines 354-429**: **Second loop** - Computing Softmax·V
     - Lines 397-413: Loading V from paged cache
     - Lines 426: Accumulating `acc += softmax * v`
  6. **Lines 431-490**: Reduction and output writing

- [`csrc/attention/attention_kernels.cuh`](file:///home/paperspace/projects/vllm/csrc/attention/attention_kernels.cuh)
  - `paged_attention_v2_reduce_kernel()` (lines 562-664) - V2 reduction kernel
  - Study how it combines partial results from multiple partitions

### Step 9: Cache Operations
**Goal**: Understand how KV cache is updated

- [`csrc/cache_kernels.cu`](file:///home/paperspace/projects/vllm/csrc/cache_kernels.cu)
  - `reshape_and_cache_kernel()` - Writes new K/V to cache
  - `copy_blocks_kernel()` - Copies blocks (for beam search, etc.)
  - `swap_blocks_kernel()` - Swaps blocks between GPU and CPU

---

## Phase 6: Advanced Topics

### Step 10: Optimizations
**Goal**: Understand specific optimizations

- [`csrc/attention/dtype_float16.cuh`](file:///home/paperspace/projects/vllm/csrc/attention/dtype_float16.cuh)
  - FP16-specific optimizations
  
- [`csrc/attention/attention_utils.cuh`](file:///home/paperspace/projects/vllm/csrc/attention/attention_utils.cuh)
  - Vectorized load/store utilities
  - `Qk_dot` structure - Optimized dot product

### Step 11: Quantization Support
**Goal**: See how FP8 KV cache works

- Look for `KV_DTYPE` template parameter usage in kernels
- [`csrc/quantization/w8a8/fp8/`](file:///home/paperspace/projects/vllm/csrc/quantization) - FP8 utilities

---

## Phase 7: Hands-On Exploration

### Step 12: Add Debugging/Logging
Try adding print statements or logging to trace execution:

1. Add prints in `PagedAttention.forward_decode()` to see when V1 vs V2 is chosen
2. Log block table contents to see the logical→physical mapping
3. Use `torch.cuda.synchronize()` and timing to measure kernel performance

### Step 13: Simple Example
Create a minimal example:

```python
# Run a simple inference and trace the attention calls
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", max_model_len=512, block_size=16)
outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=20))
```

Use a debugger to step through the attention forward pass.

---

## Key Concepts Summary

After going through all phases, you should understand:

1. **Block Table**: Maps logical KV positions to physical memory blocks
2. **KV Cache Layout**: How keys and values are stored in memory
3. **V1 vs V2**: When to use simple kernel vs partitioned kernel
4. **Prefill vs Decode**: Different kernels for different phases
5. **Memory Management**: How blocks are allocated, freed, and reused

---

## Additional Resources

- **vLLM Documentation**: https://docs.vllm.ai
- **Developer Slack**: https://slack.vllm.ai
- **GitHub Discussions**: https://github.com/vllm-project/vllm/discussions

Good luck with your deep dive into PagedAttention! 🚀
