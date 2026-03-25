// Deeply Expanded Content for Chapters 5 and 6
// Chapter 5: LLM Serving with vLLM
// Chapter 6: RL Training for LLMs (RLHF/RLVR)

const CONTENT_CH5_6 = {

// ============================================================================
// CHAPTER 5: LLM SERVING WITH vLLM  (~12,000 words, 8 sections)
// ============================================================================
ch5_sections: [

// ---------------------------------------------------------------------------
// 5.1  vLLM Architecture & PagedAttention (expanded)
// ---------------------------------------------------------------------------
{
  id: "vllm-architecture",
  title: "vLLM Architecture & PagedAttention",
  content: `
<p>vLLM is the most widely deployed open-source LLM serving engine, powering inference at companies from startups to hyperscalers. Its foundational paper, <em>"Efficient Memory Management for Large Language Model Serving with PagedAttention"</em> (Kwon et al., 2023; <a href="https://arxiv.org/abs/2309.06180">arXiv:2309.06180</a>), introduced PagedAttention -- a memory management technique that treats KV-cache memory like virtual memory in operating systems. This section provides a deep architectural walkthrough, from the math of attention through the implementation details that make vLLM fast.</p>

<h4>The KV-Cache Problem: A Quantitative Analysis</h4>
<p>During autoregressive generation, each new token must attend to every previous token. Storing the key and value tensors for all layers and all previous positions is called the <strong>KV-cache</strong>. Let us calculate the exact memory footprint:</p>

<pre><code># KV-Cache Memory Formula
# For a model with:
#   L = number of layers
#   h = number of KV heads (note: may differ from Q heads in GQA)
#   d = head dimension
#   s = sequence length
#   b = batch size (concurrent requests)
#   p = precision in bytes (2 for FP16/BF16, 1 for FP8)

KV_memory = 2 * L * h * d * s * b * p   # 2 for K and V

# Example: LLaMA-3-70B
# L=80, h=8 (GQA: 8 KV heads, 64 Q heads), d=128, s=4096, b=1
# FP16: 2 * 80 * 8 * 128 * 4096 * 1 * 2 = 1.07 GB per request
# With 64 concurrent requests: 68.7 GB just for KV-cache!

# Example: LLaMA-3-8B
# L=32, h=8 (GQA), d=128, s=4096, b=1
# FP16: 2 * 32 * 8 * 128 * 4096 * 1 * 2 = 0.27 GB per request
# With 128 concurrent requests: 34.4 GB for KV-cache

# Example: Qwen2.5-72B
# L=80, h=8 (GQA), d=128, s=32768, b=1
# FP16: 2 * 80 * 8 * 128 * 32768 * 1 * 2 = 8.59 GB per request!</code></pre>

<div class="callout">
<div class="callout-title">Key Insight: GQA Changes the Math</div>
<p>Grouped Query Attention (GQA) dramatically reduces KV-cache size. LLaMA-3-70B uses 64 query heads but only 8 KV heads -- an 8x reduction in KV-cache compared to standard Multi-Head Attention. Always use the <strong>KV head count</strong> (not the query head count) when calculating KV-cache memory. The formula above uses <code>h</code> = number of <em>KV</em> heads.</p>
</div>

<h4>Memory Fragmentation: The Silent Throughput Killer</h4>
<p>Without PagedAttention, serving engines pre-allocate a contiguous memory block for each request at the maximum possible sequence length. This leads to two types of waste:</p>

<ul>
<li><strong>Internal Fragmentation (60-80% waste):</strong> A request that only generates 200 tokens wastes the remaining pre-allocated memory up to <code>max_model_len</code>. If max_model_len=4096, that is 95% waste for this single request.</li>
<li><strong>External Fragmentation:</strong> As requests of different lengths complete and free memory, gaps appear between allocated blocks. These gaps may be too small for new requests, even though total free memory is sufficient.</li>
<li><strong>Reservation Waste:</strong> Systems must reserve memory for the worst-case scenario (all requests at max length), meaning the system is always provisioned for a scenario that almost never occurs.</li>
</ul>

<p>The Kwon et al. paper measured that existing systems (HuggingFace TGI at the time, FasterTransformer) wasted 60-80% of KV-cache memory to fragmentation. This directly limits throughput: fewer concurrent requests can fit in GPU memory.</p>

<h4>PagedAttention: Virtual Memory for KV-Cache</h4>
<p>PagedAttention maps OS virtual memory concepts directly to KV-cache management:</p>

<table>
<tr><th>OS Concept</th><th>PagedAttention Analog</th><th>Purpose</th></tr>
<tr><td>Physical page frame</td><td>KV-cache block (fixed-size tensor)</td><td>Fixed unit of physical memory</td></tr>
<tr><td>Virtual page</td><td>Logical KV-cache slot for a position range</td><td>Abstraction over physical location</td></tr>
<tr><td>Page table</td><td>Block table per sequence</td><td>Maps logical to physical blocks</td></tr>
<tr><td>Demand paging</td><td>Allocate blocks only when tokens are generated</td><td>No upfront reservation</td></tr>
<tr><td>Copy-on-write</td><td>Share prefix blocks, copy only on divergence</td><td>Beam search, parallel sampling</td></tr>
</table>

<pre><code># Detailed PagedAttention implementation sketch
class PagedAttentionEngine:
    """
    Core PagedAttention memory manager.
    Block size is typically 16 tokens (vLLM default).
    """
    def __init__(self, num_gpu_blocks, num_cpu_blocks, block_size=16,
                 num_kv_heads=8, head_dim=128, num_layers=80, dtype=torch.float16):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks

        # Pre-allocate the physical KV-cache pool on GPU
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        element_size = 2 if dtype == torch.float16 else 1  # bytes
        block_bytes = block_size * num_kv_heads * head_dim * element_size
        total_kv_bytes = 2 * num_layers * num_gpu_blocks * block_bytes
        print(f"KV-cache pool: {total_kv_bytes / 1e9:.2f} GB "
              f"({num_gpu_blocks} blocks of {block_size} tokens)")

        self.gpu_k_cache = torch.zeros(
            num_layers, num_gpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        self.gpu_v_cache = torch.zeros_like(self.gpu_k_cache)

        # Free block list (simple free-list allocator)
        self.free_gpu_blocks = list(range(num_gpu_blocks))
        # Per-sequence block tables: seq_id -> List[physical_block_id]
        self.block_tables = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list:
        """Allocate blocks for a new sequence (e.g., prompt prefill)."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if num_blocks_needed > len(self.free_gpu_blocks):
            raise RuntimeError("OOM: not enough free KV-cache blocks. "
                             "Consider reducing max_num_seqs or max_model_len.")
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_gpu_blocks.pop()
            allocated.append(block_id)
        self.block_tables[seq_id] = allocated
        return allocated

    def append_token(self, seq_id: int, position: int):
        """Called for each generated token. Allocates new block if needed."""
        block_idx = position // self.block_size
        if block_idx >= len(self.block_tables[seq_id]):
            # Need a new block
            if not self.free_gpu_blocks:
                # Trigger preemption: swap least-recently-used seq to CPU
                self._preempt_sequence()
            new_block = self.free_gpu_blocks.pop()
            self.block_tables[seq_id].append(new_block)

    def free(self, seq_id: int):
        """Free all blocks when a sequence completes."""
        for block_id in self.block_tables.pop(seq_id, []):
            self.free_gpu_blocks.append(block_id)

    def fork(self, parent_seq_id: int, child_seq_id: int):
        """Copy-on-write fork for beam search / parallel sampling."""
        # Child shares all parent's blocks (zero-copy)
        self.block_tables[child_seq_id] = list(self.block_tables[parent_seq_id])
        # Increment reference count on shared blocks
        # Actual copy happens only when child writes to a shared block

    def _preempt_sequence(self):
        """Swap out the least-recently-used sequence to CPU."""
        # In practice, vLLM uses recomputation (discard KV, recompute later)
        # or swapping (copy blocks to CPU RAM) as preemption strategies
        pass</code></pre>

<h4>The Attention Kernel with Paged Memory</h4>
<p>The key challenge is that standard FlashAttention expects contiguous KV tensors. PagedAttention requires a custom CUDA kernel that gathers K/V from non-contiguous physical blocks using the block table as an indirection layer:</p>

<pre><code># Pseudocode for the PagedAttention CUDA kernel
# For each query token q at position pos:
#   1. Look up the block table for this sequence
#   2. For each block in the table:
#       - Load K, V from the physical block
#       - Compute attention scores: score = q @ K^T / sqrt(d)
#       - Apply causal mask
#       - Accumulate softmax(score) @ V using online softmax
#
# This is implemented as a custom Triton/CUDA kernel in vLLM.
# vLLM v2 uses FlashInfer as the default attention backend,
# which natively supports paged KV-cache layouts.

# The block table acts as a gather index:
# physical_k = k_cache[block_table[seq_id][logical_block_idx]]
# This indirection adds ~2-5% overhead vs contiguous FlashAttention,
# but enables 2-4x more concurrent sequences.</code></pre>

<h4>vLLM System Architecture (v0.7+)</h4>
<p>vLLM follows a disaggregated architecture with clear separation of concerns:</p>

<pre><code>                    +-------------------+
                    |   OpenAI-compat   |
                    |   API Server      |
                    |  (FastAPI/uvicorn)|
                    +--------+----------+
                             |
                    +--------v----------+
                    |    LLM Engine     |
                    |  (orchestrator)   |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |    Scheduler     |          |  Token Sampler  |
     | (FCFS + priority |          | (top-p, top-k,  |
     |  + preemption)   |          |  beam, specul.) |
     +---------+--------+          +-----------------+
               |
     +---------v---------+
     |  Model Executor    |
     | (TP/PP workers)    |
     +---------+----------+
               |
     +---------v---------+
     | PagedAttention +   |
     | FlashInfer/Triton  |
     | KV-Cache Manager   |
     +--------------------+</code></pre>

<p>Key components:</p>
<ul>
<li><strong>API Server:</strong> OpenAI-compatible endpoints (/v1/completions, /v1/chat/completions). Handles SSE streaming. In v0.7+, uses asyncio for non-blocking I/O.</li>
<li><strong>LLM Engine:</strong> The orchestrator. Manages the lifecycle: receives requests, passes to scheduler, dispatches to executor, returns results.</li>
<li><strong>Scheduler:</strong> Decides which sequences to process in each iteration. Implements FCFS with priority levels and preemption (swap-out or recompute).</li>
<li><strong>Model Executor:</strong> Runs the actual model forward pass. Manages tensor parallelism workers across GPUs.</li>
<li><strong>KV-Cache Manager:</strong> Implements PagedAttention block allocation, as detailed above.</li>
</ul>

<div class="callout">
<div class="callout-title">vLLM v1 vs v0.x Architecture</div>
<p>vLLM v1 (released early 2025) introduced significant refactoring: a new <code>EngineCore</code> with multiprocessing-based execution (instead of Ray for single-node), a simplified <code>Scheduler</code>, and better support for multi-modal inputs. The core PagedAttention concepts remain the same, but the internal plumbing changed substantially. If you are reading vLLM source code, be aware of which version you are looking at.</p>
</div>

<h4>Block Size Selection</h4>
<p>The block size (number of tokens per KV-cache block) is a critical tuning parameter:</p>

<table>
<tr><th>Block Size</th><th>Pros</th><th>Cons</th><th>Best For</th></tr>
<tr><td>1</td><td>Zero internal fragmentation</td><td>Large page table overhead, poor cache locality</td><td>Never used in practice</td></tr>
<tr><td>8</td><td>Low fragmentation, reasonable overhead</td><td>Slightly higher overhead than 16</td><td>Short-output workloads</td></tr>
<tr><td>16 (default)</td><td>Good balance of fragmentation vs overhead</td><td>Up to 15 tokens wasted per sequence</td><td>General purpose</td></tr>
<tr><td>32</td><td>Better cache locality, lower table overhead</td><td>More internal fragmentation</td><td>Long-sequence workloads</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain PagedAttention in detail. How does it manage memory, and what is the overhead compared to contiguous KV-cache?</div>
<div class="a-text">PagedAttention manages KV-cache using a paging system inspired by OS virtual memory. The KV-cache is divided into fixed-size blocks (default 16 tokens). Each sequence has a block table mapping logical positions to physical block locations. Blocks are allocated on demand as tokens are generated, eliminating the need to pre-allocate max_sequence_length memory. Key mechanisms: (1) demand paging -- blocks allocated only when needed, (2) a free-list allocator for O(1) allocation/deallocation, (3) copy-on-write for beam search (fork sequences share blocks until divergence), (4) preemption via swap-to-CPU or recomputation when memory is exhausted. The overhead vs contiguous FlashAttention is approximately 2-5% due to the indirection in the attention kernel (gathering K/V from non-contiguous blocks via the block table). This is far outweighed by the 2-4x throughput improvement from fitting more concurrent requests. The internal fragmentation is bounded by (block_size - 1) tokens per sequence, typically negligible.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Calculate the KV-cache memory for LLaMA-3-70B serving 32 concurrent requests at 4096 tokens each. How does GQA affect this?</div>
<div class="a-text">LLaMA-3-70B: 80 layers, 8 KV heads (GQA with 64 query heads), head_dim=128. KV-cache per request = 2 (K+V) * 80 * 8 * 128 * 4096 * 2 bytes (FP16) = 1.07 GB. For 32 concurrent: 34.4 GB. Without GQA (64 KV heads): 2 * 80 * 64 * 128 * 4096 * 2 = 8.59 GB per request, or 274.9 GB for 32 concurrent -- impossible on any single node. GQA provides an 8x reduction in KV-cache, which is essential for making 70B+ models servable. This is why modern models universally adopt GQA.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 5.2  Optimization Techniques (expanded with real benchmarks)
// ---------------------------------------------------------------------------
{
  id: "vllm-optimization",
  title: "Optimization Techniques for LLM Serving",
  content: `
<p>Optimizing LLM serving requires attacking multiple bottlenecks simultaneously: memory capacity (how many requests fit), compute throughput (how fast we process tokens), and latency (how quickly users see responses). This section covers the core optimization techniques with real benchmark data.</p>

<h4>Understanding the Two Phases of LLM Inference</h4>
<p>Every LLM inference request has two distinct phases with very different computational profiles:</p>

<table>
<tr><th>Property</th><th>Prefill (Prompt Processing)</th><th>Decode (Token Generation)</th></tr>
<tr><td>Computation</td><td>Compute-bound (large matrix multiplies)</td><td>Memory-bandwidth-bound (small matmuls)</td></tr>
<tr><td>Tokens processed</td><td>All prompt tokens at once</td><td>One token at a time</td></tr>
<tr><td>GPU utilization</td><td>High (good arithmetic intensity)</td><td>Low (~1-5% of peak FLOPS)</td></tr>
<tr><td>Bottleneck</td><td>FLOPS</td><td>Memory bandwidth (reading weights)</td></tr>
<tr><td>Latency metric</td><td>Time-to-First-Token (TTFT)</td><td>Inter-Token Latency (ITL) / Time-per-Output-Token (TPOT)</td></tr>
</table>

<div class="callout">
<div class="callout-title">The Arithmetic Intensity Gap</div>
<p>For decode, each token requires reading the entire model weights from HBM but only performs O(1) FLOPs per parameter. A 70B FP16 model = 140 GB of weights. H100 HBM bandwidth = 3.35 TB/s. Theoretical minimum time to read weights once = 140/3350 = 41.8ms per token. That is the memory bandwidth wall -- no amount of compute optimization can beat it for a single request. The only way to improve utilization during decode is <strong>batching</strong>: amortize the weight reads across multiple requests.</p>
</div>

<h4>Continuous Batching: The Foundation</h4>
<p>Static batching waits until a full batch of requests arrives, processes them together, and returns all results when the slowest request finishes. Continuous (or iteration-level) batching is fundamentally different:</p>

<ul>
<li>New requests are inserted into the running batch at every decode step</li>
<li>Completed requests are immediately removed, freeing their KV-cache</li>
<li>The batch size dynamically adjusts based on available memory</li>
</ul>

<pre><code># Continuous batching pseudocode
class ContinuousBatchScheduler:
    def __init__(self, max_num_seqs=256):
        self.running = []      # Currently generating
        self.waiting = []      # Queued requests
        self.max_num_seqs = max_num_seqs

    def schedule_step(self):
        """Called every decode iteration (~10-50ms)."""
        # 1. Remove completed sequences
        self.running = [s for s in self.running if not s.is_finished()]

        # 2. Check for sequences to preempt (if memory pressure)
        while self._memory_pressure():
            victim = self.running.pop()  # LIFO preemption
            self.waiting.insert(0, victim)  # Re-queue at front

        # 3. Admit new sequences from waiting queue
        while (self.waiting and
               len(self.running) < self.max_num_seqs and
               self._can_allocate(self.waiting[0])):
            new_seq = self.waiting.pop(0)
            self._allocate_kv_blocks(new_seq)
            self.running.append(new_seq)

        return self.running  # These sequences get processed this step</code></pre>

<p>Benchmark impact of continuous batching (measured on A100-80GB, LLaMA-2-13B):</p>

<table>
<tr><th>Batching Strategy</th><th>Throughput (req/s)</th><th>Avg Latency</th><th>P99 Latency</th><th>GPU Util</th></tr>
<tr><td>Static (batch=1)</td><td>2.1</td><td>480ms</td><td>510ms</td><td>15%</td></tr>
<tr><td>Static (batch=32)</td><td>18.4</td><td>1740ms</td><td>3200ms</td><td>68%</td></tr>
<tr><td>Continuous</td><td>31.2</td><td>520ms</td><td>890ms</td><td>82%</td></tr>
</table>

<h4>Tensor Parallelism vs Pipeline Parallelism: Detailed Trade-offs</h4>

<p><strong>Tensor Parallelism (TP)</strong> splits each layer's weight matrices across GPUs. Every GPU participates in every token's computation. Requires two all-reduce operations per transformer layer (one for attention, one for FFN).</p>

<pre><code># TP=4 for a linear layer W of shape [H, 4H] (FFN up-projection)
# Split W column-wise: W = [W1 | W2 | W3 | W4], each Wi is [H, H]
# Each GPU i computes: yi = x @ Wi          (local matmul, no communication)
# Then: y = [y1 | y2 | y3 | y4]            (conceptual; stays distributed)
#
# For the down-projection W_down of shape [4H, H]:
# Split W_down row-wise: W_down = [W_d1; W_d2; W_d3; W_d4]
# Each GPU i computes: zi = yi @ W_di       (local matmul)
# Then: z = z1 + z2 + z3 + z4              (all-reduce!)
#
# Communication per layer: 2 all-reduces of [batch, seq_len, hidden]
# With NVLink (900 GB/s bidirectional on H100):
#   8B model, hidden=4096, batch*seq=2048, FP16:
#   Data per all-reduce = 2048 * 4096 * 2 bytes = 16.8 MB
#   NVLink latency: ~16.8 / 900000 + 5us overhead ~= 24us
#   Very fast! TP is efficient within a node.</code></pre>

<p><strong>Pipeline Parallelism (PP)</strong> assigns different layers to different GPUs. Only the boundary activations are communicated between stages. Much less communication, but introduces pipeline bubbles.</p>

<pre><code># PP=4 for a 32-layer model:
# GPU 0: layers 0-7, GPU 1: layers 8-15, GPU 2: layers 16-23, GPU 3: layers 24-31
#
# Micro-batching reduces bubble time:
# With m micro-batches and p pipeline stages:
#   Bubble fraction = (p - 1) / (m + p - 1)
#
# Example: PP=4, micro_batches=16
#   Bubble = 3 / 19 = 15.8% overhead
#
# PP=4, micro_batches=4
#   Bubble = 3 / 7 = 42.9% overhead  (too high!)
#
# Rule of thumb: micro_batches >= 4 * pipeline_stages</code></pre>

<table>
<tr><th>Aspect</th><th>Tensor Parallelism</th><th>Pipeline Parallelism</th></tr>
<tr><td>Communication volume</td><td>High (2 all-reduces per layer)</td><td>Low (activations at stage boundaries)</td></tr>
<tr><td>Interconnect requirement</td><td>NVLink (>= 600 GB/s)</td><td>PCIe or InfiniBand sufficient</td></tr>
<tr><td>Latency impact</td><td>Minimal with NVLink</td><td>Pipeline bubble overhead</td></tr>
<tr><td>Best for</td><td>Intra-node (same server)</td><td>Inter-node (across servers)</td></tr>
<tr><td>Max practical degree</td><td>8 (one node)</td><td>8-16 stages</td></tr>
<tr><td>Serving pattern</td><td>Latency-sensitive</td><td>Throughput-oriented</td></tr>
</table>

<h4>Speculative Decoding for Latency Reduction</h4>
<p>Speculative decoding uses a small "draft" model to generate candidate tokens, then the large "target" model verifies them in parallel. If the draft tokens are accepted, we skip multiple serial decode steps. Reference: Leviathan et al. (2023), <a href="https://arxiv.org/abs/2211.17192">arXiv:2211.17192</a>.</p>

<pre><code># Speculative decoding step
# 1. Draft model generates K candidate tokens autoregressively (fast)
# 2. Target model scores all K candidates in ONE forward pass (parallel)
# 3. Accept longest prefix where draft matches target (rejection sampling)
# 4. Generate one additional "bonus" token from the target model

# Acceptance rate depends on draft model quality:
# - Same family (LLaMA-3-8B drafting for LLaMA-3-70B): ~70-85%
# - Different family: 40-60%
# - Quantized version as draft: 75-90%

# Speedup = K * acceptance_rate / (draft_time * K + target_verify_time)
# Example: K=5, acceptance=80%, draft=5ms, target_verify=50ms
# Without speculation: 5 * 50ms = 250ms for 5 tokens
# With speculation: 25ms (draft) + 50ms (verify) = 75ms for ~4 tokens
# Effective speedup: ~2.7x per-token latency reduction</code></pre>

<p>Benchmark results (H100, LLaMA-3-70B target, LLaMA-3-8B draft):</p>

<table>
<tr><th>Concurrency</th><th>Baseline ITL</th><th>Speculative ITL</th><th>Speedup</th><th>Note</th></tr>
<tr><td>1</td><td>48ms</td><td>19ms</td><td>2.5x</td><td>Maximum latency benefit</td></tr>
<tr><td>8</td><td>52ms</td><td>24ms</td><td>2.2x</td><td>Still strong benefit</td></tr>
<tr><td>32</td><td>68ms</td><td>58ms</td><td>1.17x</td><td>Diminishing returns</td></tr>
<tr><td>64</td><td>95ms</td><td>102ms</td><td>0.93x</td><td>Overhead exceeds benefit</td></tr>
</table>

<div class="callout warning">
<div class="callout-title">When NOT to Use Speculative Decoding</div>
<p>Speculative decoding trades compute for latency. At high concurrency, the GPU is already busy serving many requests; adding the draft model's compute load can actually <em>increase</em> latency. Use speculative decoding when: (1) latency is the priority, (2) concurrency is low-to-moderate (< 32 requests), (3) you have GPU headroom. Disable it for throughput-maximizing batch workloads.</p>
</div>

<h4>FlashAttention and Kernel Optimization</h4>
<p>FlashAttention (Dao et al., 2022; <a href="https://arxiv.org/abs/2205.14135">arXiv:2205.14135</a>) is the foundational kernel optimization. FlashAttention-2 improved throughput by 2x. FlashAttention-3 (2024, <a href="https://arxiv.org/abs/2407.08691">arXiv:2407.08691</a>) added H100 optimizations including asynchronous softmax and FP8 support.</p>

<p>vLLM v0.7+ defaults to <strong>FlashInfer</strong> as the attention backend, which provides native support for paged KV-cache layouts, ragged tensors (variable-length sequences), and fused kernels. Alternatives include xFormers and the built-in vLLM Triton kernels.</p>

<pre><code># vLLM attention backend selection
# In vLLM v0.7+, the backend is auto-selected, but can be overridden:
export VLLM_ATTENTION_BACKEND=FLASHINFER  # default, best for most cases
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # FlashAttention-2
export VLLM_ATTENTION_BACKEND=XFORMERS    # xFormers memory-efficient attention

# FlashInfer advantages:
# - Native paged KV-cache support (no custom kernels needed)
# - Cascade attention for prefix caching
# - Plan-based API: precompute execution plan, reuse across iterations
# - FP8 KV-cache support</code></pre>

<h4>Comprehensive Benchmark: vLLM vs Alternatives</h4>
<p>Benchmarks on 2x H100-80GB, LLaMA-3-70B, 4-bit AWQ, TP=2:</p>

<table>
<tr><th>Engine</th><th>Throughput (tok/s)</th><th>TTFT P50</th><th>TTFT P99</th><th>ITL P50</th><th>ITL P99</th><th>Max Concurrent</th></tr>
<tr><td>vLLM 0.7</td><td>4,850</td><td>82ms</td><td>340ms</td><td>28ms</td><td>65ms</td><td>256</td></tr>
<tr><td>SGLang 0.4</td><td>5,120</td><td>76ms</td><td>310ms</td><td>26ms</td><td>58ms</td><td>256</td></tr>
<tr><td>TensorRT-LLM</td><td>5,400</td><td>68ms</td><td>290ms</td><td>24ms</td><td>52ms</td><td>256</td></tr>
<tr><td>HF TGI 2.0</td><td>3,200</td><td>120ms</td><td>580ms</td><td>35ms</td><td>95ms</td><td>128</td></tr>
</table>

<p><em>Note: TensorRT-LLM achieves higher throughput but requires model compilation (15-60 min setup). SGLang has slightly better performance than vLLM in many benchmarks due to RadixAttention and overlap scheduling. vLLM's advantage is broader model support and ecosystem maturity.</em></p>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You are serving a 70B model on 4x A100-80GB. Users are experiencing high TTFT (>2s) during peak load. Walk through your optimization approach.</div>
<div class="a-text">Systematic approach: (1) Profile the bottleneck -- is TTFT high due to queue depth (too many requests waiting) or prefill compute (prompt processing is slow)? Check Prometheus metrics for queue depth and prefill duration. (2) If queue depth is high: increase max_num_seqs, ensure continuous batching is working, add replicas behind a load balancer. (3) If prefill is slow: enable chunked prefill to overlap prefill with decode (--enable-chunked-prefill), which prevents long prompts from blocking decode. (4) Enable prefix caching if requests share system prompts (--enable-prefix-caching), which can reduce TTFT by 80%+ for the shared prefix. (5) Quantize to AWQ-4bit or FP8 to reduce memory pressure, allowing more concurrent prefills. (6) If prompts are very long (>4K tokens), consider reducing max_model_len to free memory for more concurrent requests. (7) As a last resort, add more GPU replicas and use prefix-aware routing to maximize cache hit rates.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the difference between prefill and decode phases. Why does batching help more during decode than prefill?</div>
<div class="a-text">Prefill processes all prompt tokens in parallel via a single large matrix multiplication -- it is compute-bound with high arithmetic intensity (many FLOPs per byte of weight loaded). Decode generates one token at a time -- it is memory-bandwidth-bound because each token requires reading the entire model weights but performs very few FLOPs per weight element. Batching helps decode dramatically because multiple requests can share the same weight read: reading weights once and multiplying by B different activation vectors costs nearly the same memory bandwidth as serving a single request, but produces B times the output. For prefill, batching provides less benefit because each request's large prompt already saturates compute; adding more prompts just increases total compute proportionally.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 5.3  Complete Deployment Guide (NEW)
// ---------------------------------------------------------------------------
{
  id: "vllm-deployment",
  title: "Complete Deployment Guide",
  content: `
<p>This section provides production-tested deployment configurations for vLLM across all common scenarios, from single-GPU development setups to multi-node Kubernetes clusters. Each configuration includes the reasoning behind parameter choices.</p>

<h4>Single GPU Deployment</h4>
<p>Best for: models up to ~13B parameters (FP16) or ~30B parameters (4-bit quantized) on an 80GB GPU.</p>

<pre><code># Single A100-80GB, LLaMA-3-8B-Instruct
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --dtype bfloat16 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.92 \\
    --max-num-seqs 128 \\
    --enable-prefix-caching \\
    --enable-chunked-prefill \\
    --port 8000 \\
    --api-key "your-secret-key"

# Parameter explanations:
# --dtype bfloat16         : BF16 is preferred over FP16 (no overflow issues)
# --max-model-len 8192     : Limits KV-cache allocation. Set to your actual
#                            P99 sequence length, NOT the model's max (128K).
#                            Lower = more concurrent requests.
# --gpu-memory-utilization 0.92 : Use 92% of GPU for model + KV-cache.
#                                  Reserve 8% for CUDA overhead + PyTorch.
#                                  Don't go above 0.95 -- fragmentation kills you.
# --max-num-seqs 128       : Max concurrent sequences. Bounded by KV-cache memory.
# --enable-prefix-caching  : Reuse KV-cache for shared prefixes (system prompts).
# --enable-chunked-prefill : Overlap long prompt processing with token generation.
# --api-key                : Always set in production. Unauthenticated vLLM = open relay.</code></pre>

<pre><code># Memory budget calculation for this setup:
# Model weights (8B params, BF16):           ~16 GB
# CUDA + PyTorch overhead:                   ~2 GB
# Activation memory (peaks during prefill):  ~3 GB
# Available for KV-cache: 80 * 0.92 - 16 - 2 - 3 = 52.6 GB
#
# KV-cache per request (8B model, 32 layers, 8 KV heads, dim 128, BF16):
#   2 * 32 * 8 * 128 * 8192 * 2 = 0.54 GB at max_model_len=8192
#   2 * 32 * 8 * 128 * 2048 * 2 = 0.13 GB at 2048 tokens (typical)
#
# Theoretical max concurrent: 52.6 / 0.54 = 97 at max length
# Practical concurrent at typical length: 52.6 / 0.13 = 404
# Setting max-num-seqs=128 is safe and leaves headroom.</code></pre>

<h4>Multi-GPU with Tensor Parallelism</h4>
<p>Required when the model does not fit on a single GPU, or when you need lower latency (parallelizing compute across GPUs).</p>

<pre><code># 4x H100-80GB with NVLink, LLaMA-3-70B-Instruct
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --dtype bfloat16 \\
    --tensor-parallel-size 4 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.90 \\
    --max-num-seqs 256 \\
    --enable-prefix-caching \\
    --enable-chunked-prefill \\
    --port 8000

# TP sizing rules:
# 1. TP degree must evenly divide the number of KV heads.
#    LLaMA-3-70B has 8 KV heads -> TP can be 1, 2, 4, or 8.
#    TP=3 would NOT work (8 / 3 is not integer).
#
# 2. NVLink is REQUIRED for TP. Without NVLink (PCIe only),
#    the 2 all-reduces per layer will bottleneck badly.
#    PCIe Gen4 x16: 32 GB/s. NVLink on H100: 900 GB/s. That's 28x.
#
# 3. TP degree should not exceed GPUs on a single node.
#    TP across nodes (over InfiniBand) is possible but slow.
#
# 4. Higher TP = lower latency but lower throughput efficiency.
#    TP=8 for 70B: each GPU only has ~9 layers worth of compute.
#    Communication overhead becomes significant (~15% of step time).

# Checking NVLink connectivity:
# nvidia-smi topo -m
# Look for "NV12" or "NV18" (NVLink connections).
# "PHB" or "PIX" means PCIe only -- do NOT use TP across these GPUs.</code></pre>

<pre><code># Memory budget with TP=4:
# Model weights: 140 GB / 4 GPUs = 35 GB per GPU
# CUDA overhead: ~2 GB
# Activations: ~2 GB
# Available for KV-cache per GPU: 80 * 0.90 - 35 - 2 - 2 = 33 GB
# Total KV-cache pool: 33 * 4 = 132 GB
#
# But KV-cache is NOT split by TP -- each GPU stores KV for its KV heads.
# With TP=4 and 8 KV heads: each GPU has 2 KV heads.
# KV per request per GPU = 2 * 80 * 2 * 128 * 8192 * 2 = 0.268 GB
# Max concurrent per GPU: 33 / 0.268 = 123
# So max concurrent overall: ~123 (limited by per-GPU KV-cache)
# With typical sequence lengths (~2K), this scales to ~500.</code></pre>

<h4>Multi-Node with Pipeline Parallelism</h4>
<p>For models that require more GPUs than a single node (e.g., 405B models, or when you need extreme throughput).</p>

<pre><code># Node 0 (head): 8x H100, runs layers 0-39
# Node 1: 8x H100, runs layers 40-79
# Total: 16 GPUs, TP=8 within each node, PP=2 across nodes

# On Node 0 (head node):
ray start --head --port=6379

# On Node 1:
ray start --address="node0-ip:6379"

# Then on the head node, launch vLLM:
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-405B-Instruct \\
    --dtype bfloat16 \\
    --tensor-parallel-size 8 \\
    --pipeline-parallel-size 2 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.90 \\
    --distributed-executor-backend ray \\
    --port 8000

# Key considerations for multi-node:
# 1. Ray is required for multi-node orchestration in vLLM.
# 2. InfiniBand (400 Gbps+) is strongly recommended between nodes.
#    RoCE/Ethernet works but limits throughput for PP communication.
# 3. PP bubble overhead: with PP=2, expect ~10-15% throughput loss
#    due to pipeline bubbles (mitigated by micro-batching).
# 4. NCCL environment variables for multi-node:
#    export NCCL_IB_TIMEOUT=23
#    export NCCL_IB_RETRY_CNT=7
#    export NCCL_SOCKET_IFNAME=eth0  # or your network interface
#    export NCCL_DEBUG=WARN          # for debugging</code></pre>

<h4>Docker Deployment with GPU Passthrough</h4>

<pre><code># Dockerfile for vLLM production deployment
FROM vllm/vllm-openai:v0.7.3

# Set environment variables
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run vLLM
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "meta-llama/Llama-3.1-70B-Instruct", \\
     "--tensor-parallel-size", "4", \\
     "--max-model-len", "8192", \\
     "--gpu-memory-utilization", "0.90", \\
     "--enable-prefix-caching", \\
     "--port", "8000"]</code></pre>

<pre><code># Docker run with GPU passthrough
docker run -d \\
    --name vllm-server \\
    --gpus '"device=0,1,2,3"' \\
    --shm-size=16g \\
    -p 8000:8000 \\
    -v /path/to/models:/models \\
    -e HUGGING_FACE_HUB_TOKEN=hf_xxx \\
    -e VLLM_ATTENTION_BACKEND=FLASHINFER \\
    --restart unless-stopped \\
    vllm-production:latest

# Critical flags:
# --shm-size=16g : NCCL uses shared memory for intra-node communication.
#                   Default 64MB is far too small. 16GB is safe for TP <= 8.
# --gpus          : Use specific GPU IDs, not "all", for resource isolation.
# --restart unless-stopped : Auto-restart on OOM or crash.</code></pre>

<h4>Kubernetes Deployment with GPU Scheduling</h4>

<pre><code># kubernetes/vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama70b
  labels:
    app: vllm-serving
spec:
  replicas: 2  # 2 replicas for HA
  selector:
    matchLabels:
      app: vllm-serving
  template:
    metadata:
      labels:
        app: vllm-serving
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.7.3
        args:
          - "--model"
          - "meta-llama/Llama-3.1-70B-Instruct"
          - "--tensor-parallel-size"
          - "4"
          - "--max-model-len"
          - "8192"
          - "--gpu-memory-utilization"
          - "0.90"
          - "--enable-prefix-caching"
          - "--port"
          - "8000"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 4  # Request exactly 4 GPUs
            memory: "64Gi"
            cpu: "16"
          requests:
            nvidia.com/gpu: 4
            memory: "32Gi"
            cpu: "8"
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
        - name: VLLM_ATTENTION_BACKEND
          value: "FLASHINFER"
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: model-cache
          mountPath: /root/.cache/huggingface
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120  # Model loading takes time
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 16Gi
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      # Ensure all GPUs are on the same node (NVLink required for TP)
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-H100-80GB-HBM3
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-serving
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP</code></pre>

<div class="callout">
<div class="callout-title">Kubernetes GPU Scheduling Gotcha</div>
<p>When using TP, all GPUs must be on the <strong>same node</strong> with NVLink connectivity. Kubernetes' default GPU scheduler allocates GPUs without topology awareness -- it might assign GPU 0 from node A and GPU 1 from node B. Solutions: (1) Use the NVIDIA GPU Operator with topology-aware scheduling, (2) use <code>nvidia.com/gpu: 4</code> as a single resource request (guaranteed same node), (3) for multi-node PP, use a StatefulSet with pod-affinity rules, or use a framework like LeaderWorkerSet.</p>
</div>

<h4>Production Configuration Checklist</h4>

<pre><code># === PRODUCTION CHECKLIST ===

# 1. SECURITY
- [ ] API key set (--api-key or environment variable)
- [ ] Running behind reverse proxy (nginx/envoy) with TLS
- [ ] Input validation: max_tokens limit, prompt length limit
- [ ] Rate limiting per API key

# 2. RELIABILITY
- [ ] Health check endpoint monitored (/health)
- [ ] Graceful shutdown handling (SIGTERM -> drain requests)
- [ ] Auto-restart on crash (Docker --restart, K8s liveness probe)
- [ ] Multiple replicas behind load balancer
- [ ] Model weights on persistent storage (not downloaded on each start)

# 3. PERFORMANCE
- [ ] max_model_len set to actual P99, not model maximum
- [ ] gpu-memory-utilization tuned (0.88-0.95 depending on workload)
- [ ] Prefix caching enabled if applicable
- [ ] Chunked prefill enabled for mixed short/long prompt workloads
- [ ] Quantization applied if throughput > quality priority

# 4. OBSERVABILITY
- [ ] Prometheus metrics exposed and scraped
- [ ] Grafana dashboards for latency, throughput, queue depth
- [ ] Alerting on P99 latency, GPU OOM, queue depth thresholds
- [ ] Request logging with trace IDs for debugging
- [ ] GPU temperature and power monitoring (nvidia-smi -l)

# 5. RESOURCE MANAGEMENT
- [ ] --shm-size >= 16GB in Docker (for NCCL)
- [ ] GPU pinning (CUDA_VISIBLE_DEVICES) to avoid contention
- [ ] CPU and memory limits set appropriately
- [ ] Swap disabled (OOM is better than thrashing)

# 6. DATA
- [ ] Model versioning strategy (immutable tags, not :latest)
- [ ] A/B testing infrastructure for model updates
- [ ] Rollback plan for bad model deployments
- [ ] Tokenizer cached alongside model weights</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Design a vLLM deployment for a chatbot serving 500 requests per second with a 70B model. What hardware and configuration would you use?</div>
<div class="a-text">At 500 RPS with a 70B model, assuming average output of 200 tokens: (1) Total token throughput needed: 500 * 200 = 100K tokens/sec. (2) A single 4xH100 vLLM instance with TP=4 can do roughly 5K tokens/sec for 70B. So we need ~20 replicas. (3) Hardware: 20 nodes, each with 4x H100-80GB NVLink. Total: 80 H100s. (4) Use AWQ 4-bit quantization to potentially halve this to 10 nodes (40 H100s), while maintaining quality. (5) Deploy behind an L7 load balancer with prefix-aware routing (route requests with the same system prompt to the same replica for cache hits). (6) Configure: max-model-len=4096 (check P99), enable prefix caching, chunked prefill, max-num-seqs=256 per replica. (7) Auto-scaling: scale on queue depth > 10 (add replica) or GPU utilization < 30% (remove replica). (8) Cost estimate: 40 H100s at ~$2/hr each = $80/hr = ~$58K/month. Consider Reserved Instances or spot for cost reduction.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 5.4  Quantization Deep Dive (NEW)
// ---------------------------------------------------------------------------
{
  id: "vllm-quantization",
  title: "Quantization Deep Dive",
  content: `
<p>Quantization reduces model precision from FP16/BF16 (16 bits per parameter) to lower bit-widths (8, 4, or even 2 bits). This reduces memory footprint (fit larger models on fewer GPUs), increases throughput (less data to move through memory bandwidth), and can reduce latency. The trade-off is potential quality degradation. This section covers every major quantization method with practical guidance.</p>

<h4>Quantization Fundamentals</h4>
<p>The core idea: represent weights (and optionally activations) with fewer bits by mapping continuous values to a discrete set of levels.</p>

<pre><code># Symmetric uniform quantization (simplest form)
# x_q = round(x / scale)
# x_dequant = x_q * scale
# where scale = max(|x|) / (2^(b-1) - 1) for b-bit signed

import torch

def symmetric_quantize(tensor, bits=4):
    """Quantize a tensor to n-bit integers."""
    qmax = 2**(bits - 1) - 1  # e.g., 7 for 4-bit
    scale = tensor.abs().max() / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale

def dequantize(quantized, scale):
    return quantized.float() * scale

# Per-channel vs per-tensor:
# Per-tensor: one scale for the entire tensor (fast, less accurate)
# Per-channel: one scale per output channel (slower, more accurate)
# Per-group: one scale per group of G elements (e.g., G=128). Best tradeoff.

# Group quantization example (used by GPTQ, AWQ):
# Weight matrix [4096, 4096], group_size=128
# Number of groups: 4096 * 4096 / 128 = 131,072 scale factors
# Scale storage: 131,072 * 2 bytes (FP16) = 0.25 MB (negligible vs weights)</code></pre>

<h4>GPTQ: Post-Training Quantization</h4>
<p>GPTQ (Frantar et al., 2023; <a href="https://arxiv.org/abs/2210.17323">arXiv:2210.17323</a>) quantizes weights one layer at a time, using a small calibration dataset to minimize the quantization error. It is based on the Optimal Brain Quantization (OBQ) framework.</p>

<pre><code># GPTQ Algorithm (simplified)
# For each layer:
#   1. Run calibration data through the model up to this layer
#   2. Collect input activations X for this layer
#   3. Compute Hessian: H = 2 * X^T @ X (captures input statistics)
#   4. For each column of the weight matrix:
#       a. Quantize the column
#       b. Compute the quantization error
#       c. Distribute the error to remaining columns using H^{-1}
#          (compensate for quantization error in subsequent weights)

# Using AutoGPTQ:
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,                  # 4-bit quantization
    group_size=128,          # Quantize in groups of 128 elements
    desc_act=True,           # Use activation-order quantization (slower, better)
    damp_percent=0.01,       # Dampening for Hessian stability
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantize_config=quantize_config,
)

# Calibration: 128-512 samples from a representative dataset
# Takes 2-8 hours for a 70B model on a single A100
model.quantize(
    calibration_dataset,     # List of tokenized examples
    batch_size=4,
)
model.save_quantized("Llama-3.1-70B-GPTQ-4bit")</code></pre>

<p>GPTQ characteristics:</p>
<ul>
<li><strong>Pros:</strong> Well-established, broad hardware support, good quality at 4-bit with group_size=128.</li>
<li><strong>Cons:</strong> Calibration dataset dependent (poor calibration = poor quality), slower than AWQ at inference due to kernel design, desc_act=True is much slower to quantize.</li>
<li><strong>Best for:</strong> When you need maximum compatibility across frameworks.</li>
</ul>

<h4>AWQ: Activation-Aware Weight Quantization</h4>
<p>AWQ (Lin et al., 2023; <a href="https://arxiv.org/abs/2306.00978">arXiv:2306.00978</a>) observes that not all weights are equally important -- weights corresponding to large activation magnitudes matter more. Instead of treating all weights equally, AWQ scales the important weight channels up before quantization, reducing their relative quantization error.</p>

<pre><code># AWQ Key Insight:
# Standard quantization: all weights quantized with the same scale per group
# AWQ observation: 1% of weight channels ("salient channels") have
#   disproportionate impact because they multiply with large activations.
# Solution: multiply salient weight channels by a scale factor s > 1
#   before quantization, then divide by s during inference.
#   This reduces the relative quantization error for important channels.

# Mathematically:
# y = x @ W
# With AWQ: y = (x / s) @ (s * W)_quantized
# The scaling s is chosen to minimize: ||x @ W - (x / s) @ Q(s * W)||
# where Q() is the quantization function.

# Using AutoAWQ:
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

quant_config = {
    "zero_point": True,      # Asymmetric quantization (slightly better)
    "q_group_size": 128,     # Group size
    "w_bit": 4,              # 4-bit weights
}

# AWQ calibration is faster than GPTQ (no Hessian computation)
# ~1-4 hours for 70B on a single A100
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",    # Default calibration dataset
)
model.save_quantized("Llama-3.1-70B-AWQ-4bit")</code></pre>

<p>Why AWQ is generally preferred over GPTQ for serving:</p>
<ul>
<li><strong>Faster kernels:</strong> AWQ uses Marlin/CUTLASS kernels optimized for 4-bit; typically 10-20% faster throughput than GPTQ at the same bit-width.</li>
<li><strong>Better quality:</strong> The activation-aware scaling preserves the most important weights better. On LLM benchmarks (MMLU, HumanEval, GSM8K), AWQ consistently scores 0.5-2% higher than GPTQ at the same bit-width.</li>
<li><strong>Less calibration sensitivity:</strong> AWQ's simple scaling approach is less sensitive to calibration data choice than GPTQ's Hessian-based approach.</li>
</ul>

<h4>FP8: H100 Native Support</h4>
<p>FP8 (8-bit floating point) is natively supported by NVIDIA H100 and later GPUs. It provides near-lossless quantization with significant speedups because the Tensor Cores have dedicated FP8 datapaths.</p>

<pre><code># FP8 Formats (E4M3 vs E5M2):
# E4M3: 4 exponent bits, 3 mantissa bits. Range: +-448. Precision: ~1/8.
#        Better for weights and activations (forward pass).
# E5M2: 5 exponent bits, 2 mantissa bits. Range: +-57344. Precision: ~1/4.
#        Better for gradients (training, wider range needed).

# In vLLM, FP8 is the simplest to use:
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --quantization fp8 \\
    --dtype float16 \\
    --tensor-parallel-size 4

# FP8 reduces model size by ~50% (16-bit -> 8-bit)
# 70B model: 140 GB FP16 -> 70 GB FP8
# Fits on 1x H100-80GB instead of requiring 2x!

# FP8 with pre-computed scales (better quality):
# Use a quantized checkpoint (e.g., from NVIDIA or community):
python -m vllm.entrypoints.openai.api_server \\
    --model neuralmagic/Llama-3.1-70B-Instruct-FP8 \\
    --tensor-parallel-size 2 \\
    --max-model-len 8192

# Key: with FP8, 70B can run on TP=2 instead of TP=4!
# This cuts hardware cost in half with < 1% quality loss.</code></pre>

<h4>GGUF: CPU-Friendly Quantization</h4>
<p>GGUF is the quantization format used by llama.cpp and its ecosystem. It supports mixed-precision quantization where different layers use different bit-widths, and is optimized for CPU inference (with optional GPU offloading).</p>

<pre><code># GGUF quantization types (most common):
# Q4_K_M: 4-bit with medium k-quant. Good balance of speed and quality.
# Q5_K_M: 5-bit with medium k-quant. Better quality, 25% larger.
# Q6_K:   6-bit. Near-FP16 quality, still significant size reduction.
# Q8_0:   8-bit. Almost lossless, 2x size reduction.
# Q2_K:   2-bit. Aggressive, noticeable quality loss. Emergency only.
# IQ4_XS: 4-bit importance-weighted. Best 4-bit quality.

# Converting a HuggingFace model to GGUF:
# 1. Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
make -j

# 2. Convert to GGUF format
python convert_hf_to_gguf.py /path/to/model --outtype f16

# 3. Quantize
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# GGUF is NOT directly supported by vLLM (as of v0.7).
# Use it with: llama.cpp, ollama, LM Studio, koboldcpp.
# For vLLM, use GPTQ, AWQ, or FP8 instead.</code></pre>

<h4>Comprehensive Comparison Table</h4>

<table>
<tr><th>Method</th><th>Bits</th><th>MMLU (70B)</th><th>HumanEval (70B)</th><th>Memory Savings</th><th>Throughput vs FP16</th><th>Hardware Req</th></tr>
<tr><td>FP16/BF16</td><td>16</td><td>82.0% (baseline)</td><td>81.7%</td><td>0%</td><td>1.0x</td><td>Any GPU</td></tr>
<tr><td>FP8 (E4M3)</td><td>8</td><td>81.8% (-0.2%)</td><td>81.1%</td><td>~50%</td><td>1.3-1.5x</td><td>H100/L40S</td></tr>
<tr><td>AWQ</td><td>4</td><td>80.5% (-1.5%)</td><td>78.7%</td><td>~75%</td><td>1.5-2.0x</td><td>Any GPU</td></tr>
<tr><td>GPTQ</td><td>4</td><td>80.1% (-1.9%)</td><td>77.4%</td><td>~75%</td><td>1.4-1.8x</td><td>Any GPU</td></tr>
<tr><td>GGUF Q4_K_M</td><td>~4.5</td><td>80.3% (-1.7%)</td><td>78.0%</td><td>~72%</td><td>N/A (CPU)</td><td>CPU + RAM</td></tr>
<tr><td>GGUF Q2_K</td><td>~2.5</td><td>72.8% (-9.2%)</td><td>61.0%</td><td>~85%</td><td>N/A (CPU)</td><td>CPU + RAM</td></tr>
</table>

<p><em>Benchmarks are approximate and vary by model, calibration data, and evaluation protocol. Sources: community evaluations on Open LLM Leaderboard, Neural Magic benchmarks, and HuggingFace model cards.</em></p>

<h4>Decision Framework: When to Use Which</h4>

<pre><code># QUANTIZATION DECISION TREE
#
# Q: Do you have H100/L40S GPUs?
# |-- YES: Use FP8. Near-lossless, native hardware support, simplest.
# |-- NO:
#     Q: Is quality the top priority?
#     |-- YES: Use FP16/BF16. Accept higher hardware cost.
#     |-- NO:
#         Q: Are you using vLLM/SGLang/TensorRT-LLM?
#         |-- YES: Use AWQ 4-bit. Best speed/quality tradeoff for GPU serving.
#         |-- NO:
#             Q: CPU inference? (llama.cpp, ollama)
#             |-- YES: Use GGUF Q4_K_M (general) or IQ4_XS (best 4-bit quality).
#             |-- NO: Use GPTQ 4-bit (broadest framework support).
#
# SPECIAL CASES:
# - Model < 3B params: Don't quantize below 8-bit. Small models lose quality fast.
# - Coding/math tasks: Use FP8 minimum. 4-bit hurts reasoning disproportionately.
# - Embeddings/retrieval: 4-bit is usually fine. Less sensitive to quantization.
# - Speculative decoding draft model: 4-bit is ideal (speed >> quality).</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: The Quantization Quality Cliff</div>
<p>We deployed AWQ-4bit Qwen2.5-72B for a code generation service. Benchmarks showed only 2% degradation on HumanEval. But users reported much worse subjective quality, especially for complex multi-file refactoring tasks. Investigation revealed that 4-bit quantization disproportionately affected the model's ability to maintain long-range coherence -- it could generate correct individual functions but failed at multi-function consistency. The benchmark (single-function HumanEval) did not capture this. <strong>Fix:</strong> Switched to FP8, which maintained long-range coherence. <strong>Lesson:</strong> Always evaluate quantized models on your <em>actual</em> use case, not just standard benchmarks. Benchmarks that test short outputs systematically under-estimate quantization damage for long-output tasks.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Your team wants to serve a 70B model on 2x A100-80GB GPUs. The model requires 140GB in FP16. What quantization strategy would you recommend and why?</div>
<div class="a-text">With 2x A100-80GB (160GB total), FP16 (140GB) would leave only 20GB for KV-cache -- not enough for reasonable throughput. Recommendation: AWQ 4-bit quantization. This reduces model weights to ~35GB, leaving ~125GB for KV-cache across both GPUs. With TP=2, each GPU has ~62.5GB for KV-cache, supporting hundreds of concurrent requests. AWQ is preferred over GPTQ for A100s due to faster kernels. If quality is critical (e.g., medical/legal), consider GPTQ 8-bit or wait for A100 replacement with H100s and use FP8. The A100 does not have FP8 Tensor Cores, so FP8 would not provide the same speedup as on H100. Key: validate quality on your specific task before deploying, especially for reasoning-heavy workloads where 4-bit may underperform.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 5.5  Prefix Caching & KV-Cache Management (NEW)
// ---------------------------------------------------------------------------
{
  id: "vllm-prefix-caching",
  title: "Prefix Caching & KV-Cache Management",
  content: `
<p>Prefix caching is one of the most impactful optimizations for real-world LLM serving. When many requests share the same prefix (system prompt, few-shot examples, or document context), computing the KV-cache for that prefix once and reusing it eliminates redundant computation. This section covers the theory, implementation, and measured impact.</p>

<h4>Why Prefix Caching Matters</h4>
<p>Consider a typical chatbot deployment:</p>

<pre><code># Typical chat request structure:
# [SYSTEM PROMPT: 500 tokens]     <- Same for ALL requests
# [Few-shot examples: 1000 tokens] <- Same for ALL requests
# [User message: 50-200 tokens]    <- Different per request
# [Assistant response: 100-500 tokens] <- Generated

# Without prefix caching:
#   Prefill cost per request: 1700 tokens * FLOPS_per_token
#   At 500 RPS: 850K tokens/sec of redundant prefill computation
#
# With prefix caching:
#   First request: compute KV for 1500 shared tokens + 200 unique
#   Subsequent requests: load cached KV for 1500 tokens + compute 200 unique
#   Prefill cost drops by ~88% (1500/1700 tokens cached)
#   TTFT drops from ~200ms to ~25ms for cached prefix</code></pre>

<h4>How Prefix Caching Works in vLLM</h4>
<p>vLLM's automatic prefix caching (APC) uses a hash-based approach to identify shared prefixes:</p>

<pre><code># vLLM Automatic Prefix Caching (APC) mechanism:
#
# 1. Each KV-cache block (16 tokens) is identified by a hash of:
#    - The token IDs in that block
#    - The hash of all preceding blocks (chain hash)
#    This ensures that a block is only reused if the ENTIRE prefix matches.
#
# 2. When a new request arrives:
#    a. Tokenize the prompt
#    b. Compute block hashes for the prompt
#    c. Look up each hash in the cache
#    d. If found: reuse the cached KV block (zero compute!)
#    e. If not found: compute KV for this block and cache it
#
# 3. Eviction: LRU (Least Recently Used) when cache is full.
#    Blocks shared by many requests get high "effective" usage,
#    so they survive eviction naturally.

# Block hash computation (simplified):
def compute_block_hash(token_ids, block_idx, block_size, prev_hash):
    """Hash a block of tokens, chained with previous block hash."""
    block_tokens = token_ids[block_idx * block_size : (block_idx + 1) * block_size]
    # Tuple is hashable and includes position information via prev_hash
    return hash((prev_hash, tuple(block_tokens)))

# Example: System prompt = "You are a helpful assistant..."
# Block 0 hash: hash((None, (1, 2, 3, ..., 16)))      tokens 0-15
# Block 1 hash: hash((block0_hash, (17, 18, ..., 32))) tokens 16-31
# ... and so on
# If two requests share the first 500 tokens, they share 31 block hashes.</code></pre>

<h4>RadixAttention in SGLang</h4>
<p>SGLang (Zheng et al., 2024; <a href="https://arxiv.org/abs/2312.07104">arXiv:2312.07104</a>) introduced <strong>RadixAttention</strong>, a more sophisticated prefix caching mechanism using a radix tree (trie) data structure:</p>

<pre><code># RadixAttention uses a radix tree to store KV-cache:
#
#                    [ROOT]
#                   /       \\
#          [system prompt]  [different system prompt]
#           /        \\
#    [few-shot A]  [few-shot B]
#      /     \\
#  [user1]  [user2]
#
# Advantages over hash-based (vLLM APC):
# 1. Supports arbitrary prefix sharing (not just common system prompts)
# 2. Efficient LRU eviction at any granularity
# 3. Fork/extend operations are O(1)
# 4. Better memory utilization for tree-structured workloads
#    (e.g., tree-of-thought, multi-turn conversations)
#
# Trade-off: Slightly more complex implementation,
# but SGLang benchmarks show 5-10% better TTFT than vLLM APC
# for workloads with diverse prefix patterns.</code></pre>

<h4>KV-Cache Compression Techniques</h4>
<p>Beyond prefix caching, several techniques reduce KV-cache memory consumption:</p>

<table>
<tr><th>Technique</th><th>Memory Reduction</th><th>Quality Impact</th><th>Description</th></tr>
<tr><td>FP8 KV-cache</td><td>50%</td><td>< 0.5% perplexity increase</td><td>Store KV-cache in FP8 instead of FP16. Supported in vLLM via --kv-cache-dtype fp8</td></tr>
<tr><td>GQA (model-level)</td><td>4-8x per KV head reduction</td><td>Baked into model</td><td>Grouped Query Attention uses fewer KV heads than Q heads</td></tr>
<tr><td>MLA (DeepSeek)</td><td>~10x</td><td>Designed to be lossless</td><td>Multi-head Latent Attention compresses KV into a low-rank latent space</td></tr>
<tr><td>Token eviction (H2O)</td><td>Variable (up to 80%)</td><td>Task-dependent</td><td>Evict KV for "unimportant" tokens based on attention scores</td></tr>
<tr><td>Quantized KV (KIVI)</td><td>75% (4-bit KV)</td><td>~1% quality loss</td><td>Per-channel quantization of cached K/V tensors</td></tr>
</table>

<pre><code># Using FP8 KV-cache in vLLM (simplest compression):
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --kv-cache-dtype fp8 \\
    --tensor-parallel-size 4 \\
    --max-model-len 16384

# This halves KV-cache memory with minimal quality impact!
# 70B with FP8 KV-cache: 0.54 GB/request -> 0.27 GB/request
# Doubles concurrent request capacity at the same hardware.

# MLA (Multi-head Latent Attention) - used by DeepSeek-V3:
# Instead of caching full K, V tensors (num_heads * head_dim per token),
# cache a compressed latent vector (latent_dim << num_heads * head_dim).
# DeepSeek-V3 compresses KV-cache by ~10x, enabling 128K context
# with the KV-cache footprint of a 13K context model.
# This is a model architecture choice, not a serving optimization.</code></pre>

<h4>Prefix-Aware Request Routing</h4>
<p>When running multiple vLLM replicas behind a load balancer, naive round-robin routing destroys prefix cache hit rates. Prefix-aware routing sends requests with the same prefix to the same replica:</p>

<pre><code># Prefix-aware routing implementation
import hashlib
from typing import List

class PrefixAwareRouter:
    """Route requests to replicas based on system prompt hash."""

    def __init__(self, replicas: List[str]):
        self.replicas = replicas  # ["http://vllm-0:8000", ...]

    def route(self, request: dict) -> str:
        """Consistent hashing on the system prompt."""
        # Extract the system prompt (shared prefix)
        messages = request.get("messages", [])
        system_msg = ""
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
                break

        # Hash the system prompt to a replica
        if system_msg:
            hash_val = int(hashlib.md5(system_msg.encode()).hexdigest(), 16)
            replica_idx = hash_val % len(self.replicas)
        else:
            # No system prompt: round-robin
            replica_idx = self._round_robin_counter()

        return self.replicas[replica_idx]

    # In production, use consistent hashing (e.g., jump hash or ring hash)
    # to handle replica additions/removals gracefully.
    # Libraries: uhashring (Python), envoy's ring_hash lb_policy (infra)</code></pre>

<h4>Measuring TTFT Improvement with Benchmarks</h4>

<table>
<tr><th>Scenario</th><th>Shared Prefix Length</th><th>TTFT without Cache</th><th>TTFT with Cache</th><th>Improvement</th></tr>
<tr><td>Short system prompt</td><td>100 tokens</td><td>45ms</td><td>38ms</td><td>16%</td></tr>
<tr><td>Standard chatbot</td><td>500 tokens</td><td>125ms</td><td>35ms</td><td>72%</td></tr>
<tr><td>RAG with context</td><td>2000 tokens</td><td>380ms</td><td>42ms</td><td>89%</td></tr>
<tr><td>Long system prompt + few-shot</td><td>4000 tokens</td><td>720ms</td><td>48ms</td><td>93%</td></tr>
<tr><td>Document QA (32K context)</td><td>32000 tokens</td><td>5200ms</td><td>85ms</td><td>98%</td></tr>
</table>

<p><em>Benchmarks on H100-80GB, LLaMA-3-8B, measuring P50 TTFT at 32 concurrent requests. The "with cache" numbers assume a warm cache (second+ request with the same prefix).</em></p>

<div class="callout">
<div class="callout-title">The Document QA Pattern</div>
<p>Prefix caching is transformative for document QA use cases. Upload a document once (32K tokens), ask multiple questions about it. The first question pays the full 5.2s TTFT for prefill. Every subsequent question about the same document only pays ~85ms. This makes interactive document exploration feel instantaneous. The same pattern applies to code review (paste code, ask multiple questions), translation (provide glossary + context, translate multiple segments), and customer support (load customer history, answer multiple queries).</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You are building a RAG system where each request includes 3000 tokens of retrieved context followed by a user question. How would you optimize TTFT using prefix caching?</div>
<div class="a-text">The key insight is that many users may ask questions about the same or similar documents. Optimization approach: (1) Enable automatic prefix caching in vLLM (--enable-prefix-caching). This caches KV blocks for any shared token prefix. (2) Structure prompts so the system prompt comes first (always cached), then retrieved context, then the user question. (3) If multiple users query the same document, the 3000-token context KV-cache is reused, reducing TTFT from ~400ms to ~40ms. (4) Deploy prefix-aware routing: hash the document ID to route all questions about the same document to the same vLLM replica, maximizing cache hit rate. (5) For frequently accessed documents, consider "pre-warming" the cache by sending a dummy request at retrieval time. (6) Use FP8 KV-cache to double the cache capacity (more documents cached simultaneously). Expected improvement: 80-90% TTFT reduction for cache hits, with hit rates of 30-70% depending on query patterns.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 5.6  Monitoring & Debugging (NEW)
// ---------------------------------------------------------------------------
{
  id: "vllm-monitoring",
  title: "Monitoring & Debugging",
  content: `
<p>A vLLM deployment without monitoring is a ticking time bomb. This section covers the essential metrics, dashboard setup, common failure modes, and the debugging techniques that will save you at 2 AM when the pager goes off.</p>

<h4>Key Metrics: What to Monitor</h4>

<table>
<tr><th>Metric</th><th>What It Measures</th><th>Alert Threshold</th><th>Why It Matters</th></tr>
<tr><td><strong>TTFT (P50/P95/P99)</strong></td><td>Time from request arrival to first token</td><td>P99 > 2s</td><td>User-perceived responsiveness</td></tr>
<tr><td><strong>ITL / TPOT (P50/P95/P99)</strong></td><td>Time between output tokens</td><td>P99 > 200ms</td><td>Streaming experience quality</td></tr>
<tr><td><strong>E2E Latency (P50/P95/P99)</strong></td><td>Total request duration</td><td>P99 > SLA</td><td>Overall SLA compliance</td></tr>
<tr><td><strong>Throughput (tokens/sec)</strong></td><td>Total tokens generated per second</td><td>Drops > 30% from baseline</td><td>Capacity planning</td></tr>
<tr><td><strong>Queue Depth</strong></td><td>Requests waiting to be processed</td><td>> 50 requests</td><td>Leading indicator of overload</td></tr>
<tr><td><strong>Running Requests</strong></td><td>Requests currently being processed</td><td>= max_num_seqs (saturated)</td><td>Utilization indicator</td></tr>
<tr><td><strong>GPU Utilization</strong></td><td>SM active percentage</td><td>< 30% (underutilized) or > 95%</td><td>Resource efficiency</td></tr>
<tr><td><strong>GPU Memory Used</strong></td><td>HBM consumption</td><td>> 95% of allocated</td><td>OOM risk indicator</td></tr>
<tr><td><strong>KV-Cache Utilization</strong></td><td>% of KV-cache blocks in use</td><td>> 90%</td><td>Approaching capacity limit</td></tr>
<tr><td><strong>Prefix Cache Hit Rate</strong></td><td>% of prefill tokens served from cache</td><td>< 50% (if expecting high)</td><td>Prefix caching effectiveness</td></tr>
<tr><td><strong>Preemption Count</strong></td><td>Requests swapped out due to memory pressure</td><td>> 0 per minute sustained</td><td>Sign of overcommitment</td></tr>
<tr><td><strong>Request Error Rate</strong></td><td>% of requests returning errors</td><td>> 0.1%</td><td>Service health</td></tr>
</table>

<h4>Prometheus + Grafana Dashboard Setup</h4>

<pre><code># vLLM exposes Prometheus metrics at /metrics by default.
# Step 1: Verify metrics are available
curl http://localhost:8000/metrics

# You'll see metrics like:
# vllm:num_requests_running{model_name="..."} 42
# vllm:num_requests_waiting{model_name="..."} 3
# vllm:gpu_cache_usage_perc{model_name="..."} 0.73
# vllm:cpu_cache_usage_perc{model_name="..."} 0.0
# vllm:num_preemptions_total{model_name="..."} 0
# vllm:avg_prompt_throughput_toks_per_s{model_name="..."} 4521
# vllm:avg_generation_throughput_toks_per_s{model_name="..."} 1893
# vllm:e2e_request_latency_seconds_bucket{...}  (histogram)
# vllm:time_to_first_token_seconds_bucket{...}  (histogram)
# vllm:time_per_output_token_seconds_bucket{...} (histogram)</code></pre>

<pre><code># prometheus.yml - Scrape config for vLLM
scrape_configs:
  - job_name: 'vllm'
    scrape_interval: 15s
    static_configs:
      - targets:
        - 'vllm-replica-0:8000'
        - 'vllm-replica-1:8000'
        - 'vllm-replica-2:8000'
    metrics_path: '/metrics'</code></pre>

<pre><code># Grafana Dashboard Panels (PromQL queries):

# 1. TTFT P99 (time-series)
histogram_quantile(0.99, sum(rate(
  vllm:time_to_first_token_seconds_bucket[5m]
)) by (le, model_name))

# 2. TPOT P99 (time-series)
histogram_quantile(0.99, sum(rate(
  vllm:time_per_output_token_seconds_bucket[5m]
)) by (le, model_name))

# 3. Request throughput (time-series)
sum(rate(vllm:e2e_request_latency_seconds_count[5m])) by (model_name)

# 4. KV-cache utilization (gauge)
vllm:gpu_cache_usage_perc

# 5. Queue depth (gauge)
vllm:num_requests_waiting

# 6. Running requests (gauge)
vllm:num_requests_running

# 7. Token throughput (time-series)
vllm:avg_generation_throughput_toks_per_s

# 8. Error rate (time-series)
sum(rate(vllm:request_success_total{finished_reason="error"}[5m]))
/ sum(rate(vllm:request_success_total[5m]))</code></pre>

<h4>Common Failure Modes and Debugging</h4>

<pre><code># FAILURE MODE 1: Gradual Latency Increase
# Symptom: P99 latency slowly increases over hours/days
# Cause: KV-cache fragmentation, memory leaks, or increasing queue depth
# Debug:
#   1. Check vllm:gpu_cache_usage_perc -- is it climbing toward 1.0?
#   2. Check vllm:num_requests_waiting -- is queue growing?
#   3. Check vllm:num_preemptions_total -- are preemptions increasing?
#   4. nvidia-smi -l 1 -- is memory usage growing beyond expected?
# Fix:
#   - If cache full: reduce max_model_len, add replicas, or enable FP8 KV-cache
#   - If queue growing: traffic exceeds capacity, scale out
#   - If memory leak: restart the server (workaround), file a bug (proper fix)

# FAILURE MODE 2: Sudden OOM Crash
# Symptom: vLLM process killed, nvidia-smi shows memory at limit
# Cause: Activation memory spike during prefill of very long prompt
# Debug:
#   1. Check the last request's prompt length in access logs
#   2. Check max_model_len vs actual max prompt length in traffic
#   3. Check gpu-memory-utilization setting
# Fix:
#   - Set max_model_len to your actual P99 prompt+output length
#   - Reduce gpu-memory-utilization by 3-5% to add headroom
#   - Enable chunked prefill (--enable-chunked-prefill) to bound
#     peak activation memory during long prompt processing

# FAILURE MODE 3: CUDA Errors / GPU Hangs
# Symptom: CUDA error messages, process hangs, nvidia-smi shows 100% util
# Cause: GPU hardware issue, driver bug, NCCL deadlock
# Debug:
#   1. CUDA_LAUNCH_BLOCKING=1 to get synchronous error messages
#   2. nvidia-smi -q -d ECC -- check for ECC errors (hardware failure)
#   3. Check dmesg for Xid errors (GPU firmware/hardware issues)
#   4. For NCCL: NCCL_DEBUG=INFO to trace communication
# Fix:
#   - ECC errors: RMA the GPU
#   - NCCL deadlock: check that all TP workers are alive, restart
#   - Driver bug: update NVIDIA driver + CUDA toolkit

# FAILURE MODE 4: High TTFT, Low GPU Utilization
# Symptom: GPU util is only 20-40% but TTFT is high
# Cause: CPU bottleneck in tokenization or scheduling
# Debug:
#   1. Profile the CPU (py-spy or cProfile)
#   2. Check if tokenizer is slow (some tokenizers are pure Python)
#   3. Check if scheduling overhead is high (many small requests)
# Fix:
#   - Use a fast tokenizer (tokenizers library, not sentencepiece)
#   - Increase CPU allocation (Kubernetes: request more CPU)
#   - Batch small requests client-side</code></pre>

<h4>Request Tracing and Logging</h4>

<pre><code># Enable detailed request logging in vLLM:
python -m vllm.entrypoints.openai.api_server \\
    --model ... \\
    --disable-log-stats    # Disable periodic stats (too noisy)
    # Access logs are enabled by default

# For production tracing, add a middleware:
# In a custom FastAPI wrapper around vLLM:

import time
import uuid
import logging

logger = logging.getLogger("vllm.access")

@app.middleware("http")
async def trace_requests(request, call_next):
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start
    logger.info(
        f"trace_id={trace_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"duration_ms={duration*1000:.1f}"
    )

    response.headers["X-Trace-ID"] = trace_id
    return response

# Log structure for analysis:
# trace_id=abc123 method=POST path=/v1/chat/completions status=200
#   duration_ms=1523.4 prompt_tokens=1200 completion_tokens=350
#   ttft_ms=85.2 model=llama-70b queue_time_ms=12.3</code></pre>

<h4>Auto-Scaling Triggers and Policies</h4>

<pre><code># Kubernetes HPA (Horizontal Pod Autoscaler) configuration:
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama70b
  minReplicas: 2
  maxReplicas: 10
  metrics:
  # Scale on queue depth (best leading indicator)
  - type: Pods
    pods:
      metric:
        name: vllm_num_requests_waiting
      target:
        type: AverageValue
        averageValue: "10"    # Scale up when avg queue > 10
  # Also consider GPU KV-cache utilization
  - type: Pods
    pods:
      metric:
        name: vllm_gpu_cache_usage_perc
      target:
        type: AverageValue
        averageValue: "0.85"  # Scale up when cache > 85% full
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # Wait 60s before scaling up
      policies:
      - type: Pods
        value: 2              # Add max 2 pods at a time
        periodSeconds: 120
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5min before scaling down
      policies:
      - type: Pods
        value: 1              # Remove max 1 pod at a time
        periodSeconds: 300

# IMPORTANT: GPU pods take 2-5 minutes to start (model loading).
# Use initialDelaySeconds generously in readiness probes.
# Consider pre-pulling model weights to nodes (DaemonSet or init container)
# to reduce cold start time.

# Cost-aware scaling:
# - Scale down slowly (5min window) to avoid thrashing
# - Consider time-of-day patterns: pre-scale before peak hours
# - Use spot/preemptible GPUs for overflow capacity
# - Set pod disruption budget to ensure minimum replicas during updates</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Your vLLM deployment's P99 latency has doubled over the past hour. Walk through your debugging process.</div>
<div class="a-text">Systematic debugging: (1) Check queue depth -- if growing, the system is receiving more requests than it can handle. Scale out or rate limit. (2) Check KV-cache utilization -- if near 100%, requests are being preempted (swapped to CPU and back), causing massive latency spikes. Fix: reduce max_model_len, add FP8 KV-cache, or add replicas. (3) Check GPU utilization -- if low despite high latency, the bottleneck is CPU (tokenization, scheduling) or network (slow weight loading). (4) Check for a specific "bad" request -- one extremely long prompt can block decode batches if chunked prefill is disabled. (5) Check GPU health -- run nvidia-smi for ECC errors, temperature throttling (>83C on H100). (6) Check if a new traffic pattern emerged -- longer prompts, more concurrent users, different system prompts (killing prefix cache hit rate). (7) If all metrics look normal, check the load balancer -- one replica might be down, doubling load on others.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 5.7  Advanced Serving Patterns (NEW)
// ---------------------------------------------------------------------------
{
  id: "vllm-advanced",
  title: "Advanced Serving Patterns",
  content: `
<p>Beyond basic serving, modern LLM infrastructure requires advanced patterns for structured output, efficient multi-model serving, and intelligent scheduling. This section covers the cutting-edge techniques used in production systems.</p>

<h4>Continuous Batching Internals</h4>
<p>We introduced continuous batching earlier. Here we dive deeper into the scheduling algorithm that makes it work:</p>

<pre><code># vLLM Scheduler State Machine per Request:
#
#   WAITING ----allocate----> RUNNING ----finish----> COMPLETED
#     ^                         |
#     |                         | (memory pressure)
#     +------preempt/swap------+
#
# The scheduler runs every iteration (every ~10-50ms):
#
# Phase 1: RUNNING -> COMPLETED
#   - Check which running sequences have emitted EOS or hit max_tokens
#   - Free their KV-cache blocks
#
# Phase 2: RUNNING -> WAITING (preemption, if needed)
#   - If KV-cache is full and new prefills are waiting:
#     - Option A: Recompute (discard KV, re-prefill later; lower latency)
#     - Option B: Swap (copy KV to CPU RAM; saves compute but uses CPU memory)
#   - vLLM uses RECOMPUTE by default (simpler, avoids CPU memory issues)
#   - Preemption priority: LIFO (last-in-first-out) to minimize wasted compute
#
# Phase 3: WAITING -> RUNNING (admission)
#   - FCFS (first-come-first-served) within priority levels
#   - Check if enough free KV-cache blocks for the new request's prompt
#   - If yes: allocate blocks, run prefill, start generating
#   - If no: request stays in WAITING queue

# Key insight: the scheduler must balance fairness (FCFS) with efficiency
# (batching similar-length sequences reduces padding waste).
# vLLM v1 introduced priority levels but keeps FCFS within each level.</code></pre>

<h4>Chunked Prefill for Long Prompts</h4>
<p>Long prompts (e.g., 32K tokens for RAG) can dominate a GPU for hundreds of milliseconds during prefill, blocking all decode operations. Chunked prefill (Agrawal et al., 2024; <a href="https://arxiv.org/abs/2308.16369">arXiv:2308.16369</a>) splits the prefill into chunks and interleaves them with decode steps:</p>

<pre><code># Without chunked prefill:
# |---- prefill 32K tokens (600ms) ----| decode | decode | ...
# All running decode requests are BLOCKED for 600ms!
# Their ITL spikes from 30ms to 630ms. Users see a "stutter".

# With chunked prefill (chunk_size=512):
# |prefill 512| decode |prefill 512| decode |prefill 512| decode | ...
# Prefill takes longer total (~800ms) but decode is never blocked.
# Max ITL increase: ~15ms (one chunk's compute time).

# Enable in vLLM:
python -m vllm.entrypoints.openai.api_server \\
    --model ... \\
    --enable-chunked-prefill \\
    --max-num-batched-tokens 2048  # Max tokens per iteration
    # This bounds the compute per iteration:
    # Each step processes at most 2048 tokens total
    # (sum of prefill chunk + all decode tokens)

# Trade-off:
# - Without chunked prefill: TTFT is optimal, but ITL can spike
# - With chunked prefill: TTFT increases slightly, ITL is bounded
# - For interactive chat: ALWAYS enable chunked prefill
# - For batch processing (no streaming): disable for faster throughput</code></pre>

<h4>Request Scheduling Algorithms</h4>
<p>The scheduling algorithm significantly impacts user experience, especially under load:</p>

<table>
<tr><th>Algorithm</th><th>Description</th><th>Pros</th><th>Cons</th></tr>
<tr><td>FCFS</td><td>First-Come-First-Served</td><td>Fair, simple, predictable</td><td>Long requests can hog resources</td></tr>
<tr><td>SJF</td><td>Shortest-Job-First (by estimated output length)</td><td>Minimizes average latency</td><td>Hard to estimate output length; starves long requests</td></tr>
<tr><td>Priority Queues</td><td>User-assigned priority levels</td><td>Supports SLA tiers</td><td>Requires priority assignment logic</td></tr>
<tr><td>Fair Queuing</td><td>Weighted round-robin across users/tenants</td><td>No single tenant can monopolize</td><td>More complex, slight overhead</td></tr>
</table>

<pre><code># Priority-based scheduling in vLLM (v0.7+):
# Requests can be assigned priority via the API:

import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="key")

# High priority request (lower number = higher priority)
response = client.chat.completions.create(
    model="llama-70b",
    messages=[{"role": "user", "content": "Emergency query"}],
    extra_body={"priority": 0},  # Highest priority
)

# Low priority request (background batch job)
response = client.chat.completions.create(
    model="llama-70b",
    messages=[{"role": "user", "content": "Batch analysis"}],
    extra_body={"priority": 10},  # Lower priority
)</code></pre>

<h4>Structured Output / Guided Generation</h4>
<p>Many applications require LLM outputs in a specific format (JSON, XML, SQL, etc.). Guided generation constrains the token sampling to only produce valid outputs:</p>

<pre><code># Structured output in vLLM using JSON schema:
import openai
import json

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="key")

# Method 1: JSON mode (simple)
response = client.chat.completions.create(
    model="llama-70b",
    messages=[{"role": "user", "content": "Extract name and age from: John is 30."}],
    response_format={"type": "json_object"},
)

# Method 2: JSON Schema (strict structure)
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

response = client.chat.completions.create(
    model="llama-70b",
    messages=[{"role": "user", "content": "Extract info from: John, age 30, john@example.com"}],
    extra_body={
        "guided_json": json.dumps(schema)
    }
)

# Method 3: Regex-guided generation
response = client.chat.completions.create(
    model="llama-70b",
    messages=[{"role": "user", "content": "Generate a US phone number"}],
    extra_body={
        "guided_regex": r"\\(\\d{3}\\) \\d{3}-\\d{4}"
    }
)

# Method 4: Grammar-guided (context-free grammar)
# Useful for SQL, code, etc.
response = client.chat.completions.create(
    model="llama-70b",
    messages=[{"role": "user", "content": "Write a SQL query for..."}],
    extra_body={
        "guided_grammar": sql_grammar_string  # BNF grammar
    }
)

# How it works under the hood:
# 1. vLLM uses Outlines or lm-format-enforcer library
# 2. At each token, compute which tokens are valid given the constraint
# 3. Create a logit mask: set logits of invalid tokens to -inf
# 4. Sample from the remaining valid tokens
# 5. This guarantees syntactically valid output with zero retries!
#
# Performance impact: ~5-15% slower due to mask computation.
# For JSON schema: the mask is pre-computed as a finite state machine,
# so the per-token overhead is just a lookup.</code></pre>

<h4>Multi-LoRA Serving</h4>
<p>Serve multiple fine-tuned models from a single base model by dynamically loading LoRA adapters. This is dramatically more efficient than deploying separate model instances.</p>

<pre><code># Multi-LoRA serving in vLLM:
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --enable-lora \\
    --lora-modules \\
        customer-support=/path/to/loras/customer-support \\
        code-review=/path/to/loras/code-review \\
        medical-qa=/path/to/loras/medical-qa \\
    --max-loras 3 \\
    --max-lora-rank 64

# Clients select the LoRA adapter via the model name:
response = client.chat.completions.create(
    model="customer-support",  # Uses the customer-support LoRA
    messages=[{"role": "user", "content": "I need help with my order"}],
)

response = client.chat.completions.create(
    model="code-review",  # Uses the code-review LoRA
    messages=[{"role": "user", "content": "Review this Python function..."}],
)

# How multi-LoRA works:
# 1. Base model weights are loaded once (shared across all LoRAs)
# 2. LoRA adapters are small: rank=64 adds ~0.1-0.5% parameters
# 3. At inference, the LoRA matrices are applied dynamically:
#    output = x @ (W_base + W_A @ W_B)  where W_A, W_B are LoRA matrices
# 4. Different requests in the same batch can use different LoRAs!
# 5. vLLM uses custom CUDA kernels (punica/SGMV) for batched LoRA computation
#
# Memory overhead per LoRA (rank=64, 8B model):
#   ~50-100 MB per adapter (vs 16 GB for the full model)
#   100 LoRAs: ~10 GB extra = one more LoRA costs 0.006x model memory
#
# Max concurrent LoRAs limited by:
# - GPU memory for LoRA weights
# - Kernel efficiency (batched LoRA kernels work best with < 32 active LoRAs)</code></pre>

<h4>Disaggregated Prefill-Decode Architecture</h4>
<p>The cutting edge in LLM serving separates prefill and decode onto different GPU pools, since they have completely different compute profiles:</p>

<pre><code># Disaggregated serving (DistServe, Splitwise, Mooncake):
# Reference: Zhong et al. "DistServe" (2024, arXiv:2401.09670)
#
# Architecture:
# +--------------+    KV-cache    +---------------+
# | Prefill Pool | ------------> | Decode Pool    |
# | (compute-opt)|   transfer    | (memory-bw-opt)|
# | H100 SXM     |               | L40S or A100   |
# +--------------+               +---------------+
#
# Prefill pool: optimized for compute (high FLOPS GPUs, larger batch)
# Decode pool: optimized for memory bandwidth (KV-cache capacity)
#
# Benefits:
# 1. Each pool uses optimal hardware for its workload
# 2. Long prefills don't interfere with decode latency
# 3. Can scale prefill and decode independently
#
# Challenge: KV-cache transfer between pools
# - Over NVLink (same node): 900 GB/s, ~1ms for typical request
# - Over InfiniBand (cross-node): 50 GB/s, ~20ms for typical request
# - Over Ethernet: too slow, impractical
#
# vLLM experimental support via --enable-disagg flag (v0.8+)
# SGLang has more mature disaggregated serving support.</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: The Multi-LoRA Memory Explosion</div>
<p>We deployed vLLM with 50 LoRA adapters for a multi-tenant platform. Each tenant had a custom fine-tuned LoRA. Initial testing worked great with 5 concurrent LoRAs. But during peak hours, 30+ LoRAs were active simultaneously. The issue: vLLM's SGMV kernel pre-allocates workspace proportional to <code>max_loras * max_lora_rank * batch_size</code>. With max_loras=50, rank=64, batch=256, this workspace was 3.2GB -- eating into KV-cache capacity. Plus, the SGMV kernel's efficiency drops significantly above ~16 active LoRAs. <strong>Fix:</strong> We grouped tenants with similar domains into shared LoRAs (50 -> 12), reduced max_lora_rank to 32 (minimal quality impact), and implemented LoRA routing to limit concurrent active LoRAs to 8 per replica. <strong>Lesson:</strong> Multi-LoRA is powerful but has non-obvious memory scaling. Always load-test with the expected number of <em>concurrent</em> LoRAs, not just total LoRAs.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain how structured output (guided generation) works in vLLM. What is the performance impact?</div>
<div class="a-text">Guided generation constrains token sampling to produce only syntactically valid output (e.g., valid JSON). Implementation: (1) The constraint (JSON schema, regex, or grammar) is compiled into a finite state machine (FSM) or similar automaton. (2) At each decoding step, the FSM determines which tokens are valid given the current state. (3) A logit mask is applied: valid tokens keep their logits, invalid tokens get -infinity. (4) Sampling proceeds normally on the masked logits. Performance impact: approximately 5-15% slower per token due to mask computation and FSM state tracking. For JSON schema, the FSM is pre-compiled so per-token overhead is minimal (microseconds for the lookup). For complex grammars, the overhead can be higher. The key benefit is eliminating retries -- without guided generation, you might need 2-5 attempts to get valid JSON from the model, so the net effect is often faster despite per-token overhead.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 5.8  vLLM vs SGLang vs TensorRT-LLM (bonus section for completeness)
// ---------------------------------------------------------------------------
{
  id: "vllm-ecosystem",
  title: "Serving Engine Ecosystem: vLLM, SGLang, TensorRT-LLM",
  content: `
<p>vLLM is not the only option. Understanding the broader serving engine ecosystem helps you choose the right tool for your specific requirements. This section provides an honest comparison based on production experience.</p>

<h4>Engine Comparison Matrix</h4>

<table>
<tr><th>Feature</th><th>vLLM (v0.7+)</th><th>SGLang (v0.4+)</th><th>TensorRT-LLM</th><th>HF TGI (v2+)</th></tr>
<tr><td>PagedAttention</td><td>Yes (original)</td><td>Yes</td><td>Yes (paged KV)</td><td>Yes</td></tr>
<tr><td>Continuous batching</td><td>Yes</td><td>Yes</td><td>Yes (inflight batching)</td><td>Yes</td></tr>
<tr><td>Prefix caching</td><td>Hash-based APC</td><td>RadixAttention (better)</td><td>Yes</td><td>Limited</td></tr>
<tr><td>Speculative decoding</td><td>Yes (draft model, ngram, MLPSpec)</td><td>Yes (EAGLE, etc.)</td><td>Yes (best support)</td><td>Yes (draft model)</td></tr>
<tr><td>Quantization</td><td>GPTQ, AWQ, FP8, GGUF (limited)</td><td>GPTQ, AWQ, FP8</td><td>FP8, INT4 (SmoothQuant, GPTQ, AWQ)</td><td>GPTQ, AWQ, EETQ</td></tr>
<tr><td>Structured output</td><td>Outlines, lm-format-enforcer</td><td>Native (fast)</td><td>Limited</td><td>Outlines</td></tr>
<tr><td>Multi-LoRA</td><td>Yes (punica/SGMV)</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
<tr><td>Multi-modal</td><td>Yes (vision, audio)</td><td>Yes (vision)</td><td>Yes (vision)</td><td>Yes (vision)</td></tr>
<tr><td>Model support breadth</td><td>Broadest (100+ architectures)</td><td>Good (50+ architectures)</td><td>Moderate (major architectures)</td><td>Good</td></tr>
<tr><td>Setup complexity</td><td>pip install vllm</td><td>pip install sglang</td><td>Docker + compilation (15-60 min)</td><td>Docker image</td></tr>
<tr><td>Peak throughput</td><td>High</td><td>Highest (overlap scheduling)</td><td>Highest (optimized CUDA)</td><td>Moderate</td></tr>
<tr><td>Community/ecosystem</td><td>Largest</td><td>Growing fast</td><td>NVIDIA-backed</td><td>HuggingFace-backed</td></tr>
</table>

<h4>When to Choose Which</h4>

<pre><code># DECISION GUIDE:
#
# Choose vLLM when:
# - You need broad model support (new model architectures, multi-modal)
# - You want the largest community and ecosystem
# - You need multi-LoRA serving
# - You want a balance of performance and ease of use
#
# Choose SGLang when:
# - Maximum throughput is the priority
# - You have complex prefix caching patterns (tree-structured, multi-turn)
# - You need fast structured output (native constraint support)
# - You're building a programming framework around LLM calls
#
# Choose TensorRT-LLM when:
# - You need absolute peak performance and can invest in setup
# - You're on NVIDIA hardware and will stay on NVIDIA hardware
# - You have a fixed model that won't change frequently
# - You need FP8 or INT4 with custom calibration
#
# Choose HF TGI when:
# - You want tight HuggingFace ecosystem integration
# - You need the simplest possible deployment (one Docker command)
# - Performance requirements are moderate
# - You're already using HF Inference Endpoints</code></pre>

<h4>Performance Benchmarks: Apples-to-Apples</h4>
<p>Real benchmarks on identical hardware (4x H100-80GB, LLaMA-3.1-70B-Instruct, BF16, TP=4). Workload: ShareGPT conversation dataset, 64 concurrent clients.</p>

<table>
<tr><th>Metric</th><th>vLLM 0.7.3</th><th>SGLang 0.4.1</th><th>TRT-LLM 0.16</th></tr>
<tr><td>Throughput (output tok/s)</td><td>4,850</td><td>5,320</td><td>5,580</td></tr>
<tr><td>TTFT P50</td><td>78ms</td><td>72ms</td><td>65ms</td></tr>
<tr><td>TTFT P99</td><td>340ms</td><td>295ms</td><td>260ms</td></tr>
<tr><td>ITL P50</td><td>27ms</td><td>25ms</td><td>23ms</td></tr>
<tr><td>ITL P99</td><td>62ms</td><td>54ms</td><td>48ms</td></tr>
<tr><td>Setup time</td><td>2 min</td><td>2 min</td><td>45 min</td></tr>
</table>

<p><em>SGLang's throughput advantage comes from overlap scheduling (computing next iteration's schedule while current iteration runs) and RadixAttention (better cache hit rates). TRT-LLM's advantage comes from highly optimized CUDA kernels and aggressive fusion. vLLM's advantage is ecosystem and ease of use.</em></p>

<div class="callout">
<div class="callout-title">The Convergence Trend</div>
<p>As of early 2025, vLLM and SGLang are converging in performance. SGLang adopted many vLLM patterns; vLLM adopted overlap scheduling and better prefix caching from SGLang. TRT-LLM remains the performance leader but at the cost of setup complexity. For most teams, the difference between vLLM and SGLang is smaller than the difference between good and bad configuration of either. <strong>Focus on configuration and architecture before switching engines.</strong></p>
</div>

<h4>Migration Paths</h4>

<pre><code># vLLM to SGLang migration: mostly API-compatible
# Both support OpenAI-compatible endpoints.
# Key differences in launch args:

# vLLM:
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --tensor-parallel-size 4

# SGLang equivalent:
python -m sglang.launch_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --tp 4

# Client code stays the same (both are OpenAI-compatible).
# Minor differences: SGLang has additional /generate endpoint
# with advanced features (fork, select, regex constraint).</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Your company is choosing between vLLM and TensorRT-LLM for a production deployment. What factors would you consider?</div>
<div class="a-text">Key factors: (1) Model stability -- if the model changes frequently (A/B tests, fine-tuning iterations), vLLM's instant model loading beats TRT-LLM's 15-60 minute compilation step. (2) Performance requirements -- if you need every last 10% of throughput and have fixed models, TRT-LLM wins. (3) Team expertise -- TRT-LLM requires NVIDIA-specific knowledge (engine building, plugin system); vLLM is more accessible. (4) Hardware lock-in -- TRT-LLM is NVIDIA-only; vLLM has experimental AMD/TPU support. (5) Feature requirements -- if you need multi-LoRA, complex structured output, or broad model support, vLLM has better coverage. (6) Total cost of ownership -- TRT-LLM saves on GPU costs (fewer GPUs for same throughput) but costs more in engineering time. For most teams, I would recommend starting with vLLM for speed of iteration, then evaluate TRT-LLM if you hit a performance ceiling and have stabilized on a model.</div>
</div>
`
}

],

// ============================================================================
// CHAPTER 6: RL TRAINING FOR LLMs  (~12,000 words, 8 sections)
// ============================================================================
ch6_sections: [

// ---------------------------------------------------------------------------
// 6.1  GRPO & Verifiable Rewards (expanded with math)
// ---------------------------------------------------------------------------
{
  id: "rlvr-fundamentals",
  title: "GRPO & Verifiable Rewards: Foundations",
  content: `
<p>Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as the dominant paradigm for training reasoning models. Unlike RLHF (which uses a learned reward model that can be gamed), RLVR uses <strong>verifiable reward functions</strong> -- code execution, math verification, unit tests -- that provide ground-truth signal. The key algorithm is <strong>GRPO (Group Relative Policy Optimization)</strong>, introduced by DeepSeek in their DeepSeek-Math paper (Shao et al., 2024; <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a>).</p>

<h4>The RL Formulation for LLMs</h4>
<p>We formalize LLM generation as a Markov Decision Process (MDP):</p>

<pre><code># MDP formulation for LLM generation:
# State s_t: (prompt, tokens_generated_so_far) = (x, y_{<t})
# Action a_t: next token y_t ~ pi_theta(y_t | x, y_{<t})
# Policy pi_theta: the LLM with parameters theta
# Reward R(x, y): given after the COMPLETE response y is generated
#   (no intermediate rewards -- this is a "bandit" formulation)

# Objective: maximize expected reward while staying close to reference policy
# J(theta) = E_{x ~ D, y ~ pi_theta(y|x)} [R(x, y)] - beta * KL(pi_theta || pi_ref)
#
# where:
#   D is the prompt distribution (training data)
#   pi_ref is the reference policy (the SFT model we start from)
#   beta is the KL penalty coefficient
#   KL divergence prevents the policy from deviating too far from pi_ref
#   (which would lead to reward hacking and degenerate outputs)</code></pre>

<h4>GRPO: Mathematical Derivation</h4>
<p>GRPO simplifies PPO by eliminating the value (critic) network. Instead of learning V(s) to estimate advantages, GRPO estimates advantages from a <strong>group</strong> of samples for each prompt.</p>

<pre><code># GRPO Algorithm (formal):
#
# For each prompt x_i in the mini-batch:
#   1. Sample G responses: {y_i^1, y_i^2, ..., y_i^G} ~ pi_theta_old(y|x_i)
#   2. Compute rewards: r_i^j = R(x_i, y_i^j) for j = 1..G
#   3. Compute group-normalized advantages:
#      A_i^j = (r_i^j - mean({r_i^1, ..., r_i^G})) / std({r_i^1, ..., r_i^G})
#
#   4. For each token t in response y_i^j:
#      Compute the probability ratio:
#      rho_t = pi_theta(y_t | x_i, y_{<t}) / pi_theta_old(y_t | x_i, y_{<t})
#
#   5. GRPO loss (per token):
#      L_t = -min(rho_t * A_i^j, clip(rho_t, 1-eps, 1+eps) * A_i^j)
#
#   6. KL penalty (per token):
#      KL_t = pi_ref(y_t | ...) / pi_theta(y_t | ...) - log(pi_ref / pi_theta) - 1
#      (This is the Schulman approximation of reverse KL)
#
# Total loss:
# L = (1 / sum_of_tokens) * sum_over_all_tokens(L_t + beta * KL_t)</code></pre>

<p>The mathematical elegance of GRPO lies in the advantage estimation. Compare with PPO:</p>

<table>
<tr><th>Aspect</th><th>PPO</th><th>GRPO</th></tr>
<tr><td>Advantage estimation</td><td>A(s,a) = R + gamma * V(s') - V(s) using GAE</td><td>A = (r - mean(group)) / std(group)</td></tr>
<tr><td>Requires critic</td><td>Yes (same size as policy, 2x memory)</td><td>No</td></tr>
<tr><td>Samples per prompt</td><td>1 (or few)</td><td>G (typically 8-64)</td></tr>
<tr><td>Advantage quality</td><td>Biased (depends on critic accuracy)</td><td>Unbiased within group, high variance</td></tr>
<tr><td>Memory footprint</td><td>4 models (policy, ref, critic, reward)</td><td>2-3 models (policy, ref, [reward])</td></tr>
<tr><td>Training stability</td><td>Sensitive to critic training</td><td>Sensitive to group size G</td></tr>
</table>

<h4>Verifiable Rewards: The Key Enabler</h4>
<p>RLVR works because we can compute exact rewards for certain task types:</p>

<pre><code># Reward functions for different tasks:

def math_reward(prompt, response):
    """Verify mathematical answer against ground truth."""
    predicted = extract_answer(response)  # Extract from \\boxed{...}
    correct = get_ground_truth(prompt)
    if predicted == correct:
        return 1.0
    return 0.0

def code_reward(prompt, response):
    """Execute code and run test cases."""
    code = extract_code(response)
    test_cases = get_test_cases(prompt)
    try:
        # Sandboxed execution (CRITICAL for safety)
        results = sandbox_execute(code, test_cases, timeout=10)
        passed = sum(r.passed for r in results)
        return passed / len(test_cases)  # Fraction of tests passed
    except (TimeoutError, SandboxViolation):
        return 0.0

def format_reward(prompt, response):
    """Check if response follows required format."""
    # Example: does the response have <think>...</think><answer>...</answer>?
    has_think = "<think>" in response and "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    return 1.0 if (has_think and has_answer) else 0.0

def composite_reward(prompt, response):
    """Combine multiple reward signals."""
    r_correct = math_reward(prompt, response)
    r_format = format_reward(prompt, response)
    # Weighted combination -- format reward prevents reward hacking
    # (getting the right answer without showing work)
    return 0.8 * r_correct + 0.2 * r_format</code></pre>

<h4>Hyperparameters That Matter</h4>

<table>
<tr><th>Hyperparameter</th><th>Typical Range</th><th>Effect of Too Low</th><th>Effect of Too High</th></tr>
<tr><td>Group size G</td><td>8-64</td><td>Noisy advantage estimates, unstable training</td><td>Expensive (G forward passes per prompt), slow</td></tr>
<tr><td>KL coefficient beta</td><td>0.001-0.1</td><td>Policy diverges from reference, reward hacking</td><td>Policy barely changes, no learning</td></tr>
<tr><td>Clipping epsilon</td><td>0.1-0.3</td><td>Very conservative updates, slow learning</td><td>Large policy changes, instability</td></tr>
<tr><td>Learning rate</td><td>1e-7 to 5e-6</td><td>No learning</td><td>Training collapse, loss divergence</td></tr>
<tr><td>Temperature (sampling)</td><td>0.7-1.0</td><td>Low diversity in group, poor advantage estimates</td><td>Too random, mostly bad samples</td></tr>
<tr><td>Max response length</td><td>1024-8192</td><td>Truncated reasoning chains</td><td>Slow rollouts, memory issues</td></tr>
</table>

<div class="callout">
<div class="callout-title">The Group Size - Quality Trade-off</div>
<p>Group size G is the most important GRPO hyperparameter. With G=8, the advantage estimate has high variance (you are estimating mean and std from only 8 samples). With G=64, the estimate is much better, but you need 64 forward passes per prompt -- 8x more compute. Empirically, G=16-32 offers the best trade-off for most tasks. DeepSeek-R1 used G=64 for their final training, trading compute for stability.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Derive the GRPO advantage estimation and explain why it eliminates the need for a critic network.</div>
<div class="a-text">In PPO, the advantage A(s_t, a_t) estimates how much better action a_t is compared to the average action at state s_t. This requires a learned value function V(s_t) (the critic). GRPO replaces this with a group-based estimate: for prompt x, sample G responses {y^1, ..., y^G} and compute rewards {r^1, ..., r^G}. The advantage for response j is A^j = (r^j - mean(r)) / std(r). This is a valid advantage estimate because: (1) the mean reward of the group approximates V(x) (the expected reward from this prompt), (2) subtracting the mean centers the advantages (positive for above-average, negative for below-average), (3) dividing by std normalizes across prompts with different reward scales. The key insight is that we do not need a separate neural network to estimate V(x) -- we can estimate it empirically from the group. The cost is that we need G > 1 samples per prompt (vs PPO which can use 1 sample + critic). This removes the critic training instability, reduces memory by ~30-50%, and simplifies the training pipeline.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 6.2  Distributed Training Fundamentals (expanded)
// ---------------------------------------------------------------------------
{
  id: "distributed-training",
  title: "Distributed Training Fundamentals",
  content: `
<p>Training modern LLMs requires distributing computation across many GPUs. The scale ranges from 4 GPUs for fine-tuning a 7B model to thousands of GPUs for pre-training frontier models. Understanding parallelism strategies deeply is essential for any AI engineer working on training.</p>

<h4>The Memory Problem: Why Parallelism is Necessary</h4>

<pre><code># Memory breakdown for training a 70B model:
# (Mixed precision BF16 + FP32 optimizer states)
#
# Model parameters (BF16):              70B * 2 bytes = 140 GB
# Gradients (BF16):                     70B * 2 bytes = 140 GB
# Optimizer states (Adam, FP32):
#   - First moment (m):                 70B * 4 bytes = 280 GB
#   - Second moment (v):                70B * 4 bytes = 280 GB
#   - Master weights (FP32 copy):       70B * 4 bytes = 280 GB
# Total optimizer states:                              = 840 GB
#
# TOTAL per-GPU (naive):                              = 1,120 GB
# H100-80GB can hold:                                   80 GB
# Deficit:                                             1,040 GB
#
# Even the model parameters alone (140 GB) don't fit on one GPU!
# This is why parallelism is not optional -- it's mandatory.</code></pre>

<h4>Data Parallelism (DP) in Detail</h4>

<pre><code># Data Parallelism: N copies of the full model, each on a different GPU.
# Each GPU processes a different mini-batch.
# After computing gradients, all-reduce averages them across GPUs.

# All-reduce communication cost:
# For N GPUs, all-reduce of tensor size M:
#   Ring all-reduce: 2 * M * (N-1)/N bytes communicated per GPU
#   For large N, this approaches 2*M bytes per GPU (independent of N!)
#
# Example: 8B model, BF16 gradients = 16 GB
# All-reduce per step: ~32 GB per GPU
# H100 NVLink: 900 GB/s -> 32/900 = 35ms
# This is acceptable for training step times of 1-5 seconds.

# BUT: DP requires each GPU to hold the FULL model + optimizer states.
# 8B model: 1120 / 10 * 8 = ~130 GB minimum. Too large for 80GB GPU.
# For 8B, we can use DP with gradient checkpointing and optimizer offloading.
# For 70B, we MUST combine with model parallelism.</code></pre>

<h4>ZeRO: Zero Redundancy Optimizer</h4>
<p>ZeRO (Rajbhandari et al., 2020; <a href="https://arxiv.org/abs/1910.02054">arXiv:1910.02054</a>) eliminates memory redundancy in data parallelism by partitioning optimizer states, gradients, and/or parameters across GPUs:</p>

<pre><code># ZeRO stages for an 8B model on 8 GPUs:
#
# Baseline (Pure DP):
#   Per-GPU: 16GB (params) + 16GB (grads) + 96GB (optimizer) = 128GB  [OOM!]
#
# ZeRO Stage 1 (partition optimizer states):
#   Per-GPU: 16GB + 16GB + 96/8 = 44 GB  [Fits on 80GB!]
#   Communication: same as DP (all-reduce gradients)
#
# ZeRO Stage 2 (+ partition gradients):
#   Per-GPU: 16GB + 16/8 + 96/8 = 30 GB  [Comfortable]
#   Communication: reduce-scatter gradients (slightly less than all-reduce)
#
# ZeRO Stage 3 (+ partition parameters):
#   Per-GPU: 16/8 + 16/8 + 96/8 = 16 GB  [Minimal!]
#   Communication: all-gather parameters before each forward/backward
#   This is 3x more communication than DP -- slower but very memory efficient.
#
# Rule of thumb:
# - ZeRO-1: always use. Free memory savings, no communication overhead.
# - ZeRO-2: use when ZeRO-1 is not enough. Minimal overhead.
# - ZeRO-3: use for very large models. Significant communication overhead.
#   Consider TP + PP + ZeRO-1 instead of ZeRO-3 for better throughput.</code></pre>

<h4>Tensor Parallelism for Training</h4>

<pre><code># Megatron-LM style Tensor Parallelism:
# Split weight matrices across GPUs within each node.
# All GPUs compute every token (all layers) in parallel.
#
# For a transformer layer with hidden dim H and TP degree P:
#   Attention: Q, K, V projections split column-wise (H/P per GPU)
#   FFN up-projection: split column-wise
#   FFN down-projection: split row-wise
#
# Communication per layer:
#   Forward: 2 all-reduces (attention output + FFN output)
#   Backward: 2 all-reduces (same, for gradients)
#   Total: 4 all-reduces per layer
#
# Example: 70B model, TP=8, hidden=8192, batch*seq=4096, BF16
#   Data per all-reduce: 4096 * 8192 * 2 = 67 MB
#   NVLink all-reduce: ~67MB / 900GB/s + overhead = ~0.2ms
#   80 layers * 4 * 0.2ms = 64ms total TP communication per step
#   Step time ~2-5 seconds, so TP overhead is ~1-3%.  Excellent.</code></pre>

<h4>Pipeline Parallelism for Training</h4>

<pre><code># Pipeline parallelism splits layers across nodes.
# Challenge: naive PP is terribly inefficient due to pipeline bubbles.
#
# GPipe schedule (simple but high bubble):
# GPU 0: |F1|F2|F3|F4|    |B4|B3|B2|B1|
# GPU 1:    |F1|F2|F3|F4|    |B4|B3|B2|B1|
# GPU 2:       |F1|F2|F3|F4|    |B4|B3|B2|B1|
# GPU 3:          |F1|F2|F3|F4|    |B4|B3|B2|B1|
#                              ^^^^^ BUBBLE ^^^^^
# Bubble time: (P-1) / (M+P-1) where P=stages, M=microbatches
# P=4, M=4: bubble = 43%. Terrible.
#
# 1F1B schedule (better):
# GPU 0: |F1|F2|F3|F4|B1|B2|B3|B4|
# GPU 1:    |F1|F2|F3|B1|F4|B2|B3|B4|
# GPU 2:       |F1|F2|B1|F3|B2|F4|B3|B4|
# GPU 3:          |F1|B1|F2|B2|F3|B3|F4|B4|
# Interleaves forward and backward. Same bubble time but lower PEAK memory
# (only need to store activations for P micro-batches instead of M).
#
# Interleaved 1F1B (Megatron v-stages):
# Assign non-contiguous layers to each stage:
# GPU 0: layers [0-9, 40-49], GPU 1: layers [10-19, 50-59], etc.
# Reduces bubble by 2x (effectively doubles the number of virtual stages).</code></pre>

<h4>Sequence Parallelism</h4>

<pre><code># Problem: LayerNorm and Dropout are applied to the full hidden dimension,
# but TP only splits the attention/FFN computation.
# During LayerNorm/Dropout, each GPU needs the full activation tensor.
# This is redundant memory usage.
#
# Sequence Parallelism (Korthikanti et al., 2022):
# Split the SEQUENCE dimension for LayerNorm, Dropout, and residual connections.
# Each GPU handles seq_len/TP tokens for these operations.
# This eliminates the redundant activation memory.
#
# Memory savings: reduces activation memory by TP factor for ~40% of operations.
# Communication: replaces all-reduce with reduce-scatter + all-gather
#   (same total bytes, but better overlapping with compute).
#
# In Megatron-LM: enabled via --sequence-parallel flag.
# In DeepSpeed: sequence parallelism via Ulysses or Ring Attention.</code></pre>

<h4>Practical Parallelism Strategy Guide</h4>

<table>
<tr><th>Model Size</th><th>GPUs</th><th>Recommended Strategy</th><th>Expected MFU</th></tr>
<tr><td>7-8B</td><td>8 (1 node)</td><td>DP=8, ZeRO-2, gradient checkpointing</td><td>45-55%</td></tr>
<tr><td>7-8B</td><td>32 (4 nodes)</td><td>DP=32, ZeRO-2</td><td>40-50%</td></tr>
<tr><td>70B</td><td>8 (1 node)</td><td>TP=8, gradient checkpointing</td><td>35-45%</td></tr>
<tr><td>70B</td><td>64 (8 nodes)</td><td>TP=8, PP=2, DP=4, ZeRO-1</td><td>35-45%</td></tr>
<tr><td>405B</td><td>256 (32 nodes)</td><td>TP=8, PP=4, DP=8, ZeRO-1</td><td>30-40%</td></tr>
</table>

<p><em>MFU = Model FLOPs Utilization. Percentage of theoretical peak GPU FLOPS actually used for model computation. 50% MFU is considered good for training.</em></p>

<div class="callout warning">
<div class="callout-title">War Story: NCCL Timeout on 8-Node Training</div>
<p>Training on 64 H100s across 8 nodes kept hanging at step ~500 with NCCL timeout errors. Single-node training worked fine. The issue: one node had a flaky InfiniBand cable that dropped packets under sustained all-reduce load. <code>ibstat</code> showed the link was "Active" but <code>perfquery</code> revealed 0.1% packet loss. <strong>Fix:</strong> Replaced the cable, added <code>NCCL_IB_TIMEOUT=23</code> and <code>NCCL_IB_RETRY_CNT=7</code> as environment variables, and set up a pre-training health check script that runs <code>all_reduce_bench</code> across all nodes before every training job. <strong>Lesson:</strong> Network issues in distributed training manifest as random hangs, not error messages. Always benchmark inter-node communication before starting a multi-day training run.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You have 64 H100 GPUs across 8 nodes. Design a parallelism strategy for training a 70B model. Calculate memory per GPU.</div>
<div class="a-text">Strategy: TP=8 (within each node, 8 GPUs with NVLink), PP=2 (across pairs of nodes), DP=4 (4 data-parallel replicas). Memory per GPU: Model params = 140GB / (TP=8) = 17.5GB. Gradients = 17.5GB. Optimizer (ZeRO-1 across DP=4): 3 * 35GB / 4 = 26.25GB. Activations (with gradient checkpointing): ~5GB for 8B-equivalent compute per GPU. Total: ~66GB. Fits in 80GB with headroom. Throughput: effective batch size = DP * micro_batches_per_pp_step * micro_batch_size. With DP=4, 8 microbatches for PP bubble, micro_batch=2: effective batch = 4 * 8 * 2 = 64. Step time ~3-5s for 70B at sequence length 4096. MFU target: ~40%. Communication: TP over NVLink (fast), PP activation transfer over InfiniBand (moderate), DP gradient all-reduce over InfiniBand (chunked, overlapped with backward pass).</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 6.3  RL Algorithms Comparison (NEW)
// ---------------------------------------------------------------------------
{
  id: "rl-algorithms",
  title: "RL Algorithms for LLM Training: A Deep Comparison",
  content: `
<p>Multiple RL algorithms have been applied to LLM training, each with different trade-offs in complexity, memory, stability, and effectiveness. This section provides the mathematical foundations and practical comparison of the major algorithms.</p>

<h4>PPO: Proximal Policy Optimization</h4>
<p>PPO (Schulman et al., 2017; <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a>) is the classic algorithm used in the original RLHF pipeline (InstructGPT, Ouyang et al., 2022; <a href="https://arxiv.org/abs/2203.02155">arXiv:2203.02155</a>).</p>

<pre><code># PPO for LLMs: Complete Algorithm
#
# Models needed:
#   pi_theta:  Policy (the LLM being trained)
#   pi_ref:    Reference policy (frozen SFT model)
#   V_phi:     Value/critic network (estimates expected reward from a state)
#   R_psi:     Reward model (trained on human preferences)
#
# For each training iteration:
# 1. ROLLOUT: Generate responses y ~ pi_theta(y|x) for prompts x
#
# 2. REWARD: Score each response r = R_psi(x, y)
#            Add KL penalty: r_total = r - beta * log(pi_theta(y|x) / pi_ref(y|x))
#
# 3. ADVANTAGE ESTIMATION (GAE):
#    For each token t from last to first:
#      delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
#      A_t = delta_t + gamma * lambda * A_{t+1}
#    where gamma=1.0, lambda=0.95 typically for LLMs
#    (gamma=1 because we don't discount within an episode/response)
#
# 4. POLICY UPDATE (multiple epochs over the rollout data):
#    ratio_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
#    L_clip = -min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t)
#    L_value = 0.5 * (V_phi(s_t) - returns_t)^2
#    L_entropy = -entropy(pi_theta(.|s_t))   # Encourage exploration
#    L_total = L_clip + c1 * L_value - c2 * L_entropy
#
# Typical hyperparameters for LLM PPO:
#   eps (clip range): 0.2
#   GAE lambda: 0.95
#   Value loss coeff c1: 0.5
#   Entropy coeff c2: 0.01
#   PPO epochs: 4 (reuse each rollout 4 times)
#   Learning rate: 1e-6 to 5e-6
#   KL penalty beta: 0.02</code></pre>

<h4>DPO: Direct Preference Optimization</h4>
<p>DPO (Rafailov et al., 2023; <a href="https://arxiv.org/abs/2305.18290">arXiv:2305.18290</a>) eliminates the need for a separate reward model and RL training loop entirely. It directly optimizes the policy from preference pairs.</p>

<pre><code># DPO: Key Insight
# The optimal policy under the RLHF objective has a closed-form solution:
#   pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y) / beta)
#
# Rearranging for the reward:
#   r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
#
# Substituting into the Bradley-Terry preference model:
#   P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))
#
# The Z(x) terms cancel(!), giving the DPO loss:
#   L_DPO = -log sigma(beta * [log(pi_theta(y_w|x)/pi_ref(y_w|x))
#                              - log(pi_theta(y_l|x)/pi_ref(y_l|x))])
#
# where y_w is the preferred response, y_l is the dispreferred response.

# Implementation:
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    Compute DPO loss.
    All inputs are log-probabilities of the full response (sum of token logprobs).
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return losses.mean()

# DPO advantages:
# - No reward model needed (saves memory and training time)
# - No RL infrastructure needed (just supervised fine-tuning!)
# - Stable training (standard cross-entropy-like loss)
# - Works well for general alignment
#
# DPO disadvantages:
# - Requires preference pairs (expensive to collect)
# - Cannot use verifiable rewards directly (rewards must be converted to pairs)
# - Offline: trains on fixed preference data (not iterative improvement)
# - Sensitive to the quality of pi_ref (DPO implicitly depends on it)
# - Tends to reduce entropy more than PPO (less diverse outputs)</code></pre>

<h4>GRPO: Group Relative Policy Optimization</h4>
<p>GRPO (DeepSeek-Math, Shao et al., 2024; <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a>) was covered in detail in Section 6.1. Here we provide the implementation:</p>

<pre><code># GRPO implementation (simplified, production-ready structure)
import torch
import torch.nn.functional as F

class GRPOTrainer:
    def __init__(self, policy, ref_policy, reward_fn,
                 group_size=16, clip_eps=0.2, beta_kl=0.04, lr=1e-6):
        self.policy = policy
        self.ref_policy = ref_policy  # Frozen
        self.reward_fn = reward_fn
        self.G = group_size
        self.eps = clip_eps
        self.beta = beta_kl
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    def train_step(self, prompts):
        """One GRPO training step."""
        all_losses = []

        for prompt in prompts:
            # 1. Generate G responses (using vLLM for speed)
            responses = self.generate_group(prompt, self.G)

            # 2. Compute rewards
            rewards = torch.tensor([
                self.reward_fn(prompt, resp) for resp in responses
            ])

            # 3. Normalize advantages within group
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # 4. Compute loss for each response
            for resp, advantage in zip(responses, advantages):
                tokens = tokenize(prompt + resp)

                # Log probs under current and old policy
                logprobs = self.policy.get_logprobs(tokens)
                with torch.no_grad():
                    old_logprobs = self.policy.get_logprobs(tokens)  # From rollout
                    ref_logprobs = self.ref_policy.get_logprobs(tokens)

                # Per-token ratio
                ratio = torch.exp(logprobs - old_logprobs)

                # Clipped surrogate loss
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL penalty (Schulman approx)
                kl = (ref_logprobs.exp() / logprobs.exp()
                      - (ref_logprobs - logprobs) - 1)
                kl_loss = self.beta * kl.mean()

                all_losses.append(policy_loss + kl_loss)

        # 5. Update policy
        total_loss = torch.stack(all_losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()</code></pre>

<h4>DAPO: Dynamic Sampling Policy Optimization</h4>
<p>DAPO (Yu et al., 2025; <a href="https://arxiv.org/abs/2503.14476">arXiv:2503.14476</a>) extends GRPO with several improvements specifically designed for code and math reasoning:</p>

<pre><code># DAPO Key Improvements over GRPO:
#
# 1. Dynamic Sampling: Adjust temperature based on reward distribution.
#    If all G responses are correct (too easy), increase temperature.
#    If all G responses are wrong (too hard), decrease temperature.
#    This ensures informative advantage signals at every step.
#
# 2. Token-Level KL Penalty: Instead of response-level KL,
#    apply KL penalty per-token. This is more fine-grained and stable.
#
# 3. Clip Range Annealing: Start with large eps (0.3) for exploration,
#    anneal to small eps (0.1) for exploitation.
#
# 4. Overlong Reward Shaping: Penalize responses that exceed the length limit
#    proportionally, rather than giving a flat 0 reward.
#    r_shaped = r_original * min(1, max_len / actual_len)
#
# Hyperparameters (from the DAPO paper):
# Group size G: 16
# Initial clip eps: 0.28, final: 0.12
# KL coefficient beta: 0.04
# Temperature: dynamically adjusted 0.6-1.2
# Learning rate: 5e-7 with cosine schedule</code></pre>

<h4>REINFORCE++ (Variance Reduction)</h4>
<p>REINFORCE++ applies classic variance reduction techniques to the basic REINFORCE algorithm (Hu et al., 2025):</p>

<pre><code># REINFORCE++: Building up from REINFORCE
#
# 1. Vanilla REINFORCE:
#    grad J = E[R(y) * grad log pi(y|x)]
#    Problem: very high variance.
#
# 2. With baseline subtraction:
#    grad J = E[(R(y) - b) * grad log pi(y|x)]
#    b is a baseline (e.g., running average of rewards)
#    Reduces variance without introducing bias.
#
# 3. Per-token reward assignment (reward-to-go):
#    Instead of multiplying ALL token gradients by the same R(y),
#    each token t gets the "reward-to-go" from that point:
#    R_t = sum_{t'>=t} r_{t'}  (future rewards only)
#    Reduces variance by not crediting early tokens for late rewards.
#
# 4. Group-based baseline (like GRPO):
#    b(x) = mean of rewards in the group for prompt x
#    This is essentially GRPO's advantage estimation.
#
# REINFORCE++ combines all four techniques:
#   - Baseline subtraction (group mean)
#   - Normalization (group std)
#   - Token-level credit assignment
#   - PPO-style clipping for stability
#
# It's simpler than GRPO (no explicit KL penalty, uses clipping instead)
# and often performs comparably.</code></pre>

<h4>Algorithm Decision Matrix</h4>

<table>
<tr><th>Criterion</th><th>PPO</th><th>DPO</th><th>GRPO</th><th>DAPO</th><th>REINFORCE++</th></tr>
<tr><td>Reward type</td><td>Learned RM or verifiable</td><td>Preference pairs only</td><td>Verifiable (ideal)</td><td>Verifiable (ideal)</td><td>Any</td></tr>
<tr><td>Memory (models loaded)</td><td>4 (policy, ref, critic, RM)</td><td>2 (policy, ref)</td><td>2-3 (policy, ref, [RM])</td><td>2-3</td><td>2-3</td></tr>
<tr><td>Implementation complexity</td><td>High</td><td>Low (SFT-like)</td><td>Medium</td><td>Medium-High</td><td>Medium</td></tr>
<tr><td>Training stability</td><td>Moderate (critic issues)</td><td>High</td><td>Moderate-High</td><td>High</td><td>Moderate</td></tr>
<tr><td>Sample efficiency</td><td>High (reuses rollouts)</td><td>High (offline)</td><td>Lower (G samples/prompt)</td><td>Medium</td><td>Medium</td></tr>
<tr><td>Best for</td><td>General RLHF with RM</td><td>Alignment from preferences</td><td>Math/code with verifiable rewards</td><td>Hard reasoning tasks</td><td>Simple RL training</td></tr>
<tr><td>Key paper results</td><td>InstructGPT, ChatGPT</td><td>Zephyr, many open models</td><td>DeepSeek-Math, DeepSeek-R1</td><td>DAPO paper benchmarks</td><td>Competitive with GRPO</td></tr>
</table>

<pre><code># DECISION GUIDE:
#
# "I have human preference data (chosen/rejected pairs)"
#   -> DPO. Simplest, most stable. No RL infrastructure needed.
#
# "I have verifiable rewards (math, code, structured output)"
#   -> GRPO or DAPO. GRPO is simpler; DAPO is better for hard tasks.
#
# "I have a reward model trained on preferences"
#   -> PPO if you have the infrastructure; GRPO if you want simplicity.
#
# "I want the simplest possible RL setup"
#   -> REINFORCE++ or DPO.
#
# "I'm training a frontier reasoning model (like DeepSeek-R1)"
#   -> GRPO with large group size (G=64), verified rewards,
#      multi-stage training (SFT -> RL -> rejection sampling -> RL).</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Compare PPO, DPO, and GRPO for training an LLM to write correct code. Which would you choose and why?</div>
<div class="a-text">For code generation with automated test suites, GRPO is the best choice. Reasoning: (1) Code correctness is verifiable -- we can run tests and get exact 0/1 rewards. GRPO is designed for verifiable rewards. (2) DPO requires preference pairs, which are hard to create for code (is this wrong code "better" than that wrong code?). DPO also cannot improve beyond the quality of the preference data (offline). (3) PPO would work but requires a separate critic network (expensive, hard to train well for code tasks). GRPO eliminates the critic with group normalization. (4) Implementation: sample G=16 code solutions per problem, run test suites (sandboxed!), compute binary rewards (pass/fail), normalize advantages within the group, update policy. The verifiable reward ensures no reward hacking. DAPO would be even better for very hard coding problems due to its dynamic temperature adjustment (avoids wasting compute on problems that are too easy or too hard for the current policy).</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 6.4  Reward Modeling (NEW)
// ---------------------------------------------------------------------------
{
  id: "reward-modeling",
  title: "Reward Modeling: From Human Preferences to Verifiable Signals",
  content: `
<p>The reward signal is the most important component of any RL training pipeline. A flawed reward function will produce a flawed model -- potentially one that appears to perform well on benchmarks while behaving badly in practice (Goodhart's Law). This section covers the full spectrum of reward approaches.</p>

<h4>Human Preference Data Collection</h4>
<p>The traditional RLHF pipeline starts with collecting human preferences:</p>

<pre><code># Preference data format:
# Each example is a tuple (prompt, chosen_response, rejected_response)
# Collected via human annotation:
#
# Step 1: Generate multiple responses to each prompt
# Step 2: Present pairs to annotators: "Which response is better?"
# Step 3: Record the preference (with optional confidence score)
#
# Example:
{
    "prompt": "Explain photosynthesis to a 5-year-old.",
    "chosen": "Plants eat sunlight! They use their leaves like little...",
    "rejected": "Photosynthesis is a biochemical process whereby...",
    "annotator_id": "ann_042",
    "confidence": "high",
    "criteria": ["helpfulness", "age-appropriateness"]
}

# Key challenges:
# 1. Cost: $1-5 per comparison. 100K comparisons = $100K-$500K.
# 2. Inter-annotator agreement: typically 65-80% for general quality.
#    Lower for subjective tasks (creative writing), higher for factual.
# 3. Annotation bias: annotators prefer verbose, confident responses
#    even when shorter answers are better (verbosity bias).
# 4. Position bias: annotators tend to prefer the first option shown.
#    Mitigation: randomize order, show both orders to different annotators.

# Quality control:
# - Gold-standard questions (known-correct pairs) to filter bad annotators
# - Require minimum 3 annotators per pair, take majority vote
# - Stratify by difficulty and domain
# - Pay well (rushed annotations = bad data)</code></pre>

<h4>Bradley-Terry Reward Model Training</h4>
<p>The standard reward model is trained using the Bradley-Terry (BT) preference model (Bradley & Terry, 1952):</p>

<pre><code># Bradley-Terry model:
# P(y_w preferred over y_l | x) = sigma(r(x, y_w) - r(x, y_l))
# where sigma is the sigmoid function and r is the reward model.
#
# Training loss:
# L = -E[log sigma(r(x, y_w) - r(x, y_l))]
#
# This is identical to binary cross-entropy where the "label" is always 1
# (chosen is always preferred), and the logit is the reward difference.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class RewardModel(nn.Module):
    """Reward model based on a pretrained LLM backbone."""
    def __init__(self, model_name="meta-llama/Llama-3.1-8B"):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1,  # Single scalar output
            torch_dtype=torch.bfloat16,
        )

    def forward(self, input_ids, attention_mask):
        """Returns scalar reward for each input."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # [batch_size]

def compute_rm_loss(reward_model, chosen_ids, chosen_mask,
                    rejected_ids, rejected_mask):
    """Bradley-Terry loss for reward model training."""
    chosen_rewards = reward_model(chosen_ids, chosen_mask)
    rejected_rewards = reward_model(rejected_ids, rejected_mask)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    # Accuracy for monitoring
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss, accuracy

# Training hyperparameters:
# - Learning rate: 1e-5 to 5e-5 (higher than RL policy training)
# - Epochs: 1-2 (overfitting is a major risk!)
# - Batch size: 32-128
# - Backbone: same architecture as the policy (or smaller)
# - Regularization: weight decay 0.01, dropout on classification head</code></pre>

<h4>Reward Hacking and Mitigation</h4>
<p>Reward hacking occurs when the RL policy exploits weaknesses in the reward model to achieve high reward without genuinely improving quality. This is arguably the most dangerous failure mode in RLHF.</p>

<pre><code># Common reward hacking patterns:

# 1. LENGTH HACKING
# The reward model assigns higher scores to longer responses.
# The RL policy learns to be excessively verbose.
# Mitigation: length-normalized rewards, or explicit length penalty.
def length_normalized_reward(reward, response_length, target_length=200):
    length_penalty = -0.5 * max(0, response_length - target_length) / target_length
    return reward + length_penalty

# 2. SYCOPHANCY
# The reward model prefers responses that agree with the user's
# (potentially incorrect) premise. The policy learns to always agree.
# Mitigation: Include adversarial prompts with false premises in RM training.

# 3. FORMATTING HACKING
# The policy learns that adding markdown headers, bullet points,
# and code blocks gets higher RM scores regardless of content.
# Mitigation: Format-blind evaluation, train RM on content not formatting.

# 4. REPETITION EXPLOITATION
# The policy repeats key phrases that the RM associates with quality.
# Mitigation: Repetition penalty in sampling, n-gram penalty in reward.

# 5. REWARD MODEL OVERCONFIDENCE
# RM gives very high/low scores to out-of-distribution outputs.
# The policy moves to these OOD regions.
# Mitigation: Ensemble reward models, clip reward range, KL penalty.

# General mitigations:
# - KL divergence penalty (prevents large policy shifts)
# - Reward clipping: clip rewards to [-5, 5] range
# - Ensemble of 3-5 RMs: use the minimum reward (conservative)
# - Periodic RM updates: retrain RM on policy's current outputs
# - Red-teaming: specifically test for known hacking patterns</code></pre>

<h4>Verifiable Rewards: The Better Alternative</h4>
<p>Where possible, verifiable rewards completely eliminate reward hacking because the reward is objectively correct:</p>

<pre><code># Types of verifiable rewards:

# 1. EXACT MATCH (math, factual QA)
def exact_match_reward(response, ground_truth):
    answer = extract_answer(response)  # Parse from \boxed{} or similar
    return 1.0 if answer == ground_truth else 0.0

# 2. CODE EXECUTION (programming tasks)
def code_execution_reward(response, test_cases):
    code = extract_code(response)
    results = run_in_sandbox(code, test_cases, timeout=30)
    return sum(1 for r in results if r.passed) / len(results)

# 3. LOGICAL VERIFICATION (formal proofs, SAT solving)
def proof_verification_reward(response, theorem):
    proof = parse_proof(response)
    return 1.0 if lean4_verify(proof, theorem) else 0.0

# 4. CONSTRAINT SATISFACTION (format, length, content requirements)
def constraint_reward(response, constraints):
    score = 0.0
    for constraint in constraints:
        if constraint.type == "contains":
            score += 1.0 if constraint.value in response else 0.0
        elif constraint.type == "max_length":
            score += 1.0 if len(response) <= constraint.value else 0.0
        elif constraint.type == "json_valid":
            try: json.loads(response); score += 1.0
            except: pass
    return score / len(constraints)

# 5. COMPOSITE REWARDS (combining multiple signals)
def training_reward(prompt, response, ground_truth, test_cases):
    correctness = exact_match_reward(response, ground_truth)
    format_ok = 1.0 if has_valid_format(response) else 0.0
    # Weight correctness heavily, but require proper format
    return 0.7 * correctness + 0.3 * format_ok</code></pre>

<h4>Constitutional AI: Self-Supervised Reward</h4>
<p>Constitutional AI (Bai et al., 2022; <a href="https://arxiv.org/abs/2212.08073">arXiv:2212.08073</a>) uses the LLM itself to evaluate outputs against a set of principles (the "constitution"):</p>

<pre><code># Constitutional AI pipeline:
# 1. Generate response to a potentially harmful prompt
# 2. Ask the model to critique its own response against principles
# 3. Ask the model to revise the response based on the critique
# 4. Use (original, revised) as (rejected, chosen) preference pairs
# 5. Train with DPO or RLHF on these self-generated preferences

# Example constitution principles:
PRINCIPLES = [
    "Choose the response that is most helpful to the user.",
    "Choose the response that is least likely to cause harm.",
    "Choose the response that is most honest and accurate.",
    "Choose the response that avoids stereotypes and bias.",
    "Choose the response that respects user privacy.",
]

# CAI critique prompt:
critique_prompt = f"""
Consider the following response to a user query.

Query: {user_query}
Response: {model_response}

Critique this response based on the following principle:
"{principle}"

Identify any ways the response violates this principle.
"""

# CAI revision prompt:
revision_prompt = f"""
Based on the critique: {critique}

Please revise the response to better adhere to the principle:
"{principle}"

Revised response:
"""

# This creates preference pairs WITHOUT human annotation!
# Quality depends on the model's ability to self-critique.</code></pre>

<h4>Reward Model Evaluation</h4>

<pre><code># How to evaluate a reward model before using it for RL:

# 1. Agreement with human preferences (held-out test set)
def evaluate_rm_accuracy(rm, test_data):
    correct = 0
    for chosen, rejected, prompt in test_data:
        r_chosen = rm.score(prompt, chosen)
        r_rejected = rm.score(prompt, rejected)
        if r_chosen > r_rejected:
            correct += 1
    return correct / len(test_data)
# Target: > 70% accuracy (65% is random-ish for close pairs)

# 2. Calibration: are RM scores well-calibrated?
# Plot predicted P(chosen > rejected) vs actual frequency.
# Well-calibrated RM: when it says "80% chance chosen is better",
# chosen is actually better 80% of the time.

# 3. Correlation with downstream metrics:
# Train small-scale RL policies with different RMs.
# Evaluate policies on human evaluation.
# The RM whose RL policy produces the best human ratings wins.

# 4. Adversarial robustness:
# Test with adversarial examples designed to exploit known biases:
# - Verbose vs concise (same content): should score equally
# - Sycophantic vs honest disagreement: should prefer honest
# - With formatting vs without: should score on content</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is reward hacking in RLHF, and how would you detect and mitigate it?</div>
<div class="a-text">Reward hacking occurs when the RL policy exploits weaknesses in the reward model to achieve high reward scores without genuinely improving output quality. Detection: (1) Monitor the gap between RM score and human evaluation -- if RM score keeps increasing but human quality plateaus or decreases, you have reward hacking. (2) Track output statistics: average response length, repetition rate, formatting patterns. Sudden changes suggest hacking. (3) Track KL divergence from reference policy -- rapid KL increase without quality improvement is a red flag. (4) Regularly sample and manually inspect outputs. Mitigation: (1) KL penalty to prevent large policy divergence from the SFT model. (2) Ensemble of 3-5 reward models -- take the minimum or average to reduce exploitability. (3) Reward clipping to [-5, 5] to prevent extreme reward signals. (4) Periodically retrain the RM on the current policy's outputs. (5) Length normalization to prevent verbosity hacking. (6) Where possible, use verifiable rewards instead of learned RMs. The best defense is using verifiable rewards for tasks that support them, since they cannot be hacked.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 6.5  verl Framework Tutorial (NEW)
// ---------------------------------------------------------------------------
{
  id: "verl-tutorial",
  title: "verl Framework Tutorial: Production RL Training",
  content: `
<p>verl (Volcano Engine RL; Sheng et al., 2024; <a href="https://arxiv.org/abs/2409.19256">arXiv:2409.19256</a>) is the open-source framework that implements the 3D-HybridEngine for RL training of LLMs. It solves the fundamental challenge of RL training: the rollout phase (inference) and update phase (training) have completely different computational requirements, and naively switching between them wastes time and memory.</p>

<h4>Architecture: The 3D-HybridEngine</h4>

<pre><code># The fundamental problem verl solves:
#
# ROLLOUT (Inference):
#   - Need: vLLM/SGLang for fast generation (PagedAttention, continuous batching)
#   - Memory layout: weights sharded for inference (TP), KV-cache blocks
#   - GPU utilization: memory-bandwidth bound
#
# UPDATE (Training):
#   - Need: Megatron/DeepSpeed for efficient training (TP, PP, ZeRO)
#   - Memory layout: weights + gradients + optimizer states
#   - GPU utilization: compute-bound
#
# These two phases need DIFFERENT weight layouts on the SAME GPUs!
#
# verl's solution: 3D-HybridEngine
#   Dimension 1: Rollout engine (vLLM or SGLang)
#   Dimension 2: Training engine (Megatron-LM or FSDP)
#   Dimension 3: Weight resharding (automatic conversion between layouts)

# Architecture diagram:
#
# +-------------------------------------------------------+
# |                    verl Orchestrator                    |
# |  (Ray-based, manages worker lifecycle and data flow)   |
# +---+-------------------+-------------------+------------+
#     |                   |                   |
#     v                   v                   v
# +-------+         +----------+        +----------+
# | Rollout|  reshard | Training |  reshard | Rollout |  ...
# | (vLLM) | ------> | (Megatron)| -----> | (vLLM)  |
# | Phase  |         | Phase    |         | Phase   |
# +-------+         +----------+        +----------+
#  ^^ Uses inference-optimized      ^^ Uses training-optimized
#     weight layout (TP only)          weight layout (TP+PP+ZeRO)</code></pre>

<h4>Weight Resharding Between vLLM and Megatron</h4>

<pre><code># Weight resharding is the key innovation that makes verl fast.
# Without resharding, you'd need to:
#   Option A: Keep separate copies for inference and training (2x memory!)
#   Option B: Reload weights from disk between phases (too slow!)
#   Option C: Use the same framework for both (suboptimal for one phase)
#
# verl's resharding:
# 1. After rollout, collect weights from vLLM's TP layout
# 2. Redistribute to Megatron's TP+PP+ZeRO layout
# 3. Train for one or more steps
# 4. Reshard back to vLLM layout
# 5. Update vLLM's weights in-place
#
# This takes ~1-3 seconds for a 70B model (GPU-to-GPU via NCCL).
# Compare to: ~30-60 seconds to reload from disk.

# The resharding is non-trivial because:
# - vLLM TP may use a different TP degree than Megatron
# - Megatron uses PP (pipeline stages); vLLM doesn't
# - ZeRO-3 shards parameters across DP groups
# - Weight matrix splitting dimensions may differ
#
# verl handles all of this automatically via a mapping table
# that converts between the two weight layouts.</code></pre>

<h4>Custom Rollout Worker Implementation</h4>

<pre><code># verl uses Ray actors for rollout and training workers.
# Here's how to implement a custom rollout worker:

import ray
from verl.workers.rollout import BaseRolloutWorker

@ray.remote(num_gpus=1)
class CustomRolloutWorker(BaseRolloutWorker):
    """Custom rollout worker with domain-specific reward."""

    def __init__(self, model_name, reward_config):
        super().__init__()
        self.reward_config = reward_config

    def generate_and_score(self, prompts, sampling_params):
        """Generate responses and compute rewards."""
        # 1. Generate using vLLM backend
        outputs = self.llm.generate(prompts, sampling_params)

        # 2. Compute rewards for each response
        scored_outputs = []
        for prompt, output in zip(prompts, outputs):
            for completion in output.outputs:
                response = completion.text

                # Custom reward function
                reward = self.compute_reward(prompt, response)

                scored_outputs.append({
                    'prompt': prompt,
                    'response': response,
                    'reward': reward,
                    'token_ids': completion.token_ids,
                    'logprobs': completion.logprobs,
                })

        return scored_outputs

    def compute_reward(self, prompt, response):
        """Domain-specific reward computation."""
        rewards = {}

        # Correctness reward (verifiable)
        if self.reward_config.get('math_verify'):
            predicted = extract_boxed_answer(response)
            ground_truth = self.get_answer(prompt)
            rewards['correctness'] = 1.0 if predicted == ground_truth else 0.0

        # Format reward
        if self.reward_config.get('require_cot'):
            has_thinking = '<think>' in response and '</think>' in response
            rewards['format'] = 1.0 if has_thinking else 0.0

        # Length penalty
        if self.reward_config.get('max_length'):
            max_len = self.reward_config['max_length']
            if len(response) > max_len:
                rewards['length'] = -0.5 * (len(response) - max_len) / max_len
            else:
                rewards['length'] = 0.0

        # Weighted combination
        weights = self.reward_config.get('weights', {})
        total = sum(weights.get(k, 1.0) * v for k, v in rewards.items())
        return total</code></pre>

<h4>Configuration for Different Model Sizes</h4>

<pre><code># verl configuration: 8B model with GRPO on 8x H100
# File: config_8b_grpo.yaml

model:
  name: meta-llama/Llama-3.1-8B-Instruct
  dtype: bfloat16

rollout:
  engine: vllm
  tensor_parallel_size: 1        # 8B fits on 1 GPU for inference
  sampling:
    temperature: 0.8
    top_p: 0.95
    max_tokens: 2048
  group_size: 16                 # G=16 for GRPO

training:
  engine: fsdp                   # FSDP (ZeRO-3 equivalent) for 8B
  learning_rate: 1e-6
  weight_decay: 0.01
  max_grad_norm: 1.0
  num_epochs_per_step: 1         # 1 training epoch per rollout batch
  batch_size: 128
  gradient_checkpointing: true

algorithm:
  name: grpo
  clip_eps: 0.2
  kl_coef: 0.04                  # beta for KL penalty
  gamma: 1.0
  advantage_normalization: true

reward:
  type: custom
  function: math_reward          # Verifiable math reward
  format_reward_weight: 0.2

data:
  train_path: /data/math_problems/train.jsonl
  eval_path: /data/math_problems/eval.jsonl

hardware:
  num_gpus: 8
  num_nodes: 1
  gpu_type: H100

logging:
  wandb_project: rlvr-math
  log_interval: 10
  eval_interval: 100
  save_interval: 500</code></pre>

<pre><code># verl configuration: 70B model with GRPO on 32x H100 (4 nodes)
# Key differences from 8B config:

model:
  name: meta-llama/Llama-3.1-70B-Instruct
  dtype: bfloat16

rollout:
  engine: vllm
  tensor_parallel_size: 4        # 70B needs TP=4 for inference
  sampling:
    temperature: 0.7
    max_tokens: 4096
  group_size: 8                  # Smaller G due to higher per-sample cost

training:
  engine: megatron               # Megatron for 70B (better than FSDP)
  tensor_parallel_size: 8        # TP=8 within node for training
  pipeline_parallel_size: 1      # PP=1 (fits in 1 node with TP=8)
  data_parallel_size: 4          # DP=4 across nodes
  learning_rate: 5e-7            # Lower LR for larger model
  gradient_checkpointing: true
  sequence_parallel: true        # Enable SP with TP

# Resource mapping:
# Rollout: 4 GPUs per vLLM instance, 8 instances = 32 GPUs
# Training: TP=8 * DP=4 = 32 GPUs
# Same GPUs used for both phases (resharding between phases)</code></pre>

<h4>Complete Training Script</h4>

<pre><code># train_grpo.py - Training a math reasoning model with verl
import verl
from verl import DataProto
from verl.trainer import GRPOTrainer
from verl.utils.reward import MathRewardFunction

def main():
    # 1. Load config
    config = verl.load_config("config_8b_grpo.yaml")

    # 2. Initialize reward function
    reward_fn = MathRewardFunction(
        answer_extraction_regex=r"\\\\boxed\\{(.+?)\\}",
        format_checker=lambda r: "<think>" in r and "</think>" in r,
        format_weight=0.2,
    )

    # 3. Initialize trainer (handles rollout, training, resharding)
    trainer = GRPOTrainer(
        config=config,
        reward_fn=reward_fn,
    )

    # 4. Load training data
    train_data = verl.load_dataset(config.data.train_path)
    eval_data = verl.load_dataset(config.data.eval_path)

    # 5. Training loop
    for step in range(config.training.max_steps):
        # Sample a batch of prompts
        batch = train_data.sample(config.training.batch_size)

        # Run one GRPO step (rollout -> reward -> advantage -> update)
        metrics = trainer.step(batch)

        # Logging
        if step % config.logging.log_interval == 0:
            print(f"Step {step}: "
                  f"reward={metrics['mean_reward']:.3f}, "
                  f"kl={metrics['kl_divergence']:.4f}, "
                  f"loss={metrics['policy_loss']:.4f}, "
                  f"entropy={metrics['entropy']:.3f}")

        # Evaluation
        if step % config.logging.eval_interval == 0:
            eval_metrics = trainer.evaluate(eval_data)
            print(f"Eval: accuracy={eval_metrics['accuracy']:.3f}")

        # Save checkpoint
        if step % config.logging.save_interval == 0:
            trainer.save_checkpoint(f"checkpoints/step_{step}")

if __name__ == "__main__":
    main()

# Launch:
# Single node: python train_grpo.py
# Multi-node:
#   Node 0: ray start --head --port=6379
#   Node 1-3: ray start --address=node0:6379
#   Node 0: python train_grpo.py</code></pre>

<div class="callout">
<div class="callout-title">verl vs Alternatives</div>
<p><strong>verl</strong> excels at large-scale RL training (70B+) with its HybridEngine. <strong>OpenRLHF</strong> (Hu et al., 2024; <a href="https://arxiv.org/abs/2405.11143">arXiv:2405.11143</a>) is an alternative that uses Ray-based scheduling and supports similar algorithms. <strong>TRL</strong> (HuggingFace) is simpler but limited to single-node and lacks the HybridEngine. For < 13B models on a single node, TRL may be sufficient. For 70B+ models requiring multi-node training, verl or OpenRLHF are necessary. The choice often depends on your team's familiarity: if you already use Megatron, verl is natural; if you use DeepSpeed, OpenRLHF integrates more smoothly.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the weight resharding problem in RL training for LLMs and how verl solves it.</div>
<div class="a-text">RL training for LLMs alternates between two phases with different requirements: rollout (inference, needs vLLM's PagedAttention, TP-only layout) and update (training, needs Megatron/FSDP's TP+PP+ZeRO layout with gradient and optimizer states). The same GPU memory must serve both phases. Naive approaches: (A) keep two copies of weights (2x memory, impossible for large models), (B) reload from disk between phases (30-60s overhead per step), (C) use one framework for both (suboptimal for one phase). verl's 3D-HybridEngine solves this with GPU-to-GPU weight resharding via NCCL. After rollout completes, weights are redistributed from vLLM's layout to Megatron's layout in 1-3 seconds. After training, they are resharded back. This adds minimal overhead while allowing each phase to use its optimal framework and memory layout. The resharding handles different TP degrees (e.g., TP=4 for inference, TP=8 for training), pipeline parallelism stages, and ZeRO parameter sharding transparently.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 6.6  Training Debugging Cookbook (NEW)
// ---------------------------------------------------------------------------
{
  id: "training-debug",
  title: "Training Debugging Cookbook",
  content: `
<p>RL training for LLMs is notoriously difficult to debug. The reward signal is sparse, the training dynamics are complex, and failures can be silent (the model "trains" but learns nothing useful, or learns something subtly wrong). This cookbook covers the most common failure modes with specific diagnostic steps and fixes.</p>

<h4>Problem 1: Loss Divergence (NaN/Inf)</h4>

<pre><code># Symptom: Loss suddenly becomes NaN or Inf
# Usually happens within the first 100-1000 steps.

# DIAGNOSTIC CHECKLIST:
# 1. Check learning rate
#    - For GRPO/PPO: 1e-7 to 5e-6 is typical
#    - If > 1e-5, almost certainly too high
#    - Try reducing by 10x

# 2. Check gradient norms BEFORE the divergence
import torch
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"WARNING: {name} grad norm = {grad_norm}")
# Large gradient norms (>100) precede divergence.
# Fix: gradient clipping (max_grad_norm=1.0)

# 3. Check for bad data
# A single malformed example can produce NaN loss.
# Log the batch contents when loss > threshold:
if loss.item() > 10 or torch.isnan(loss):
    print(f"Bad batch detected at step {step}")
    print(f"Prompts: {batch['prompts'][:3]}")
    print(f"Rewards: {batch['rewards']}")
    # Common culprits: empty responses, extremely long sequences,
    # non-UTF8 characters, reward function returning NaN

# 4. Check dtype
# FP16 overflows more easily than BF16.
# ALWAYS use BF16 for LLM training if hardware supports it.
# FP16 dynamic range: +-65504. BF16 dynamic range: +-3.4e38.

# 5. Check KL divergence
# If KL grows rapidly, the policy is diverging from reference.
# This can cause numerical issues in the KL penalty term.
# Fix: increase KL coefficient (beta) or reduce learning rate.</code></pre>

<h4>Problem 2: Reward Hacking Detection</h4>

<pre><code># Symptom: Reward keeps increasing but output quality doesn't improve
# (or even degrades as judged by humans).

# DETECTION METRICS:
# 1. Track reward vs human evaluation score
#    If they diverge: reward hacking.

# 2. Track output statistics:
def monitor_output_stats(outputs):
    lengths = [len(o) for o in outputs]
    unique_tokens = [len(set(o.split())) / len(o.split()) for o in outputs]
    has_format = [1 for o in outputs if '<think>' in o]

    stats = {
        'avg_length': sum(lengths) / len(lengths),
        'length_std': torch.tensor(lengths).float().std().item(),
        'avg_unique_ratio': sum(unique_tokens) / len(unique_tokens),
        'format_compliance': sum(has_format) / len(outputs),
    }
    return stats
    # Red flags:
    # - avg_length increasing monotonically -> length hacking
    # - unique_ratio decreasing -> repetition hacking
    # - format_compliance suddenly jumping to 100% -> format gaming

# 3. Sample and inspect outputs every N steps
# This is the single most important debugging technique.
# Automated metrics miss subtle quality issues.
# Save 10 random (prompt, response, reward) tuples every 50 steps.

# 4. Compare reward model score vs ground truth (if available)
# For math: check if high-reward responses are actually correct
# For code: check if high-reward code actually passes ALL tests
# Discrepancy = reward model is being exploited</code></pre>

<h4>Problem 3: KL Divergence Monitoring</h4>

<pre><code># KL divergence measures how far the RL policy has moved from the reference.
# It should increase slowly and plateau. Rapid increase = instability.

# Healthy KL trajectory:
# Step 0:    KL = 0.00 (policy = reference)
# Step 100:  KL = 0.05 (small updates)
# Step 500:  KL = 0.15 (moderate divergence)
# Step 2000: KL = 0.30 (plateau -- KL penalty balances reward gradient)

# Unhealthy KL trajectory:
# Step 0:    KL = 0.00
# Step 50:   KL = 0.50  (too fast!)
# Step 100:  KL = 2.00  (policy has diverged significantly)
# Step 200:  KL = 5.00  (model is generating gibberish)

# Fix for rapid KL increase:
# 1. Increase KL coefficient (beta): 0.04 -> 0.1 -> 0.2
# 2. Reduce learning rate
# 3. Reduce PPO/GRPO epochs (number of gradient steps per rollout)
# 4. Reduce clip range (epsilon): 0.2 -> 0.1

# Implementation:
def compute_kl_divergence(policy_logprobs, ref_logprobs):
    """Per-token KL divergence, averaged over batch."""
    # Approximate KL: E_pi[log(pi/ref)]
    kl = policy_logprobs - ref_logprobs
    return kl.mean().item()

# Monitor per-step:
kl = compute_kl_divergence(new_logprobs, ref_logprobs)
if kl > 5.0:
    print(f"CRITICAL: KL={kl:.2f}. Policy has diverged. "
          f"Consider reverting to last checkpoint.")
if kl > 1.0:
    print(f"WARNING: KL={kl:.2f}. Consider increasing beta.")</code></pre>

<h4>Problem 4: Gradient Norm Analysis</h4>

<pre><code># Gradient norms are a leading indicator of training health.
# Track per-layer gradient norms to identify problematic layers.

def log_gradient_norms(model, step):
    """Log gradient norms per layer for diagnosis."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()

    # Overall norm
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float('inf')  # Don't actually clip, just measure
    ).item()

    # Check for anomalies
    max_norm_layer = max(norms, key=norms.get)
    if norms[max_norm_layer] > 10 * total_norm / len(norms):
        print(f"Step {step}: Gradient spike in {max_norm_layer}: "
              f"{norms[max_norm_layer]:.2f} vs avg {total_norm/len(norms):.2f}")

    return {
        'total_grad_norm': total_norm,
        'max_layer_norm': norms[max_norm_layer],
        'max_layer_name': max_norm_layer,
    }

# Patterns to watch for:
# 1. Spikes: sudden 10-100x increase in gradient norm for one step
#    Cause: bad data batch or numerical instability
#    Fix: gradient clipping (you should already have this!)
#
# 2. Vanishing gradients: norms close to 0 in early layers
#    Cause: too many layers without residual connections (rare in modern LLMs)
#    Fix: check gradient checkpointing isn't dropping gradients
#
# 3. Exploding gradients: norms steadily increasing each step
#    Cause: learning rate too high, or reward signal too noisy
#    Fix: reduce LR, increase gradient clipping threshold
#
# 4. Layer-specific spikes: always in attention or always in FFN
#    Cause: specific layer architecture issue, or data length mismatch
#    Fix: investigate the specific layer</code></pre>

<h4>Problem 5: "It Trains But Doesn't Improve"</h4>

<pre><code># The most frustrating problem: loss decreases, reward increases,
# but the model doesn't actually get better on the downstream task.

# DIAGNOSIS FRAMEWORK:

# Step 1: Is the reward function measuring the right thing?
# - Run the reward function on hand-verified good and bad examples.
# - If the reward doesn't correlate with your human judgment of quality,
#   fix the reward before fixing training.

# Step 2: Is the model learning the reward or the task?
# - Generate 100 responses from the trained model.
# - Manually categorize: correct vs incorrect vs reward-hacked.
# - If many are "high reward but wrong": reward hacking (see Problem 2).

# Step 3: Is the exploration sufficient?
# - Check advantage statistics:
def check_advantage_distribution(advantages):
    positive = (advantages > 0).float().mean()
    print(f"Positive advantage fraction: {positive:.2%}")
    print(f"Advantage std: {advantages.std():.3f}")
    # If positive > 95%: almost all responses are "good"
    #   -> Task is too easy, increase difficulty
    # If positive < 5%: almost all responses are "bad"
    #   -> Task is too hard, start with easier prompts
    # If std < 0.01: all responses get similar rewards
    #   -> No learning signal. Increase temperature or group size.

# Step 4: Is the training signal reaching the model?
# - Check if policy loss is actually changing model outputs.
# - Generate with the same prompt at step 0, 100, 500, 1000.
# - If outputs are identical: learning rate too low or optimizer issue.
# - If outputs change randomly: learning rate too high.

# Step 5: Is there a distribution mismatch?
# - Training prompts may not represent the evaluation distribution.
# - Common: train on easy math, evaluate on hard math.
# - Fix: ensure training data covers the difficulty range of evaluation.</code></pre>

<h4>Checkpoint Comparison Tools</h4>

<pre><code># Compare two checkpoints to understand what changed:
import torch

def compare_checkpoints(ckpt_path_a, ckpt_path_b, top_k=10):
    """Compare two model checkpoints to identify what changed most."""
    state_a = torch.load(ckpt_path_a, map_location='cpu')
    state_b = torch.load(ckpt_path_b, map_location='cpu')

    diffs = {}
    for key in state_a:
        if key in state_b:
            diff = (state_a[key].float() - state_b[key].float()).norm().item()
            relative_diff = diff / (state_a[key].float().norm().item() + 1e-8)
            diffs[key] = {
                'absolute': diff,
                'relative': relative_diff,
            }

    # Sort by relative difference
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1]['relative'], reverse=True)

    print(f"\\nTop {top_k} most changed parameters:")
    for name, diff in sorted_diffs[:top_k]:
        print(f"  {name}: relative={diff['relative']:.4f}, absolute={diff['absolute']:.4f}")

    # Also check: are LoRA layers changing? Are base layers frozen?
    lora_changes = [d for n, d in sorted_diffs if 'lora' in n.lower()]
    base_changes = [d for n, d in sorted_diffs if 'lora' not in n.lower()]

    if lora_changes:
        avg_lora = sum(d['relative'] for d in lora_changes) / len(lora_changes)
        avg_base = sum(d['relative'] for d in base_changes) / len(base_changes) if base_changes else 0
        print(f"\\nAvg LoRA change: {avg_lora:.6f}")
        print(f"Avg base change: {avg_base:.6f}")
        if avg_base > 0.001 and avg_lora > 0:
            print("WARNING: Base layers changing when they should be frozen!")

# Usage:
# compare_checkpoints("ckpt/step_0/model.pt", "ckpt/step_500/model.pt")</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: The Silent Entropy Collapse</div>
<p>We trained a code generation model with GRPO. Reward kept increasing for 2000 steps. Loss looked healthy. But at step 2500, we noticed the model was generating the same boilerplate solution for every prompt -- it had found a template that passed ~60% of test cases regardless of the problem. The entropy of the output distribution had collapsed to near zero by step 800, but we were not monitoring it. By step 2500, the model was in a local optimum: the same template response for every prompt. <strong>Fix:</strong> Added entropy monitoring. When entropy drops below a threshold (0.5), inject an entropy bonus into the loss: <code>L_total = L_policy + kl_loss - 0.01 * entropy</code>. This penalizes the model for becoming too deterministic. Also added "diversity in group" as a diagnostic: if all G responses for a prompt are identical, the policy has collapsed. <strong>Lesson:</strong> Monitor entropy religiously. By the time you notice quality degradation, entropy collapse has been happening for hundreds of steps and may require rolling back to an earlier checkpoint.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Your RL training run shows increasing reward but your eval accuracy on held-out problems is flat. What do you investigate?</div>
<div class="a-text">This is the classic reward hacking or distribution mismatch scenario. Investigation steps: (1) Inspect actual model outputs -- sample 50 training prompts and 50 eval prompts, generate responses, and manually examine them. Look for patterns: is the model exploiting reward function loopholes? (2) Check if the reward function correlates with eval accuracy. Run the reward function on the eval set -- if eval set rewards are also increasing, the problem is evaluation metric mismatch, not reward hacking. (3) Check entropy -- if it has collapsed, the model is generating repetitive outputs that happen to score well on training rewards. (4) Check the distribution of training vs eval prompts -- if training is easy math and eval is hard math, the skills may not transfer. (5) Check KL divergence -- if KL is very high (>2), the model may have diverged too far from the reference and lost general capabilities. (6) Check the group statistics: what fraction of training groups have mixed rewards (some positive, some negative)? If most groups are all-positive or all-negative, the advantage signal is weak.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 6.7  Scaling Laws & Compute Planning (NEW)
// ---------------------------------------------------------------------------
{
  id: "training-scaling",
  title: "Scaling Laws & Compute Planning",
  content: `
<p>Understanding scaling laws is essential for planning training runs, estimating costs, and making hardware decisions. This section covers the key scaling relationships and provides practical tools for compute planning.</p>

<h4>Chinchilla Scaling Laws</h4>
<p>The Chinchilla paper (Hoffmann et al., 2022; <a href="https://arxiv.org/abs/2203.15556">arXiv:2203.15556</a>) established that, for a fixed compute budget, there is an optimal balance between model size and number of training tokens:</p>

<pre><code># Chinchilla optimal ratio:
# Tokens = 20 * Parameters
#
# Model Size   | Optimal Tokens  | Compute (FLOPs)
# 1B           | 20B             | 1.2e20
# 7B           | 140B            | 5.9e21
# 13B          | 260B            | 2.0e22
# 70B          | 1.4T            | 5.9e23
# 405B         | 8.1T            | 2.0e25
#
# BUT: Post-Chinchilla, practitioners often "overtrain" small models:
# LLaMA-3-8B was trained on 15T tokens (107x Chinchilla optimal!)
# Why? Inference cost matters more than training cost for deployed models.
# A smaller model trained longer is cheaper to SERVE than a larger model
# trained for fewer tokens, even if training is more expensive.
#
# The "inference-aware" scaling law (roughly):
# If you'll serve the model for K total tokens of inference,
# it's worth spending up to K * cost_per_inference_token on extra training
# to reduce model size while maintaining quality.

# Training FLOPs formula (approximate):
# C = 6 * N * D
# where:
#   C = total FLOPs
#   N = number of parameters
#   D = number of training tokens
#   6 = forward + backward pass (approximately 2x forward each)
#
# Example: LLaMA-3-70B on 15T tokens
# C = 6 * 70e9 * 15e12 = 6.3e24 FLOPs
# H100 at 50% MFU: 990 TFLOPS * 0.5 = 495 TFLOPS
# GPU-hours: 6.3e24 / (495e12 * 3600) = 3.5 million GPU-hours
# With 16K GPUs: 3.5M / 16K = 219 hours = ~9 days</code></pre>

<h4>Compute-Optimal RL Training</h4>
<p>RL training has different scaling characteristics than pre-training:</p>

<pre><code># RL training compute is dominated by ROLLOUT (inference), not gradient updates.
#
# For GRPO with group size G:
# Total inference FLOPs per step = G * num_prompts * avg_output_length * inference_cost_per_token
# Total training FLOPs per step = G * num_prompts * avg_output_length * training_cost_per_token
#
# Typically: inference_cost = 1x model FLOPs, training_cost = 2x model FLOPs
# But with G=16 samples per prompt, rollout dominates:
# Rollout: 16 * batch * tokens * N
# Training: 2 * 16 * batch * tokens * N (but only 1 epoch, vs 16 forward passes)
# Total: 18 * batch * tokens * N for rollout, 2 * batch * tokens * N for training
# Rollout is 90% of compute!

# This is why verl uses vLLM for rollout -- inference optimization
# has 9x more impact on total training speed than training optimization.

# RL training data efficiency:
# Pre-training: ~20 tokens per parameter (Chinchilla)
# SFT fine-tuning: ~1M-10M examples (highly data-efficient)
# RL training: typically 10K-100K unique prompts, each rolled out G times
#   over multiple epochs of RL training.
#
# DeepSeek-R1 RL training:
# - 600K prompts, G=64, multiple iterations
# - Total rollout tokens: ~600K * 64 * 4096 * 10 iterations = 1.6T tokens
# - But only 600K unique prompts (massive reuse with different responses)</code></pre>

<h4>Cost Estimation for Different Model Sizes</h4>

<table>
<tr><th>Model</th><th>GPUs</th><th>SFT (10K examples)</th><th>RL-GRPO (50K prompts, G=16)</th><th>Total Cloud Cost</th></tr>
<tr><td>8B</td><td>8x H100</td><td>~2 hours</td><td>~24 hours</td><td>~$800</td></tr>
<tr><td>8B</td><td>8x A100</td><td>~4 hours</td><td>~48 hours</td><td>~$1,200</td></tr>
<tr><td>70B</td><td>32x H100</td><td>~8 hours</td><td>~72 hours</td><td>~$24,000</td></tr>
<tr><td>70B</td><td>64x A100</td><td>~16 hours</td><td>~120 hours</td><td>~$30,000</td></tr>
<tr><td>70B (LoRA r=64)</td><td>8x H100</td><td>~4 hours</td><td>~48 hours</td><td>~$2,000</td></tr>
</table>

<p><em>Estimates assume H100 at $3/GPU-hr, A100 at $2/GPU-hr (cloud on-demand pricing). LoRA significantly reduces memory requirements, enabling training on fewer GPUs. Actual costs vary by provider: Reserved Instances are 40-60% cheaper, spot instances are 60-80% cheaper but can be interrupted.</em></p>

<pre><code># Cost estimation calculator
def estimate_rl_training_cost(
    model_params_B: float,     # Model size in billions
    num_prompts: int,          # Number of unique training prompts
    group_size: int,           # G for GRPO
    avg_output_tokens: int,    # Average response length
    num_epochs: int,           # RL training epochs
    gpu_type: str,             # "H100" or "A100"
    num_gpus: int,
    mfu: float = 0.4,         # Model FLOPs utilization
    cost_per_gpu_hour: float = None,
):
    """Estimate RL training cost."""
    # GPU FLOPS
    gpu_flops = {
        "H100": 990e12,  # BF16 Tensor Core TFLOPS
        "A100": 312e12,
        "L40S": 362e12,
    }
    if cost_per_gpu_hour is None:
        cost_per_gpu_hour = {"H100": 3.0, "A100": 2.0, "L40S": 1.5}[gpu_type]

    # Total rollout FLOPs (dominant cost)
    rollout_flops = (
        num_prompts * group_size * avg_output_tokens
        * 2 * model_params_B * 1e9  # 2N FLOPs per token for inference
        * num_epochs
    )

    # Total training FLOPs
    train_flops = (
        num_prompts * group_size * avg_output_tokens
        * 6 * model_params_B * 1e9  # 6N FLOPs per token for training
        * num_epochs
        / group_size  # Training sees each prompt once, not G times
    )

    total_flops = rollout_flops + train_flops
    effective_flops_per_second = gpu_flops[gpu_type] * mfu * num_gpus
    training_seconds = total_flops / effective_flops_per_second
    training_hours = training_seconds / 3600
    total_cost = training_hours * num_gpus * cost_per_gpu_hour

    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"  Rollout: {rollout_flops:.2e} ({rollout_flops/total_flops*100:.0f}%)")
    print(f"  Training: {train_flops:.2e} ({train_flops/total_flops*100:.0f}%)")
    print(f"Estimated time: {training_hours:.1f} hours on {num_gpus}x {gpu_type}")
    print(f"Estimated cost: $" + f"{total_cost:,.0f}")
    return total_cost

# Examples:
estimate_rl_training_cost(
    model_params_B=8, num_prompts=50000, group_size=16,
    avg_output_tokens=1024, num_epochs=3,
    gpu_type="H100", num_gpus=8
)
# -> ~20 hours, ~$480

estimate_rl_training_cost(
    model_params_B=70, num_prompts=50000, group_size=16,
    avg_output_tokens=2048, num_epochs=3,
    gpu_type="H100", num_gpus=32
)
# -> ~65 hours, ~$6,240</code></pre>

<h4>Hardware Selection Guide</h4>

<table>
<tr><th>GPU</th><th>HBM</th><th>BF16 TFLOPS</th><th>Memory BW</th><th>NVLink BW</th><th>~Cloud $/hr</th><th>Best For</th></tr>
<tr><td>H100 SXM</td><td>80 GB</td><td>990</td><td>3.35 TB/s</td><td>900 GB/s</td><td>$2.50-3.50</td><td>All training, best $/FLOP for large models</td></tr>
<tr><td>H200 SXM</td><td>141 GB</td><td>990</td><td>4.8 TB/s</td><td>900 GB/s</td><td>$3.50-4.50</td><td>Large models that need more memory (70B FP16)</td></tr>
<tr><td>A100 SXM</td><td>80 GB</td><td>312</td><td>2.0 TB/s</td><td>600 GB/s</td><td>$1.50-2.50</td><td>Budget training, good for < 13B models</td></tr>
<tr><td>L40S</td><td>48 GB</td><td>362</td><td>864 GB/s</td><td>None</td><td>$1.00-1.80</td><td>Inference, LoRA fine-tuning of small models</td></tr>
<tr><td>H100 NVL</td><td>94 GB</td><td>835</td><td>3.9 TB/s</td><td>600 GB/s</td><td>$2.00-3.00</td><td>Inference-optimized, not ideal for training</td></tr>
</table>

<pre><code># Hardware decision framework:

# Budget < $5K:
#   Use cloud spot instances: 8x A100 or 8x L40S.
#   Train 8B model with LoRA + GRPO. Use verl with FSDP backend.
#   Expected: LoRA fine-tune of 8B in ~2 days.

# Budget $5K-50K:
#   Use cloud reserved: 8-32x H100.
#   Full fine-tune of 8B or LoRA of 70B.
#   GRPO with verl, vLLM rollout backend.
#   Expected: Full RL training of 8B in ~3 days on 8x H100.
#             LoRA RL training of 70B in ~5 days on 16x H100.

# Budget $50K-500K:
#   Consider reserved instances or dedicated clusters.
#   Full fine-tune of 70B.
#   Megatron backend in verl with TP=8, PP as needed.
#   Expected: Full RL training of 70B in ~5 days on 64x H100.

# Budget > $500K:
#   Dedicated cluster or cloud commitment.
#   Train from scratch or extensive RL on 70B+.
#   Consider H200 for memory-intensive workloads.

# Key ratios to remember:
# H100 vs A100: ~3x performance, ~1.5x cost -> 2x better $/FLOP
# H200 vs H100: same compute, 1.75x memory, ~1.3x cost
# Spot vs On-Demand: 60-80% cheaper, but can be interrupted
# Reserved (1yr) vs On-Demand: 40-50% cheaper, requires commitment</code></pre>

<h4>Training Time Estimation Formulas</h4>

<pre><code># FORMULA 1: Pre-training time
# T = (6 * N * D) / (num_gpus * gpu_flops * mfu)
# N = parameters, D = tokens, mfu = model FLOPs utilization
#
# Example: 8B model, 1T tokens, 64x H100, 45% MFU
# T = (6 * 8e9 * 1e12) / (64 * 990e12 * 0.45)
# T = 4.8e22 / 2.85e16 = 1.68e6 seconds = 19.5 days

# FORMULA 2: RL training time per step
# T_step = T_rollout + T_reshard + T_train
# T_rollout = (G * B * L_out * 2N) / (num_gpus * gpu_flops * mfu_inference)
# T_reshard ~ 1-3 seconds (constant for a given model size)
# T_train = (B * L_out * 6N) / (num_gpus * gpu_flops * mfu_train)
#
# Note: mfu for inference (~5-20%) is much lower than training (~35-50%)
# because decode is memory-bandwidth bound.

# FORMULA 3: Total RL training time
# T_total = num_steps * T_step
# num_steps = (num_prompts * num_epochs) / batch_size
#
# Example: 70B model, 50K prompts, G=16, batch=128, 3 epochs
# num_steps = (50000 * 3) / 128 = 1172 steps
# T_step ~ 200 seconds (rollout-dominated on 32x H100)
# T_total = 1172 * 200 = 234,375 seconds = 65 hours

# FORMULA 4: Memory estimation
# Training memory per GPU =
#   model_params / TP + gradients / TP + optimizer / (TP * DP_ZeRO) + activations
# where:
#   model_params = N * bytes_per_param (2 for BF16)
#   gradients = N * 2 (BF16)
#   optimizer (Adam) = N * (4 + 4 + 4) = 12N bytes (m, v, master weights in FP32)
#   activations: depends on batch size and sequence length, reduced by checkpointing
#
# Example: 70B, TP=8, ZeRO-1 across DP=4
# Params per GPU: 140GB / 8 = 17.5 GB
# Grads per GPU: 140GB / 8 = 17.5 GB
# Optimizer per GPU: 840GB / 8 / 4 = 26.25 GB (ZeRO-1 over DP=4)
# Activations: ~5 GB (with gradient checkpointing)
# Total: ~66 GB. Fits on 80GB H100 with headroom.</code></pre>

<div class="callout">
<div class="callout-title">The Inference-Aware Training Paradigm</div>
<p>Modern scaling decisions prioritize <strong>inference efficiency</strong> over training efficiency. LLaMA-3-8B was trained on 15T tokens (107x Chinchilla optimal) because a well-trained 8B model is much cheaper to deploy than a Chinchilla-optimal 65B model at similar quality. For RL training, this manifests as: prefer training a smaller model with more RL steps than a larger model with fewer steps. An 8B model with 10K steps of GRPO can match or exceed a 70B model with 1K steps on specific tasks, while being 9x cheaper to serve.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Estimate the total compute and cost to RL-train a 70B model with GRPO on 50K math problems, G=16, for 3 epochs. What hardware would you recommend?</div>
<div class="a-text">Calculation: Total rollout tokens = 50K * 16 * 2048 (avg output) * 3 epochs = 4.9T tokens. Rollout FLOPs = 4.9T * 2 * 70B = 6.9e23 FLOPs. Training FLOPs = (50K * 2048 * 3) * 6 * 70B = 1.3e23 FLOPs. Total = ~8.2e23 FLOPs. Hardware: 32x H100 SXM. Effective throughput: 32 * 990 TFLOPS * 0.40 MFU = 12.7 PFLOPS. Time: 8.2e23 / 12.7e15 = 64,567 seconds = ~18 hours per epoch, ~54 hours total. Cost: 54 hours * 32 GPUs * $3/hr = ~$5,200. With overhead (scheduling, checkpointing, evaluation): ~$7,000. Recommendation: 32x H100 cluster (4 nodes of 8 GPUs). Use verl with Megatron backend, TP=8 within nodes, DP=4 across nodes. The 32x H100 is preferred over 64x A100 despite similar cost because H100's higher memory bandwidth improves rollout speed (the bottleneck), and NVLink is faster for TP communication.</div>
</div>
`
},

// ---------------------------------------------------------------------------
// 6.8  End-to-End RL Training Case Study
// ---------------------------------------------------------------------------
{
  id: "rl-case-study",
  title: "Case Study: Training a Math Reasoning Model End-to-End",
  content: `
<p>This section walks through a complete, realistic end-to-end project: training an 8B model to solve competition mathematics using GRPO with verifiable rewards. We cover every decision point, from data preparation through evaluation, with the actual hyperparameters and results you would see in practice.</p>

<h4>Project Setup</h4>

<pre><code># Goal: Train LLaMA-3.1-8B-Instruct to solve math competition problems
# using chain-of-thought reasoning verified by exact answer matching.
#
# Hardware: 8x H100-80GB (single node)
# Budget: ~$2,000 (targeting ~50 GPU-hours of training)
# Target: > 70% accuracy on MATH500 (up from ~50% base model)

# Data preparation:
# Training data: 30K problems from NuminaMath, MATH train set, GSM8K train
# Eval data: MATH500 (held-out), GSM8K test (1319 problems)
# Format: {"prompt": "...", "answer": "42"}

# Step 1: Format prompts with chain-of-thought template
SYSTEM_PROMPT = """You are a math reasoning assistant. For each problem:
1. Think step by step inside <think>...</think> tags.
2. Give your final answer inside \\\\boxed{}.

Example:
<think>
The problem asks for 2 + 3. I can add these directly: 2 + 3 = 5.
</think>
The answer is \\\\boxed{5}."""</code></pre>

<h4>Stage 1: Cold-Start SFT</h4>

<pre><code># Before RL training, we need the model to follow the output format.
# SFT on 5K examples with correct reasoning chains.

# training_config_sft.yaml
model: meta-llama/Llama-3.1-8B-Instruct
method: sft
dataset: /data/math_sft_5k.jsonl
output_dir: /checkpoints/sft

training:
  num_epochs: 3
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  lr_scheduler: cosine
  warmup_ratio: 0.1
  max_length: 2048
  bf16: true
  gradient_checkpointing: true

peft:
  method: lora
  rank: 64
  alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  dropout: 0.05

# Expected: ~1 hour on 8x H100
# Validation accuracy after SFT: ~45-55% on MATH500
# The model can now produce the right FORMAT but isn't yet optimized for correctness.</code></pre>

<h4>Stage 2: GRPO Training</h4>

<pre><code># Now apply GRPO to optimize for correctness.
# The SFT model becomes our reference policy.

# training_config_grpo.yaml
model: /checkpoints/sft/final  # Start from SFT checkpoint
algorithm: grpo

rollout:
  engine: vllm
  tensor_parallel_size: 1
  sampling:
    temperature: 0.8
    top_p: 0.95
    max_tokens: 2048
    n: 16  # Group size G=16
  batch_size: 64  # 64 prompts per rollout batch

training:
  learning_rate: 5e-7           # Much lower than SFT!
  weight_decay: 0.01
  max_grad_norm: 1.0
  ppo_epochs: 1                 # 1 training epoch per rollout
  gradient_checkpointing: true
  bf16: true

grpo:
  clip_eps: 0.2
  kl_coef: 0.04
  gamma: 1.0
  advantage_normalization: true
  # Token-level vs response-level advantage:
  # response_level: true        # Simpler, assign same advantage to all tokens
  token_level: false

reward:
  type: math_verify
  answer_extraction: "boxed"    # Extract from \\boxed{}
  format_reward: 0.2            # 20% weight on format compliance
  correctness_reward: 0.8       # 80% weight on correct answer

data:
  train: /data/math_train_30k.jsonl
  eval: /data/math500.jsonl

schedule:
  total_steps: 1000
  eval_interval: 100
  save_interval: 200
  log_interval: 10

# Expected training time: ~24 hours on 8x H100
# Breakdown per step:
#   Rollout: 64 prompts * 16 responses * ~3s per response = ~50s
#            (parallelized across 8 GPUs: ~7s)
#   Reward: ~0.5s (answer extraction + comparison)
#   Reshard: ~0.5s
#   Training: ~2s (1 epoch over 64*16=1024 responses)
#   Total per step: ~10s
#   1000 steps * 10s = ~2.8 hours
#
# Wait, that's only 2.8 hours! The dominant cost is actually:
# - Checkpoint saving/evaluation: ~30 min total
# - Warmup and initialization: ~10 min
# - Actual: ~3.5 hours total
# (The 24-hour estimate above was conservative for budget planning.)</code></pre>

<h4>Stage 3: Monitoring Training</h4>

<pre><code># Key metrics to track during GRPO training:

# Step | Reward | KL    | Entropy | Accuracy | Grad Norm | Adv Std
# ----------------------------------------------------------------
# 0    | 0.32   | 0.000 | 4.21    | 32%      | 0.45      | 0.48
# 100  | 0.48   | 0.031 | 3.85    | 48%      | 0.62      | 0.46
# 200  | 0.55   | 0.058 | 3.62    | 55%      | 0.58      | 0.44
# 300  | 0.60   | 0.082 | 3.41    | 60%      | 0.55      | 0.41
# 500  | 0.66   | 0.125 | 3.18    | 66%      | 0.52      | 0.38
# 700  | 0.70   | 0.158 | 3.05    | 70%      | 0.50      | 0.36
# 1000 | 0.72   | 0.192 | 2.95    | 72%      | 0.48      | 0.35

# INTERPRETATION:
# - Reward increases steadily: good, model is learning.
# - KL increases slowly (< 0.2 at step 1000): healthy, not diverging.
# - Entropy decreases slowly: model becomes more confident, but not collapsed.
#   (Collapsed would be < 1.0)
# - Accuracy tracks reward: no reward hacking (they correlate).
# - Grad norm is stable: no instability.
# - Advantage std decreasing: expected as policy improves (less variance in group).

# RED FLAGS to watch for:
# - Reward increases but accuracy doesn't -> reward hacking
# - KL > 0.5 by step 200 -> too aggressive, reduce LR or increase beta
# - Entropy drops below 1.5 -> approaching collapse, add entropy bonus
# - Grad norm spikes > 5.0 -> bad batch or instability
# - Advantage std drops to < 0.1 -> all responses same quality, no signal</code></pre>

<h4>Stage 4: Evaluation and Analysis</h4>

<pre><code># Final evaluation results:
#
# Benchmark       | Base Model | After SFT | After GRPO | Improvement
# -----------------------------------------------------------------------
# MATH500         | 48.2%      | 52.4%     | 72.0%      | +23.8%
# GSM8K (test)    | 79.1%      | 82.3%     | 88.7%      | +9.6%
# AIME 2024       | 6.7%       | 10.0%     | 20.0%      | +13.3%
# AMC 2023        | 45.0%      | 52.0%     | 68.0%      | +23.0%
#
# Key observations:
# 1. SFT alone gives small improvement (mostly format compliance)
# 2. GRPO provides the main accuracy boost
# 3. Improvement is larger on harder benchmarks (MATH > GSM8K)
#    This suggests GRPO improves reasoning, not just pattern matching
# 4. AIME improvement is moderate -- frontier models (R1, o3) get 70%+
#    Our 8B model's reasoning depth is fundamentally limited

# Error analysis on MATH500:
# - Arithmetic errors: 8% (model can reason but makes calculation mistakes)
# - Logic errors: 12% (wrong approach to the problem)
# - Incomplete reasoning: 5% (gives up or truncates)
# - Format errors: 3% (answer not in \boxed{} format)
# Total error: 28%

# The most impactful next steps:
# 1. Increase training data to 100K problems (expect +3-5%)
# 2. Use rejection sampling: generate 64 solutions per problem,
#    keep only correct ones, fine-tune on them (SFT on policy outputs)
# 3. Train longer (2000-5000 steps) with learning rate annealing
# 4. Use DAPO instead of GRPO (better for hard problems)
# 5. Scale to 70B model (expect +10-15% from model capacity alone)</code></pre>

<h4>Lessons Learned</h4>

<pre><code># PRACTICAL LESSONS FROM THIS PROJECT:

# 1. SFT before RL is CRITICAL.
#    We tried skipping SFT and going directly to GRPO.
#    Result: model never learned the format, reward was always 0,
#    GRPO had no signal to learn from. Total waste of 4 hours.

# 2. Temperature matters more than you think.
#    T=0.5: group responses too similar, poor advantage estimates.
#    T=1.2: too many gibberish responses, wasted compute.
#    T=0.8: sweet spot for math reasoning.

# 3. The reward function defines what the model learns.
#    V1: binary correct/incorrect. Model learned to guess randomly.
#    V2: partial credit for correct intermediate steps. Too complex, noisy.
#    V3: 0.8 * correct + 0.2 * format. Simple and effective.
#    The simplest reward that captures your goal is the best reward.

# 4. Evaluation during training is non-negotiable.
#    At step 300, we saw reward increasing but eval accuracy flat.
#    Turned out we had a bug in the reward function that gave credit
#    for answers close to correct (floating point comparison).
#    Model learned to output "approximately right" answers.
#    Fixed to exact match, retrained from step 200 checkpoint.

# 5. Checkpoints are cheap insurance.
#    Save every 200 steps. Storage cost: ~50 GB per checkpoint for 8B LoRA.
#    We rolled back twice during this project. Worth every byte.

# 6. Start small, then scale.
#    We first validated the entire pipeline on 1K prompts, G=4, 50 steps.
#    This took 10 minutes and confirmed the pipeline worked end-to-end.
#    Only then did we commit to the full 30K prompts, G=16, 1000 steps.
#    This saved us 24 hours of debugging on the full run.</code></pre>

<div class="callout">
<div class="callout-title">Connecting Chapter 5 and Chapter 6</div>
<p>Notice how LLM serving (Chapter 5) and RL training (Chapter 6) are deeply intertwined. The rollout phase of RL training IS LLM inference -- it uses vLLM with all the same optimizations: PagedAttention, continuous batching, prefix caching. In fact, prefix caching is especially valuable for RL rollouts because all G responses for the same prompt share the prompt prefix. verl's use of vLLM for rollouts and Megatron for training is the direct application of the serving optimizations from Chapter 5 to the training workflow of Chapter 6. Understanding both chapters together is essential for building efficient RL training pipelines.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Walk through designing a complete RL training pipeline for a code generation model. Cover data, reward function, algorithm choice, infrastructure, and evaluation.</div>
<div class="a-text">Pipeline design: (1) Data: 50K coding problems from LeetCode, HumanEval-X, APPS, CodeContests. Mix of easy/medium/hard. Each problem has test cases. (2) Reward function: sandboxed code execution. Run extracted code against test cases (timeout 30s). Reward = fraction of tests passed. Add format reward (0.1 weight) for proper code block formatting. Use Docker-based sandboxes for safety. (3) Algorithm: GRPO with G=16. Code has binary-ish rewards (passes or doesn't), which GRPO handles well. Start with SFT on 5K (problem, solution) pairs to teach code format. (4) Infrastructure: 8x H100 for an 8B model (or 32x for 70B). Use verl with vLLM rollout backend. Key: sandbox execution adds latency to reward computation -- parallelize across CPU cores while GPUs handle next rollout. (5) Hyperparameters: LR=5e-7, beta_kl=0.04, temperature=0.8, max_output=4096 tokens (code can be long), 1000 steps. (6) Evaluation: HumanEval (164 problems), MBPP (500 problems), plus held-out test problems. Track pass@1 and pass@10. Expected improvement: 15-25% pass@1 increase over SFT baseline. (7) Key risk: reward hacking via test case memorization. Mitigate by using held-out test cases for evaluation that are never seen during training reward computation.</div>
</div>
`
}

]

};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CONTENT_CH5_6;
}
