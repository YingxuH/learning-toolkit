// Deeply Expanded Content for Chapters 11-12
// Chapter 11: Transformer Architecture Deep Dive (8 sections, ~12,000 words)
// Chapter 12: General LLMs (8 sections, ~12,000 words)

const CONTENT_CH11_12 = {

  // ============================================================
  // CHAPTER 11: Transformer Architecture Deep Dive
  // ============================================================
  ch11_sections: [
    // ----------------------------------------------------------
    // 11.1 Historical Context: From RNNs to Attention
    // ----------------------------------------------------------
    {
      id: "transformer-overview",
      title: "Historical Context: From RNNs to Attention",
      content: `
<p>The transformer architecture, introduced in the landmark paper "Attention Is All You Need" (arXiv:1706.03762, Vaswani et al., 2017), did not emerge in a vacuum. It was the culmination of a decade of research in sequence modeling, where each generation of models solved one problem while revealing the next. To truly understand why transformers work and where their design choices come from, you must understand what came before and why it was insufficient.</p>

<div class="callout">
<div class="callout-title">Key Insight</div>
<p>The history of sequence modeling is a history of solving the <strong>long-range dependency problem</strong>: how to let information at position 0 influence computation at position 10,000. RNNs compressed everything into a fixed-size state vector (bottleneck). Attention mechanisms let every position directly look at every other position (no bottleneck, but quadratic cost). The transformer's genius was making attention the <em>only</em> mechanism, removing recurrence entirely.</p>
</div>

<h4>1. Recurrent Neural Networks (1986-2014)</h4>
<p>Recurrent Neural Networks (RNNs), formalized for backpropagation by Rumelhart, Hinton, and Williams (1986), process sequences one token at a time, maintaining a hidden state <code>h_t = f(W_h * h_{t-1} + W_x * x_t + b)</code>. The hidden state acts as a "memory" that summarizes everything the network has seen so far. This elegant formulation has a critical flaw: the hidden state is a fixed-size vector (typically 256-1024 dimensions), meaning a 10,000-token sequence must be compressed into the same-sized representation as a 10-token sequence.</p>

<p>The practical consequence is the <strong>vanishing gradient problem</strong>. During backpropagation through time (BPTT), gradients flow backward through the chain of hidden states. At each step, they are multiplied by the recurrent weight matrix. If the spectral radius (largest eigenvalue) of this matrix is less than 1, gradients shrink exponentially; if greater than 1, they explode. For a sequence of length T, the gradient signal from position T reaching position 0 is attenuated by roughly <code>lambda^T</code>, where <code>lambda</code> is the spectral radius. For T=100 and lambda=0.9, this is <code>0.9^100 = 2.66e-5</code> &mdash; essentially zero.</p>

<h4>2. LSTMs and GRUs (1997-2016)</h4>
<p>Long Short-Term Memory (LSTM, Hochreiter and Schmidhuber, 1997) addressed vanishing gradients by introducing a <strong>cell state</strong> with gated access: forget gate, input gate, and output gate. The cell state acts as a "highway" for gradient flow &mdash; information can pass through unchanged if the forget gate is close to 1 and the input gate is close to 0. The Gated Recurrent Unit (GRU, Cho et al., 2014) simplified this to two gates (reset and update) with comparable performance.</p>

<p>LSTMs extended the effective memory window from ~10-20 tokens (vanilla RNN) to ~200-500 tokens. This was sufficient for many NLP tasks of the era but still inadequate for documents, conversations, or code. Moreover, LSTMs retained the fundamental sequential bottleneck: position T cannot be computed until positions 0 through T-1 have been processed. This makes them inherently difficult to parallelize across sequence positions, limiting training speed on modern GPU hardware.</p>

<table>
<tr><th>Architecture</th><th>Year</th><th>Effective Memory</th><th>Parallelizable?</th><th>Training Speed (relative)</th></tr>
<tr><td>Vanilla RNN</td><td>1986</td><td>~10-20 tokens</td><td>No</td><td>1x</td></tr>
<tr><td>LSTM</td><td>1997</td><td>~200-500 tokens</td><td>No</td><td>0.7x (more ops per step)</td></tr>
<tr><td>GRU</td><td>2014</td><td>~200-500 tokens</td><td>No</td><td>0.85x</td></tr>
<tr><td>Bidirectional LSTM</td><td>2005</td><td>~200-500 tokens</td><td>No (per direction)</td><td>0.35x</td></tr>
<tr><td>Transformer</td><td>2017</td><td>Full context window</td><td>Yes</td><td>5-50x</td></tr>
</table>

<h4>3. The Sequence-to-Sequence Revolution (2014)</h4>
<p>Sutskever, Vinyals, and Le (2014, arXiv:1409.3215) introduced the encoder-decoder framework for sequence-to-sequence (seq2seq) tasks, particularly machine translation. An encoder LSTM reads the source sentence and produces a fixed-size "thought vector" (the final hidden state). A decoder LSTM then generates the target sentence conditioned on this vector. This worked remarkably well but suffered from the information bottleneck: the entire source sentence had to be compressed into a single vector.</p>

<h4>4. Attention: The Breakthrough (2014-2015)</h4>
<p>Bahdanau, Cho, and Bengio (2014, arXiv:1409.0473) proposed <strong>additive attention</strong> to solve the bottleneck. Instead of relying solely on the encoder's final state, the decoder could attend to all encoder hidden states at each generation step. For each decoder step t, the model computes an attention weight for each encoder position j:</p>

<pre><code># Bahdanau (additive) attention
# s_t: decoder state at time t
# h_j: encoder hidden state at position j

e_{t,j} = v^T * tanh(W_s * s_t + W_h * h_j)  # alignment score
alpha_{t,j} = softmax(e_{t,:})_j               # attention weight
context_t = sum_j(alpha_{t,j} * h_j)           # weighted sum of encoder states</code></pre>

<p>Luong et al. (2015, arXiv:1508.04025) simplified this to <strong>multiplicative (dot-product) attention</strong>: <code>e_{t,j} = s_t^T * h_j</code>, which is cheaper to compute and works equally well. This dot-product form is the direct ancestor of transformer attention.</p>

<p>Attention solved the information bottleneck &mdash; the decoder could directly access any encoder position &mdash; but the model still used RNNs for the encoder and decoder. The recurrence remained, limiting parallelization.</p>

<h4>5. "Attention Is All You Need" (June 2017)</h4>
<p>The transformer paper, published by Vaswani et al. at Google Brain (arXiv:1706.03762), asked a radical question: what if we removed the recurrence entirely and built a model using <em>only</em> attention? The key innovations were:</p>

<ul>
<li><strong>Self-attention:</strong> Each position attends to all other positions in the same sequence (not just encoder-to-decoder). This allows the model to build rich contextual representations without recurrence.</li>
<li><strong>Multi-head attention:</strong> Instead of a single attention function, the model runs multiple attention "heads" in parallel, each learning different relationship patterns.</li>
<li><strong>Positional encoding:</strong> Since attention is permutation-equivariant (order-agnostic), sinusoidal position encodings are added to inject sequence order information.</li>
<li><strong>Layer normalization + residual connections:</strong> Every sub-layer (attention, feed-forward) has a residual connection and layer normalization for training stability.</li>
</ul>

<p>The results were striking. On the WMT 2014 English-to-German translation benchmark, the Transformer achieved 28.4 BLEU, surpassing the previous best (including all ensemble models) by over 2 BLEU points. More importantly, it trained in 3.5 days on 8 P100 GPUs &mdash; a fraction of the time required for comparable LSTM models. The parallelization advantage was decisive.</p>

<div class="callout warning">
<div class="callout-title">War Story: The Eight-GPU Moment</div>
<p>The original transformer was trained on just 8 NVIDIA P100 GPUs (16GB each). By today's standards, this is a tiny cluster. The "big" transformer model had 213M parameters &mdash; smaller than BERT-large (340M), which came just a year later. The architecture's efficiency was so profound that it scaled from 8 GPUs to 10,000+ GPUs (GPT-4 scale) without fundamental architectural changes. Few other architectures in the history of deep learning have demonstrated this kind of scaling range. The lesson: if you get the computational primitives right (matrix multiplications, attention), hardware scaling takes care of the rest.</p>
</div>

<h4>6. The Transformer's Progeny (2018-Present)</h4>
<p>Within two years, the transformer spawned three major lineages:</p>

<ul>
<li><strong>Encoder-only (BERT, 2018):</strong> Devlin et al. (arXiv:1810.04805) used the transformer encoder with masked language modeling (MLM). Bidirectional attention over the full context. Dominated NLP benchmarks for 2 years. Descendants: RoBERTa, ALBERT, DeBERTa, ELECTRA.</li>
<li><strong>Decoder-only (GPT, 2018):</strong> Radford et al. (OpenAI) used the transformer decoder with causal (left-to-right) language modeling. This lineage led to GPT-2, GPT-3, GPT-4, LLaMA, Mistral, and essentially all modern LLMs. The decoder-only architecture won because it naturally supports generation and scales with more data.</li>
<li><strong>Encoder-decoder (T5, 2019):</strong> Raffel et al. (arXiv:1910.10683) used the full original transformer architecture, framing all NLP tasks as text-to-text. Used for translation, summarization, question answering. Descendants: mT5, FLAN-T5, UL2.</li>
</ul>

<p>The decoder-only architecture became dominant for language modeling by 2022, primarily because: (1) it naturally supports autoregressive generation, (2) it requires no task-specific heads &mdash; everything is text generation, (3) it scales efficiently with more data and compute under the scaling laws discovered by Kaplan et al., and (4) instruction tuning and RLHF work particularly well with generative models.</p>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why did the decoder-only architecture (GPT-style) win over encoder-only (BERT-style) for general-purpose language AI?</div>
<div class="a-text">Several factors: (1) <strong>Generative flexibility:</strong> Decoder-only models naturally generate text token by token, supporting open-ended tasks (writing, coding, reasoning) without task-specific output heads. BERT required a classification head for each task. (2) <strong>Scaling efficiency:</strong> Every token in the training data provides a training signal (next-token prediction), while BERT only trains on the ~15% of tokens that are masked. This makes decoder-only models more data-efficient at scale. (3) <strong>Zero/few-shot ability:</strong> GPT-3 demonstrated that large decoder-only models can perform new tasks via in-context learning (prompting), eliminating the need for fine-tuning. BERT always required fine-tuning. (4) <strong>Unified interface:</strong> All tasks become text generation &mdash; translation, summarization, QA, code generation &mdash; simplifying both training and deployment.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the vanishing gradient problem in RNNs and how the transformer architecture solves it.</div>
<div class="a-text">In RNNs, gradients during backpropagation through time are multiplied by the recurrent weight matrix at each step. For a sequence of length T, the gradient from position T to position 0 is multiplied T times, causing exponential decay (if spectral radius < 1) or explosion (if > 1). This means RNNs cannot learn dependencies beyond ~200 tokens effectively. The transformer solves this in two ways: (1) <strong>Self-attention creates direct connections</strong> between any two positions regardless of distance, so the gradient path from position T to position 0 is O(1) layers deep instead of O(T) steps. (2) <strong>Residual connections</strong> around every sub-layer create "gradient highways" that allow gradients to flow through the network without attenuation. The combination means transformers can learn dependencies across their entire context window (thousands to millions of tokens).</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What are the key computational advantages of transformers over LSTMs for training on modern hardware?</div>
<div class="a-text">Three primary advantages: (1) <strong>Parallelization:</strong> In an LSTM, computing h_t requires h_{t-1}, creating a sequential dependency chain of length T. The transformer's self-attention computes all positions simultaneously using matrix multiplication, fully utilizing GPU parallelism. Training speedups of 5-50x are typical. (2) <strong>Hardware utilization:</strong> Transformers are dominated by matrix multiplications (GEMM operations), which GPUs and TPUs are specifically designed to accelerate. LSTMs involve element-wise operations and gating that underutilize hardware. (3) <strong>Memory access patterns:</strong> Transformer attention involves large, contiguous matrix operations with high arithmetic intensity (FLOPs per byte of memory access), achieving better GPU memory bandwidth utilization. LSTMs have lower arithmetic intensity due to their sequential, element-wise structure.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 11.2 Self-Attention Mechanism
    // ----------------------------------------------------------
    {
      id: "self-attention",
      title: "Self-Attention Mechanism: Q/K/V and Multi-Head Attention",
      content: `
<p>Self-attention is the core computational primitive of the transformer. Every other component &mdash; feed-forward networks, normalization, residual connections &mdash; serves to support and regulate the representations that self-attention builds. This section provides a complete mathematical derivation of the self-attention mechanism, from first principles through multi-head attention, with full complexity analysis.</p>

<h4>1. Intuition: Attention as Soft Dictionary Lookup</h4>
<p>Think of attention as a soft, differentiable dictionary lookup. In a traditional dictionary, you have a query (the word you're looking up), keys (the index entries), and values (the definitions). You find the key that exactly matches your query and return the corresponding value. Self-attention generalizes this: instead of exact matching, you compute a <strong>similarity score</strong> between your query and every key, then return a <strong>weighted combination</strong> of all values, where the weights are proportional to the similarity scores.</p>

<p>In self-attention, every token in the sequence simultaneously acts as a query (asking "what should I attend to?"), a key (announcing "here's what I contain"), and a value (offering "here's the information I provide if attended to").</p>

<h4>2. Mathematical Derivation: Scaled Dot-Product Attention</h4>

<p>Let the input sequence be <code>X in R^{n x d_model}</code>, where <code>n</code> is the sequence length and <code>d_model</code> is the model dimension. We project X into three different spaces using learned weight matrices:</p>

<pre><code># Projection matrices
W_Q in R^{d_model x d_k}    # Query projection
W_K in R^{d_model x d_k}    # Key projection
W_V in R^{d_model x d_v}    # Value projection

# Compute Q, K, V matrices
Q = X * W_Q    # Shape: (n, d_k)  - queries
K = X * W_K    # Shape: (n, d_k)  - keys
V = X * W_V    # Shape: (n, d_v)  - values

# The attention computation (3 steps):

# Step 1: Compute raw attention scores via dot product
S = Q * K^T    # Shape: (n, n)
# S[i,j] = dot product of query_i and key_j
# Measures similarity between position i's query and position j's key

# Step 2: Scale by sqrt(d_k) and apply softmax
A = softmax(S / sqrt(d_k))    # Shape: (n, n)
# A[i,j] = attention weight from position i to position j
# Each row sums to 1 (probability distribution)

# Step 3: Weighted sum of values
Output = A * V    # Shape: (n, d_v)
# Output[i] = weighted combination of all value vectors,
# weighted by how much position i attends to each position</code></pre>

<p>The complete formula in one line:</p>
<pre><code>Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V</code></pre>

<h4>3. Why Scale by sqrt(d_k)?</h4>
<p>This is a subtle but critical detail that interviewers love to ask about. If Q and K have entries drawn from a distribution with mean 0 and variance 1, then the dot product <code>q_i . k_j = sum_{l=1}^{d_k} q_{i,l} * k_{j,l}</code> has mean 0 and variance <code>d_k</code> (since it's a sum of d_k independent products, each with variance 1). For large d_k (e.g., 64 or 128), the dot products become large in magnitude, pushing the softmax into regions where it has extremely small gradients (the "saturation" region where the output is nearly one-hot).</p>

<p>Scaling by <code>1/sqrt(d_k)</code> normalizes the variance back to 1, keeping the softmax in a regime where gradients flow well. Without this scaling, training is unstable and convergence is slow. The original paper found this was essential for models with d_k >= 64.</p>

<pre><code>import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.

    Args:
        Q: Queries, shape (batch, n_heads, seq_len, d_k)
        K: Keys, shape (batch, n_heads, seq_len, d_k)
        V: Values, shape (batch, n_heads, seq_len, d_v)
        mask: Optional mask, shape (batch, 1, seq_len, seq_len) or broadcastable

    Returns:
        output: shape (batch, n_heads, seq_len, d_v)
        attention_weights: shape (batch, n_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Step 1: Compute scaled attention scores
    # (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len)
    # -> (batch, n_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Step 2: Apply mask (for causal attention or padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights</code></pre>

<h4>4. Causal (Autoregressive) Masking</h4>
<p>For decoder-only models (GPT, LLaMA, etc.), each position can only attend to itself and earlier positions &mdash; it cannot "see the future." This is enforced by a causal mask: an upper-triangular matrix of negative infinities applied to the attention scores before softmax. After softmax, these positions become zero.</p>

<pre><code># Causal mask for sequence length n
# mask[i,j] = 0 if j > i (future positions), 1 otherwise
causal_mask = torch.tril(torch.ones(n, n))  # Lower triangular matrix

# Example for n=5:
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]

# Applied: scores.masked_fill(causal_mask == 0, -inf)
# After softmax, masked positions become 0</code></pre>

<h4>5. Multi-Head Attention: Full Derivation</h4>
<p>A single attention head computes one set of attention weights &mdash; one "view" of how tokens relate. Multi-head attention runs <code>h</code> attention heads in parallel, each with its own learned projections, then concatenates and projects the results. This allows the model to simultaneously attend to information from different representation subspaces at different positions.</p>

<pre><code># Multi-head attention parameters:
# h = number of heads (e.g., 8, 16, 32)
# d_k = d_v = d_model / h (each head operates on a fraction of the model dimension)
#
# For each head i in {1, ..., h}:
#   W_Q^i in R^{d_model x d_k}
#   W_K^i in R^{d_model x d_k}
#   W_V^i in R^{d_model x d_v}
#
# Output projection:
#   W_O in R^{(h * d_v) x d_model}

# Computation:
head_i = Attention(X * W_Q^i, X * W_K^i, X * W_V^i)   # Shape: (n, d_v)

MultiHead(X) = Concat(head_1, ..., head_h) * W_O
# Concat shape: (n, h * d_v) = (n, d_model)
# Output shape: (n, d_model)</code></pre>

<p>In practice, the per-head projections are implemented as a single large projection followed by a reshape, which is more efficient for GPU computation:</p>

<pre><code>import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Single large projections (equivalent to h separate small projections)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and reshape: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Final projection
        output = self.W_O(attn_output)
        return output</code></pre>

<h4>6. Complexity Analysis</h4>
<p>Understanding the computational complexity of self-attention is essential for engineering decisions about sequence length, model size, and hardware requirements.</p>

<table>
<tr><th>Operation</th><th>FLOPs</th><th>Memory</th><th>Bottleneck For</th></tr>
<tr><td>Q, K, V projections</td><td>3 * 2 * n * d_model * d_model = 6n * d^2</td><td>O(n * d)</td><td>Short sequences</td></tr>
<tr><td>Q * K^T (attention scores)</td><td>2 * n * n * d_k * h = 2 * n^2 * d</td><td>O(n^2 * h) = O(n^2 * h)</td><td>Long sequences</td></tr>
<tr><td>softmax</td><td>O(n^2 * h)</td><td>O(n^2 * h)</td><td>Long sequences</td></tr>
<tr><td>A * V (weighted sum)</td><td>2 * n * n * d_v * h = 2 * n^2 * d</td><td>O(n * d)</td><td>Long sequences</td></tr>
<tr><td>Output projection</td><td>2 * n * d * d</td><td>O(n * d)</td><td>Short sequences</td></tr>
<tr><td><strong>Total</strong></td><td><strong>O(n^2 * d + n * d^2)</strong></td><td><strong>O(n^2 * h + n * d)</strong></td><td></td></tr>
</table>

<p>The critical insight: self-attention has two terms in its complexity:</p>
<ul>
<li><code>O(n^2 * d)</code>: The attention score computation, which is <strong>quadratic in sequence length</strong>. This dominates for long sequences.</li>
<li><code>O(n * d^2)</code>: The projection layers, which are <strong>quadratic in model dimension</strong>. This dominates for short sequences with large models.</li>
</ul>

<p>For a typical LLaMA-2-7B configuration (d_model=4096, n_heads=32, d_k=128, context length 4096): the attention score computation is <code>2 * 4096^2 * 4096 = 137 billion FLOPs</code> per layer, while the projections are <code>8 * 4096 * 4096^2 = 549 billion FLOPs</code> per layer. The projections actually dominate! The n^2 term only dominates when n >> d_model &mdash; which happens at very long context lengths (e.g., 32K+ tokens for 4096-dim models).</p>

<p>The memory bottleneck is the attention matrix itself: storing the full <code>n x n</code> attention scores for all heads requires <code>n^2 * h * sizeof(float)</code> bytes. For n=4096, h=32, in float16: <code>4096^2 * 32 * 2 = 1 GB</code> per layer. This is why FlashAttention (Section 11.7) is so important &mdash; it avoids materializing this full matrix.</p>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Walk me through the self-attention computation step by step for a 3-token sequence. What is the shape at each step?</div>
<div class="a-text">Let's say d_model=4, n_heads=2, d_k=2. Input X has shape (3, 4). Step 1: Project to Q, K, V. Q = X @ W_Q: (3,4) @ (4,4) = (3,4). Reshape to (3, 2 heads, 2 d_k) then transpose to (2, 3, 2). Same for K, V. Step 2: For each head, compute scores: Q @ K^T: (3, 2) @ (2, 3) = (3, 3) &mdash; this is the n x n attention matrix. Each entry [i,j] is the dot product of query_i and key_j. Step 3: Scale by sqrt(d_k)=sqrt(2)=1.414. Step 4: Apply causal mask if decoder (set upper triangle to -inf). Step 5: Softmax each row to get attention weights: still (3, 3), each row sums to 1. Step 6: Multiply by V: (3, 3) @ (3, 2) = (3, 2). Step 7: Concatenate heads: (2 heads, 3, 2) -> (3, 4). Step 8: Output projection: (3, 4) @ (4, 4) = (3, 4). Total output shape = input shape = (3, 4).</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why is the scaling factor 1/sqrt(d_k) and not 1/d_k or some other value?</div>
<div class="a-text">The scaling factor normalizes the variance of the dot products. If q and k have entries with mean 0 and variance 1 (which is the case after standard initialization and layer normalization), then their dot product q.k = sum of d_k products of independent random variables, each with variance 1. By the central limit theorem, this sum has variance d_k and standard deviation sqrt(d_k). Dividing by sqrt(d_k) gives the dot product a standard deviation of 1, keeping it in a range where softmax has meaningful gradients. Dividing by d_k would over-normalize (variance would be 1/d_k, making all attention weights too uniform). The 1/sqrt(d_k) scaling achieves a "goldilocks" normalization where the softmax is neither too sharp nor too flat.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 11.3 Positional Encoding
    // ----------------------------------------------------------
    {
      id: "positional-encoding",
      title: "Positional Encoding: Sinusoidal, Learned, RoPE, ALiBi",
      content: `
<p>Self-attention is permutation-equivariant: if you shuffle the input tokens, the output is shuffled in exactly the same way. The attention mechanism has no inherent notion of position or order. This is both a strength (it enables full parallelization) and a weakness (language is fundamentally sequential). Positional encodings inject order information into the model, and the choice of encoding scheme profoundly affects the model's ability to generalize to different sequence lengths.</p>

<div class="callout">
<div class="callout-title">Key Principle</div>
<p>A good positional encoding should satisfy: (1) <strong>Uniqueness:</strong> each position gets a distinct encoding. (2) <strong>Bounded:</strong> values should not grow with sequence length. (3) <strong>Relative distance awareness:</strong> the model should easily compute the distance between two positions. (4) <strong>Length generalization:</strong> the model should handle sequences longer than those seen during training.</p>
</div>

<h4>1. Sinusoidal Positional Encoding (Original Transformer)</h4>
<p>The original transformer used fixed sinusoidal functions at different frequencies:</p>

<pre><code># For position pos and dimension i:
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

# Each dimension has a different frequency, creating a unique "fingerprint" for each position.
# Low dimensions have high frequency (rapid oscillation across positions).
# High dimensions have low frequency (slow oscillation).

# The key property: PE(pos+k) can be expressed as a linear function of PE(pos)
# This means relative positions are encoded in a way the model can learn to decode.</code></pre>

<p>The sinusoidal encoding is added to the token embeddings: <code>input = token_embedding + positional_encoding</code>. Since it's fixed (not learned), it costs zero parameters and can theoretically generalize to any sequence length. In practice, generalization beyond 2x the training length is limited.</p>

<h4>2. Learned Positional Embeddings</h4>
<p>GPT-2 and BERT used learned positional embeddings: a lookup table of shape <code>(max_seq_len, d_model)</code> where each position has a trainable embedding vector. This adds <code>max_seq_len * d_model</code> parameters (e.g., 2048 * 768 = 1.5M for BERT). The disadvantage: the model cannot generalize beyond <code>max_seq_len</code> at all &mdash; there is simply no embedding for position max_seq_len + 1.</p>

<h4>3. Rotary Position Embeddings (RoPE)</h4>
<p>RoPE (Su et al., 2021, arXiv:2104.09864) is the dominant positional encoding in modern LLMs (LLaMA, Mistral, Qwen, DeepSeek). It encodes position information by <strong>rotating</strong> the query and key vectors in a way that makes the dot product between them depend only on their relative position.</p>

<p><strong>The key idea:</strong> Instead of adding a positional vector to the embedding, RoPE <em>multiplies</em> the query/key vectors by a rotation matrix that depends on position. The rotation is applied in 2D subspaces &mdash; pairs of consecutive dimensions are rotated together.</p>

<p><strong>Mathematical derivation:</strong> Consider a 2D case first. For position m, we rotate a 2D vector by angle <code>m * theta</code>:</p>

<pre><code># 2D rotation matrix for position m:
R(m) = [[cos(m*theta), -sin(m*theta)],
        [sin(m*theta),  cos(m*theta)]]

# For a query vector q at position m and key vector k at position n:
# q_rotated = R(m) * q
# k_rotated = R(n) * k
#
# Their dot product:
# q_rotated^T * k_rotated = q^T * R(m)^T * R(n) * k
#                         = q^T * R(n - m) * k
#
# The rotation matrices are orthogonal: R(m)^T = R(-m)
# So R(m)^T * R(n) = R(-m) * R(n) = R(n-m)
#
# KEY RESULT: The dot product depends only on (n - m), the RELATIVE position!
# This is why RoPE naturally encodes relative positions.</code></pre>

<p>For the full d-dimensional case, RoPE pairs up dimensions (0,1), (2,3), ..., (d-2, d-1) and applies a 2D rotation to each pair with a different frequency:</p>

<pre><code># Full RoPE rotation matrix for position m in d dimensions:
# theta_i = 10000^{-2i/d} for i = 0, 1, ..., d/2 - 1
#
# The rotation is applied as:
# For each pair (2i, 2i+1):
#   [q_{2i}', q_{2i+1}'] = [[cos(m*theta_i), -sin(m*theta_i)],
#                            [sin(m*theta_i),  cos(m*theta_i)]] * [q_{2i}, q_{2i+1}]
#
# In block-diagonal form:
# R(m) = diag(R_0(m), R_1(m), ..., R_{d/2-1}(m))
# where R_i(m) is the 2x2 rotation matrix with angle m * theta_i</code></pre>

<p><strong>Implementation from scratch:</strong></p>

<pre><code>import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding.

    Used in LLaMA, Mistral, Qwen, DeepSeek, and most modern LLMs.
    Encodes relative position by rotating Q and K vectors.
    """
    def __init__(self, d_model, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute rotation frequencies: theta_i = base^{-2i/d}
        # i = 0, 1, ..., d/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)  # (d_model/2,)

        # Precompute sin/cos for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """Precompute cos and sin values for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        # Outer product: (seq_len,) x (d/2,) -> (seq_len, d/2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate for pairs: (seq_len, d)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        """Rotate adjacent pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, positions=None):
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor, shape (batch, n_heads, seq_len, d_k)
            k: Key tensor, shape (batch, n_heads, seq_len, d_k)
            positions: Optional position indices, shape (seq_len,)

        Returns:
            q_rotated, k_rotated: same shapes as input
        """
        seq_len = q.shape[2]

        if positions is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[positions].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[positions].unsqueeze(0).unsqueeze(0)

        # Apply rotation: x' = x * cos + rotate_half(x) * sin
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated

# Usage example:
d_model = 128  # dimension per head
rope = RotaryPositionalEmbedding(d_model, max_seq_len=4096)

# Dummy Q, K of shape (batch=2, heads=8, seq=512, d_k=128)
q = torch.randn(2, 8, 512, 128)
k = torch.randn(2, 8, 512, 128)
q_rot, k_rot = rope(q, k)
# q_rot, k_rot now have positional information baked in
# Their dot product will naturally encode relative positions</code></pre>

<h4>4. ALiBi (Attention with Linear Biases)</h4>
<p>ALiBi (Press et al., 2022, arXiv:2108.12409) takes a radically different approach: instead of modifying the embeddings, it adds a <strong>linear bias</strong> directly to the attention scores. The bias penalizes distant positions with a fixed slope m that varies per head:</p>

<pre><code># ALiBi adds a bias to attention scores:
# score(i, j) = q_i . k_j - m * |i - j|
#
# where m is a per-head slope, typically set to geometric sequence:
# For 8 heads: m = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
# = 2^{-8/8}, 2^{-8/7}, ..., 2^{-8/1}  (different heads have different slopes)
#
# Advantages:
# - No learned parameters, no cos/sin computation
# - Natural length extrapolation: longer distances just get bigger penalties
# - Proven to generalize well beyond training length (up to 6x in some experiments)</code></pre>

<p>ALiBi is used in the BLOOM model and some Falcon variants. Its main advantage is simplicity and strong length generalization, but it has been largely overtaken by RoPE in practice, partly because RoPE's relative position encoding is more expressive and partly because RoPE works better with techniques like FlashAttention.</p>

<h4>5. Relative Position Bias (T5-Style)</h4>
<p>T5 (Raffel et al., 2019) uses a learned relative position bias: a lookup table indexed by the relative position <code>j - i</code>, clipped to a maximum distance. Each attention head has its own bias table. This is flexible (learned biases) and supports relative positions, but adds parameters proportional to the number of heads times the number of distinct relative positions.</p>

<h4>6. Comparison of Positional Encoding Methods</h4>

<table>
<tr><th>Method</th><th>Used In</th><th>Params</th><th>Length Generalization</th><th>Relative Position</th><th>Efficiency</th></tr>
<tr><td>Sinusoidal</td><td>Original Transformer</td><td>0</td><td>Moderate (2x)</td><td>Implicit</td><td>High</td></tr>
<tr><td>Learned</td><td>GPT-2, BERT</td><td>n_max * d</td><td>None</td><td>No</td><td>High</td></tr>
<tr><td>RoPE</td><td>LLaMA, Mistral, Qwen</td><td>0</td><td>Good (with scaling)</td><td>Yes (exact)</td><td>High</td></tr>
<tr><td>ALiBi</td><td>BLOOM, Falcon</td><td>0</td><td>Excellent (6x+)</td><td>Yes</td><td>Highest</td></tr>
<tr><td>Relative Bias</td><td>T5, DeBERTa</td><td>h * n_buckets</td><td>Good</td><td>Yes</td><td>Medium</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why does RoPE encode relative positions, and what is the mathematical proof?</div>
<div class="a-text">RoPE applies a position-dependent rotation to Q and K vectors. For position m, q is rotated by R(m); for position n, k is rotated by R(n). The dot product between rotated q and k is: q^T R(m)^T R(n) k = q^T R(n-m) k, because rotation matrices are orthogonal (R(m)^T = R(-m)) and compose as R(-m)R(n) = R(n-m). This means the attention score between positions m and n depends only on the relative offset (n-m), not on the absolute positions. This is the defining property of a relative position encoding. The proof relies on the multiplicative group structure of rotation matrices: R(a) * R(b) = R(a+b).</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you extend a model trained with max context 4096 to handle 32768 tokens? What are the tradeoffs?</div>
<div class="a-text">For RoPE-based models (LLaMA, Mistral), several approaches exist: (1) <strong>Position interpolation</strong> (Chen et al., 2023): Scale all position indices by 4096/32768, so position 32768 maps to position 4096. Requires some continued pretraining. (2) <strong>NTK-aware scaling</strong>: Modify the RoPE base frequency from 10000 to a larger value (e.g., 10000 * (32768/4096)^{d/(d-2)}), which spreads the rotation frequencies to cover longer distances without compressing nearby positions. (3) <strong>YaRN</strong> (Yet Another RoPE Extension): Combines NTK scaling for high frequencies, interpolation for low frequencies, and a temperature factor. Best results among scaling methods. For ALiBi models, extension is simpler since ALiBi naturally extrapolates. Tradeoffs: all methods slightly degrade performance on short sequences while enabling long sequences. Position interpolation is simplest but requires fine-tuning. NTK-aware scaling works out-of-the-box but may lose precision at medium distances.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 11.4 Feed-Forward Networks
    // ----------------------------------------------------------
    {
      id: "ffn-architecture",
      title: "Feed-Forward Networks: FFN, SwiGLU, and Knowledge Storage",
      content: `
<p>Each transformer layer consists of two sub-layers: self-attention and a position-wise feed-forward network (FFN). While attention gets most of the press, the FFN contains the majority of the model's parameters (typically 2/3 of total) and is where the model stores factual knowledge. Understanding FFN design choices is critical for both understanding how LLMs work and for making informed engineering decisions.</p>

<h4>1. Standard FFN (Original Transformer)</h4>
<p>The original transformer FFN is a simple two-layer MLP applied independently to each position:</p>

<pre><code># Standard FFN:
FFN(x) = W_2 * ReLU(W_1 * x + b_1) + b_2

# W_1 in R^{d_model x d_ff}     (up-projection)
# W_2 in R^{d_ff x d_model}     (down-projection)
# d_ff = 4 * d_model typically   (expansion ratio of 4x)

# For d_model = 4096 (LLaMA-7B):
# W_1: 4096 x 16384 = 67M parameters
# W_2: 16384 x 4096 = 67M parameters
# Total: 134M parameters per FFN layer
# With 32 layers: 134M * 32 = 4.3B FFN parameters (out of ~7B total)</code></pre>

<p>The FFN applies the same transformation to every position independently &mdash; it's "position-wise." Each position's representation is projected up to a higher dimension (d_ff), passed through a nonlinearity, and projected back down. This expansion-contraction pattern is thought to enable the network to represent complex functions: the high-dimensional intermediate space allows the network to disentangle features that are entangled in the lower-dimensional space.</p>

<h4>2. SwiGLU: The Modern FFN (LLaMA, Mistral, Qwen)</h4>
<p>Nearly all modern LLMs have replaced the standard FFN with SwiGLU (Shazeer, 2020, arXiv:2002.05202), a gated linear unit with the SiLU (Swish) activation:</p>

<pre><code># SwiGLU FFN:
SwiGLU(x) = (Swish(W_gate * x) * (W_up * x)) * W_down

# Where Swish(z) = z * sigmoid(z) = z * (1 / (1 + exp(-z)))
# Also known as SiLU (Sigmoid Linear Unit)

# Three weight matrices instead of two:
# W_gate in R^{d_model x d_ff}    (gate projection)
# W_up   in R^{d_model x d_ff}    (up-projection, no activation)
# W_down in R^{d_ff x d_model}    (down-projection)

# To keep parameter count similar to standard FFN with d_ff = 4 * d_model,
# SwiGLU uses d_ff = (2/3) * 4 * d_model = 8/3 * d_model
# (rounded to nearest multiple of 256 for hardware efficiency)
#
# LLaMA-7B: d_model=4096, d_ff=11008 (closest multiple of 256 to 8/3 * 4096 = 10922.67)</code></pre>

<p><strong>Why SwiGLU works better:</strong> The gating mechanism (element-wise multiplication of two parallel projections) allows the network to selectively activate different features. The gate projection decides "which features are relevant," while the up-projection computes "what the feature values should be." This is more expressive than a single projection followed by a fixed nonlinearity. Empirically, SwiGLU gives ~0.5-1% improvement on perplexity benchmarks compared to standard FFN, which compounds across the many layers of a deep transformer.</p>

<pre><code>import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network as used in LLaMA, Mistral, etc."""

    def __init__(self, d_model, d_ff=None, bias=False):
        super().__init__()
        if d_ff is None:
            # Standard ratio: 8/3 * d_model, rounded to multiple of 256
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = 256 * ((d_ff + 255) // 256)  # round up

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        # Gate: apply SiLU activation to gate projection
        gate = F.silu(self.w_gate(x))
        # Up: linear projection (no activation)
        up = self.w_up(x)
        # Element-wise multiply, then down-project
        return self.w_down(gate * up)

# Parameter count comparison for d_model=4096:
# Standard FFN: 2 * 4096 * 16384 = 134M params
# SwiGLU FFN:   3 * 4096 * 11008 = 135M params (similar total)</code></pre>

<h4>3. GeGLU and Other Gated Variants</h4>
<p>SwiGLU is part of a family of gated linear units (GLU variants) studied by Shazeer:</p>

<table>
<tr><th>Variant</th><th>Formula</th><th>Activation</th><th>Used In</th></tr>
<tr><td>GLU</td><td>sigmoid(Wx) * Vx</td><td>Sigmoid</td><td>Original (Dauphin 2016)</td></tr>
<tr><td>ReGLU</td><td>ReLU(Wx) * Vx</td><td>ReLU</td><td>Research</td></tr>
<tr><td>GeGLU</td><td>GELU(Wx) * Vx</td><td>GELU</td><td>Some GPT variants, PaLM</td></tr>
<tr><td>SwiGLU</td><td>Swish(Wx) * Vx</td><td>SiLU/Swish</td><td>LLaMA, Mistral, Qwen</td></tr>
</table>

<p>PaLM (Google, 2022) uses GeGLU, while LLaMA and most open-source models use SwiGLU. The differences in practice are small (~0.1% perplexity), but SwiGLU has become the de facto standard.</p>

<h4>4. How FFNs Store Knowledge</h4>
<p>A remarkable line of research has shown that FFN layers function as <strong>key-value memories</strong> that store factual knowledge (Geva et al., 2021, arXiv:2012.14913). The first layer (W_1/W_gate) acts as "keys" that match input patterns, and the second layer (W_2/W_down) acts as "values" that contribute to the output when a key is matched. Each row of W_1 can be thought of as a pattern detector, and the corresponding column of W_2 is the information retrieved when that pattern fires.</p>

<p>This has practical implications: when you fine-tune a model, you are primarily updating these key-value memories. When a model "hallucinates" a wrong fact, it's because the wrong FFN neurons are firing. Knowledge editing techniques (ROME, MEMIT) work by directly modifying specific rows/columns of FFN weight matrices.</p>

<h4>5. Mixture of Experts (MoE) FFN</h4>
<p>Mixture of Experts replaces the single FFN with multiple "expert" FFNs, of which only a few are activated for each token. This allows scaling model parameters without proportionally scaling compute.</p>

<pre><code># Standard FFN: every token goes through the same FFN
# MoE FFN: a router selects top-k experts for each token

# Mixtral 8x7B architecture:
# - 8 expert FFNs per layer, each identical to a 7B-class FFN
# - Router network selects top-2 experts per token
# - Total parameters: ~46.7B (8 experts * ~5.8B per expert + shared params)
# - Active parameters per token: ~12.9B (only 2 experts active)
# - Inference cost similar to a 13B dense model, quality closer to a 40B+ model

class MoELayer(nn.Module):
    """Simplified Mixture of Experts layer."""
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: linear layer mapping token to expert scores
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert FFNs
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch*seq, d_model)

        # Compute routing scores
        router_logits = self.router(x_flat)  # (batch*seq, n_experts)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # both (batch*seq, top_k)

        # Softmax over selected experts only
        top_k_weights = torch.softmax(top_k_logits, dim=-1)

        # Compute expert outputs (simplified; real impl uses sparse dispatch)
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]      # (batch*seq,)
            expert_weight = top_k_weights[:, i]    # (batch*seq,)

            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask].unsqueeze(-1) * expert_output

        return output.view(batch_size, seq_len, d_model)</code></pre>

<p>The router training is tricky: without balancing, tokens converge to just 1-2 experts ("expert collapse"). Solutions include auxiliary load-balancing losses that penalize uneven expert usage, and "expert choice" routing (Zhou et al., 2022) where experts choose their tokens instead of tokens choosing experts.</p>

<div class="callout warning">
<div class="callout-title">War Story: Expert Collapse in Production</div>
<p>A team training a MoE model for code generation found that after 20% of training, 6 out of 8 experts received nearly zero tokens. The router had learned to route everything to the "safest" 2 experts, wasting 75% of the model's parameters. The fix: (1) adding an auxiliary load-balancing loss with coefficient 0.01, (2) adding random noise to router logits during training (Gaussian, std=0.1), and (3) initializing the router with small random weights (std=0.02) instead of zeros. After these changes, all 8 experts received between 10-15% of tokens each (ideal uniform would be 12.5%). The quality improvement from using all experts was equivalent to a 40% increase in model size.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why does the FFN in modern transformers use 8/3 * d_model as the intermediate dimension instead of 4 * d_model?</div>
<div class="a-text">The 8/3 ratio is specifically for SwiGLU/GeGLU architectures. Standard FFN has 2 weight matrices (W_1, W_2) with d_ff = 4 * d_model, giving 2 * d_model * 4 * d_model = 8 * d_model^2 parameters. SwiGLU has 3 weight matrices (W_gate, W_up, W_down), so to keep the total parameter count the same: 3 * d_model * d_ff = 8 * d_model^2, giving d_ff = 8/3 * d_model. This ensures a fair comparison: same parameter budget, but SwiGLU uses 3 smaller matrices instead of 2 larger ones. In practice, the value is rounded to the nearest multiple of 256 (or 128) for GPU efficiency, which is why LLaMA-7B uses d_ff = 11008 instead of the exact 10923.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 11.5 Normalization
    // ----------------------------------------------------------
    {
      id: "normalization",
      title: "Layer Normalization: Pre-LN, Post-LN, RMSNorm, DeepNorm",
      content: `
<p>Normalization layers are the unsung heroes of transformer training. Without them, training deep transformers (32+ layers) is essentially impossible &mdash; gradients either explode or vanish, and the model fails to converge. The choice of normalization scheme and its placement within the transformer block has a significant impact on training stability, final performance, and convergence speed.</p>

<h4>1. Layer Normalization Basics</h4>
<p>Layer normalization (Ba et al., 2016, arXiv:1607.06450) normalizes the activations across the feature dimension for each individual token:</p>

<pre><code># Layer Normalization for a vector x of dimension d:
mu = mean(x)                    # Mean across feature dimension
sigma = sqrt(var(x) + epsilon)  # Standard deviation (epsilon for numerical stability)

LayerNorm(x) = gamma * (x - mu) / sigma + beta

# gamma, beta: learned affine parameters (scale and shift), both of shape (d,)
# epsilon: small constant, typically 1e-5 or 1e-6

# Key difference from Batch Normalization:
# - BatchNorm normalizes across the batch dimension (all samples, one feature)
# - LayerNorm normalizes across the feature dimension (one sample, all features)
# - LayerNorm is batch-size independent, making it suitable for variable-length sequences</code></pre>

<h4>2. Post-LN vs Pre-LN: Where to Normalize</h4>
<p>The original transformer used <strong>Post-LN</strong>: normalization is applied after the residual connection. Modern transformers universally use <strong>Pre-LN</strong>: normalization is applied before the attention/FFN sub-layer.</p>

<pre><code># Post-LN (original transformer):
x = x + Attention(x)
x = LayerNorm(x)           # Normalize AFTER residual
x = x + FFN(x)
x = LayerNorm(x)

# Pre-LN (GPT-2, LLaMA, all modern models):
x = x + Attention(LayerNorm(x))   # Normalize BEFORE sub-layer
x = x + FFN(LayerNorm(x))</code></pre>

<p><strong>Why Pre-LN won:</strong> In Post-LN, the normalization is applied after the residual addition. If the sub-layer's output has large magnitude, the normalization "absorbs" the residual connection's contribution, weakening the gradient highway. This makes deep Post-LN models (>12 layers) difficult to train without careful learning rate warmup. Pre-LN normalizes the input to each sub-layer, ensuring it has unit variance, which stabilizes the forward pass. The residual connection then adds the sub-layer's output to the unnormalized input, preserving the gradient highway.</p>

<p>Xiong et al. (2020, arXiv:2002.04745) proved theoretically that Pre-LN transformers have better-behaved gradients: the gradient norm is bounded regardless of depth, while Post-LN gradient norms can grow with depth. This is why GPT-2 (2019) switched to Pre-LN and every model since has followed.</p>

<h4>3. RMSNorm: Faster Normalization</h4>
<p>Root Mean Square Layer Normalization (Zhang and Sennrich, 2019, arXiv:1910.07467) simplifies LayerNorm by removing the mean centering step:</p>

<pre><code># Standard LayerNorm:
mu = mean(x)
x_centered = x - mu                    # Center to zero mean
rms = sqrt(mean(x_centered^2) + eps)   # RMS of centered x
output = gamma * x_centered / rms + beta

# RMSNorm (simplified):
rms = sqrt(mean(x^2) + eps)            # RMS of x (no centering!)
output = gamma * x / rms               # No beta (no shift)

# RMSNorm removes:
# 1. The mean computation and subtraction
# 2. The beta (shift) parameter
# This saves ~10-15% computation and parameters in the normalization layer</code></pre>

<p><strong>Why it works:</strong> The key insight is that the re-centering (mean subtraction) in LayerNorm is often unnecessary because the learned gamma and beta parameters can compensate. The scaling (division by RMS) is the critical operation that keeps activations bounded. Empirically, RMSNorm achieves identical performance to LayerNorm while being faster.</p>

<p><strong>Implementation from scratch:</strong></p>

<pre><code>import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used in LLaMA, LLaMA-2, LLaMA-3, Mistral, Mixtral, Qwen, DeepSeek.
    Faster than LayerNorm with equivalent performance.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # gamma (scale)
        # No bias (beta) parameter - this is a key simplification

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., d_model)
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        # Using float32 for numerical stability even if x is float16/bfloat16
        input_dtype = x.dtype
        x = x.float()

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_normed = x / rms

        return (self.weight * x_normed).to(input_dtype)

# Performance comparison:
# For d_model=4096, seq_len=2048, batch=4:
# LayerNorm: ~0.42ms per call (mean computation + centering + variance + normalization)
# RMSNorm:   ~0.35ms per call (just RMS + normalization)
# Speedup:   ~17% per normalization call
# Total impact: with 2 norms per layer * 32 layers = 64 calls,
# saves ~4.5ms per forward pass, which adds up at scale</code></pre>

<h4>4. DeepNorm: Training Very Deep Transformers</h4>
<p>DeepNorm (Wang et al., 2022, arXiv:2203.00555) enables training of extremely deep transformers (up to 1000 layers in experiments) by modifying the residual connection with a scaling factor:</p>

<pre><code># Standard Pre-LN:
x = x + SubLayer(LayerNorm(x))

# DeepNorm:
x = LayerNorm(alpha * x + SubLayer(x))

# Where alpha = (2 * N)^{1/4} for a model with N layers
# Additionally, the sub-layer weights are initialized with a scale of beta = (8 * N)^{-1/4}

# For a 100-layer model: alpha = (200)^{0.25} = 3.76, beta = (800)^{-0.25} = 0.188
# The larger alpha amplifies the residual connection relative to the sub-layer output,
# which stabilizes gradient flow through very deep models.</code></pre>

<p>DeepNorm was key to training Microsoft's 2.5B-parameter, 200-layer language model that outperformed shallower but larger models. It's a niche technique but important if you're exploring deep architectures.</p>

<h4>5. Normalization in Modern LLMs: A Summary</h4>

<table>
<tr><th>Model</th><th>Normalization</th><th>Placement</th><th>Extra Details</th></tr>
<tr><td>Original Transformer</td><td>LayerNorm</td><td>Post-LN</td><td>With bias terms</td></tr>
<tr><td>GPT-2</td><td>LayerNorm</td><td>Pre-LN</td><td>Additional final LayerNorm</td></tr>
<tr><td>BERT</td><td>LayerNorm</td><td>Post-LN</td><td>Standard formulation</td></tr>
<tr><td>LLaMA / LLaMA-2 / LLaMA-3</td><td>RMSNorm</td><td>Pre-LN</td><td>No bias, eps=1e-6</td></tr>
<tr><td>Mistral / Mixtral</td><td>RMSNorm</td><td>Pre-LN</td><td>Same as LLaMA</td></tr>
<tr><td>Qwen / Qwen-2</td><td>RMSNorm</td><td>Pre-LN</td><td>Same as LLaMA</td></tr>
<tr><td>DeepSeek-V2</td><td>RMSNorm</td><td>Pre-LN</td><td>With DeepNorm for stability</td></tr>
<tr><td>Gemma</td><td>RMSNorm</td><td>Pre-LN + Post-LN</td><td>Both pre and post normalization</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is RMSNorm and why is it preferred over LayerNorm in modern LLMs?</div>
<div class="a-text">RMSNorm normalizes by dividing by the root mean square of the input: RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps). It differs from LayerNorm in two ways: (1) it removes the mean subtraction step (no centering), and (2) it has no bias/shift parameter. This saves compute (~15% faster per normalization call) and parameters. The insight is that the mean centering in LayerNorm is redundant: the learned gamma can compensate, and the critical operation for training stability is the scaling (keeping activations bounded), which RMSNorm preserves. Empirically, RMSNorm achieves the same model quality as LayerNorm across all tested scales (from 125M to 70B+ parameters).</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the difference between Pre-LN and Post-LN and why Pre-LN is universally used in modern models.</div>
<div class="a-text">In Post-LN (original transformer): output = LayerNorm(x + SubLayer(x)). In Pre-LN (modern): output = x + SubLayer(LayerNorm(x)). The critical difference is gradient flow. In Post-LN, the normalization after the residual addition can attenuate the residual signal, making gradients unstable in deep models. The gradient through the residual path in Post-LN passes through a normalization layer that depends on the sub-layer output, creating coupling. In Pre-LN, the residual path is "clean": gradients can flow directly from output to input through the addition, unimpeded by normalization. Xiong et al. proved that Pre-LN gradient norms are bounded independent of depth, while Post-LN gradients can grow proportional to depth. Practically, Post-LN requires careful warmup (the original paper used 4000 warmup steps), while Pre-LN trains stably even without warmup.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 11.6 Modern Architectures
    // ----------------------------------------------------------
    {
      id: "modern-architectures",
      title: "Modern Architectures: GPT, BERT, LLaMA, Mistral, Mixtral",
      content: `
<p>The transformer has been instantiated in dozens of architectures, but a handful dominate the landscape. Understanding the specific design choices in each architecture &mdash; and, critically, the reasoning behind those choices &mdash; is essential for any AI engineer. This section provides detailed architectural walkthroughs with parameter count calculations.</p>

<h4>1. The Three Paradigms</h4>

<table>
<tr><th>Paradigm</th><th>Attention Pattern</th><th>Training Objective</th><th>Best For</th><th>Examples</th></tr>
<tr><td><strong>Encoder-only</strong></td><td>Bidirectional (full)</td><td>Masked LM (MLM)</td><td>Classification, NER, retrieval</td><td>BERT, RoBERTa, DeBERTa</td></tr>
<tr><td><strong>Decoder-only</strong></td><td>Causal (left-to-right)</td><td>Next-token prediction</td><td>Generation, chat, reasoning</td><td>GPT, LLaMA, Mistral</td></tr>
<tr><td><strong>Encoder-decoder</strong></td><td>Full (encoder) + causal (decoder) + cross-attention</td><td>Denoising / seq2seq</td><td>Translation, summarization</td><td>T5, BART, FLAN-T5</td></tr>
</table>

<h4>2. BERT Architecture (Encoder-Only)</h4>
<p>BERT (Devlin et al., 2018, arXiv:1810.04805) uses only the encoder half of the transformer. Key design choices:</p>
<ul>
<li><strong>Bidirectional attention:</strong> Every token attends to every other token (no causal mask). This gives BERT a richer understanding of context &mdash; it can see both left and right &mdash; but makes generation impractical.</li>
<li><strong>Training objective:</strong> Masked Language Modeling (randomly mask 15% of tokens, predict them) + Next Sentence Prediction (NSP). RoBERTa later showed NSP is unnecessary.</li>
<li><strong>Positional encoding:</strong> Learned positional embeddings, max 512 tokens.</li>
<li><strong>BERT-base:</strong> 12 layers, d_model=768, 12 heads, d_ff=3072, 110M params</li>
<li><strong>BERT-large:</strong> 24 layers, d_model=1024, 16 heads, d_ff=4096, 340M params</li>
</ul>

<h4>3. GPT Architecture (Decoder-Only)</h4>
<p>GPT (Radford et al., 2018) uses only the decoder half with causal masking:</p>
<ul>
<li><strong>Causal attention:</strong> Each token can only attend to previous tokens. This enables autoregressive generation.</li>
<li><strong>Training objective:</strong> Next-token prediction. Every token provides a training signal, making it more data-efficient than MLM.</li>
<li><strong>GPT-2 innovations (2019):</strong> Pre-LN (moved LayerNorm to before sub-layers), larger scale (1.5B params)</li>
<li><strong>GPT-3 innovations (2020, arXiv:2005.14165):</strong> 175B parameters, alternating dense and sparse attention layers, in-context learning via prompting</li>
</ul>

<h4>4. LLaMA Architecture (Detailed Walkthrough)</h4>
<p>LLaMA (Touvron et al., 2023, arXiv:2302.13971) established the modern open-source LLM blueprint. Nearly every open model since follows its architecture with minor variations. Here is the complete specification:</p>

<pre><code># LLaMA-2-7B Architecture (Touvron et al., 2023):
# -----------------------------------------------
config = {
    "d_model": 4096,          # Hidden dimension
    "n_layers": 32,            # Number of transformer layers
    "n_heads": 32,             # Number of attention heads
    "n_kv_heads": 32,          # Number of KV heads (LLaMA-2-7B: same as n_heads)
    "d_k": 128,                # d_model / n_heads = 4096 / 32
    "d_ff": 11008,             # SwiGLU intermediate: round(8/3 * 4096) to multiple of 256
    "vocab_size": 32000,       # SentencePiece BPE vocabulary
    "max_seq_len": 4096,       # Context window (LLaMA-2)
    "norm": "RMSNorm",         # Pre-LN RMSNorm, eps=1e-6
    "activation": "SwiGLU",    # Gated activation in FFN
    "positional": "RoPE",      # Rotary positional embeddings, base=10000
    "bias": False,             # No bias terms anywhere (attention, FFN, norms)
}

# Per-layer parameter count:
# Attention:
#   Q projection: d_model * d_model = 4096 * 4096 = 16.8M
#   K projection: d_model * d_model = 4096 * 4096 = 16.8M
#   V projection: d_model * d_model = 4096 * 4096 = 16.8M
#   O projection: d_model * d_model = 4096 * 4096 = 16.8M
#   Subtotal: 4 * 4096^2 = 67.1M
#
# FFN (SwiGLU):
#   W_gate: d_model * d_ff = 4096 * 11008 = 45.1M
#   W_up:   d_model * d_ff = 4096 * 11008 = 45.1M
#   W_down: d_ff * d_model = 11008 * 4096 = 45.1M
#   Subtotal: 3 * 4096 * 11008 = 135.3M
#
# RMSNorm (2 per layer): 2 * 4096 = 8.2K (negligible)
#
# Per-layer total: 67.1M + 135.3M = 202.4M
# 32 layers: 202.4M * 32 = 6.48B
#
# Embeddings:
#   Token embedding: vocab_size * d_model = 32000 * 4096 = 131.1M
#   (No separate positional embedding - RoPE is applied at runtime)
#
# Final RMSNorm + output head:
#   RMSNorm: 4096 params
#   LM head: d_model * vocab_size = 4096 * 32000 = 131.1M
#   (Often weight-tied with token embedding, saving 131.1M)
#
# TOTAL: 6.48B + 131.1M + 131.1M = ~6.74B parameters
# (Published: 6.7B - matches!)</code></pre>

<h4>5. LLaMA-2 vs LLaMA-3 Changes</h4>

<table>
<tr><th>Feature</th><th>LLaMA-2</th><th>LLaMA-3</th><th>Reasoning</th></tr>
<tr><td>Vocabulary</td><td>32K (SentencePiece)</td><td>128K (tiktoken)</td><td>Better multilingual, code, reduced token count</td></tr>
<tr><td>GQA</td><td>Only 70B model</td><td>All sizes</td><td>Memory efficiency at all scales</td></tr>
<tr><td>Context</td><td>4096</td><td>8192 (base), 128K (extended)</td><td>Longer documents, complex reasoning</td></tr>
<tr><td>RoPE base</td><td>10000</td><td>500000</td><td>Better long-context via higher base frequency</td></tr>
<tr><td>Training data</td><td>2T tokens</td><td>15T tokens</td><td>More data, higher quality filtering</td></tr>
</table>

<h4>6. Mistral and Sliding Window Attention</h4>
<p>Mistral 7B (Jiang et al., 2023, arXiv:2310.06825) introduced sliding window attention (SWA) as a practical efficiency improvement:</p>

<pre><code># Standard causal attention: each token attends to ALL previous tokens
# Sliding Window Attention: each token attends to only the last W tokens

# Mistral 7B: W = 4096 (sliding window size)
# At each layer, each token attends to positions [max(0, i-W), i]
# This reduces the attention matrix from n*n to n*W

# But information propagates further than W through stacking!
# Layer 1: token i sees [i-W, i]
# Layer 2: token i sees [i-2W, i] (because tokens at i-W already saw [i-2W, i-W])
# Layer L: token i effectively sees [i-L*W, i]
# With L=32 layers and W=4096: effective context = 32*4096 = 131072 tokens!

# Memory savings: for n >> W:
# Standard: O(n^2) attention memory -> 4096^2 = 16M entries per head
# SWA: O(n * W) attention memory    -> 4096 * 4096 = 16M (same at max seq len)
# But at seq_len=32768: standard = 1B entries, SWA = 134M entries (7.5x savings)</code></pre>

<h4>7. Mixtral MoE Architecture</h4>
<p>Mixtral 8x7B (Jiang et al., 2024, arXiv:2401.04088) combines Mistral's architecture with Mixture of Experts:</p>

<pre><code># Mixtral 8x7B:
# - Same attention as Mistral 7B (GQA, SWA, RoPE)
# - FFN replaced with MoE: 8 SwiGLU experts, top-2 routing
# - Each expert has the same structure as Mistral-7B's FFN
# - d_model=4096, n_heads=32, n_kv_heads=8 (GQA)
# - Total parameters: ~46.7B
# - Active parameters per token: ~12.9B (2 of 8 experts + shared attention)
# - Performance: competitive with LLaMA-2-70B at 1/3 the inference cost</code></pre>

<h4>8. Parameter Count Calculation Guide</h4>
<p>Here is a general formula for calculating transformer parameters:</p>

<pre><code>def count_transformer_params(
    d_model, n_layers, n_heads, d_ff, vocab_size,
    n_kv_heads=None, tie_embeddings=True, has_bias=False
):
    """Calculate total parameters for a transformer LLM."""
    if n_kv_heads is None:
        n_kv_heads = n_heads

    d_k = d_model // n_heads

    # Per-layer attention parameters
    # Q: d_model -> n_heads * d_k = d_model (standard)
    attn_q = d_model * (n_heads * d_k)
    # K: d_model -> n_kv_heads * d_k (may be smaller with GQA)
    attn_k = d_model * (n_kv_heads * d_k)
    # V: same as K
    attn_v = d_model * (n_kv_heads * d_k)
    # Output projection
    attn_o = (n_heads * d_k) * d_model

    attn_total = attn_q + attn_k + attn_v + attn_o

    # Per-layer FFN parameters (SwiGLU: 3 matrices)
    ffn_total = 3 * d_model * d_ff

    # Normalization: 2 RMSNorm per layer
    norm_per_layer = 2 * d_model

    # Per-layer total
    per_layer = attn_total + ffn_total + norm_per_layer
    if has_bias:
        per_layer += (attn_q + attn_k + attn_v + attn_o) // d_model  # bias terms
        per_layer += 3 * d_ff + d_model  # FFN biases

    # Embedding
    embedding = vocab_size * d_model

    # LM head (output projection)
    lm_head = 0 if tie_embeddings else vocab_size * d_model

    # Final normalization
    final_norm = d_model

    total = n_layers * per_layer + embedding + lm_head + final_norm

    print(f"Per-layer attention: {attn_total/1e6:.1f}M")
    print(f"Per-layer FFN:       {ffn_total/1e6:.1f}M")
    print(f"Per-layer total:     {per_layer/1e6:.1f}M")
    print(f"All layers:          {n_layers * per_layer/1e9:.2f}B")
    print(f"Embeddings:          {embedding/1e6:.1f}M")
    print(f"TOTAL:               {total/1e9:.2f}B")
    return total

# LLaMA-2-7B
count_transformer_params(4096, 32, 32, 11008, 32000, tie_embeddings=False)
# Per-layer attention: 67.1M, Per-layer FFN: 135.3M
# All layers: 6.48B, Embeddings: 131.1M, TOTAL: 6.74B

# LLaMA-2-70B (with GQA: 8 KV heads instead of 64)
count_transformer_params(8192, 80, 64, 28672, 32000, n_kv_heads=8, tie_embeddings=False)
# Total: ~68.98B (published: 70B - close match)</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Walk me through the LLaMA architecture. What are the key design choices and why were they made?</div>
<div class="a-text">LLaMA uses: (1) <strong>Pre-LN with RMSNorm</strong> for training stability and speed (RMSNorm is ~15% faster than LayerNorm). (2) <strong>SwiGLU FFN</strong> with d_ff = 8/3 * d_model for better quality at same parameter count. (3) <strong>RoPE</strong> for relative positional encoding that enables context length extension. (4) <strong>No bias terms</strong> anywhere &mdash; this slightly reduces parameters and doesn't hurt quality. (5) <strong>GQA (Grouped Query Attention)</strong> in larger models (70B uses 8 KV heads for 64 query heads) to reduce KV cache size during inference. (6) <strong>SentencePiece BPE tokenizer</strong> with 32K vocabulary (LLaMA-2) or 128K (LLaMA-3). Each choice is motivated by specific empirical findings at scale.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 11.7 Efficient Attention
    // ----------------------------------------------------------
    {
      id: "efficient-attention",
      title: "Efficient Attention: FlashAttention, GQA, MQA",
      content: `
<p>The quadratic memory and compute cost of standard attention is the primary bottleneck for serving LLMs, especially at long context lengths. This section covers the key techniques for making attention efficient, with particular focus on FlashAttention (the single most impactful systems-level optimization in modern LLM infrastructure) and KV cache compression techniques (GQA, MQA).</p>

<h4>1. The Memory Wall Problem</h4>
<p>Before diving into solutions, let's understand the problem precisely. Modern GPUs have high compute throughput but limited memory bandwidth. An A100 GPU can perform 312 TFLOPS (312 trillion floating-point operations per second in float16) but can only transfer 2 TB/s of data from HBM (high-bandwidth memory). The <strong>arithmetic intensity</strong> (FLOPs per byte of memory access) determines whether an operation is compute-bound or memory-bound.</p>

<pre><code># Attention is memory-bound, not compute-bound!

# For a self-attention with seq_len=4096, d=128, in float16:
# Computing QK^T: 2 * 4096 * 4096 * 128 = 4.29 GFLOPs
# Reading Q and K from HBM: 2 * 4096 * 128 * 2 bytes = 2 MB
# Writing the 4096x4096 attention matrix to HBM: 4096 * 4096 * 2 = 32 MB
# Total memory: 34 MB
# Arithmetic intensity: 4.29e9 / 34e6 = 126 FLOPs/byte
# A100 needs: 312e12 / 2e12 = 156 FLOPs/byte to be compute-bound
# Attention is JUST below the threshold -> memory-bound for shorter sequences

# The real issue: the n x n attention matrix is materialized in HBM
# For n=32768, h=32: 32768^2 * 32 * 2 bytes = 64 GB -> does not fit in GPU memory!</code></pre>

<h4>2. FlashAttention: IO-Aware Attention (Dao et al., 2022)</h4>
<p>FlashAttention (Dao et al., 2022, arXiv:2205.14135) was a breakthrough that recomputes attention without materializing the full n x n attention matrix. The key ideas are <strong>tiling</strong> and <strong>online softmax</strong>.</p>

<p><strong>Tiling:</strong> Instead of computing the entire attention matrix at once, FlashAttention divides Q, K, V into blocks that fit in GPU SRAM (the fast on-chip memory, ~20 MB on A100). It processes attention one block at a time, accumulating results without writing the full attention matrix to HBM.</p>

<p><strong>Online softmax:</strong> The standard softmax requires knowing the maximum value across the entire row to compute <code>exp(x_i - max) / sum(exp(x_j - max))</code>. FlashAttention uses the "online softmax" trick: it processes blocks sequentially, maintaining a running maximum and running sum, and rescaling previous partial results when a new maximum is found. This produces the exact same result as standard softmax but block-by-block.</p>

<pre><code># Pseudocode for FlashAttention (simplified)
# Q, K, V each have shape (n, d)
# Block size B_r (rows of Q) and B_c (columns of K)

def flash_attention(Q, K, V, B_r, B_c):
    """FlashAttention: block-by-block attention without materializing n x n matrix."""
    n, d = Q.shape
    O = zeros(n, d)       # Output accumulator (in HBM)
    l = zeros(n)          # Row-sum accumulator (in HBM)
    m = full(n, -inf)     # Row-max accumulator (in HBM)

    # Process Q in blocks of B_r rows
    for i in range(0, n, B_r):
        Q_i = Q[i:i+B_r]                # Load block of queries to SRAM
        O_i = zeros(B_r, d)             # Local output (in SRAM)
        l_i = zeros(B_r)                # Local row sums (in SRAM)
        m_i = full(B_r, -inf)           # Local row maxes (in SRAM)

        # Process K, V in blocks of B_c columns
        for j in range(0, n, B_c):
            K_j = K[j:j+B_c]            # Load block of keys to SRAM
            V_j = V[j:j+B_c]            # Load block of values to SRAM

            # Compute block attention scores (in SRAM)
            S_ij = Q_i @ K_j.T / sqrt(d)   # Shape: (B_r, B_c) - fits in SRAM!

            # Online softmax update
            m_ij = S_ij.max(dim=-1)         # New block max
            m_new = max(m_i, m_ij)           # Updated running max

            # Rescale previous results
            alpha = exp(m_i - m_new)         # Rescaling factor for old results
            beta = exp(m_ij - m_new)         # Scale factor for new block

            P_ij = beta * exp(S_ij - m_ij)  # Softmax numerators for this block

            l_new = alpha * l_i + P_ij.sum(dim=-1)  # Updated row sums

            # Update output: rescale old output, add new contribution
            O_i = (alpha * l_i / l_new).unsqueeze(-1) * O_i + \
                  (P_ij / l_new.unsqueeze(-1)) @ V_j

            m_i = m_new
            l_i = l_new

        # Write block output back to HBM
        O[i:i+B_r] = O_i

    return O

# Memory: O(n) instead of O(n^2) - the n x n matrix is NEVER stored!
# Speed: 2-4x faster than standard attention due to reduced HBM access
# Result: Mathematically EXACT (no approximation)</code></pre>

<h4>3. FlashAttention-2 Improvements</h4>
<p>FlashAttention-2 (Dao, 2023, arXiv:2307.08691) improved upon the original with:</p>
<ul>
<li><strong>Better work partitioning:</strong> Parallelizes over the sequence length dimension (rows of Q) across thread blocks, and over the number of heads across warps within a thread block. This achieves much better GPU occupancy.</li>
<li><strong>Reduced non-matmul FLOPs:</strong> Optimizes the rescaling operations in online softmax to minimize non-tensor-core operations.</li>
<li><strong>Causal masking optimization:</strong> When computing causal attention (decoder), FlashAttention-2 skips entire blocks where all entries would be masked, saving ~50% of computation for causal models.</li>
</ul>

<p>FlashAttention-2 achieves ~70% of theoretical peak FLOPS on A100, compared to ~35% for standard PyTorch attention.</p>

<h4>4. Grouped Query Attention (GQA)</h4>
<p>GQA (Ainslie et al., 2023, arXiv:2305.13245) is a KV cache compression technique that shares K and V heads across multiple Q heads:</p>

<pre><code># Multi-Head Attention (MHA):
# n_heads Q heads, n_heads K heads, n_heads V heads
# Each has independent learned projections
# KV cache per token: 2 * n_heads * d_k * sizeof(dtype) bytes

# Multi-Query Attention (MQA, Shazeer 2019):
# n_heads Q heads, 1 K head, 1 V head
# All Q heads share the SAME K and V
# KV cache per token: 2 * 1 * d_k * sizeof(dtype)
# Memory reduction: n_heads times less KV cache
# Quality cost: slight degradation (~0.5% on benchmarks)

# Grouped Query Attention (GQA):
# n_heads Q heads, n_kv_heads K heads, n_kv_heads V heads
# Groups of (n_heads / n_kv_heads) Q heads share one K/V head
# Interpolates between MHA and MQA

# Example: LLaMA-2-70B
# n_heads = 64, n_kv_heads = 8
# Each group of 8 Q heads shares 1 K head and 1 V head
# KV cache reduction: 64/8 = 8x smaller
# Quality: negligible loss (within noise of MHA)

# KV cache size comparison for LLaMA-2-70B at seq_len=4096:
# MHA:   2 * 64 * 128 * 4096 * 2 bytes = 128 MB per layer * 80 layers = 10 GB
# GQA-8: 2 * 8 * 128 * 4096 * 2 bytes  = 16 MB per layer * 80 layers = 1.25 GB
# Savings: 8x less KV cache memory, enabling longer sequences or larger batches</code></pre>

<h4>5. Sliding Window Attention</h4>
<p>As described in the Mistral section, sliding window attention restricts each token to attending to only the nearest W tokens. Combined with FlashAttention, this reduces both memory and compute:</p>

<pre><code># Memory and compute comparison at seq_len=32768:
# Standard causal attention:
#   FLOPs: O(n^2 * d) = 32768^2 * 128 = 137 GFLOPs per head
#   Memory (naively): n^2 = 32768^2 * 2 bytes = 2 GB per head (n^2 attention matrix)
#   With FlashAttention: O(n * d) memory, same FLOPs
#
# Sliding window (W=4096):
#   FLOPs: O(n * W * d) = 32768 * 4096 * 128 = 17 GFLOPs per head (8x reduction)
#   Memory: O(n * d) with FlashAttention (same)
#   Quality: minimal loss due to information propagation through layers</code></pre>

<h4>6. Memory Analysis: Putting It All Together</h4>

<table>
<tr><th>Component</th><th>Standard (LLaMA-2-7B, seq=4096)</th><th>Optimized (Mistral-7B, seq=4096)</th></tr>
<tr><td>Model weights (float16)</td><td>13.4 GB</td><td>14.2 GB (slightly different d_ff)</td></tr>
<tr><td>KV cache (float16)</td><td>2 * 32 * 128 * 4096 * 32 * 2B = 2 GB</td><td>2 * 8 * 128 * 4096 * 32 * 2B = 512 MB (GQA)</td></tr>
<tr><td>Attention matrix</td><td>N/A (FlashAttention)</td><td>N/A (FlashAttention + SWA)</td></tr>
<tr><td>Activations (batch=1)</td><td>~1.5 GB</td><td>~1 GB</td></tr>
<tr><td><strong>Total inference</strong></td><td><strong>~17 GB</strong></td><td><strong>~15.7 GB</strong></td></tr>
</table>

<div class="callout warning">
<div class="callout-title">War Story: The KV Cache That Killed Our Batch Size</div>
<p>A serving team deployed LLaMA-2-70B on 4x A100-80GB GPUs using tensor parallelism. The model weights consumed 140 GB (70B * 2 bytes, float16). With 320 GB total GPU memory, they had 180 GB for KV cache and activations. They expected to serve batch_size=64 at seq_len=4096. The reality: the KV cache for LLaMA-2-70B with MHA (64 KV heads) required 10 GB per sample at 4096 tokens. Batch=64 would need 640 GB &mdash; 3.5x their available memory. They could only serve batch_size=16. Switching to a GQA-equipped variant (8 KV heads) reduced KV cache to 1.25 GB per sample, enabling batch_size=128 on the same hardware. The lesson: at production scale, KV cache dominates memory for long sequences, not model weights.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain FlashAttention. Why is it faster if it does the same number of FLOPs?</div>
<div class="a-text">FlashAttention computes the exact same result as standard attention but rearranges the computation to minimize memory access. The key insight: standard attention materializes the full n x n attention matrix in GPU HBM (slow main memory), requiring O(n^2) memory reads/writes. FlashAttention uses tiling: it divides Q, K, V into blocks that fit in SRAM (fast on-chip memory, ~20 MB), computes attention block-by-block using an online softmax trick that maintains running statistics, and never writes the full attention matrix to HBM. This reduces HBM access from O(n^2) to O(n^2 * d / SRAM_size). Since modern GPUs are memory-bandwidth-bound for attention (not compute-bound), reducing memory access directly translates to wall-clock speedup. Result: 2-4x faster, O(n) memory instead of O(n^2), mathematically exact.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is Grouped Query Attention (GQA) and why is it important for serving?</div>
<div class="a-text">GQA shares Key and Value heads across groups of Query heads. For example, LLaMA-2-70B has 64 Q heads but only 8 KV heads &mdash; each group of 8 Q heads shares one K and one V head. This reduces the KV cache size by 8x without significantly affecting model quality (Ainslie et al. showed GQA-8 performs within noise of full MHA). Why it matters for serving: during autoregressive generation, the KV cache stores the K and V tensors for all previous tokens. For a 70B model with MHA at 4096 tokens, the KV cache is ~10 GB per sample. With GQA-8, it's ~1.25 GB. This means you can either (a) serve 8x more concurrent users on the same hardware, (b) serve the same users but with 8x longer context, or (c) some combination. At $3/hour per A100, this 8x efficiency improvement translates directly to 8x cost reduction for the KV-cache-bound portion of serving.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 11.8 Training Transformers
    // ----------------------------------------------------------
    {
      id: "transformer-training",
      title: "Training Transformers: Initialization, Scheduling, Stability",
      content: `
<p>Training large transformers is an art backed by theory. Small mistakes in initialization, learning rate scheduling, or numerical precision can waste hundreds of thousands of dollars in compute. This section covers the practical knowledge you need to train transformers successfully, from weight initialization through mixed precision training.</p>

<h4>1. Weight Initialization</h4>
<p>Proper initialization ensures that activations and gradients have reasonable magnitudes at the start of training. The wrong initialization can cause immediate divergence or extremely slow convergence.</p>

<pre><code># Xavier (Glorot) Initialization:
# Used when activation is roughly linear (tanh, sigmoid) near zero
# Goal: keep variance of activations constant across layers
#
# W ~ Uniform(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))
# or equivalently:
# W ~ Normal(0, sqrt(2 / (fan_in + fan_out)))
#
# fan_in = number of input features
# fan_out = number of output features

# Kaiming (He) Initialization:
# Used when activation is ReLU or its variants
# Accounts for the fact that ReLU zeros out ~50% of activations
#
# W ~ Normal(0, sqrt(2 / fan_in))
# The factor of 2 compensates for the ReLU zeroing effect

# GPT-2 / LLaMA Initialization:
# Most modern LLMs use a modified initialization:
# - Embedding layers: Normal(0, 0.02)
# - Linear layers: Normal(0, 0.02)
# - Output projection of each residual block: Normal(0, 0.02 / sqrt(2 * n_layers))
#   This scales down the contribution of each layer, preventing
#   the residual stream from growing with depth

import torch.nn as nn
import math

def init_transformer_weights(model, n_layers, std=0.02):
    """Initialize transformer weights following GPT-2/LLaMA convention."""
    for name, param in model.named_parameters():
        if param.dim() == 1:
            # Bias terms and norm weights: initialize to 0 or 1
            if 'norm' in name or 'ln' in name:
                nn.init.ones_(param)  # Scale parameters start at 1
            else:
                nn.init.zeros_(param)  # Bias starts at 0
        elif param.dim() == 2:
            if 'output_proj' in name or 'w_down' in name:
                # Output projections of residual blocks: scaled down
                nn.init.normal_(param, mean=0.0, std=std / math.sqrt(2 * n_layers))
            else:
                # All other weight matrices
                nn.init.normal_(param, mean=0.0, std=std)</code></pre>

<h4>2. Learning Rate Warmup + Cosine Decay</h4>
<p>The standard learning rate schedule for transformer training consists of two phases:</p>

<pre><code># Phase 1: Linear warmup (typically 0.1-2% of total steps)
# Start from near-zero LR and increase linearly to peak LR
# Reason: at the start, the model's representations are random.
# Large gradients on random features would cause instability.
# Warmup lets the model establish reasonable representations before
# applying the full learning rate.

# Phase 2: Cosine decay to a minimum LR (typically 10% of peak)
# The learning rate follows a cosine curve from peak to min.
# This gradually reduces the step size as the model converges.

import math

class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay.

    Used by GPT-3, LLaMA, Mistral, and nearly all modern LLMs.
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 peak_lr, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * min_lr_ratio

    def get_lr(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / self.warmup_steps
        elif step >= self.total_steps:
            return self.min_lr
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay

    def step(self, step):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# Typical hyperparameters for different scales:
# GPT-3 175B: peak_lr=0.6e-4, warmup=375 steps, total=300B tokens
# LLaMA-7B:   peak_lr=3e-4,   warmup=2000 steps, total=1T tokens
# LLaMA-65B:  peak_lr=1.5e-4, warmup=2000 steps, total=1.4T tokens
# Rule of thumb: larger models use smaller peak LR</code></pre>

<h4>3. Gradient Clipping</h4>
<p>Gradient clipping prevents training instability caused by occasional large gradients (from outlier batches, numerical issues, or data artifacts):</p>

<pre><code># Global gradient norm clipping:
# 1. Compute the total norm of all gradients: ||g|| = sqrt(sum(g_i^2))
# 2. If ||g|| > max_norm, scale all gradients by max_norm / ||g||
#
# Typical max_norm: 1.0 (used by GPT-3, LLaMA, Mistral)

import torch

def clip_grad_norm_(parameters, max_norm=1.0):
    """Clip gradient norm. Standard practice for transformer training."""
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return total_norm  # Returns the original norm (before clipping)

# In a training loop:
optimizer.zero_grad()
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Monitor grad_norm: if it frequently exceeds max_norm by 10x+,
# the learning rate may be too high or the data has issues</code></pre>

<h4>4. Mixed Precision Training</h4>
<p>Modern LLM training uses mixed precision to reduce memory and increase throughput while maintaining training quality:</p>

<pre><code># Mixed precision strategy (following Micikevicius et al., 2017):
# - Forward pass: float16/bfloat16 (half the memory, 2x the throughput)
# - Loss computation: float32 (for numerical stability)
# - Backward pass: float16/bfloat16
# - Weight update: float32 (critical: optimizer states must be float32)
# - Master weights: float32 copy of weights for the optimizer

# bfloat16 vs float16:
# float16: 1 sign, 5 exponent, 10 mantissa bits. Range: ~6e-8 to 65504.
# bfloat16: 1 sign, 8 exponent, 7 mantissa bits. Range: ~1e-38 to 3e38.
# bfloat16 has same range as float32 but lower precision.
# Modern LLMs prefer bfloat16 because it NEVER overflows (same range as float32).
# float16 can overflow at values > 65504, requiring loss scaling.

# PyTorch mixed precision example:
from torch.cuda.amp import autocast, GradScaler

# Option 1: bfloat16 (preferred on A100/H100, no scaler needed)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input_ids)
    loss = criterion(output, labels)
loss.backward()
optimizer.step()

# Option 2: float16 (needed for V100 and older GPUs, requires scaler)
scaler = GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input_ids)
    loss = criterion(output, labels)
scaler.scale(loss).backward()     # Scale loss to prevent underflow
scaler.step(optimizer)             # Unscale gradients, then step
scaler.update()                    # Adjust scale factor

# Memory savings from mixed precision:
# Model weights: 2x reduction (float32 -> float16)
# Activations: 2x reduction
# Optimizer states (Adam): NO reduction (must stay float32)
# Adam states for 7B model: 7B * 2 * 4 bytes = 56 GB (momentum + variance)
# This is why Adam's memory cost is 3x the model size</code></pre>

<h4>5. Optimizer Choice: AdamW</h4>
<p>Nearly all transformer training uses AdamW (Loshchilov and Hutter, 2019, arXiv:1711.05101), which decouples weight decay from the gradient update:</p>

<pre><code># AdamW update rule:
# m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t        # Momentum (first moment)
# v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2      # Variance (second moment)
# m_hat = m_t / (1 - beta_1^t)                        # Bias correction
# v_hat = v_t / (1 - beta_2^t)                        # Bias correction
# theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
#           Adam update (based on gradients)                  Weight decay (independent)

# Standard hyperparameters:
# beta_1 = 0.9, beta_2 = 0.95 (GPT-3, LLaMA)
# Note: beta_2 = 0.95 is lower than the default 0.999
# This makes the optimizer more responsive to recent gradient magnitudes,
# which helps with the non-stationary loss landscape of LLM training
# weight_decay = 0.1 (applied to all 2D weight matrices, NOT biases/norms)
# eps = 1e-8</code></pre>

<h4>6. Training Stability Tips</h4>

<div class="callout">
<div class="callout-title">Production Training Stability Checklist</div>
<p>Hard-won lessons from training runs at scale:</p>
</div>

<ul>
<li><strong>Monitor loss spikes:</strong> Sudden increases in loss (>2x) indicate instability. Common causes: bad data batches (corrupted text, extremely long sequences), learning rate too high, gradient explosion. Log and investigate every spike.</li>
<li><strong>Use z-loss regularization:</strong> Add a small penalty on the log-partition function of the output softmax: <code>z_loss = 1e-4 * log(sum(exp(logits)))^2</code>. This prevents logit magnitudes from growing unbounded, a common cause of training instability in later stages.</li>
<li><strong>Checkpoint frequently:</strong> Save checkpoints every 100-500 steps at minimum. A training run that diverges at step 50,000 with the last checkpoint at step 45,000 loses 5,000 steps of compute. At scale, this can be worth $10,000+.</li>
<li><strong>Watch for embedding norm growth:</strong> If the norm of token embeddings grows steadily during training, it can cause overflow in float16. Some teams normalize embeddings or use a separate learning rate for the embedding layer.</li>
<li><strong>Data quality > data quantity:</strong> A single batch of corrupted or adversarial data can cause a loss spike that takes thousands of steps to recover from. Pre-filter your data aggressively.</li>
<li><strong>Reproducibility:</strong> Set random seeds for all sources (PyTorch, NumPy, CUDA, data sampling). Use deterministic algorithms where possible. Log the seed and all hyperparameters for every run.</li>
</ul>

<pre><code># Example: Minimal but complete training loop for a transformer LLM
import torch
from torch.optim import AdamW

def train_step(model, batch, optimizer, scheduler, scaler, step, max_grad_norm=1.0):
    """One training step with all best practices."""
    model.train()
    optimizer.zero_grad(set_to_none=True)  # set_to_none=True is slightly faster

    input_ids = batch['input_ids'].cuda()
    labels = batch['labels'].cuda()

    # Mixed precision forward pass
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(input_ids)
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        # Optional: z-loss for stability
        z_loss = 1e-4 * torch.logsumexp(shift_logits, dim=-1).pow(2).mean()
        total_loss = loss + z_loss

    # Backward
    total_loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Optimizer step
    optimizer.step()
    scheduler.step(step)

    return {
        'loss': loss.item(),
        'z_loss': z_loss.item(),
        'grad_norm': grad_norm.item(),
        'lr': scheduler.get_lr(step)
    }</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why does transformer training use learning rate warmup? What happens without it?</div>
<div class="a-text">Without warmup, the model starts with random weights and immediately receives the full learning rate. At initialization, the model's representations are meaningless, so the gradients point in essentially random directions with potentially large magnitudes. Applying a large learning rate to these random gradients causes the model to make huge, erratic weight updates that can (a) cause loss spikes, (b) destabilize normalization statistics, or (c) push weights into regions from which recovery is impossible (particularly with float16 where the dynamic range is limited). Warmup starts with a tiny learning rate and gradually increases it, giving the model time to establish reasonable representations in the first few hundred steps. Once the representations are meaningful, gradients are more informative, and the full learning rate can be applied safely. Empirically, without warmup, Pre-LN transformers can sometimes train but converge slower; Post-LN transformers typically diverge entirely.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is the memory cost breakdown for training a 7B parameter model with AdamW?</div>
<div class="a-text">For a 7B parameter model: (1) <strong>Model weights (float32 master copy):</strong> 7B * 4 bytes = 28 GB. (2) <strong>Model weights (float16 for forward/backward):</strong> 7B * 2 bytes = 14 GB. (3) <strong>Adam momentum (float32):</strong> 7B * 4 bytes = 28 GB. (4) <strong>Adam variance (float32):</strong> 7B * 4 bytes = 28 GB. (5) <strong>Gradients (float16):</strong> 7B * 2 bytes = 14 GB. Total optimizer state: 28 + 28 = 56 GB. Total training memory (without activations): 28 + 14 + 56 + 14 = 112 GB. (6) <strong>Activations</strong> (depends on batch size, sequence length, and gradient checkpointing): typically 20-100 GB. Total: ~130-210 GB, requiring 2-4 A100-80GB GPUs minimum. This is why techniques like ZeRO (sharding optimizer states across GPUs), gradient checkpointing (recompute activations during backward to save memory), and model parallelism are essential for training.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the difference between float16 and bfloat16. When would you choose one over the other?</div>
<div class="a-text">Both are 16-bit formats but with different bit allocations. float16 uses 5 exponent bits and 10 mantissa bits: high precision (~3.3 decimal digits) but limited range (max ~65504). bfloat16 uses 8 exponent bits and 7 mantissa bits: lower precision (~2.4 decimal digits) but the same range as float32 (max ~3.4e38). For LLM training: use bfloat16 if your hardware supports it (A100, H100, TPU v3+). Its float32-equivalent range means you never need loss scaling and gradient values never overflow, simplifying the training code. Use float16 on older hardware (V100, consumer GPUs) with a GradScaler to handle the limited range. For inference: bfloat16 is standard. float16 can also work but occasionally produces NaN values on models trained in bfloat16 due to range mismatches in certain layers.</div>
</div>
`
    }
  ],

  // ============================================================
  // CHAPTER 12: General LLMs
  // ============================================================
  ch12_sections: [
    // ----------------------------------------------------------
    // 12.1 The LLM Landscape
    // ----------------------------------------------------------
    {
      id: "llm-landscape",
      title: "The LLM Landscape: GPT, LLaMA, Mistral, and Beyond",
      content: `
<p>The large language model landscape has evolved at breathtaking speed since GPT-1 in 2018. Understanding the lineage, capabilities, and design philosophy of the major model families is essential for making informed engineering decisions. This section provides a comprehensive survey of the most important LLM families, their architectural innovations, and how they compare.</p>

<h4>1. The GPT Family (OpenAI)</h4>

<table>
<tr><th>Model</th><th>Year</th><th>Parameters</th><th>Training Data</th><th>Context</th><th>Key Innovation</th></tr>
<tr><td>GPT-1</td><td>2018</td><td>117M</td><td>BookCorpus (7K books)</td><td>512</td><td>Demonstrated pretraining + fine-tuning paradigm</td></tr>
<tr><td>GPT-2</td><td>2019</td><td>1.5B</td><td>WebText (40GB)</td><td>1024</td><td>Zero-shot task performance, "too dangerous to release"</td></tr>
<tr><td>GPT-3</td><td>2020</td><td>175B</td><td>300B tokens (570GB)</td><td>2048</td><td>In-context learning, few-shot prompting</td></tr>
<tr><td>GPT-3.5</td><td>2022</td><td>~175B</td><td>+ Code, RLHF</td><td>4096</td><td>InstructGPT + ChatGPT interface</td></tr>
<tr><td>GPT-4</td><td>2023</td><td>~1.8T (rumored MoE)</td><td>~13T tokens</td><td>8K/32K/128K</td><td>Multimodal, substantially improved reasoning</td></tr>
<tr><td>GPT-4o</td><td>2024</td><td>Undisclosed</td><td>Undisclosed</td><td>128K</td><td>Native multimodal, faster inference</td></tr>
</table>

<p>GPT-3 (arXiv:2005.14165) was the inflection point for the field. It demonstrated that scaling a simple architecture (decoder-only transformer with next-token prediction) to 175B parameters produced emergent capabilities: the model could perform translation, arithmetic, and code generation via in-context learning, without any task-specific training. This single result launched the modern LLM era.</p>

<h4>2. The LLaMA Family (Meta)</h4>
<p>LLaMA (Large Language Model Meta AI) democratized LLM research by releasing high-quality open-weight models:</p>

<table>
<tr><th>Model</th><th>Year</th><th>Sizes</th><th>Training Data</th><th>Context</th><th>Key Innovation</th></tr>
<tr><td>LLaMA</td><td>Feb 2023</td><td>7B, 13B, 33B, 65B</td><td>1.4T tokens</td><td>2048</td><td>Open weights, trained on public data only</td></tr>
<tr><td>LLaMA-2</td><td>Jul 2023</td><td>7B, 13B, 70B</td><td>2T tokens</td><td>4096</td><td>GQA (70B), commercial license, RLHF chat models</td></tr>
<tr><td>LLaMA-3</td><td>Apr 2024</td><td>8B, 70B</td><td>15T tokens</td><td>8192 (128K extended)</td><td>128K vocab, GQA all sizes, 15T tokens</td></tr>
<tr><td>LLaMA-3.1</td><td>Jul 2024</td><td>8B, 70B, 405B</td><td>15T+ tokens</td><td>128K</td><td>405B dense model, multilingual, tool use</td></tr>
</table>

<p>LLaMA's key contribution was showing that <strong>smaller models trained on more data</strong> could match or exceed larger models trained on less data. LLaMA-13B matched GPT-3 (175B) on many benchmarks. This validated the Chinchilla scaling findings and made powerful LLMs accessible to the research community.</p>

<h4>3. Mistral and Mixtral</h4>
<p>Mistral AI, founded by ex-DeepMind and ex-Meta researchers, has focused on efficiency-optimized architectures:</p>

<ul>
<li><strong>Mistral 7B (Oct 2023, arXiv:2310.06825):</strong> Introduced sliding window attention (W=4096) and GQA (8 KV heads). Outperformed LLaMA-2-13B despite being half the size. Released under Apache 2.0 license.</li>
<li><strong>Mixtral 8x7B (Jan 2024, arXiv:2401.04088):</strong> MoE with 8 experts, top-2 routing. 46.7B total params, 12.9B active. Matched LLaMA-2-70B quality at 1/3 the inference cost.</li>
<li><strong>Mistral Large / Mistral Medium:</strong> Proprietary larger models competing with GPT-4 class. Specific architectures not disclosed.</li>
</ul>

<h4>4. Qwen (Alibaba)</h4>
<p>Qwen has emerged as the strongest Chinese LLM family:</p>
<ul>
<li><strong>Qwen-2.5 (Sep 2024):</strong> Available in 0.5B to 72B sizes. Particularly strong in mathematics and code. Uses GQA, SwiGLU, RoPE (same as LLaMA architecture). Trained on 18T tokens of multilingual data.</li>
<li><strong>Qwen-2.5-Coder:</strong> Code-specialized variant, competitive with GPT-4 on coding benchmarks like HumanEval and MBPP.</li>
</ul>

<h4>5. DeepSeek</h4>
<p>DeepSeek has pushed the frontier on efficiency and reasoning:</p>
<ul>
<li><strong>DeepSeek-V2 (May 2024):</strong> 236B total parameters using MoE (21B active). Introduced Multi-head Latent Attention (MLA) &mdash; a compression technique that reduces KV cache to a low-dimensional latent space, cutting memory by 93.3% compared to standard MHA.</li>
<li><strong>DeepSeek-V3 (Dec 2024):</strong> 671B total parameters (37B active per token). Trained for only $5.5M in compute &mdash; remarkably efficient. Uses auxiliary-loss-free load balancing for MoE, multi-token prediction objective, and FP8 mixed precision training.</li>
<li><strong>DeepSeek-R1 (Jan 2025):</strong> Reasoning-focused model using reinforcement learning to develop chain-of-thought reasoning. Competitive with o1-class models on math and code reasoning.</li>
</ul>

<h4>6. Other Notable Families</h4>

<table>
<tr><th>Family</th><th>Organization</th><th>Key Models</th><th>Notable Feature</th></tr>
<tr><td>Gemini</td><td>Google</td><td>Gemini 1.5 Pro/Flash</td><td>1M+ token context window, native multimodal</td></tr>
<tr><td>Claude</td><td>Anthropic</td><td>Claude 3 Opus/Sonnet/Haiku, Claude 3.5/4</td><td>Constitutional AI, strong instruction following, 200K context</td></tr>
<tr><td>Falcon</td><td>TII</td><td>Falcon-40B, Falcon-180B</td><td>Early open model, used ALiBi positioning</td></tr>
<tr><td>Phi</td><td>Microsoft</td><td>Phi-3, Phi-3.5</td><td>Small but capable (3.8B), "textbook quality" training data</td></tr>
<tr><td>Command R</td><td>Cohere</td><td>Command R+</td><td>RAG-optimized, citation generation</td></tr>
<tr><td>Gemma</td><td>Google</td><td>Gemma-2 (2B, 9B, 27B)</td><td>Open weights, knowledge distillation from Gemini</td></tr>
</table>

<h4>7. Architecture Comparison</h4>

<table>
<tr><th>Feature</th><th>LLaMA-3</th><th>Mistral 7B</th><th>Qwen-2.5</th><th>DeepSeek-V3</th><th>Gemma-2</th></tr>
<tr><td>Normalization</td><td>RMSNorm</td><td>RMSNorm</td><td>RMSNorm</td><td>RMSNorm</td><td>RMSNorm (pre+post)</td></tr>
<tr><td>FFN</td><td>SwiGLU</td><td>SwiGLU</td><td>SwiGLU</td><td>SwiGLU (MoE)</td><td>GeGLU</td></tr>
<tr><td>Position</td><td>RoPE (base=500K)</td><td>RoPE (base=10K)</td><td>RoPE</td><td>RoPE (YaRN)</td><td>RoPE</td></tr>
<tr><td>Attention</td><td>GQA</td><td>GQA + SWA</td><td>GQA</td><td>MLA (latent)</td><td>GQA + local/global</td></tr>
<tr><td>Vocab Size</td><td>128K</td><td>32K</td><td>152K</td><td>128K</td><td>256K</td></tr>
<tr><td>Bias Terms</td><td>No</td><td>No</td><td>No (attention), Yes (QKV)</td><td>No</td><td>No</td></tr>
</table>

<div class="callout">
<div class="callout-title">Key Observation</div>
<p>Despite the apparent diversity in the LLM landscape, modern architectures have converged on a remarkably similar design: <strong>decoder-only transformer + Pre-LN RMSNorm + SwiGLU FFN + RoPE + GQA</strong>. The main differentiators are now (1) training data quality and scale, (2) post-training (instruction tuning, RLHF, DPO), and (3) efficiency innovations like MoE and MLA. Architecture itself has become table stakes.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Compare LLaMA-3-8B, Mistral-7B, and Gemma-2-9B. If you had to pick one for a production chat application, which would you choose and why?</div>
<div class="a-text">All three are similar in quality for general tasks, but the choice depends on requirements: (1) <strong>LLaMA-3-8B:</strong> Best for multilingual applications (128K vocab with strong non-English coverage), longest context (8K base, extendable to 128K), and the largest community ecosystem (most fine-tuned variants available on HuggingFace). (2) <strong>Mistral-7B:</strong> Most efficient for inference (smallest model at 7B, GQA + SWA reduce memory and compute), Apache 2.0 license is maximally permissive. Best if you're memory-constrained or need maximum throughput. (3) <strong>Gemma-2-9B:</strong> Benefits from knowledge distillation from Gemini (trained by Google), has strong safety alignment built in, and dual pre+post normalization may improve quality. Best for safety-sensitive applications. For a general production chat app, I'd choose LLaMA-3-8B for its ecosystem and community support, unless inference cost is the primary constraint (then Mistral-7B).</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 12.2 Scaling Laws
    // ----------------------------------------------------------
    {
      id: "scaling-laws",
      title: "Scaling Laws: Kaplan, Chinchilla, and Compute-Optimal Training",
      content: `
<p>Scaling laws are the closest thing the LLM field has to "physics" &mdash; empirical relationships that predict model performance as a function of compute, data, and parameters. Understanding scaling laws is critical for deciding how to allocate your training budget: should you train a bigger model on less data, or a smaller model on more data?</p>

<h4>1. Kaplan et al. Scaling Laws (OpenAI, 2020)</h4>
<p>The foundational paper "Scaling Laws for Neural Language Models" (arXiv:2001.08361) discovered that language model loss follows power laws in three dimensions:</p>

<pre><code># Kaplan scaling laws (2020):
# Loss as a function of parameters (N), data (D), and compute (C):

L(N) = (N_c / N)^{alpha_N}      where alpha_N ~ 0.076, N_c ~ 8.8e13
L(D) = (D_c / D)^{alpha_D}      where alpha_D ~ 0.095, D_c ~ 5.4e13
L(C) = (C_c / C)^{alpha_C}      where alpha_C ~ 0.050, C_c ~ 3.1e8

# Combined (when neither N nor D is bottleneck):
L(N, D) = [(N_c/N)^{alpha_N/alpha_D} + D_c/D]^{alpha_D}

# Key finding: performance improves as a smooth power law
# over many orders of magnitude (from 10M to 10B parameters)

# Kaplan's compute-optimal allocation:
# Given a fixed compute budget C:
# - N should scale as C^0.73 (invest most of budget in larger model)
# - D should scale as C^0.27 (data can be relatively small)
# This implies: train BIG models on relatively LITTLE data</code></pre>

<p>Kaplan's key conclusion was that <strong>model size matters more than data size</strong>: given a fixed compute budget, you should make the model as large as possible even if it means training on fewer tokens. This led to GPT-3 (175B parameters, 300B tokens &mdash; heavily over-parameterized relative to data).</p>

<h4>2. Chinchilla Scaling Laws (DeepMind, 2022)</h4>
<p>The Chinchilla paper "Training Compute-Optimal Large Language Models" (Hoffmann et al., arXiv:2203.15556) fundamentally revised Kaplan's findings:</p>

<pre><code># Chinchilla scaling law:
# For compute-optimal training, parameters and tokens should scale EQUALLY:

N_opt ~ C^{0.50}    # Parameters scale as sqrt(compute)
D_opt ~ C^{0.50}    # Data scales as sqrt(compute)

# The "Chinchilla ratio": optimal tokens ~ 20 * parameters
# A 10B model should be trained on ~200B tokens
# A 70B model should be trained on ~1.4T tokens

# Chinchilla (70B, 1.4T tokens) outperformed:
# - Gopher (280B, 300B tokens) - 4x larger but undertrained
# - GPT-3 (175B, 300B tokens) - 2.5x larger but undertrained
# Chinchilla was trained to "completion" - near-optimal for its compute budget</code></pre>

<p><strong>Why the difference from Kaplan?</strong> Kaplan's experiments used a fixed training schedule and extrapolated. Chinchilla trained each model to convergence and measured the final loss, revealing that Kaplan's models were systematically undertrained. When you train to convergence, parameters and data contribute equally to loss reduction.</p>

<h4>3. The Chinchilla Ratio in Practice</h4>

<table>
<tr><th>Model</th><th>Parameters</th><th>Training Tokens</th><th>Tokens/Params Ratio</th><th>Chinchilla Optimal?</th></tr>
<tr><td>GPT-3</td><td>175B</td><td>300B</td><td>1.7x</td><td>No (severely undertrained)</td></tr>
<tr><td>Chinchilla</td><td>70B</td><td>1.4T</td><td>20x</td><td>Yes (by design)</td></tr>
<tr><td>LLaMA-1</td><td>65B</td><td>1.4T</td><td>21.5x</td><td>~Yes</td></tr>
<tr><td>LLaMA-2</td><td>70B</td><td>2T</td><td>28.6x</td><td>Over-trained (intentionally)</td></tr>
<tr><td>LLaMA-3-8B</td><td>8B</td><td>15T</td><td>1875x</td><td>Massively over-trained</td></tr>
<tr><td>Mistral-7B</td><td>7B</td><td>~8T (estimated)</td><td>~1143x</td><td>Massively over-trained</td></tr>
</table>

<h4>4. Inference-Aware Scaling: Beyond Chinchilla</h4>
<p>The Chinchilla ratio optimizes for the <em>cheapest training cost</em> to reach a given quality. But in production, the dominant cost is <strong>inference</strong>, not training. A model is trained once but serves millions of requests. This changes the optimization:</p>

<pre><code># Inference-aware scaling:
# Total cost = Training cost + Inference cost * number_of_inference_requests
#
# Training cost ~ C = 6 * N * D  (FLOPs)
# Inference cost per token ~ 2 * N  (FLOPs for forward pass)
# If the model serves R tokens total:
#
# Total cost = 6*N*D + 2*N*R
#
# When R >> D (inference dominates):
# Minimize N (use the smallest model that achieves target quality)
# This means OVER-TRAIN: train a smaller model on MORE data
#
# LLaMA-3-8B trained on 15T tokens (1875x Chinchilla ratio) because:
# - Training cost: 6 * 8B * 15T = 720e21 FLOPs (~$5M on H100s)
# - Per-inference savings: 8B vs 70B = 8.75x cheaper per request
# - Break-even after ~10M requests (reached in hours for a popular API)
# The over-training strategy pays for itself almost immediately</code></pre>

<h4>5. The Emergent Abilities Debate</h4>
<p>Certain capabilities appear to "emerge" suddenly as models scale, rather than improving smoothly:</p>

<ul>
<li><strong>Claimed emergent abilities (Wei et al., 2022, arXiv:2206.07682):</strong> multi-step arithmetic, chain-of-thought reasoning, word unscrambling, and ~100 other tasks that show near-zero performance below a threshold scale and rapid improvement above it.</li>
<li><strong>Counter-argument (Schaeffer et al., 2023, arXiv:2304.15004):</strong> "Emergence" may be an artifact of using non-linear or discontinuous evaluation metrics (e.g., exact-match accuracy). When measured with continuous metrics (e.g., log-likelihood), performance improves smoothly. The claim: emergence is a measurement artifact, not a real phenomenon.</li>
<li><strong>Current consensus:</strong> The truth is nuanced. Some capabilities do appear suddenly (particularly those requiring composing multiple skills), but the threshold depends on the metric. For practical purposes, you should assume that larger models will be qualitatively better at complex tasks, but don't rely on specific "emergence points."</li>
</ul>

<h4>6. Compute-Optimal Training in Practice</h4>

<pre><code># How to estimate your optimal training configuration:
def estimate_training_config(target_quality_loss, compute_budget_flops):
    """
    Estimate compute-optimal N and D given a FLOP budget.
    Uses Chinchilla scaling: N_opt ~ sqrt(C/6/20), D_opt ~ 20 * N_opt
    """
    # C = 6 * N * D (approximate FLOPs for training)
    # D = 20 * N (Chinchilla ratio)
    # C = 6 * N * 20 * N = 120 * N^2
    # N = sqrt(C / 120)

    import math
    N_opt = math.sqrt(compute_budget_flops / 120)
    D_opt = 20 * N_opt

    # For inference-aware scaling (overtraining), multiply D by a factor:
    # If inference cost dominates, train 5-50x longer
    D_inference_aware = 100 * N_opt  # 5x Chinchilla

    return {
        'chinchilla_optimal_params': N_opt,
        'chinchilla_optimal_tokens': D_opt,
        'inference_aware_tokens': D_inference_aware
    }

# Example: $1M training budget on H100s (~3e23 FLOPs)
config = estimate_training_config(None, 3e23)
# Chinchilla: N ~ 50B, D ~ 1T tokens
# Inference-aware: N ~ 50B, D ~ 5T tokens
# Or: smaller model trained longer: N ~ 10B, D ~ 25T tokens</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the Chinchilla scaling laws and why they changed how the industry trains LLMs.</div>
<div class="a-text">Chinchilla (Hoffmann et al., 2022) showed that for compute-optimal training, models should be trained on approximately 20 tokens per parameter. Before Chinchilla, Kaplan et al.'s findings suggested investing most compute in larger models with less data (GPT-3: 175B params, only 300B tokens = 1.7 tokens per parameter). Chinchilla proved this was suboptimal: a 70B model trained on 1.4T tokens (20x ratio) outperformed the 280B Gopher trained on 300B tokens. The impact was immediate: LLaMA-1 (65B, 1.4T tokens) followed the Chinchilla ratio exactly and matched much larger models. The industry then went further with inference-aware scaling: since inference cost scales with model size but training is one-time, it's often better to over-train smaller models (LLaMA-3-8B on 15T tokens = 1875x Chinchilla ratio). The key insight: the "optimal" training recipe depends on whether you're optimizing for training cost or total lifecycle cost.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why are modern models like LLaMA-3-8B trained far beyond the Chinchilla optimal ratio?</div>
<div class="a-text">The Chinchilla ratio (20 tokens per parameter) minimizes training compute to reach a given loss. But in production, inference is the dominant cost, and inference cost scales linearly with model size. A smaller model trained longer (more tokens) can match a larger Chinchilla-optimal model's quality while being much cheaper to serve. For LLaMA-3-8B trained on 15T tokens: the extra training cost is marginal (training happens once), but the 8B model is ~9x cheaper to serve than a 70B model per request. With millions of daily API calls, the inference savings dwarf the extra training cost within hours. This is "inference-aware scaling" or "over-training." The tradeoff: there are diminishing returns to more data at fixed model size, so a 8B model trained on 100T tokens won't match a 70B model. But the quality-per-inference-dollar is much better.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 12.3 Tokenization
    // ----------------------------------------------------------
    {
      id: "tokenization",
      title: "Tokenization: BPE, SentencePiece, and Vocabulary Design",
      content: `
<p>Tokenization is the process of converting raw text into a sequence of integers that the model can process. It's one of the most overlooked components in the LLM pipeline, yet it has profound effects on model performance, multilinguality, cost, and even capability. A poor tokenizer can make your model 2-3x less efficient on certain languages or domains.</p>

<h4>1. Why Subword Tokenization?</h4>
<p>Three approaches to tokenization:</p>
<ul>
<li><strong>Character-level:</strong> Vocabulary of ~256 characters. Handles any text, but sequences are very long (a word like "transformer" = 11 tokens). Long sequences mean more compute, more memory, and difficulty learning long-range patterns.</li>
<li><strong>Word-level:</strong> Vocabulary of ~100K+ words. Short sequences, but cannot handle unseen words (OOV problem), misspellings, code, or morphologically rich languages.</li>
<li><strong>Subword:</strong> Vocabulary of 32K-128K subword units. The sweet spot: common words are single tokens ("the" = 1 token), rare words are split into subwords ("transformer" might be "trans" + "former"), and any text can be encoded (no OOV). This is what all modern LLMs use.</li>
</ul>

<h4>2. Byte Pair Encoding (BPE): Step-by-Step</h4>
<p>BPE (Sennrich et al., 2016, arXiv:1508.07909) is the dominant tokenization algorithm. It starts with individual characters and iteratively merges the most frequent pair:</p>

<pre><code># BPE Algorithm:
# 1. Start with a vocabulary of individual characters (or bytes)
# 2. Count all adjacent pairs in the training corpus
# 3. Merge the most frequent pair into a new token
# 4. Repeat steps 2-3 until vocabulary reaches desired size

# Example on a tiny corpus:
# Corpus: "low low low low low lower lower newest newest newest widest"
#
# Initial vocabulary: {l, o, w, e, r, n, s, t, i, d, ' '}
#
# Step 1: Most frequent pair is ('l', 'o') - appears in "low" (5x) and "lower" (2x) = 7
# Merge 'l' + 'o' -> 'lo'
# Corpus becomes: "lo w lo w lo w lo w lo w lo w e r lo w e r ..."
#
# Step 2: Most frequent pair is ('lo', 'w') - appears 7 times
# Merge 'lo' + 'w' -> 'low'
# Corpus becomes: "low low low low low low e r low e r ..."
#
# Step 3: Most frequent pair is ('e', 'r') - appears 2 times in "lower"
# Merge 'e' + 'r' -> 'er'
#
# Step 4: Most frequent pair is ('low', 'er') -> 'lower'
#
# ... continue until vocab_size reached

# Final vocabulary might include:
# {l, o, w, e, r, n, s, t, i, d, lo, low, er, lower, ne, new, est, newest, ...}</code></pre>

<p><strong>Complete BPE implementation:</strong></p>

<pre><code>from collections import Counter, defaultdict
import re

class SimpleBPE:
    """Minimal Byte Pair Encoding implementation for educational purposes."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []        # List of (pair, merged_token) in order
        self.vocab = {}         # token -> id mapping

    def _get_stats(self, words):
        """Count frequency of adjacent pairs across all words."""
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair, words):
        """Merge all occurrences of a pair in the vocabulary."""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in words.items():
            # Replace the pair with the merged token
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = freq

        return new_words

    def train(self, text):
        """Train BPE on a text corpus."""
        # Step 1: Compute word frequencies
        word_freqs = Counter(text.split())

        # Step 2: Split each word into characters, add end-of-word marker
        words = {}
        for word, freq in word_freqs.items():
            # Space-separated characters
            char_word = ' '.join(list(word)) + ' </w>'
            words[char_word] = freq

        # Step 3: Build initial vocabulary from all characters
        all_chars = set()
        for word in words:
            for char in word.split():
                all_chars.add(char)

        current_vocab_size = len(all_chars)

        # Step 4: Iteratively merge most frequent pairs
        while current_vocab_size < self.vocab_size:
            pairs = self._get_stats(words)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge it
            words = self._merge_pair(best_pair, words)
            merged_token = ''.join(best_pair)
            self.merges.append((best_pair, merged_token))
            all_chars.add(merged_token)
            current_vocab_size += 1

        # Build final vocabulary
        self.vocab = {token: idx for idx, token in enumerate(sorted(all_chars))}
        print(f"Trained BPE with {len(self.vocab)} tokens and {len(self.merges)} merges")

    def encode(self, text):
        """Encode text into token IDs."""
        tokens = []
        for word in text.split():
            # Start with characters
            chars = list(word) + ['</w>']

            # Apply merges in order
            for (pair, merged) in self.merges:
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == pair[0] and chars[i+1] == pair[1]:
                        chars = chars[:i] + [merged] + chars[i+2:]
                    else:
                        i += 1

            # Convert to IDs
            for char in chars:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    tokens.append(self.vocab.get('<unk>', 0))

        return tokens

# Usage:
bpe = SimpleBPE(vocab_size=500)
bpe.train("the cat sat on the mat the cat ate the rat " * 100)
ids = bpe.encode("the cat sat")
print(f"Token IDs: {ids}")</code></pre>

<h4>3. SentencePiece vs tiktoken</h4>

<table>
<tr><th>Feature</th><th>SentencePiece</th><th>tiktoken</th></tr>
<tr><td>Used by</td><td>LLaMA-1/2, T5, Mistral 7B</td><td>GPT-3.5/4, LLaMA-3, Claude</td></tr>
<tr><td>Algorithm</td><td>BPE or Unigram</td><td>BPE (byte-level)</td></tr>
<tr><td>Input</td><td>Unicode codepoints</td><td>UTF-8 bytes</td></tr>
<tr><td>Whitespace handling</td><td>Replaces space with special character (U+2581, the "lower one-eighth block")</td><td>Treats bytes directly, no special whitespace handling</td></tr>
<tr><td>Speed</td><td>Fast (C++ core)</td><td>Very fast (Rust core)</td></tr>
<tr><td>Unknown tokens</td><td>Falls back to character-level, then byte-level</td><td>Never produces unknown tokens (byte-level fallback)</td></tr>
</table>

<h4>4. Vocabulary Size Tradeoffs</h4>

<pre><code># Vocabulary size affects:
# 1. Sequence length (larger vocab = fewer tokens per text = shorter sequences)
# 2. Embedding table size (larger vocab = more parameters in embedding layer)
# 3. Multilingual coverage (larger vocab = more room for non-English tokens)
# 4. Rare token quality (larger vocab = more rare tokens with fewer training examples)

# Examples:
# GPT-2:       50,257 tokens (English-centric)
# LLaMA-1/2:   32,000 tokens (SentencePiece, multilingual but English-heavy)
# LLaMA-3:    128,256 tokens (tiktoken, much better multilingual)
# Qwen-2.5:  152,064 tokens (strong CJK coverage)
# Gemma-2:   256,000 tokens (very large, best multilingual coverage)

# Impact on sequence length (tokenizing the same multilingual text):
# English: "Hello, how are you?"
#   GPT-2: 6 tokens, LLaMA-3: 6 tokens, Qwen: 6 tokens (similar)
# Chinese: "How are you" in Chinese
#   LLaMA-2 (32K vocab): ~8 tokens (each character split into bytes)
#   LLaMA-3 (128K vocab): ~4 tokens (common characters are single tokens)
#   Qwen (152K vocab): ~3 tokens (optimized for Chinese)
#
# Larger vocab = 2-3x fewer tokens for non-English text = 2-3x cheaper inference</code></pre>

<h4>5. Multilingual Tokenization Challenges</h4>
<p>Tokenization is inherently biased toward the language distribution of the training corpus. English-centric tokenizers are dramatically inefficient for other languages:</p>

<ul>
<li><strong>Fertility rate:</strong> The number of tokens per word. English: ~1.3 tokens/word. Japanese: ~2.5 tokens/word with LLaMA-2 tokenizer. Hindi: ~3.5 tokens/word. Burmese: ~8 tokens/word. This means Burmese users pay ~6x more per word in API costs.</li>
<li><strong>The "byte fallback" problem:</strong> When a character isn't in the vocabulary, it's split into UTF-8 bytes. A single CJK character is 3 bytes, a single Hindi character can be 3-6 bytes. This wastes context window and increases cost.</li>
<li><strong>Mitigation:</strong> Use larger vocabularies (128K+) with intentional multilingual coverage. Train the tokenizer on a data mix that proportionally represents target languages. LLaMA-3 (128K vocab) is ~2x more token-efficient for Chinese compared to LLaMA-2 (32K vocab).</li>
</ul>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the BPE tokenization algorithm. How does it handle unseen words?</div>
<div class="a-text">BPE starts with a base vocabulary of individual characters (or bytes). It then counts all adjacent pairs in the training corpus and merges the most frequent pair into a new token. This process repeats until the vocabulary reaches the desired size. The result is a list of merge rules applied in order. To tokenize a new word: (1) split it into characters, (2) apply merge rules in order, greedily combining the highest-priority pairs. For unseen words, BPE naturally handles them by breaking them into known subword pieces. For example, "unhappiness" might become ["un", "happiness"] or ["un", "happ", "iness"] depending on the learned merges. If a byte-level BPE is used (like tiktoken), there are truly no unknown tokens: any byte sequence can be represented. This is why modern LLMs never produce "UNK" tokens.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why does vocabulary size matter, and how would you choose it for a new model?</div>
<div class="a-text">Vocabulary size involves several tradeoffs: (1) <strong>Sequence compression:</strong> larger vocab = fewer tokens per text = shorter sequences = less compute per sample. Going from 32K to 128K vocab typically reduces token count by 10-30% for multilingual text. (2) <strong>Embedding parameters:</strong> the embedding table is vocab_size * d_model. At 128K vocab and 4096 dim, that's 524M parameters just for embeddings (~7% of a 7B model). (3) <strong>Rare token quality:</strong> with a larger vocab, rare tokens have fewer training examples each, potentially degrading quality for rare subwords. (4) <strong>Output softmax cost:</strong> computing logits over 128K tokens is 4x more expensive than over 32K. For a new model: if English-only, 32-50K is sufficient. For multilingual, 128K+ is strongly recommended. For heavily code-focused models, ensure the vocab includes common code tokens (indentation, operators). Always validate by measuring fertility rate across your target languages.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 12.4 Pretraining
    // ----------------------------------------------------------
    {
      id: "pretraining",
      title: "Pretraining: Data, Deduplication, Quality Filtering, and the Loss Curve",
      content: `
<p>Pretraining is where a language model learns the vast majority of its knowledge and capabilities. It's also the most expensive phase, costing millions of dollars for frontier models. The quality and composition of pretraining data is arguably more important than the architecture itself &mdash; all modern architectures are similar, but data pipelines differ dramatically.</p>

<h4>1. Data Sources</h4>

<table>
<tr><th>Dataset</th><th>Size</th><th>Content</th><th>Quality</th><th>Used By</th></tr>
<tr><td>Common Crawl</td><td>~250B pages (petabytes raw)</td><td>Web scrape (everything)</td><td>Very low (needs heavy filtering)</td><td>All models (after filtering)</td></tr>
<tr><td>C4 (Colossal Clean Crawled Corpus)</td><td>~750GB / ~156B tokens</td><td>Filtered Common Crawl</td><td>Medium</td><td>T5, early models</td></tr>
<tr><td>The Pile</td><td>~825GB / ~300B tokens</td><td>22 diverse sources</td><td>High (curated)</td><td>GPT-NeoX, Pythia</td></tr>
<tr><td>RedPajama-v1</td><td>~1.2T tokens</td><td>Reproduces LLaMA training data</td><td>Medium-High</td><td>RedPajama models, research</td></tr>
<tr><td>RedPajama-v2</td><td>~30T tokens (raw)</td><td>84 CommonCrawl dumps</td><td>Raw (needs filtering)</td><td>Research</td></tr>
<tr><td>FineWeb</td><td>~15T tokens</td><td>Filtered Common Crawl</td><td>High (HuggingFace quality filtering)</td><td>Open source community</td></tr>
<tr><td>StarCoder data</td><td>~800GB / ~250B tokens</td><td>GitHub code (80+ languages)</td><td>High (license-filtered)</td><td>Code models</td></tr>
<tr><td>Books (various)</td><td>~100B+ tokens</td><td>Published books</td><td>Very high</td><td>Most models (legal gray area)</td></tr>
<tr><td>Wikipedia</td><td>~4B tokens (English)</td><td>Encyclopedia</td><td>Very high</td><td>All models</td></tr>
<tr><td>arXiv</td><td>~30B tokens</td><td>Scientific papers (LaTeX)</td><td>Very high</td><td>Most models</td></tr>
</table>

<h4>2. Data Processing Pipeline</h4>

<pre><code># A typical pretraining data pipeline:

# Stage 1: Collection
# - Download Common Crawl WARC files (web archive format)
# - Extract text from HTML (trafilatura or jusText for main content extraction)
# - Deduplicate URLs

# Stage 2: Language Identification
# - Use fastText language ID model
# - Filter to desired languages (e.g., >95% confidence English)
# - For multilingual models: keep proportional mix

# Stage 3: Quality Filtering
# - Heuristic filters:
quality_filters = {
    "min_words": 50,                  # Remove very short documents
    "max_words": 100000,              # Remove extremely long documents
    "min_avg_word_length": 3,         # Filter non-language content
    "max_avg_word_length": 10,        # Filter base64/encoded data
    "max_symbol_to_word_ratio": 0.1,  # Filter symbol-heavy content
    "max_bullet_ratio": 0.9,          # Filter lists/menus
    "min_alpha_ratio": 0.7,           # Ensure mostly alphabetic text
    "max_duplicate_line_ratio": 0.3,  # Filter repeated content
    "contains_stop_words": True,      # Verify natural language
}

# - Model-based filtering (used by Phi, LLaMA-3):
#   Train a classifier to predict "educational value" or "quality"
#   using Wikipedia/textbook text as positive examples
#   and random web text as negative examples
#   Filter to top 10-30% by quality score

# Stage 4: Deduplication (critical!)
# - Exact dedup: SHA-256 hash of normalized text
# - Near-dedup: MinHash + LSH (Locality-Sensitive Hashing)
#   Detects ~80%+ similar documents
# - Paragraph-level dedup: Remove repeated paragraphs across documents

# Stage 5: PII and Safety Filtering
# - Remove documents with emails, phone numbers, SSNs
# - Filter toxic/harmful content using classifier
# - Remove known copyrighted test sets (benchmark contamination)

# Stage 6: Tokenization and Packing
# - Tokenize all documents
# - Concatenate into long sequences, separated by EOS tokens
# - Pack into fixed-length chunks for efficient batching</code></pre>

<h4>3. Deduplication: Why It Matters</h4>
<p>Deduplication is one of the most impactful preprocessing steps. Lee et al. (2022, arXiv:2107.06499) showed that training on deduplicated data:</p>
<ul>
<li>Reduces memorization (the model is less likely to regurgitate training data verbatim)</li>
<li>Improves perplexity by ~0.5-1.0 points (equivalent to 2-3x more unique data)</li>
<li>Reduces training cost (fewer redundant samples to process)</li>
<li>Improves downstream task performance</li>
</ul>

<pre><code># MinHash deduplication (approximate near-duplicate detection):
#
# 1. For each document, extract all n-grams (e.g., 5-grams)
# 2. Hash each n-gram with k different hash functions
# 3. For each hash function, keep the minimum hash value
# 4. The k minimum hashes form the "MinHash signature"
# 5. Two documents are near-duplicates if their signatures share
#    more than a threshold fraction of values (e.g., >80% overlap)
# 6. Use LSH (banding technique) to efficiently find candidate pairs

# Typical settings:
# - n-gram size: 5 (words) or 13 (characters)
# - Number of hashes: 128
# - Jaccard similarity threshold: 0.8
# - This catches ~95% of near-duplicates in web crawl data</code></pre>

<h4>4. Data Mixing Ratios</h4>
<p>The proportion of different data sources significantly affects model capabilities:</p>

<pre><code># Approximate data mix for a general-purpose LLM (LLaMA-style):
data_mix = {
    "web_text_filtered": 0.67,    # 67% - general knowledge, language patterns
    "code":              0.08,    # 8%  - coding ability, logical reasoning
    "books":             0.05,    # 5%  - long-form coherence, knowledge depth
    "academic_papers":   0.04,    # 4%  - scientific knowledge, formal reasoning
    "wikipedia":         0.04,    # 4%  - factual accuracy, structured knowledge
    "math":              0.04,    # 4%  - mathematical reasoning
    "conversations":     0.04,    # 4%  - dialogue, instruction following
    "multilingual":      0.04,    # 4%  - non-English languages
}
# Note: these are sampled proportions, not data proportions.
# High-quality data (Wikipedia, books) is often upsampled 2-5x
# relative to its actual share of the total data.

# The Pile's composition (EleutherAI, for reference):
# 40% Common Crawl, 16% PubMed Central, 10% Books3,
# 8% OpenWebText2, 7% ArXiv, 5% GitHub,
# 4% Wikipedia, 10% other (DM Mathematics, USPTO, Gutenberg, etc.)</code></pre>

<h4>5. Curriculum Learning</h4>
<p>Some training runs use curriculum learning: starting with easier/shorter/higher-quality data and gradually introducing more challenging or lower-quality data. The rationale is that early training establishes core language understanding, while later training adds breadth:</p>

<ul>
<li><strong>Quality curriculum:</strong> Start with only the highest-quality data (Wikipedia, textbooks, curated web), then gradually mix in lower-quality web text. Used by Phi models.</li>
<li><strong>Length curriculum:</strong> Start with shorter sequences and gradually increase to full context length. This improves early training efficiency.</li>
<li><strong>Domain curriculum:</strong> Start with general text, then increase the proportion of specialized domains (code, math) toward the end. This emphasizes capabilities you want the final model to be strong in.</li>
</ul>

<h4>6. The Loss Curve and When to Stop</h4>
<p>During pretraining, the loss (cross-entropy, measuring how well the model predicts the next token) follows a characteristic pattern:</p>

<pre><code># Typical loss curve for a 7B model:
# Steps 0-100:       Loss drops rapidly from ~11.0 to ~4.0 (learning basic patterns)
# Steps 100-1000:    Loss drops from ~4.0 to ~2.5 (learning grammar, common phrases)
# Steps 1000-10000:  Loss drops from ~2.5 to ~2.0 (learning facts, reasoning patterns)
# Steps 10000-100000: Loss drops from ~2.0 to ~1.7 (diminishing returns, fine details)
# Steps 100000+:     Loss asymptotes around ~1.6-1.7 (near-optimal for model size)

# When to stop training:
# 1. Chinchilla-optimal: stop at ~20 tokens per parameter
# 2. Inference-optimal: keep training until loss stops improving (100-200 tokens/param)
# 3. Practical: monitor downstream eval metrics; stop when they plateau
# 4. Budget-constrained: stop when you run out of compute

# Warning signs during training:
# - Loss spikes (> 2x normal): data issue or learning rate too high
# - Loss plateau too early: learning rate too low or data quality issue
# - Eval metrics diverge from train loss: overfitting or eval contamination</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: The Deduplication Disaster</div>
<p>A startup training a 13B model on 2T tokens discovered at month 3 (after $400K in compute) that their data pipeline had a bug: the deduplication step was only checking exact matches, not near-duplicates. As a result, 30% of their "2T tokens" were minor variants of the same documents (different HTML rendering of the same content, mirror sites, syndicated articles). The model had effectively seen only 1.4T unique tokens. When they fixed the dedup pipeline and retrained, the model's downstream performance improved by 2-3% across all benchmarks &mdash; equivalent to doubling the training compute. The lesson: invest heavily in data deduplication. A single week of engineering on MinHash dedup is worth months of training compute.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Walk me through how you would build a pretraining data pipeline from scratch.</div>
<div class="a-text">My pipeline would have six stages: (1) <strong>Collection:</strong> Download Common Crawl WARC files and extract text using trafilatura for content extraction. Separately collect high-quality sources (Wikipedia dumps, arXiv LaTeX, GitHub code with permissive licenses). (2) <strong>Language ID:</strong> Run fastText language identification, keeping documents above 95% confidence for target languages. (3) <strong>Quality filtering:</strong> Apply heuristic filters (min/max length, alphabet ratio, stop word presence), then train a quality classifier on Wikipedia vs. random web and keep top 20%. (4) <strong>Deduplication:</strong> Exact dedup via document-level SHA-256, then near-dedup via MinHash with 128 hashes and 0.8 Jaccard threshold using LSH for efficiency. This typically removes 30-50% of web data. (5) <strong>Safety:</strong> PII removal (regex for emails, phone numbers, SSNs), toxicity classification, and removal of known benchmark test sets. (6) <strong>Tokenization and packing:</strong> Tokenize with the model's tokenizer, concatenate documents with EOS separators, pack into fixed-length training sequences.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 12.5 Fine-Tuning Methods
    // ----------------------------------------------------------
    {
      id: "finetuning",
      title: "Fine-Tuning: LoRA, QLoRA, DoRA, and Adapter Methods",
      content: `
<p>Fine-tuning adapts a pretrained LLM to a specific task or behavior. Full fine-tuning (updating all parameters) is expensive and requires multiple GPUs; parameter-efficient fine-tuning (PEFT) methods update only a small fraction of parameters while achieving comparable results. Understanding these methods &mdash; their math, tradeoffs, and when to use each &mdash; is essential for any AI engineer working with LLMs.</p>

<h4>1. Full Fine-Tuning</h4>
<p>Full fine-tuning updates all model parameters on the new task/data:</p>

<pre><code># Full fine-tuning:
# - Update all parameters: ~7B for a 7B model
# - Memory: model (14GB fp16) + optimizer states (56GB fp32) + gradients (14GB) = ~84GB
# - Requires 2-4 A100-80GB GPUs minimum for a 7B model
# - Risk of catastrophic forgetting (model loses pretraining knowledge)
# - Produces the best results when you have enough data (>100K examples)
# - Creates a completely new model checkpoint (full 7B parameters to store)

# When to use full fine-tuning:
# - Large training dataset (>100K examples)
# - Task is very different from pretraining (e.g., domain-specific language)
# - You need maximum quality and have sufficient compute
# - You plan to further fine-tune or merge the model</code></pre>

<h4>2. LoRA: Low-Rank Adaptation</h4>
<p>LoRA (Hu et al., 2021, arXiv:2106.09685) is the most important PEFT method. It freezes the pretrained weights and injects trainable low-rank decomposition matrices into each layer:</p>

<pre><code># LoRA Math:
# Original forward pass for a linear layer:
#   h = W * x        where W in R^{d_out x d_in}
#
# LoRA modifies this to:
#   h = W * x + (B * A) * x
#   where:
#     W: frozen pretrained weights (d_out x d_in)
#     A: trainable down-projection (r x d_in), initialized from Normal(0, std)
#     B: trainable up-projection (d_out x r), initialized to zeros
#     r: rank (typically 8, 16, 32, 64) << min(d_in, d_out)
#
# At initialization: B * A = 0 (zero matrix), so the model starts
# exactly as the pretrained model. Training adjusts A and B.
#
# The key insight: the weight update delta_W = B * A is a RANK-r MATRIX.
# Neural network weight updates during fine-tuning are empirically low-rank,
# so this is a good approximation.

# Parameter count:
# Original W: d_out * d_in (e.g., 4096 * 4096 = 16.8M)
# LoRA A + B: r * d_in + d_out * r = r * (d_in + d_out)
# For r=16: 16 * (4096 + 4096) = 131K (128x fewer parameters!)
#
# For a 7B model with LoRA on all attention layers:
# 32 layers * 4 attention matrices * 131K = 16.8M trainable params
# That is 0.24% of the total model parameters

# Scaling factor alpha:
# In practice, the LoRA output is scaled: h = W*x + (alpha/r) * B * A * x
# alpha is a hyperparameter (typically equal to r, so the factor is 1.0)
# Higher alpha = larger LoRA contribution = faster adaptation but more instability</code></pre>

<pre><code>import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, original_linear, rank=16, alpha=16):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)  # Freeze original weights

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)  # Down-project
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))          # Up-project (init to 0!)

        self.scaling = alpha / rank

    def forward(self, x):
        # Original path (frozen)
        original_output = self.original(x)

        # LoRA path (trainable)
        # x: (..., d_in) -> A: (rank, d_in) -> (..., rank) -> B: (d_out, rank) -> (..., d_out)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return original_output + lora_output

    def merge_weights(self):
        """Merge LoRA weights into original for inference (no extra latency)."""
        self.original.weight.data += (self.lora_B @ self.lora_A * self.scaling)
        return self.original

# After training, you can merge LoRA into the base model:
# merged_linear = lora_layer.merge_weights()
# The merged model has ZERO inference overhead compared to the original!</code></pre>

<h4>3. QLoRA: Quantized LoRA</h4>
<p>QLoRA (Dettmers et al., 2023, arXiv:2305.14314) enables fine-tuning of large models on a single GPU by quantizing the base model to 4-bit while training LoRA adapters in float16:</p>

<pre><code># QLoRA components:
# 1. NF4 (NormalFloat4): 4-bit quantization format optimized for normally-distributed weights
#    - Each weight is quantized to one of 16 values (4 bits)
#    - The 16 quantization levels are spaced according to a normal distribution
#    - This preserves model quality better than uniform 4-bit quantization
#
# 2. Double Quantization: the quantization constants are themselves quantized
#    - Standard: one FP32 scale factor per 64 weights (0.5 bits/weight overhead)
#    - Double: quantize the scale factors to FP8 (0.125 bits/weight overhead)
#
# 3. Paged Optimizers: uses CPU RAM for optimizer states that don't fit in GPU memory
#    - Automatically pages optimizer states between GPU and CPU
#    - Enables training on GPUs with limited memory

# Memory comparison for fine-tuning LLaMA-2-70B:
# Full fine-tuning (fp16):   70B * 2 bytes + 70B * 8 bytes (Adam) = 700 GB -> 10+ A100s
# LoRA (fp16 base):          70B * 2 bytes + 160M * 2 bytes = 140.3 GB -> 2-4 A100s
# QLoRA (4-bit base):        70B * 0.5 bytes + 160M * 2 bytes = 35.3 GB -> 1 A100 48GB!
#
# QLoRA makes 70B model fine-tuning possible on a single A100 or even an RTX 4090 (24GB)</code></pre>

<h4>4. DoRA: Weight-Decomposed Low-Rank Adaptation</h4>
<p>DoRA (Liu et al., 2024, arXiv:2402.09353) decomposes the pretrained weight matrix into magnitude and direction components, then applies LoRA only to the direction:</p>

<pre><code># DoRA decomposition:
# W = m * (W / ||W||_c)
# where m = ||W||_c (column-wise magnitude) and W/||W||_c (direction)
#
# DoRA applies LoRA to the direction only:
# W' = m' * ((W + B*A) / ||W + B*A||_c)
# where m' is also trainable
#
# Intuition: fine-tuning often needs to change the "direction" of
# weight vectors more than their magnitude. By decomposing,
# DoRA can more efficiently learn directional changes.
#
# DoRA consistently outperforms LoRA by 0.5-1% on benchmarks
# with negligible additional parameters (just the magnitude vector m')</code></pre>

<h4>5. Other Adapter Methods</h4>

<table>
<tr><th>Method</th><th>Where Inserted</th><th>Trainable Params</th><th>Quality vs LoRA</th><th>Inference Overhead</th></tr>
<tr><td>LoRA</td><td>Attention Q, K, V, O matrices</td><td>~0.1-1% of model</td><td>Baseline</td><td>Zero (after merge)</td></tr>
<tr><td>QLoRA</td><td>Same as LoRA, 4-bit base</td><td>Same as LoRA</td><td>~Equal</td><td>Quantization overhead</td></tr>
<tr><td>DoRA</td><td>Same as LoRA + magnitude</td><td>~0.1-1% + magnitude</td><td>+0.5-1%</td><td>Small overhead</td></tr>
<tr><td>Adapter layers</td><td>After attention and FFN</td><td>~1-3% of model</td><td>Comparable</td><td>Extra forward pass through adapter</td></tr>
<tr><td>Prefix tuning</td><td>Prepended to KV at each layer</td><td>~0.1-1%</td><td>Slightly worse</td><td>Longer effective sequence</td></tr>
<tr><td>IA3</td><td>Scales K, V, FFN activations</td><td>~0.01% of model</td><td>Slightly worse</td><td>Negligible</td></tr>
</table>

<h4>6. When to Use Which Method</h4>

<pre><code># Decision framework:
#
# Q: How much data do you have?
# - < 1K examples  -> LoRA rank 8, just attention Q/V
# - 1K-10K         -> LoRA rank 16-32, all attention matrices
# - 10K-100K       -> LoRA rank 64 or full fine-tuning if compute available
# - > 100K         -> Full fine-tuning gives best results
#
# Q: What's your GPU budget?
# - 1x 24GB GPU   -> QLoRA (4-bit) for models up to 33B
# - 1x 80GB A100  -> QLoRA for 70B or LoRA for 13B
# - 4x 80GB A100  -> LoRA for 70B or full fine-tuning for 7B
# - 8x 80GB H100  -> Full fine-tuning for 70B
#
# Q: How different is your task from the base model?
# - Same domain, style adjustment -> LoRA rank 8, few epochs
# - Moderate domain shift          -> LoRA rank 16-32, more epochs
# - Very different domain          -> Full fine-tuning or LoRA rank 64+
# - New language/modality          -> Full fine-tuning strongly preferred</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain LoRA. What is the mathematical intuition behind low-rank adaptation?</div>
<div class="a-text">LoRA adds a trainable low-rank update to frozen pretrained weights: instead of learning a full weight update delta_W (d_out x d_in), LoRA decomposes it as delta_W = B * A, where B is (d_out x r) and A is (r x d_in), with r << min(d_out, d_in). The mathematical intuition: Aghajanyan et al. (2020) showed that pretrained model weight updates during fine-tuning have low intrinsic rank &mdash; the effective dimensionality of the update is much smaller than the full parameter space. LoRA exploits this by constraining updates to a low-rank subspace. With rank r=16 on a 4096x4096 matrix, you train 131K parameters instead of 16.8M (128x reduction) while capturing ~95%+ of the fine-tuning quality. B is initialized to zero so the model starts exactly at the pretrained weights. After training, LoRA can be merged into the base weights (W_new = W + B*A), adding zero inference latency.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How does QLoRA enable fine-tuning a 70B model on a single GPU? What are the tradeoffs?</div>
<div class="a-text">QLoRA combines three techniques: (1) <strong>NF4 quantization:</strong> The base model weights are quantized to 4-bit NormalFloat format, reducing the 70B model from 140GB (fp16) to ~35GB. The 16 quantization levels are optimized for normally-distributed weights, preserving quality. (2) <strong>Double quantization:</strong> The quantization scale factors are themselves quantized from fp32 to fp8, saving additional memory (~0.4 bits per parameter). (3) <strong>Paged optimizers:</strong> Optimizer states (Adam momentum, variance) are paged between GPU and CPU memory. Together, a 70B model fits in ~35GB GPU memory, enabling training on a single A100-80GB. The LoRA adapters (only ~160M parameters) are trained in fp16/bf16 on top of the quantized base. Tradeoffs: (a) Training is 20-30% slower due to quantization/dequantization overhead. (b) There may be a small quality gap vs. fp16 LoRA (typically <0.5% on benchmarks). (c) The base model quality depends on quantization quality &mdash; some tasks that depend on precise weight values may be more affected.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 12.6 Context Window Extension
    // ----------------------------------------------------------
    {
      id: "context-extension",
      title: "Context Window Extension: RoPE Scaling, YaRN, and Beyond",
      content: `
<p>The context window &mdash; the maximum number of tokens a model can process at once &mdash; is a critical limitation of transformer LLMs. A model trained with 4K context cannot natively handle an 8K input. Extending the context window after training has become a crucial capability, enabling LLMs to process long documents, codebases, and conversations. This section covers the key techniques.</p>

<h4>1. Why Context Extension Is Hard</h4>
<p>The challenge is positional encoding. RoPE (used by most modern LLMs) rotates Q/K vectors by position-dependent angles. During training, the model only sees rotation angles corresponding to positions 0 through max_seq_len. At position max_seq_len + 1, the rotation angle is out-of-distribution &mdash; the model has never learned to interpret it.</p>

<pre><code># RoPE rotation angles for position m:
# theta_i(m) = m * base^{-2i/d} for dimension pair i
#
# For a model trained with max_seq_len=4096 and base=10000:
# The model has seen theta_i values in [0, 4096 * base^{-2i/d}]
# At position 8192, theta_i = 8192 * base^{-2i/d} -> outside training range!
#
# For low-frequency dimensions (large i): the angle barely changes (< 2*pi total)
# For high-frequency dimensions (small i): the angle wraps around many times
# The challenge: high-frequency dimensions extrapolate well (cyclic),
# but low-frequency dimensions see never-before-seen angles</code></pre>

<h4>2. Position Interpolation (PI)</h4>
<p>Chen et al. (2023, arXiv:2306.15595) proposed the simplest approach: scale all positions down to fit within the original training range:</p>

<pre><code># Position Interpolation:
# Instead of using position m directly, use m * (L_train / L_target)
# For extending 4K to 32K: position m becomes m * (4096 / 32768) = m * 0.125
#
# Position 32768 -> mapped to position 4096 (in the original training range)
# Position 16384 -> mapped to position 2048
#
# Advantage: all positions are within the trained range
# Disadvantage: positions that were far apart are now close together,
#   potentially losing local resolution. Nearby tokens (positions 0 and 1)
#   now have very similar positional encodings (0 and 0.125).
#
# Requires continued pretraining (1-2B tokens) to work well.
# Performance: solid results up to 8x extension with continued training</code></pre>

<h4>3. NTK-Aware RoPE Scaling</h4>
<p>NTK-aware scaling (Reddit user "bloc97", 2023) modifies the RoPE base frequency instead of scaling positions:</p>

<pre><code># NTK-aware scaling:
# Instead of modifying positions, modify the base:
# new_base = base * (L_target / L_train)^{d/(d-2)}
#
# For extending 4K to 32K with d=128:
# scale = 32768 / 4096 = 8
# new_base = 10000 * 8^{128/126} = 10000 * 8.13 = 81,300
#
# This spreads the rotation frequencies, giving each position a more
# unique encoding at longer distances without compressing nearby positions.
#
# The "NTK" name comes from the Neural Tangent Kernel theory:
# the key insight is that high-frequency components (which already
# wrap around many times) don't need modification, while low-frequency
# components need stretching. NTK-aware scaling naturally handles this.
#
# Advantage: works reasonably well without any fine-tuning!
# Disadvantage: lower quality than methods that use continued training</code></pre>

<h4>4. YaRN: Yet Another RoPE ExtensioN</h4>
<p>YaRN (Peng et al., 2023, arXiv:2309.00071) combines the best of both approaches and adds a temperature factor:</p>

<pre><code># YaRN combines three techniques:
#
# 1. NTK-aware interpolation for HIGH-frequency dimensions
#    (these are already periodic, just need slight stretching)
#
# 2. Linear interpolation for LOW-frequency dimensions
#    (these have never completed a full rotation, need interpolation)
#
# 3. Attention temperature scaling: multiply attention scores by
#    t = 0.1 * ln(s) + 1, where s is the extension factor
#    This accounts for the increased entropy in attention weights
#    when the context is longer (more tokens to attend to)
#
# YaRN assigns each dimension pair to "high" or "low" frequency
# based on its wavelength relative to the original context length.
# Dimensions with wavelength < L_train are "high frequency" (NTK scaling).
# Dimensions with wavelength > L_train are "low frequency" (PI).
# A smooth interpolation connects the two regimes.
#
# Results: YaRN extends LLaMA-2 from 4K to 128K with only ~400M tokens
# of continued pretraining (vs. trillions for training from scratch).
# Used by many open-source models including Qwen-2.5 and DeepSeek-V2.</code></pre>

<h4>5. Sliding Window and StreamingLLM</h4>

<pre><code># Sliding Window (Mistral):
# Only attend to the nearest W tokens at each layer.
# Information propagates through layers: effective context = L * W
# For 32 layers * 4096 window = 131K effective context
# Trade-off: not ALL information propagates; details from the distant past
# may be lost. Works well for tasks that need recent context.

# StreamingLLM (Xiao et al., 2023, arXiv:2309.17453):
# Observation: attention "sinks" exist - the first few tokens always receive
# high attention regardless of content. These are "attention sinks."
#
# StreamingLLM keeps:
# 1. The first 4 tokens (attention sinks)
# 2. The most recent W tokens (sliding window)
# Discards everything in between!
#
# This enables infinite-length generation with fixed memory.
# The quality is surprisingly good for many tasks because:
# - Recent context has the most relevant information
# - Attention sinks maintain numerical stability
# - The model learns to accumulate information in hidden states</code></pre>

<h4>6. Infinite Context via Retrieval (RAG)</h4>
<p>For truly unbounded context, retrieval-augmented generation (RAG) is the practical solution:</p>

<pre><code># RAG for context extension:
# Instead of stuffing 1M tokens into the model's context window:
# 1. Chunk the documents into ~512-token segments
# 2. Embed each chunk with an embedding model
# 3. When the user asks a question, embed the query
# 4. Retrieve the top-k most relevant chunks (cosine similarity)
# 5. Insert the retrieved chunks into the model's context
#
# Advantages:
# - Handles unlimited document sizes
# - Lower cost (only process relevant chunks)
# - Can be updated without retraining (add new documents to index)
#
# Disadvantages:
# - Retrieval quality limits overall quality
# - Cannot perform tasks that require cross-document reasoning
# - Adds latency (embedding + retrieval + generation)
# - "Lost in the middle" problem: models struggle to use info in the
#   middle of long contexts (Liu et al., 2023)</code></pre>

<h4>7. Practical Limits and Comparison</h4>

<table>
<tr><th>Method</th><th>Max Extension</th><th>Fine-tuning Required</th><th>Quality at Max</th><th>Compute Cost</th></tr>
<tr><td>Position Interpolation</td><td>~8x</td><td>Yes (1-2B tokens)</td><td>Good</td><td>Low-medium</td></tr>
<tr><td>NTK-aware scaling</td><td>~4-8x</td><td>No (works zero-shot)</td><td>Moderate</td><td>Zero</td></tr>
<tr><td>YaRN</td><td>~32x</td><td>Yes (400M tokens)</td><td>Very good</td><td>Low</td></tr>
<tr><td>Sliding Window</td><td>Unlimited (with info loss)</td><td>Trained natively</td><td>Good (recent context)</td><td>Zero</td></tr>
<tr><td>StreamingLLM</td><td>Unlimited</td><td>No</td><td>Good (recent + sinks)</td><td>Zero</td></tr>
<tr><td>RAG</td><td>Unlimited</td><td>No</td><td>Depends on retrieval</td><td>Retrieval infrastructure</td></tr>
<tr><td>Native long context</td><td>128K-1M (Gemini)</td><td>Trained natively</td><td>Best</td><td>Very high (training)</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How do you extend a model's context window from 4K to 128K? What are the options and tradeoffs?</div>
<div class="a-text">Five main approaches: (1) <strong>YaRN:</strong> Best quality-to-cost ratio. Modifies RoPE frequencies with dimension-dependent interpolation + attention scaling. Requires ~400M tokens of continued pretraining. Can extend 32x (4K to 128K) with minimal quality loss. (2) <strong>Position Interpolation:</strong> Simpler, scales positions linearly. Works up to ~8x extension with ~2B tokens of continued training. (3) <strong>NTK-aware scaling:</strong> Modify RoPE base frequency. Works without any fine-tuning but quality is lower for large extensions. Good for quick experiments. (4) <strong>Native long-context training:</strong> Train from scratch or continue pretraining with long sequences. Best quality but most expensive. Used by Gemini (1M+), LLaMA-3.1 (128K). (5) <strong>RAG:</strong> Don't extend the window; retrieve relevant passages instead. Best for document QA but can't handle tasks requiring full-document reasoning. For production, I'd use YaRN if I need the model to reason over the full context, or RAG if the task is information retrieval. Most real applications combine both: extend to 32K with YaRN and use RAG for the document collection beyond 32K.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 12.7 LLM Evaluation
    // ----------------------------------------------------------
    {
      id: "llm-evaluation",
      title: "LLM Evaluation: Benchmarks, Arenas, and Custom Evals",
      content: `
<p>Evaluating LLMs is one of the hardest problems in AI engineering. Unlike traditional ML where you have clear metrics (accuracy, F1, AUC), LLMs perform open-ended generation across thousands of possible tasks. No single benchmark captures "how good" a model is. This section covers the major evaluation frameworks, their limitations, and how to build your own evaluation pipeline.</p>

<h4>1. Standard Benchmarks</h4>

<table>
<tr><th>Benchmark</th><th>What It Tests</th><th>Format</th><th>Limitations</th></tr>
<tr><td><strong>MMLU</strong> (Hendrycks et al., 2020)</td><td>World knowledge across 57 subjects</td><td>4-choice multiple choice</td><td>Saturating, contamination risk, narrow format</td></tr>
<tr><td><strong>HumanEval</strong> (Chen et al., 2021)</td><td>Code generation (Python)</td><td>Function completion, test execution</td><td>Only 164 problems, Python-only, simple problems</td></tr>
<tr><td><strong>MBPP</strong> (Austin et al., 2021)</td><td>Code generation (Python, basic)</td><td>974 simple problems</td><td>Too easy for modern models (>80%)</td></tr>
<tr><td><strong>GSM8K</strong> (Cobbe et al., 2021)</td><td>Grade school math reasoning</td><td>Word problems with numerical answers</td><td>Simple math, many models >90%</td></tr>
<tr><td><strong>MATH</strong> (Hendrycks et al., 2021)</td><td>Competition math</td><td>Hard math with LaTeX answers</td><td>Still challenging, but improving rapidly</td></tr>
<tr><td><strong>ARC</strong> (Clark et al., 2018)</td><td>Science reasoning (grade school)</td><td>4-choice multiple choice</td><td>Relatively easy, saturating</td></tr>
<tr><td><strong>HellaSwag</strong> (Zellers et al., 2019)</td><td>Commonsense reasoning</td><td>Sentence completion</td><td>Largely saturated (>95%)</td></tr>
<tr><td><strong>WinoGrande</strong> (Sakaguchi et al., 2019)</td><td>Commonsense (pronoun resolution)</td><td>Binary choice</td><td>Saturating</td></tr>
<tr><td><strong>TruthfulQA</strong> (Lin et al., 2022)</td><td>Factual accuracy (tricky questions)</td><td>Multiple choice + generation</td><td>817 questions, narrow scope</td></tr>
</table>

<h4>2. Human Evaluation and Arena-Style Benchmarks</h4>

<pre><code># The gold standard for LLM evaluation is human preference.
# Two main approaches:

# 1. MT-Bench (Zheng et al., 2023):
# - 80 multi-turn questions across 8 categories
# - Categories: writing, roleplay, extraction, reasoning, math, coding, STEM, humanities
# - Scored by GPT-4 as judge (1-10 scale)
# - Cheap, fast, reproducible
# - Limitation: GPT-4 judge has known biases (prefers verbose answers, its own style)

# 2. LMSYS Chatbot Arena (Zheng et al., 2023):
# - Live human evaluation platform
# - Users chat with two anonymous models side-by-side
# - User votes for the better response
# - Uses Elo rating system (like chess)
# - Currently the most trusted LLM ranking
# - Limitation: population bias (tech-savvy English-speaking users), slow to accumulate data
# - As of 2025: ~1.5M+ human votes collected
# - Top models: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro (Elo ~1280+)

# 3. AlpacaEval (Li et al., 2023):
# - 805 questions, LLM-as-judge evaluation
# - Reports "win rate" against reference model (GPT-4)
# - AlpacaEval 2.0 uses length-controlled win rate (penalizes verbosity)
# - Fast and cheap, but highly correlated with verbosity</code></pre>

<h4>3. The Benchmark Contamination Problem</h4>
<p>Benchmark contamination occurs when test set examples appear in the model's pretraining data. This inflates scores and makes comparisons unreliable:</p>

<pre><code># How contamination happens:
# 1. Benchmark questions are posted online (GitHub, forums, papers)
# 2. Web crawlers include them in Common Crawl
# 3. The model trains on Common Crawl
# 4. The model has "memorized" the answers, inflating its benchmark scores
#
# Detection methods:
# - N-gram overlap: check if long n-grams from test examples appear in training data
# - Canary strings: insert unique identifiers in test data, check if model completes them
# - Performance analysis: suspiciously high scores on exact benchmark wording
#   vs. paraphrased versions suggests memorization
# - Perplexity analysis: if a model has much lower perplexity on benchmark text
#   than on similar non-benchmark text, it may have seen the benchmarks
#
# The arms race:
# - GPQA Diamond (Rein et al., 2023): PhD-level science questions, hard to contaminate
# - LiveCodeBench: new coding problems posted after model training cutoffs
# - Chatbot Arena: live human evaluation, cannot be contaminated by definition
# - IFEval: instruction-following eval with verifiable constraints</code></pre>

<h4>4. Building Custom Evaluation Pipelines</h4>
<p>For production systems, standard benchmarks are insufficient. You need custom evals that measure what matters for your specific use case:</p>

<pre><code>import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EvalExample:
    """A single evaluation example."""
    input_prompt: str
    reference_answer: Optional[str] = None  # For answer-graded evals
    criteria: Optional[List[str]] = None    # For rubric-graded evals
    metadata: Optional[dict] = None         # Category, difficulty, etc.

@dataclass
class EvalResult:
    """Result of evaluating one example."""
    example: EvalExample
    model_output: str
    score: float           # 0-1 normalized score
    judge_reasoning: str   # Why this score was assigned
    latency_ms: float      # Time to generate response

class LLMEvaluator:
    """Custom evaluation pipeline for production LLM systems."""

    def __init__(self, model_under_test, judge_model, eval_set: List[EvalExample]):
        self.model = model_under_test
        self.judge = judge_model
        self.eval_set = eval_set

    def evaluate_single(self, example: EvalExample) -> EvalResult:
        """Evaluate a single example using LLM-as-judge."""
        import time

        # Generate model response
        start = time.time()
        model_output = self.model.generate(example.input_prompt)
        latency = (time.time() - start) * 1000

        # Judge the response
        judge_prompt = f"""You are an expert evaluator. Rate the following response
on a scale of 1-10.

Question: {example.input_prompt}

{"Reference answer: " + example.reference_answer if example.reference_answer else ""}

{"Evaluation criteria: " + ", ".join(example.criteria) if example.criteria else ""}

Model response: {model_output}

Provide your rating and reasoning in JSON format:
{{"score": <1-10>, "reasoning": "<explanation>"}}"""

        judge_output = self.judge.generate(judge_prompt)
        result = json.loads(judge_output)

        return EvalResult(
            example=example,
            model_output=model_output,
            score=result['score'] / 10.0,  # Normalize to 0-1
            judge_reasoning=result['reasoning'],
            latency_ms=latency
        )

    def run_full_eval(self) -> dict:
        """Run evaluation on all examples and compute aggregate metrics."""
        results = [self.evaluate_single(ex) for ex in self.eval_set]

        # Aggregate metrics
        scores = [r.score for r in results]
        latencies = [r.latency_ms for r in results]

        # Category-level breakdown
        category_scores = {}
        for r in results:
            cat = r.example.metadata.get('category', 'overall') if r.example.metadata else 'overall'
            category_scores.setdefault(cat, []).append(r.score)

        return {
            'overall_score': sum(scores) / len(scores),
            'score_std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
            'p50_latency_ms': sorted(latencies)[len(latencies)//2],
            'p95_latency_ms': sorted(latencies)[int(len(latencies)*0.95)],
            'category_scores': {
                cat: sum(s)/len(s) for cat, s in category_scores.items()
            },
            'num_examples': len(results),
            'detailed_results': results
        }

# Best practices for custom evals:
# 1. Minimum 100 examples per category (statistical significance)
# 2. Use 3+ judge models and average (reduce judge bias)
# 3. Include adversarial examples (edge cases your model should handle)
# 4. Version your eval set and never modify existing examples (only add)
# 5. Track scores over time (regression detection)
# 6. Include human evaluation for 10% of examples (calibrate judge)</code></pre>

<h4>5. Evaluation Anti-Patterns</h4>

<div class="callout warning">
<div class="callout-title">Common Evaluation Mistakes</div>
<p>Avoid these pitfalls that lead to misleading evaluation results:</p>
</div>

<ul>
<li><strong>Optimizing for benchmarks instead of users:</strong> A model that scores 90% on MMLU but gives unhelpful answers to real user questions is a failure. Always prioritize user-facing metrics.</li>
<li><strong>Using a single benchmark:</strong> No benchmark is comprehensive. Use a portfolio of benchmarks plus custom evals.</li>
<li><strong>Ignoring eval contamination:</strong> If your model scores suspiciously well on a benchmark, investigate whether it has seen the test data. Compare performance on the exact benchmark vs. paraphrased versions.</li>
<li><strong>Not measuring latency and cost:</strong> A model that scores 5% higher but costs 10x more or is 10x slower may not be the better choice for production.</li>
<li><strong>Vibes-based evaluation:</strong> "It feels better" is not a metric. Quantify everything, even if the metric is imperfect.</li>
</ul>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you evaluate an LLM for a customer support chatbot? What metrics would you use?</div>
<div class="a-text">I'd use a multi-level evaluation approach: (1) <strong>Automated metrics:</strong> Task completion rate (did the bot resolve the issue without escalation?), response relevance (semantic similarity to reference answers), factual accuracy (does the bot cite correct policies/procedures?), response latency P50/P95. (2) <strong>LLM-as-judge:</strong> Build 200+ evaluation examples from real customer queries, annotated with ideal responses. Use 3 judge models (GPT-4, Claude, Gemini) to rate responses on helpfulness (1-5), accuracy (1-5), tone (1-5), and safety (pass/fail). (3) <strong>Human evaluation:</strong> Weekly sampling of 50 conversations rated by human QA team on the same rubric. Use this to calibrate the LLM judges. (4) <strong>Production metrics:</strong> Customer satisfaction score (CSAT), escalation rate, average handle time, repeat contact rate (did the customer come back with the same issue?). The most important metric is escalation rate: it directly measures whether the model is successfully resolving issues. Track week-over-week trends and set alerts for regressions.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is benchmark contamination and how do you detect it?</div>
<div class="a-text">Benchmark contamination occurs when test set examples appear in a model's pretraining data, inflating benchmark scores without reflecting genuine capability. Detection methods: (1) <strong>N-gram overlap:</strong> Check if long n-grams (8+) from test examples appear in the training data. This is the most direct method but requires access to the training data. (2) <strong>Paraphrase comparison:</strong> Compare model performance on the exact benchmark wording vs. semantically equivalent paraphrased versions. If performance drops significantly on paraphrases, contamination is likely. (3) <strong>Perplexity analysis:</strong> Measure model perplexity on benchmark text vs. similar non-benchmark text. Abnormally low perplexity on benchmarks suggests memorization. (4) <strong>Canary strings:</strong> For new benchmarks, embed unique strings that would only appear if the benchmark leaked. (5) <strong>Temporal analysis:</strong> Use benchmarks created after the model's training cutoff date. Live benchmarks like Chatbot Arena and LiveCodeBench are immune to contamination by design.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 12.8 Practical LLM Guide
    // ----------------------------------------------------------
    {
      id: "llm-practical",
      title: "Practical LLM Guide: Choosing, Deploying, and Prompting Models",
      content: `
<p>Choosing and deploying an LLM for production is a complex decision with many tradeoffs. This section provides practical frameworks for model selection, cost analysis, the API vs self-hosted decision, and prompt engineering fundamentals.</p>

<h4>1. Model Selection Framework</h4>
<p>Start with your requirements, not with models:</p>

<pre><code># Step 1: Define your requirements
requirements = {
    "task_type": "customer_support_chat",  # What will the model do?
    "quality_threshold": "95% accuracy on domain QA",  # Minimum acceptable quality
    "latency_p95": "< 2 seconds",          # Maximum acceptable latency
    "throughput": "100 requests/second",     # Peak load
    "cost_per_request": "< $0.01",          # Budget constraint
    "context_needed": "8K tokens",           # Typical input length
    "languages": ["English", "Spanish"],     # Language requirements
    "privacy": "no data leaves our servers", # Data residency constraints
    "compliance": "SOC2, HIPAA"              # Regulatory requirements
}

# Step 2: Filter models based on hard constraints
# Privacy + compliance = must self-host (eliminates API-only models)
# Multilingual = need model with good Spanish (LLaMA-3 > Mistral for non-English)
# Cost < $0.01 = need efficient model (8B class, not 70B)
# Latency < 2s = need fast inference (smaller model or optimized serving)

# Step 3: Benchmark candidate models on YOUR task
# Don't trust general benchmarks - evaluate on your specific data
candidates = ["llama-3.1-8b-instruct", "mistral-7b-instruct", "qwen-2.5-7b-instruct"]
# Run each through your eval pipeline (Section 12.7)
# Measure: accuracy, latency, cost on 500+ representative examples

# Step 4: Cost-quality Pareto analysis
# Plot cost vs. quality for each candidate
# Choose the model on the Pareto frontier closest to your requirements</code></pre>

<h4>2. Cost Analysis</h4>

<table>
<tr><th>Model</th><th>API Cost (per 1M input tokens)</th><th>API Cost (per 1M output tokens)</th><th>Self-hosted Cost (per 1M tokens, H100)</th></tr>
<tr><td>GPT-4o</td><td>$2.50</td><td>$10.00</td><td>N/A (API only)</td></tr>
<tr><td>GPT-4o-mini</td><td>$0.15</td><td>$0.60</td><td>N/A (API only)</td></tr>
<tr><td>Claude 3.5 Sonnet</td><td>$3.00</td><td>$15.00</td><td>N/A (API only)</td></tr>
<tr><td>Claude 3.5 Haiku</td><td>$0.80</td><td>$4.00</td><td>N/A (API only)</td></tr>
<tr><td>LLaMA-3.1-8B</td><td>$0.05-0.10 (hosted)</td><td>$0.05-0.10</td><td>~$0.02-0.05 (self-hosted)</td></tr>
<tr><td>LLaMA-3.1-70B</td><td>$0.50-0.90 (hosted)</td><td>$0.50-0.90</td><td>~$0.15-0.30 (self-hosted)</td></tr>
<tr><td>Mistral-7B</td><td>$0.05-0.10 (hosted)</td><td>$0.05-0.10</td><td>~$0.02-0.04 (self-hosted)</td></tr>
</table>

<p><em>Note: API prices as of early 2025 and will likely decrease. Self-hosted costs include GPU amortization, electricity, and engineering time.</em></p>

<pre><code># Cost estimation for a production workload:
def estimate_monthly_cost(
    requests_per_day,
    avg_input_tokens,
    avg_output_tokens,
    input_cost_per_million,
    output_cost_per_million
):
    """Estimate monthly API cost."""
    daily_input_tokens = requests_per_day * avg_input_tokens
    daily_output_tokens = requests_per_day * avg_output_tokens

    daily_cost = (
        (daily_input_tokens / 1_000_000) * input_cost_per_million +
        (daily_output_tokens / 1_000_000) * output_cost_per_million
    )

    monthly_cost = daily_cost * 30
    cost_per_request = daily_cost / requests_per_day

    print(f"Daily tokens: {daily_input_tokens/1e6:.1f}M input, {daily_output_tokens/1e6:.1f}M output")
    print(f"Daily cost:   \${daily_cost:.2f}")
    print(f"Monthly cost: \${monthly_cost:,.2f}")
    print(f"Per request:  \${cost_per_request:.4f}")
    return monthly_cost

# Example: Customer support chatbot
# 10K requests/day, 2000 input tokens avg, 500 output tokens avg

# GPT-4o:
estimate_monthly_cost(10000, 2000, 500, 2.50, 10.00)
# Monthly: $3,000  |  Per request: $0.01

# GPT-4o-mini:
estimate_monthly_cost(10000, 2000, 500, 0.15, 0.60)
# Monthly: $180  |  Per request: $0.0006

# Self-hosted LLaMA-3.1-8B (1x H100, ~$3/hour):
# ~$2,160/month fixed cost for the GPU, regardless of traffic
# Cost effective when daily cost exceeds GPU cost</code></pre>

<h4>3. API vs Self-Hosted Decision Framework</h4>

<pre><code># Decision tree:

# 1. Do you have data privacy/compliance requirements?
#    YES -> Self-host (or use a compliant cloud provider with data agreements)
#    NO  -> Continue

# 2. What's your monthly request volume?
#    < 10K requests/month  -> Use API (not worth the infrastructure overhead)
#    10K - 1M              -> Depends on model size and cost sensitivity
#    > 1M                  -> Self-host likely more cost effective

# 3. What model quality do you need?
#    Frontier (GPT-4/Claude Opus class)  -> API (these models are not open)
#    Strong (GPT-4o-mini class)          -> Either (Llama-3.1-70B is competitive)
#    Good enough (GPT-3.5 class)         -> Self-host (Llama-3.1-8B is sufficient)

# 4. Do you need customization (fine-tuning)?
#    YES, extensive -> Self-host (more control, faster iteration)
#    YES, light     -> API fine-tuning (OpenAI, Together, Fireworks)
#    NO             -> API is simpler

# Break-even analysis:
# Self-hosted 8B model on 1x H100:
#   Fixed cost: ~$2,160/month ($3/hour)
#   Throughput: ~500 req/sec at 500 output tokens
#   Capacity: 500 * 86400 * 30 = 1.3B tokens/month
#
# API equivalent (GPT-4o-mini at $0.60/M output tokens):
#   1.3B tokens * $0.60/M = $780/month
#
# Wait - the API is CHEAPER for this volume?
# Yes, for small models. The break-even shifts with:
# - Larger models (70B self-hosted vs. GPT-4o API)
# - Higher volumes (fixed GPU cost is amortized)
# - Fine-tuning needs (API fine-tuning is expensive)
# - Privacy requirements (self-hosted is the only option)</code></pre>

<h4>4. Prompt Engineering Fundamentals</h4>
<p>Prompt engineering is the most cost-effective way to improve LLM performance. Good prompting can extract more value from a cheap model than bad prompting can from an expensive one.</p>

<pre><code># Technique 1: System Prompt Engineering
# Define the model's role, constraints, and output format clearly.

system_prompt = """You are a customer support agent for Acme Software.
Your role is to help users troubleshoot technical issues.

Rules:
1. Always be polite and professional
2. If you don't know the answer, say so - never make up information
3. For billing issues, direct users to billing@acme.com
4. Never share other customers' information
5. If the user is frustrated, acknowledge their frustration before solving

Response format:
- Start with a brief acknowledgment of the issue
- Provide step-by-step troubleshooting instructions
- End with "Is there anything else I can help with?"
"""

# Technique 2: Few-Shot Examples
# Show the model 2-3 examples of ideal input/output pairs.

few_shot_prompt = """
Here are examples of ideal responses:

User: My app keeps crashing when I open it.
Agent: I'm sorry to hear you're experiencing crashes! Let's troubleshoot this together.

1. First, please make sure you're running the latest version (v3.2.1). Go to Settings > About > Check for Updates.
2. If you're already on the latest version, try clearing the app cache: Settings > Storage > Clear Cache.
3. If the issue persists, please restart your device and try again.

Is there anything else I can help with?

---

User: I was charged twice for my subscription.
Agent: I understand how frustrating a double charge can be. I want to make sure this gets resolved quickly for you.

For billing inquiries, our billing team can help you directly. Please email billing@acme.com with your account email and the dates of the charges. They typically respond within 24 hours and will process any necessary refunds.

Is there anything else I can help with?
"""

# Technique 3: Chain-of-Thought Prompting
# Ask the model to reason step by step before answering.

cot_prompt = """
Analyze this customer's issue and think through it step by step before responding:

Step 1: What is the core problem?
Step 2: What are the most likely causes?
Step 3: What is the simplest fix to try first?
Step 4: What should be escalated if the simple fix doesn't work?

Then provide your response to the customer.
"""

# Technique 4: Output Format Constraints
# Be explicit about the desired output format.

json_prompt = """
Extract the following information from the customer message.
Respond in JSON format only, no other text.

Required fields:
{
  "issue_category": "billing|technical|account|feature_request|other",
  "severity": "low|medium|high|critical",
  "product": "string",
  "summary": "one sentence summary",
  "requires_escalation": true/false
}
"""</code></pre>

<h4>5. Advanced Prompting Techniques</h4>

<pre><code># Technique 5: Self-Consistency (Wang et al., 2022)
# Generate multiple responses with temperature > 0, then take the majority vote.
# Particularly effective for reasoning tasks.
# Cost: N * base cost (typically N=5 to 10)
# Quality improvement: 5-15% on reasoning benchmarks

# Technique 6: Structured Output with Schema
# Modern APIs support structured output (JSON schema enforcement)
# This guarantees the output matches your expected format.

# OpenAI example:
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Classify this ticket: ..."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ticket_classification",
            "schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["billing", "technical", "account"]},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5}
                },
                "required": ["category", "priority"]
            }
        }
    }
)

# Technique 7: Prompt Caching
# For system prompts + few-shot examples that don't change between requests,
# use prompt caching (supported by Anthropic, OpenAI, etc.)
# The static prefix is cached on the server, reducing both latency and cost.
# Typically saves 50-90% on the cached portion.</code></pre>

<h4>6. The Prompt Engineering Checklist</h4>

<div class="callout">
<div class="callout-title">Production Prompt Engineering Checklist</div>
<p>Before deploying any prompt to production:</p>
</div>

<ul>
<li><strong>Test with 100+ diverse inputs</strong> including edge cases (empty input, very long input, adversarial input, non-English input)</li>
<li><strong>Measure against your eval set</strong> and compare with the previous best prompt</li>
<li><strong>Check for prompt injection vulnerabilities</strong>: can a user make the model ignore its system prompt?</li>
<li><strong>Verify output format consistency</strong>: does the model always produce valid JSON/the expected format?</li>
<li><strong>Test with the actual model and temperature</strong> you'll use in production (don't test with GPT-4 and deploy with GPT-4o-mini)</li>
<li><strong>Version your prompts</strong> in source control with clear documentation of changes and their impact</li>
<li><strong>A/B test new prompts</strong> in production before full rollout</li>
</ul>

<div class="callout warning">
<div class="callout-title">War Story: The $50K Prompt</div>
<p>A fintech company was using GPT-4 to classify financial transactions into 47 categories. Their initial prompt was a simple instruction with the category list. Accuracy: 72%. They tried few-shot examples (5 per category = 235 examples in the prompt). Accuracy jumped to 89%, but the prompt was now 8K tokens, and each API call cost $0.024. At 500K classifications per day, that's $12,000/day. They then spent a week optimizing: reduced to 2 key examples per confusing category pair (1.5K tokens), added a decision tree in the system prompt, and switched to GPT-4o-mini for easy cases with GPT-4 only for uncertain classifications. Final accuracy: 91% (higher than the expensive version). Final cost: $800/day. The one-week prompt engineering effort saved $11,200/day &mdash; that is $336K/month. The lesson: prompt engineering is the highest-ROI activity in LLM deployment.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You need to deploy an LLM for a use case with strict data privacy requirements and 50K requests per day. Walk me through your decision process.</div>
<div class="a-text">Given strict data privacy, I must self-host (no external API calls). Decision process: (1) <strong>Estimate requirements:</strong> 50K requests/day = ~0.6 requests/second average, ~3 req/sec peak (5x burst). Assume 1500 input tokens, 500 output tokens average. (2) <strong>Model selection:</strong> Start with the smallest model that meets quality needs. Run evals on LLaMA-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, and Mistral-7B-Instruct using our custom eval set. If quality is insufficient, try 70B class. (3) <strong>Hardware sizing:</strong> For an 8B model: 1x A100 (or L40S) can serve ~50-100 requests/second with vLLM, far exceeding our needs. Total: 1 GPU for inference + 1 for redundancy. Cost: ~$4,000-6,000/month. (4) <strong>Serving stack:</strong> vLLM for inference (PagedAttention for efficient memory), Kubernetes for orchestration, with health checks and auto-restart. (5) <strong>Monitoring:</strong> Log all inputs/outputs (internally), track latency P50/P95, token throughput, error rate, and run the eval suite nightly. (6) <strong>Fine-tuning:</strong> If the base model doesn't meet quality bars, fine-tune with QLoRA on domain-specific data. This entire setup costs ~$6K/month vs. potentially $15K+/month for a premium API (and with privacy guarantees).</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What are the most effective prompt engineering techniques and when would you use each?</div>
<div class="a-text">The key techniques in order of impact: (1) <strong>Clear system prompt with constraints:</strong> Always the first thing to optimize. Define role, rules, output format explicitly. This alone fixes 50%+ of quality issues. Use for: every production deployment. (2) <strong>Few-shot examples:</strong> Include 2-5 examples of ideal input/output pairs. Use for: classification, extraction, any task with a specific output format. (3) <strong>Chain-of-thought (CoT):</strong> Add "think step by step" or provide a reasoning template. Use for: math, logic, complex reasoning, multi-step tasks. Increases latency and cost (more output tokens). (4) <strong>Structured output:</strong> Enforce JSON schema or use XML tags. Use for: any task that feeds into downstream code (APIs, databases). (5) <strong>Self-consistency:</strong> Generate N responses and take majority vote. Use for: high-stakes decisions where accuracy matters more than cost/latency. (6) <strong>Prompt caching:</strong> Cache the static prefix (system prompt + examples). Use for: any production system with a stable system prompt. Reduces cost 50-90%.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">When would you choose to self-host vs. use an API? What factors matter most?</div>
<div class="a-text">Five key factors: (1) <strong>Data privacy:</strong> If data cannot leave your infrastructure (HIPAA, GDPR, financial regulations), self-hosting is mandatory. This is the most common hard constraint. (2) <strong>Cost at scale:</strong> Self-hosting has high fixed costs (GPU rental, engineering) but low marginal costs. APIs have zero fixed cost but higher marginal cost. Break-even depends on model size and volume. For 8B models, the break-even is surprisingly high (~1M+ requests/month) because small model APIs are cheap. For 70B models, self-hosting breaks even faster. (3) <strong>Quality requirements:</strong> If you need frontier quality (GPT-4, Claude Opus class), you must use an API since these models are not open-weight. If "good enough" quality suffices, open models like LLaMA-3.1-70B are competitive. (4) <strong>Customization:</strong> If you need extensive fine-tuning, self-hosting gives more control and faster iteration. API fine-tuning exists but is more limited and expensive. (5) <strong>Operational complexity:</strong> Self-hosting requires ML infrastructure expertise (GPU management, model serving, monitoring). If your team lacks this, start with APIs and migrate to self-hosting when the ROI justifies hiring for it.</div>
</div>
`
    }
  ]
};
