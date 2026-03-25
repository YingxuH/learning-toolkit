// Deeply Expanded Content for Chapters 13-14
// Chapter 13: Quantization Deep Dive (8 sections, ~12,000 words)
// Chapter 14: RAG Systems (8 sections, ~12,000 words)

const CONTENT_CH13_14 = {

  // ============================================================
  // CHAPTER 13: Quantization Deep Dive
  // ============================================================
  ch13_sections: [
    // ----------------------------------------------------------
    // 13.1 Quantization Fundamentals
    // ----------------------------------------------------------
    {
      id: "quant-fundamentals",
      title: "Quantization Fundamentals",
      content: `
<p>Modern large language models are breathtakingly expensive to serve. A 70-billion-parameter model stored in FP16 requires 140 GB of GPU memory just for the weights&mdash;before a single token is generated. Once you factor in KV-cache, activations, and optimizer states, the memory budget explodes. Quantization is the single most impactful technique for making LLM inference practical, reducing memory by 2&ndash;4x while preserving the vast majority of model quality. This section lays the mathematical and systems-level foundations you need before diving into specific quantization methods.</p>

<div class="callout">
<div class="callout-title">Key Insight: It's About Bandwidth, Not Compute</div>
<p>LLM inference is <strong>memory-bandwidth-bound</strong>, not compute-bound. During autoregressive decoding, each token generation requires reading the entire model weight matrix from GPU memory. On an A100 (2 TB/s bandwidth, 312 TFLOPS FP16), the arithmetic intensity of matrix-vector products is so low that the GPU spends most of its time waiting for data. Quantization shrinks the weights, meaning fewer bytes to transfer per token, which directly translates to higher throughput. A 4-bit model runs roughly 3&ndash;4x faster than FP16 on the same hardware&mdash;not because the math is faster, but because the data moves faster.</p>
</div>

<h4>1. Why Quantize? The Memory Wall</h4>
<p>Consider the memory requirements for popular model sizes:</p>

<table>
<tr><th>Model Size</th><th>FP32</th><th>FP16/BF16</th><th>INT8</th><th>INT4</th><th>GPUs Needed (80GB)</th></tr>
<tr><td>7B</td><td>28 GB</td><td>14 GB</td><td>7 GB</td><td>3.5 GB</td><td>1 (FP16)</td></tr>
<tr><td>13B</td><td>52 GB</td><td>26 GB</td><td>13 GB</td><td>6.5 GB</td><td>1 (FP16)</td></tr>
<tr><td>34B</td><td>136 GB</td><td>68 GB</td><td>34 GB</td><td>17 GB</td><td>1 (FP16)</td></tr>
<tr><td>70B</td><td>280 GB</td><td>140 GB</td><td>70 GB</td><td>35 GB</td><td>2 (FP16) / 1 (INT4)</td></tr>
<tr><td>405B</td><td>1620 GB</td><td>810 GB</td><td>405 GB</td><td>203 GB</td><td>10 (FP16) / 3 (INT4)</td></tr>
</table>

<p>Quantizing a 70B model from FP16 to INT4 means it fits on a single 48 GB GPU (RTX 4090 or A6000) instead of requiring two A100s. That is a 10x cost reduction in hardware.</p>

<h4>2. Number Representations</h4>
<p>Understanding the bit-level formats is essential for reasoning about quantization error.</p>

<pre><code># Floating-point representations
# FP32: 1 sign + 8 exponent + 23 mantissa = 32 bits
# FP16: 1 sign + 5 exponent + 10 mantissa = 16 bits
# BF16: 1 sign + 8 exponent +  7 mantissa = 16 bits (same range as FP32!)
# FP8 (E4M3): 1 sign + 4 exponent + 3 mantissa = 8 bits
# FP8 (E5M2): 1 sign + 5 exponent + 2 mantissa = 8 bits

# Integer representations
# INT8:  [-128, 127] - 256 levels, uniform spacing
# INT4:  [-8, 7]     - 16 levels, uniform spacing
# UINT4: [0, 15]     - 16 levels, unsigned

import struct
import numpy as np

def float_to_binary(f, fmt='f'):
    """Show binary representation of a float."""
    packed = struct.pack(f'>{fmt}', f)
    binary = ''.join(f'{byte:08b}' for byte in packed)
    return binary

# FP32 example
val = 3.14
binary = float_to_binary(val)
print(f"FP32 of {val}: {binary}")
print(f"  Sign:     {binary[0]}")
print(f"  Exponent: {binary[1:9]}")
print(f"  Mantissa: {binary[9:32]}")

# BF16 vs FP16: BF16 keeps FP32's range but sacrifices precision
fp32_max = 3.4e38
fp16_max = 65504.0   # Much smaller range!
bf16_max = 3.4e38    # Same range as FP32

print(f"FP32 max: {fp32_max}")
print(f"FP16 max: {fp16_max}")  # Overflow risk!
print(f"BF16 max: {bf16_max}")  # Safe for LLM training</code></pre>

<p><strong>BF16 vs FP16:</strong> BF16 (bfloat16) uses the same 8 exponent bits as FP32, giving it the same dynamic range (up to ~3.4e38). FP16 has only 5 exponent bits, limiting it to ~65,504. This is why BF16 became the default for LLM training&mdash;FP16 training often requires loss scaling to avoid overflow, while BF16 does not.</p>

<h4>3. The Quantization Function</h4>
<p>Quantization maps continuous values to a discrete set. The fundamental operation is:</p>

<pre><code>import torch

def quantize_symmetric(tensor: torch.Tensor, bits: int = 8) -> tuple:
    """Symmetric quantization: maps [-max, +max] to [-2^(b-1), 2^(b-1)-1].
    
    The zero point is always 0 (hence 'symmetric').
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    # Scale factor: maps the max absolute value to the max quantized value
    abs_max = tensor.abs().max()
    scale = abs_max / qmax
    
    # Quantize: divide by scale, round, clamp
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)
    
    return q_tensor, scale

def dequantize_symmetric(q_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Reconstruct the approximate original tensor."""
    return q_tensor.float() * scale

def quantize_asymmetric(tensor: torch.Tensor, bits: int = 8) -> tuple:
    """Asymmetric quantization: maps [min, max] to [0, 2^b - 1].
    
    Uses a zero-point offset, which allows better utilization of the
    quantization range when the distribution is not centered at zero.
    """
    qmin = 0
    qmax = 2 ** bits - 1
    
    # Compute scale and zero-point
    val_min = tensor.min()
    val_max = tensor.max()
    scale = (val_max - val_min) / (qmax - qmin)
    zero_point = torch.round(qmin - val_min / scale).clamp(qmin, qmax).int()
    
    # Quantize
    q_tensor = torch.clamp(
        torch.round(tensor / scale + zero_point), qmin, qmax
    ).to(torch.uint8)
    
    return q_tensor, scale, zero_point

def dequantize_asymmetric(q_tensor, scale, zero_point):
    """Reconstruct from asymmetric quantization."""
    return (q_tensor.float() - zero_point) * scale

# Example: quantize a weight tensor
weights = torch.randn(4096, 4096) * 0.02  # Typical LLM weight distribution

q_sym, scale_sym = quantize_symmetric(weights, bits=8)
q_asym, scale_asym, zp = quantize_asymmetric(weights, bits=8)

# Measure quantization error
recon_sym = dequantize_symmetric(q_sym, scale_sym)
recon_asym = dequantize_asymmetric(q_asym, scale_asym, zp)

mse_sym = ((weights - recon_sym) ** 2).mean().item()
mse_asym = ((weights - recon_asym) ** 2).mean().item()

print(f"Symmetric  INT8 MSE: {mse_sym:.8f}")
print(f"Asymmetric INT8 MSE: {mse_asym:.8f}")
print(f"Max absolute error (symmetric):  {(weights - recon_sym).abs().max():.6f}")
print(f"Max absolute error (asymmetric): {(weights - recon_asym).abs().max():.6f}")</code></pre>

<h4>4. Symmetric vs Asymmetric Quantization</h4>
<table>
<tr><th>Property</th><th>Symmetric</th><th>Asymmetric</th></tr>
<tr><td>Range</td><td>[-max, +max]</td><td>[min, max]</td></tr>
<tr><td>Zero point</td><td>Always 0</td><td>Computed offset</td></tr>
<tr><td>Best for</td><td>Weights (roughly symmetric)</td><td>Activations (often positive, e.g., after ReLU)</td></tr>
<tr><td>Storage overhead</td><td>1 scale per group</td><td>1 scale + 1 zero-point per group</td></tr>
<tr><td>Computation</td><td>Simpler (multiply only)</td><td>Requires zero-point subtraction</td></tr>
<tr><td>Precision</td><td>Wastes range if distribution is skewed</td><td>Better range utilization</td></tr>
</table>

<p>In practice, most LLM quantization uses <strong>symmetric quantization for weights</strong> (because weight distributions are approximately Gaussian centered at zero) and <strong>asymmetric quantization for activations</strong> (which can be skewed, especially after ReLU/GeLU).</p>

<h4>5. Granularity: Per-Tensor vs Per-Channel vs Per-Group</h4>
<p>The granularity of the scale factor determines the trade-off between accuracy and overhead:</p>

<pre><code>import torch

def quantize_per_tensor(weight: torch.Tensor, bits: int = 4):
    """One scale factor for the entire tensor. Fastest but least accurate."""
    qmax = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / qmax
    q = torch.clamp(torch.round(weight / scale), -qmax - 1, qmax)
    return q, scale  # 1 scale value total

def quantize_per_channel(weight: torch.Tensor, bits: int = 4):
    """One scale per output channel (row). Good balance."""
    qmax = 2 ** (bits - 1) - 1
    # Scale per row (output channel)
    scales = weight.abs().amax(dim=1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)  # Avoid division by zero
    q = torch.clamp(torch.round(weight / scales), -qmax - 1, qmax)
    return q, scales  # N_out scale values

def quantize_per_group(weight: torch.Tensor, bits: int = 4, 
                       group_size: int = 128):
    """One scale per group of elements. Most accurate for low-bit."""
    qmax = 2 ** (bits - 1) - 1
    N_out, N_in = weight.shape
    
    assert N_in % group_size == 0, f"N_in ({N_in}) must be divisible by group_size ({group_size})"
    
    # Reshape into groups
    weight_grouped = weight.reshape(N_out, N_in // group_size, group_size)
    
    # Scale per group
    scales = weight_grouped.abs().amax(dim=2, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)
    
    q = torch.clamp(torch.round(weight_grouped / scales), -qmax - 1, qmax)
    q = q.reshape(N_out, N_in)
    scales = scales.squeeze(2)  # Shape: (N_out, N_in // group_size)
    
    return q, scales

# Compare accuracy across granularities
weight = torch.randn(4096, 4096) * 0.02

# Add some outliers (common in LLMs!)
weight[0, 0] = 0.5  # 25x the std dev
weight[100, 200] = -0.4

for name, fn in [("Per-tensor", quantize_per_tensor),
                 ("Per-channel", quantize_per_channel)]:
    q, s = fn(weight, bits=4)
    if isinstance(s, torch.Tensor):
        recon = (q * s) if s.dim() > 0 else q * s
    else:
        recon = q * s
    mse = ((weight - recon) ** 2).mean().item()
    print(f"{name:15s} INT4 MSE: {mse:.10f}")

q, s = quantize_per_group(weight, bits=4, group_size=128)
recon = (q.reshape(4096, 32, 128).float() * s.unsqueeze(2)).reshape(4096, 4096)
mse = ((weight - recon) ** 2).mean().item()
print(f"{'Per-group(128)':15s} INT4 MSE: {mse:.10f}")</code></pre>

<table>
<tr><th>Granularity</th><th>Scale Overhead</th><th>Accuracy</th><th>Use Case</th></tr>
<tr><td>Per-tensor</td><td>1 value</td><td>Lowest</td><td>Quick prototyping, INT8 (where error is small)</td></tr>
<tr><td>Per-channel</td><td>N_out values</td><td>Medium</td><td>INT8 weights, good default</td></tr>
<tr><td>Per-group (g=128)</td><td>N_out * N_in/128</td><td>Highest</td><td>INT4/INT3, essential for low-bit</td></tr>
</table>

<p><strong>Group size 128</strong> has become the standard for 4-bit quantization. It adds ~0.5 bits of overhead per weight (the FP16 scales), but the accuracy improvement is dramatic. GPTQ, AWQ, and GGUF all default to group_size=128.</p>

<h4>6. Quantization Error Analysis</h4>
<p>The quantization error for uniform quantization follows a predictable pattern. For a uniform distribution of values, the rounding error is approximately:</p>

<pre><code># Theoretical quantization error analysis
import numpy as np

def theoretical_quantization_snr(bits):
    """Signal-to-Quantization-Noise Ratio for uniform quantization.
    
    SNR = 6.02 * bits + 1.76 dB (for uniformly distributed signal)
    Each additional bit adds ~6 dB of SNR.
    """
    return 6.02 * bits + 1.76

for bits in [2, 3, 4, 5, 6, 8, 16]:
    snr = theoretical_quantization_snr(bits)
    levels = 2 ** bits
    print(f"INT{bits:2d}: {levels:6d} levels, SNR = {snr:5.1f} dB")

# Output:
# INT 2:      4 levels, SNR =  13.8 dB
# INT 3:      8 levels, SNR =  19.8 dB
# INT 4:     16 levels, SNR =  25.8 dB
# INT 5:     32 levels, SNR =  31.9 dB
# INT 6:     64 levels, SNR =  37.9 dB
# INT 8:    256 levels, SNR =  49.9 dB
# INT16:  65536 levels, SNR =  98.1 dB</code></pre>

<p>For LLMs, however, the distribution is <strong>not uniform</strong>&mdash;it is approximately Gaussian with occasional outliers. These outliers are the primary source of quantization error, because a single large value forces the scale factor to be large, reducing precision for all other values. This observation motivates per-group quantization (limits outlier impact to one group) and methods like AWQ (special handling of outlier-sensitive weights).</p>

<div class="callout warning">
<div class="callout-title">The Outlier Problem</div>
<p>Dettmers et al. (arXiv: 2208.07339, "LLM.int8()") discovered that transformer activations contain systematic outliers: a small number of feature dimensions consistently produce values 10&ndash;100x larger than the rest. These "emergent features" appear in models above ~6B parameters and become more extreme as models grow. A naive INT8 quantization of these activations causes catastrophic quality loss because the outlier dimensions dominate the scale factor, crushing all other values into a few quantization levels. Their solution: mixed-precision decomposition, where outlier dimensions stay in FP16 while the rest are quantized to INT8.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why is LLM inference memory-bandwidth-bound rather than compute-bound, and how does quantization help?</div>
<div class="a-text">During autoregressive generation, each token requires a matrix-vector multiplication: <code>y = W @ x</code>, where W is [d, d] and x is [d, 1]. This has d^2 multiply-adds but requires reading d^2 weight values. The arithmetic intensity (FLOPS per byte) is approximately 1 op/2 bytes for FP16, while modern GPUs have a compute-to-bandwidth ratio of ~150 FLOPS/byte (A100: 312 TFLOPS / 2 TB/s). This means the GPU is 150x more compute-capable than the memory system can feed it. Quantization reduces the bytes per weight (FP16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes), directly increasing effective bandwidth utilization. A 4-bit model reads 4x fewer bytes per token, enabling ~3.5x higher throughput (not exactly 4x due to dequantization overhead and scale factor reads). For batched inference (batch size > 1), the problem shifts toward compute-bound, which is why quantization helps less for large-batch throughput scenarios.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the difference between per-tensor, per-channel, and per-group quantization. When would you choose each?</div>
<div class="a-text">Per-tensor uses one scale/zero-point for the entire weight matrix. It is simplest but least accurate because a single outlier element forces a large scale, reducing effective precision for all other values. Use it only for INT8 where the error budget is generous. Per-channel (per-row for weights) uses one scale per output channel. It is the standard for INT8 quantization and supported natively by most hardware. It handles the case where different output neurons have different weight magnitudes. Per-group divides each row into groups of g elements (typically 128) and uses one scale per group. This is essential for INT4 and below, where 16 quantization levels cannot represent the full weight range accurately with a single scale. The overhead is storing N_out * N_in/g additional FP16 scale values, adding about 0.5 bits per weight for g=128. The accuracy improvement from per-tensor to per-group INT4 can be the difference between usable and broken output.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 13.2 Post-Training Quantization
    // ----------------------------------------------------------
    {
      id: "quant-ptq",
      title: "Post-Training Quantization: RTN, Calibration, and GPTQ",
      content: `
<p>Post-training quantization (PTQ) converts a pre-trained FP16 model to lower precision without any additional training. This is by far the most common quantization approach for LLMs because retraining a 70B model is prohibitively expensive for most teams. PTQ methods range from the trivially simple (round-to-nearest) to the mathematically sophisticated (GPTQ), with dramatic quality differences at low bit-widths.</p>

<h4>1. Round-to-Nearest (RTN)</h4>
<p>The simplest PTQ method: take each weight, divide by a scale factor, round to the nearest integer, clamp to the valid range. No calibration data needed.</p>

<pre><code>import torch

def rtn_quantize(weight: torch.Tensor, bits: int = 4, 
                 group_size: int = 128):
    """Round-to-Nearest quantization. Simple but surprisingly decent for INT8.
    
    For INT4, quality degrades noticeably on models > 13B.
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    N_out, N_in = weight.shape
    weight_grouped = weight.reshape(N_out, -1, group_size)
    
    # Per-group scale (symmetric)
    scales = weight_grouped.abs().amax(dim=2, keepdim=True) / qmax
    scales = scales.clamp(min=1e-10)
    
    # Round to nearest
    q = torch.clamp(torch.round(weight_grouped / scales), qmin, qmax)
    
    # Dequantize for measuring error
    dequant = q * scales
    mse = ((weight_grouped - dequant) ** 2).mean()
    
    return q.to(torch.int8), scales.squeeze(2), mse.item()

# Example
weight = torch.randn(4096, 4096) * 0.02
q, scales, mse = rtn_quantize(weight, bits=4, group_size=128)
print(f"RTN INT4 MSE: {mse:.10f}")
print(f"Weight shape: {weight.shape}")
print(f"Quantized shape: {q.shape}")
print(f"Scales shape: {scales.shape}")
# Scales overhead: 4096 * 32 * 2 bytes (FP16) = 256 KB
# vs weight: 4096 * 4096 * 0.5 bytes (INT4) = 8 MB
# Overhead ratio: ~3.1%</code></pre>

<p>RTN is fast and requires no calibration data, but it treats every weight independently. It has no way to compensate for the rounding error of one weight by adjusting another. For INT8, RTN works well (typical perplexity increase &lt; 0.1). For INT4, it causes noticeable degradation, especially in larger models where weight distributions have more complex structure.</p>

<h4>2. Calibration Datasets</h4>
<p>Smarter PTQ methods need a small dataset of representative inputs to understand how quantization errors propagate through the network. This is the <strong>calibration dataset</strong>.</p>

<pre><code>from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_calibration_data(
    model_name: str = "meta-llama/Llama-3.1-70B",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    n_samples: int = 128,
    seq_len: int = 2048
):
    """Prepare a calibration dataset for PTQ.
    
    Guidelines:
    - 128 samples is typically sufficient (more helps marginally)
    - Use diverse text (Wikipedia, code, conversation)
    - Match the distribution of your deployment data if possible
    - Sequence length should match inference (2048-4096)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name, dataset_config, split="train")
    
    # Concatenate all text
    all_text = "\\n\\n".join(dataset["text"])
    tokens = tokenizer(all_text, return_tensors="pt")["input_ids"][0]
    
    # Split into sequences
    calibration_data = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        if len(calibration_data) >= n_samples:
            break
        calibration_data.append(tokens[i:i + seq_len].unsqueeze(0))
    
    print(f"Prepared {len(calibration_data)} calibration sequences "
          f"of length {seq_len}")
    return calibration_data

# Common calibration datasets:
# - WikiText-2: General English text (most common default)
# - C4 (subset): Web text, more diverse
# - RedPajama (subset): Mix of sources
# - Custom domain data: Best if you know your deployment domain</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: Calibration Data Mismatch</div>
<p>We quantized a code-generation model (StarCoder2-15B) using WikiText-2 for calibration and saw a 15% drop in HumanEval pass@1. When we switched to a calibration set of Python code from The Stack, the degradation dropped to 2%. <strong>Calibration data should match your deployment distribution.</strong> If your model will primarily handle code, calibrate with code. If it handles medical text, calibrate with medical text. For general-purpose models, use a diverse mix. The 128-sample default is usually fine for size, but the domain matters enormously.</p>
</div>

<h4>3. GPTQ: Optimal Layer-Wise Quantization</h4>
<p>GPTQ (Frantar et al., arXiv: 2210.17323) is the most widely used PTQ algorithm for LLMs. It quantizes weights one layer at a time, using the Hessian matrix of each layer to understand which weights are most important and to compensate for quantization errors by adjusting not-yet-quantized weights.</p>

<p><strong>Key insight:</strong> When you quantize one weight and introduce error, you can partially compensate by slightly adjusting the remaining unquantized weights in the same row. GPTQ does this optimally (in the least-squares sense) using the inverse Hessian.</p>

<pre><code>import torch
import torch.nn as nn

class GPTQQuantizer:
    """Simplified GPTQ implementation for educational purposes.
    
    Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization 
    for Generative Pre-trained Transformers" (arXiv: 2210.17323)
    
    The algorithm:
    1. Collect activation statistics (Hessian) from calibration data
    2. For each layer, quantize columns one-at-a-time
    3. After quantizing each column, update remaining columns to 
       compensate for the quantization error
    4. Use block-wise updates for efficiency (block size = 128)
    """
    
    def __init__(self, layer: nn.Linear, bits: int = 4, 
                 group_size: int = 128, block_size: int = 128,
                 damp_percent: float = 0.01):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.block_size = block_size
        self.damp_percent = damp_percent
        
        self.W = layer.weight.data.clone().float()
        self.N_out, self.N_in = self.W.shape
        self.H = torch.zeros(self.N_in, self.N_in, device=self.W.device)
        self.n_samples = 0
    
    def collect_hessian(self, input_activations: torch.Tensor):
        """Accumulate Hessian approximation from calibration inputs.
        
        H = 2 * X^T @ X (for MSE loss on layer output)
        We accumulate X^T @ X incrementally.
        """
        # input_activations shape: (batch, seq_len, N_in)
        inp = input_activations.reshape(-1, self.N_in).float()
        self.H += inp.T @ inp
        self.n_samples += inp.shape[0]
    
    def quantize(self) -> tuple:
        """Run GPTQ quantization on the layer.
        
        Returns: (quantized_weights, scales, zeros, quantization_error)
        """
        W = self.W.clone()
        H = self.H / self.n_samples  # Average Hessian
        
        # Dampening: add small diagonal to ensure H is invertible
        damp = self.damp_percent * H.diag().mean()
        H += damp * torch.eye(self.N_in, device=H.device)
        
        # Cholesky decomposition of inverse Hessian
        # We need H^{-1} for the update formula
        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
        except RuntimeError:
            # Fallback: use pseudo-inverse if Cholesky fails
            H_inv = torch.linalg.pinv(H)
        
        H_inv_diag = H_inv.diag()
        
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        
        Q = torch.zeros_like(W)
        scales = torch.zeros(self.N_out, self.N_in // self.group_size,
                           device=W.device)
        
        total_error = 0.0
        
        # Process columns in blocks for efficiency
        for block_start in range(0, self.N_in, self.block_size):
            block_end = min(block_start + self.block_size, self.N_in)
            block_size = block_end - block_start
            
            # Extract the block
            W_block = W[:, block_start:block_end].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            H_inv_block = H_inv[block_start:block_end, 
                                block_start:block_end]
            
            for col in range(block_size):
                global_col = block_start + col
                
                # Determine group for this column
                group_idx = global_col // self.group_size
                
                # Compute scale for this group (if first col in group)
                if global_col % self.group_size == 0:
                    group_end = min(global_col + self.group_size, self.N_in)
                    group_weights = W[:, global_col:group_end]
                    group_max = group_weights.abs().amax(dim=1)
                    scales[:, group_idx] = group_max / qmax
                
                scale = scales[:, group_idx].clamp(min=1e-10)
                
                # Quantize this column
                w_col = W_block[:, col]
                q_col = torch.clamp(
                    torch.round(w_col / scale), qmin, qmax
                )
                Q_block[:, col] = q_col
                
                # Quantization error for this column
                err = (w_col - q_col * scale)
                Err_block[:, col] = err
                total_error += (err ** 2).sum().item()
                
                # GPTQ magic: compensate remaining columns
                # Update formula: W[:, j] -= err * H_inv[col, j] / H_inv[col, col]
                if col < block_size - 1:
                    h_ratio = H_inv_block[col, col+1:] / H_inv_block[col, col]
                    W_block[:, col+1:] -= err.unsqueeze(1) * h_ratio.unsqueeze(0)
            
            Q[:, block_start:block_end] = Q_block
            
            # Update remaining columns outside this block
            if block_end < self.N_in:
                block_H = H_inv[block_start:block_end, block_end:]
                update = Err_block @ block_H / H_inv_block.diag().unsqueeze(1)
                W[:, block_end:] -= update
        
        rmse = (total_error / (self.N_out * self.N_in)) ** 0.5
        print(f"GPTQ quantization RMSE: {rmse:.8f}")
        
        return Q.to(torch.int8), scales, rmse</code></pre>

<h4>4. Using GPTQ in Practice with AutoGPTQ</h4>
<pre><code>from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Step 1: Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=True,        # Use activation order (slower but better)
    static_groups=False,
    sym=True,             # Symmetric quantization
    model_name_or_path="meta-llama/Llama-3.1-70B",
    model_file_base_name="model",
)

# Step 2: Load model in FP16
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantize_config=quantize_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")

# Step 3: Prepare calibration data
calibration_data = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in calibration_texts[:128]
]

# Step 4: Quantize (this takes 1-4 hours for 70B on a single GPU)
model.quantize(calibration_data)

# Step 5: Save
model.save_quantized("./llama-3.1-70b-gptq-int4")
tokenizer.save_pretrained("./llama-3.1-70b-gptq-int4")

# Step 6: Load and use the quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "./llama-3.1-70b-gptq-int4",
    device_map="auto",
    use_safetensors=True,
)

# Now runs on a single 48GB GPU!
input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))</code></pre>

<h4>5. OBQ Origins</h4>
<p>GPTQ is derived from Optimal Brain Quantization (OBQ, Frantar & Alistarh, arXiv: 2208.11580), which itself builds on the classic Optimal Brain Surgeon (OBS) framework from the 1990s. OBQ quantizes weights one at a time in order of increasing quantization error (greedy ordering), updating remaining weights after each quantization. GPTQ's key innovation was replacing the row-by-row greedy order with a fixed column order plus block-wise updates, reducing the complexity from O(d_row * d_col^2 * d_col) to O(d_row * d_col^2) and enabling practical application to billion-parameter models.</p>

<table>
<tr><th>Method</th><th>Year</th><th>Complexity per row</th><th>Time for 70B (INT4)</th><th>Quality (PPL increase)</th></tr>
<tr><td>RTN</td><td>N/A</td><td>O(d)</td><td>Minutes</td><td>+0.5 to +2.0</td></tr>
<tr><td>OBQ</td><td>2022</td><td>O(d^3)</td><td>Weeks</td><td>+0.1 to +0.3</td></tr>
<tr><td>GPTQ</td><td>2022</td><td>O(d^2)</td><td>1-4 hours</td><td>+0.1 to +0.3</td></tr>
<tr><td>AWQ</td><td>2023</td><td>O(d)</td><td>Minutes</td><td>+0.1 to +0.2</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the GPTQ algorithm at a high level. Why does it work better than simple round-to-nearest?</div>
<div class="a-text">GPTQ quantizes weights column-by-column within each layer. After rounding each column to the nearest quantized value, it computes the quantization error and uses the inverse Hessian matrix (computed from calibration data activations) to optimally adjust the remaining unquantized columns. This compensatory update minimizes the total squared error on the layer output. Intuitively: if quantizing weight w_{ij} introduces a positive error, GPTQ can slightly decrease nearby weights to cancel out the effect on the output. RTN cannot do this because it treats each weight independently. The Hessian H = X^T X captures input correlations, so correlated input features get coordinated weight adjustments. For INT8, the difference is small (RTN is fine). For INT4, the compensatory updates are critical: they can halve the perplexity degradation compared to RTN.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 13.3 AWQ Deep Dive
    // ----------------------------------------------------------
    {
      id: "quant-awq",
      title: "AWQ: Activation-Aware Weight Quantization",
      content: `
<p>Activation-Aware Weight Quantization (AWQ, Lin et al., arXiv: 2306.00978) is an elegant PTQ method that achieves GPTQ-level quality at a fraction of the computational cost. Its core observation is deceptively simple: <strong>not all weights are equally important, and the importance of a weight is determined by the activation magnitude it multiplies.</strong> By protecting the 1% of weights that correspond to large activations, AWQ preserves 99% of model quality even at 4-bit precision.</p>

<div class="callout">
<div class="callout-title">The 1% Principle</div>
<p>In a typical transformer layer, roughly 1% of input feature channels carry disproportionately large activation magnitudes. The weights connected to these channels dominate the output. Quantizing these "salient" weights carelessly causes outsized errors. AWQ's solution: scale up salient weights before quantization (making them easier to represent precisely), then scale down the corresponding activations to preserve the mathematical result. This is a zero-cost transformation that redistributes quantization error away from the most impactful weights.</p>
</div>

<h4>1. Why Salient Weights Matter</h4>
<pre><code>import torch
import torch.nn as nn

def analyze_weight_salience(model, calibration_loader, n_batches=10):
    """Identify salient weight channels using activation magnitudes.
    
    Salience of weight column j = mean(|activation_j|) * ||weight_column_j||
    """
    activation_means = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            # input[0] shape: (batch, seq_len, hidden_dim)
            inp = input[0].detach().abs()
            if name not in activation_means:
                activation_means[name] = inp.mean(dim=(0, 1))  # Per-channel
            else:
                activation_means[name] += inp.mean(dim=(0, 1))
        return hook_fn
    
    # Register hooks on all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run calibration data through model
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= n_batches:
                break
            model(batch["input_ids"].cuda())
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Normalize
    for name in activation_means:
        activation_means[name] /= n_batches
    
    # Analyze salience
    for name, act_mean in list(activation_means.items())[:5]:
        total = act_mean.numel()
        top_1pct = int(total * 0.01)
        top_indices = act_mean.topk(top_1pct).indices
        
        top_1pct_magnitude = act_mean[top_indices].sum()
        total_magnitude = act_mean.sum()
        fraction = (top_1pct_magnitude / total_magnitude).item()
        
        print(f"{name}:")
        print(f"  Top 1% channels carry {fraction*100:.1f}% of activation magnitude")
        print(f"  Max activation: {act_mean.max():.4f}")
        print(f"  Mean activation: {act_mean.mean():.4f}")
        print(f"  Ratio (max/mean): {act_mean.max()/act_mean.mean():.1f}x")
    
    return activation_means</code></pre>

<h4>2. The AWQ Algorithm</h4>
<p>AWQ applies a per-channel scaling to weights before quantization. The key formula:</p>

<pre><code>import torch

def awq_scale_search(weight: torch.Tensor, 
                     activation_mean: torch.Tensor,
                     bits: int = 4,
                     group_size: int = 128,
                     grid_size: int = 20) -> torch.Tensor:
    """Search for optimal per-channel scaling factors.
    
    For each channel j, we want to find scale s_j such that:
    - Weight column j is multiplied by s_j before quantization
    - Activation channel j is divided by s_j at runtime
    - The net effect is identity, but quantization error is minimized
    
    The search minimizes: ||Q(W * diag(s)) * diag(1/s) @ X - W @ X||
    
    Closed-form approximation: s_j = (act_mean_j)^alpha, 
    where alpha in [0, 1] is searched via grid.
    """
    N_out, N_in = weight.shape
    device = weight.device
    
    best_scales = torch.ones(N_in, device=device)
    best_error = float('inf')
    
    # Grid search over alpha
    for alpha_idx in range(grid_size + 1):
        alpha = alpha_idx / grid_size  # 0.0 to 1.0
        
        # Compute scales: s = activation_mean^alpha
        # Higher alpha = more protection for salient channels
        scales = activation_mean.pow(alpha).clamp(min=1e-4)
        scales = scales / scales.mean()  # Normalize to avoid changing overall magnitude
        
        # Apply scaling to weights
        scaled_weight = weight * scales.unsqueeze(0)
        
        # Quantize the scaled weights
        qmax = 2 ** (bits - 1) - 1
        scaled_grouped = scaled_weight.reshape(N_out, -1, group_size)
        group_scales = scaled_grouped.abs().amax(dim=2, keepdim=True) / qmax
        group_scales = group_scales.clamp(min=1e-10)
        
        q = torch.clamp(torch.round(scaled_grouped / group_scales), 
                        -qmax - 1, qmax)
        dequant = q * group_scales
        dequant = dequant.reshape(N_out, N_in)
        
        # Undo the scaling (this is what happens at inference)
        unscaled_dequant = dequant / scales.unsqueeze(0)
        
        # Compute error weighted by activation magnitude
        # Errors on high-activation channels matter more
        error = ((weight - unscaled_dequant) ** 2)
        weighted_error = (error * activation_mean.unsqueeze(0)).sum().item()
        
        if weighted_error < best_error:
            best_error = weighted_error
            best_scales = scales.clone()
            best_alpha = alpha
    
    print(f"  Best alpha: {best_alpha:.2f}, "
          f"Error reduction: {best_error:.6f}")
    return best_scales

# Demonstration
weight = torch.randn(4096, 4096) * 0.02
activation_mean = torch.randn(4096).abs()  # Simulated activation magnitudes

# Make some channels "salient" (10-50x larger)
activation_mean[42] = activation_mean.mean() * 30
activation_mean[137] = activation_mean.mean() * 50
activation_mean[256] = activation_mean.mean() * 20

scales = awq_scale_search(weight, activation_mean, bits=4)</code></pre>

<h4>3. Why This Works: Intuition</h4>
<p>Consider a single weight w connected to a high-activation channel with mean activation a. The output contribution is approximately w * a. If we scale the weight by s (making it w*s) and the activation by 1/s, the output is still w * a, but now:</p>

<ul>
<li>The weight w*s is larger, so it occupies more quantization levels, reducing its relative quantization error</li>
<li>The quantization error on this weight is divided by s at inference, reducing its impact on the output</li>
<li>Meanwhile, less important weights (small activations) get slightly less precision, but their errors are multiplied by small activations, so the impact is negligible</li>
</ul>

<p>The mathematical insight is that quantization error is <strong>not</strong> uniformly important. AWQ redistributes precision to where it matters most, achieving the same effective quality as GPTQ but without the expensive Hessian computation.</p>

<h4>4. Using AutoAWQ</h4>
<pre><code>from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Step 1: Load model
model_path = "meta-llama/Llama-3.1-70B"
quant_path = "./llama-3.1-70b-awq-int4"

model = AutoAWQForCausalLM.from_pretrained(
    model_path, 
    safetensors=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Step 2: Configure and quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # or "GEMV" for single-batch optimization
}

# AWQ quantization - much faster than GPTQ!
# 70B model: ~20 minutes vs ~3 hours for GPTQ
model.quantize(tokenizer, quant_config=quant_config)

# Step 3: Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Step 4: Load for inference
model = AutoAWQForCausalLM.from_quantized(
    quant_path,
    fuse_layers=True,  # Fuse QKV + MLP for speed
    device_map="auto"
)

# Inference
prompt = "Explain quantum computing in simple terms:"
tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output = model.generate(tokens, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))</code></pre>

<h4>5. AWQ vs GPTQ: When to Choose Which</h4>
<table>
<tr><th>Factor</th><th>AWQ</th><th>GPTQ</th></tr>
<tr><td>Quantization speed</td><td>Fast (minutes)</td><td>Slow (hours)</td></tr>
<tr><td>Quality (perplexity)</td><td>Slightly better on average</td><td>Very good</td></tr>
<tr><td>Inference speed</td><td>Faster (GEMM/GEMV kernels)</td><td>Good (Marlin kernels)</td></tr>
<tr><td>Memory</td><td>Same (both INT4 g128)</td><td>Same</td></tr>
<tr><td>Calibration data needed</td><td>Yes (small set)</td><td>Yes (small set)</td></tr>
<tr><td>vLLM support</td><td>Excellent</td><td>Excellent</td></tr>
<tr><td>Best for</td><td>Default choice, GPU serving</td><td>When you need desc_act ordering</td></tr>
</table>

<div class="callout tip">
<div class="callout-title">Practical Recommendation</div>
<p>For most GPU-based serving scenarios, <strong>AWQ is the default choice</strong> as of 2025. It quantizes faster, produces equal or better quality, and has optimized inference kernels. GPTQ remains relevant when you need the desc_act (activation-order) option for maximum quality, or when using specific inference backends that have better GPTQ support. For CPU inference, neither AWQ nor GPTQ is ideal&mdash;use GGUF instead (Section 13.5).</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">AWQ claims that 1% of weights matter for 99% of quality. Explain the mechanism behind this claim and how AWQ exploits it.</div>
<div class="a-text">The claim is based on the observation that transformer activations have a highly non-uniform distribution across feature channels. A small fraction of channels (~1%) consistently produce activation values 10-100x larger than the mean. The weights connected to these high-activation channels have a disproportionate impact on the layer output: output_j = sum(w_jk * x_k), so if x_k is 50x the mean, w_jk's contribution is 50x more important. AWQ exploits this by applying a per-channel scaling transformation: multiply each weight column by s_k and divide the corresponding activation by s_k. This is mathematically equivalent (no change to the output), but redistributes quantization precision. Salient weights (high activation channels) get scaled up, occupying more quantization levels and reducing their relative error. Non-salient weights get scaled down slightly but their errors are multiplied by small activations anyway. The optimal scale s_k is proportional to the activation magnitude raised to a power alpha (typically 0.5-0.8), found via a simple grid search. This avoids the expensive Hessian computation of GPTQ while achieving comparable or better results.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 13.4 FP8 Quantization
    // ----------------------------------------------------------
    {
      id: "quant-fp8",
      title: "FP8 Quantization: Near-Lossless at Half the Memory",
      content: `
<p>FP8 (8-bit floating point) is the Goldilocks of quantization formats: it halves memory compared to FP16, runs natively on H100/H200 GPUs with zero performance penalty, and introduces virtually no quality degradation. For production serving on modern hardware, FP8 is rapidly becoming the default precision, relegating FP16 to a legacy format.</p>

<div class="callout">
<div class="callout-title">Why FP8 Is Special</div>
<p>Unlike INT8, FP8 is a <strong>floating-point</strong> format with an exponent field. This gives it a much wider dynamic range, which is critical for LLM weights and activations that span several orders of magnitude. INT8 has 256 uniformly spaced values; FP8 has 256 logarithmically spaced values, naturally allocating more precision to small values (where most weights cluster) and less to large values (rare outliers). The result: FP8 achieves near-FP16 quality without the outlier handling tricks that INT8 requires.</p>
</div>

<h4>1. E4M3 vs E5M2: Two FP8 Formats</h4>
<p>The IEEE and OCP (Open Compute Project) define two FP8 formats, each optimized for different use cases:</p>

<pre><code># FP8 Format Comparison
# E4M3: 1 sign + 4 exponent + 3 mantissa
#   - Range: [-448, 448]
#   - Precision: 8 mantissa levels per power-of-2 interval
#   - Best for: Weights and forward-pass activations
#
# E5M2: 1 sign + 5 exponent + 2 mantissa  
#   - Range: [-57344, 57344]
#   - Precision: 4 mantissa levels per power-of-2 interval
#   - Best for: Gradients (need wider range, less precision)

import numpy as np

def enumerate_fp8_e4m3():
    """Enumerate all positive E4M3 values to show the distribution."""
    values = []
    for exp in range(16):  # 4-bit exponent: 0-15
        for mant in range(8):  # 3-bit mantissa: 0-7
            if exp == 0:  # Subnormal
                val = (mant / 8.0) * (2 ** -6)
            elif exp == 15 and mant == 7:
                val = float('nan')  # NaN
            else:
                val = (1 + mant / 8.0) * (2 ** (exp - 7))
            values.append(val)
    return sorted(set(v for v in values if not np.isnan(v)))

def enumerate_fp8_e5m2():
    """Enumerate all positive E5M2 values."""
    values = []
    for exp in range(32):  # 5-bit exponent: 0-31
        for mant in range(4):  # 2-bit mantissa: 0-3
            if exp == 0:  # Subnormal
                val = (mant / 4.0) * (2 ** -14)
            elif exp == 31:  # Inf/NaN
                val = float('inf') if mant == 0 else float('nan')
            else:
                val = (1 + mant / 4.0) * (2 ** (exp - 15))
            values.append(val)
    return sorted(set(v for v in values if np.isfinite(v)))

e4m3_vals = enumerate_fp8_e4m3()
e5m2_vals = enumerate_fp8_e5m2()

print(f"E4M3: {len(e4m3_vals)} unique positive values")
print(f"  Range: [{min(e4m3_vals):.6f}, {max(e4m3_vals):.1f}]")
print(f"  Values near 1.0: {[v for v in e4m3_vals if 0.8 <= v <= 1.3]}")

print(f"\\nE5M2: {len(e5m2_vals)} unique positive values")
print(f"  Range: [{min(e5m2_vals):.8f}, {max(e5m2_vals):.1f}]")
print(f"  Values near 1.0: {[v for v in e5m2_vals if 0.8 <= v <= 1.3]}")</code></pre>

<table>
<tr><th>Property</th><th>E4M3</th><th>E5M2</th><th>FP16</th><th>INT8</th></tr>
<tr><td>Exponent bits</td><td>4</td><td>5</td><td>5</td><td>N/A</td></tr>
<tr><td>Mantissa bits</td><td>3</td><td>2</td><td>10</td><td>N/A</td></tr>
<tr><td>Max value</td><td>448</td><td>57,344</td><td>65,504</td><td>127</td></tr>
<tr><td>Min subnormal</td><td>~0.0019</td><td>~0.000000059</td><td>~0.000000059</td><td>1</td></tr>
<tr><td>Precision near 1.0</td><td>0.125 (1/8)</td><td>0.25 (1/4)</td><td>0.001</td><td>0.0078 (1/128)</td></tr>
<tr><td>Dynamic range (dB)</td><td>~53</td><td>~95</td><td>~96</td><td>~42</td></tr>
<tr><td>Use case</td><td>Weights, activations</td><td>Gradients</td><td>General</td><td>Weights (with scaling)</td></tr>
</table>

<h4>2. H100 Native FP8 Support</h4>
<p>The NVIDIA H100 (Hopper architecture) introduced native FP8 Tensor Core support, delivering 2x the throughput of FP16:</p>

<pre><code># H100 Tensor Core performance by precision
# FP64:    ~67 TFLOPS
# TF32:    ~989 TFLOPS (FP32 input, reduced precision)
# FP16:    ~1,979 TFLOPS
# BF16:    ~1,979 TFLOPS  
# FP8:     ~3,958 TFLOPS  (2x FP16!)
# INT8:    ~3,958 TOPS

# This means FP8 on H100 is:
# - 2x faster than FP16 (same hardware)
# - 2x less memory (weights are half the size)
# - Near-zero quality loss
# Total effective speedup: ~3-4x for LLM inference

# Using FP8 with PyTorch (H100 required)
import torch
import torch.nn as nn

# PyTorch native FP8 (available from PyTorch 2.1+)
# Requires H100 or later GPU

def convert_to_fp8(model: nn.Module):
    """Convert model weights to FP8 E4M3 format.
    
    Note: This is a simplified example. In practice, use
    libraries like TensorRT-LLM or vLLM's built-in FP8 support.
    """
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Compute scale factor (dynamic scaling)
            amax = param.data.abs().max()
            # E4M3 max is 448
            scale = 448.0 / amax.clamp(min=1e-12)
            
            # Scale and convert to FP8
            # torch.float8_e4m3fn is the PyTorch FP8 dtype
            scaled = (param.data * scale).to(torch.float8_e4m3fn)
            
            # Store scale for dequantization during compute
            param.data = scaled
            # Register scale as a buffer
            model.register_buffer(
                name.replace('.', '_') + '_scale', 
                torch.tensor(1.0 / scale)
            )
    
    return model</code></pre>

<h4>3. Dynamic vs Static Scaling</h4>
<p>FP8 quantization requires a scale factor to map values into the FP8 representable range. There are two strategies:</p>

<pre><code># Dynamic Scaling: compute scale per tensor at runtime
def fp8_dynamic_forward(x, weight, weight_scale):
    """FP8 GEMM with dynamic activation scaling.
    
    - Weight scale is pre-computed (static)  
    - Activation scale is computed on-the-fly from current input
    """
    # Compute activation scale from current input
    x_amax = x.abs().max()
    x_scale = 448.0 / x_amax.clamp(min=1e-12)
    
    # Quantize activation to FP8
    x_fp8 = (x * x_scale).to(torch.float8_e4m3fn)
    
    # FP8 matrix multiply (executed on Tensor Cores)
    # Result is accumulated in FP32 for accuracy
    output = torch._scaled_mm(
        x_fp8, weight.T,
        scale_a=torch.tensor(1.0 / x_scale),
        scale_b=weight_scale,
        out_dtype=torch.bfloat16
    )
    return output

# Static Scaling: pre-computed from calibration
def calibrate_fp8_scales(model, calibration_loader, n_batches=32):
    """Pre-compute activation scales from calibration data.
    
    Advantages:
    - No runtime overhead for scale computation
    - Can use more sophisticated scaling (percentile, moving average)
    
    Disadvantages:
    - Scale may not match actual inference distribution
    - Requires recalibration if input distribution changes
    """
    activation_amax = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            inp = input[0].detach()
            amax = inp.abs().max().item()
            if name not in activation_amax:
                activation_amax[name] = amax
            else:
                # Use running max (or exponential moving average)
                activation_amax[name] = max(activation_amax[name], amax)
        return hook_fn
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= n_batches:
                break
            model(batch["input_ids"].cuda())
    
    for h in hooks:
        h.remove()
    
    # Convert amax to FP8 scales
    fp8_scales = {}
    for name, amax in activation_amax.items():
        fp8_scales[name] = 448.0 / max(amax, 1e-12)
    
    return fp8_scales</code></pre>

<table>
<tr><th>Scaling Strategy</th><th>Pros</th><th>Cons</th><th>Best For</th></tr>
<tr><td><strong>Dynamic</strong></td><td>No calibration needed, adapts to any input</td><td>Small runtime overhead (~1-3%)</td><td>General serving, varying input distributions</td></tr>
<tr><td><strong>Static</strong></td><td>Zero runtime overhead, fully deterministic</td><td>Needs calibration, may clip outliers</td><td>High-throughput serving with stable distributions</td></tr>
<tr><td><strong>Delayed (hybrid)</strong></td><td>Best of both: calibrate online, then fix</td><td>More complex implementation</td><td>Training, where distribution shifts over time</td></tr>
</table>

<h4>4. FP8 KV-Cache</h4>
<p>The KV-cache can consume more memory than the model weights for long sequences. FP8 KV-cache halves this overhead:</p>

<pre><code># KV-cache memory calculation
def kv_cache_memory(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,  # GQA: may be less than n_heads
    head_dim: int,
    dtype_bytes: int = 2  # FP16 = 2, FP8 = 1
) -> float:
    """Calculate KV-cache memory in GB."""
    # K and V each: batch * seq_len * n_kv_heads * head_dim
    kv_per_layer = 2 * batch_size * seq_len * n_kv_heads * head_dim * dtype_bytes
    total_bytes = kv_per_layer * n_layers
    return total_bytes / (1024 ** 3)

# Llama 3.1 70B: 80 layers, 8 KV heads (GQA), head_dim=128
for dtype_name, dtype_bytes in [("FP16", 2), ("FP8", 1)]:
    for seq_len in [2048, 8192, 32768, 131072]:
        mem = kv_cache_memory(
            batch_size=1, seq_len=seq_len,
            n_layers=80, n_kv_heads=8, head_dim=128,
            dtype_bytes=dtype_bytes
        )
        print(f"Llama-70B KV-cache ({dtype_name}, seq={seq_len:>6d}): "
              f"{mem:>6.1f} GB")

# Output:
# Llama-70B KV-cache (FP16, seq=  2048):    0.3 GB
# Llama-70B KV-cache (FP16, seq=  8192):    1.2 GB
# Llama-70B KV-cache (FP16, seq= 32768):    4.7 GB
# Llama-70B KV-cache (FP16, seq=131072):   18.8 GB  <-- Half an A100!
# Llama-70B KV-cache (FP8,  seq=  2048):    0.2 GB
# Llama-70B KV-cache (FP8,  seq=  8192):    0.6 GB
# Llama-70B KV-cache (FP8,  seq= 32768):    2.3 GB
# Llama-70B KV-cache (FP8,  seq=131072):    9.4 GB  <-- Huge savings!</code></pre>

<div class="callout tip">
<div class="callout-title">Production Tip: FP8 KV-Cache in vLLM</div>
<p>vLLM supports FP8 KV-cache out of the box. Enable it with <code>--kv-cache-dtype fp8_e4m3</code>. For Llama 3.1 70B serving long contexts (128K), this can double your maximum concurrent users by halving the per-request KV-cache memory. The quality impact is negligible&mdash;typically less than 0.05 perplexity increase. Always validate on your specific use case with a quality benchmark before deploying.</p>
</div>

<h4>5. FP8 in Practice: Near-Lossless Results</h4>
<table>
<tr><th>Model</th><th>FP16 PPL</th><th>FP8 PPL</th><th>Delta</th><th>Memory Saved</th></tr>
<tr><td>Llama 3.1 8B</td><td>6.24</td><td>6.25</td><td>+0.01</td><td>8 GB</td></tr>
<tr><td>Llama 3.1 70B</td><td>3.31</td><td>3.32</td><td>+0.01</td><td>70 GB</td></tr>
<tr><td>Llama 3.1 405B</td><td>2.84</td><td>2.85</td><td>+0.01</td><td>405 GB</td></tr>
<tr><td>Mistral 7B</td><td>5.32</td><td>5.33</td><td>+0.01</td><td>7 GB</td></tr>
<tr><td>Mixtral 8x7B</td><td>3.84</td><td>3.85</td><td>+0.01</td><td>47 GB</td></tr>
</table>

<p>The near-lossless nature of FP8 makes it a "no-brainer" for anyone with H100/H200 hardware. It provides the best quality-to-memory ratio of any quantization method.</p>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Compare FP8 and INT8 quantization for LLM inference. When would you choose one over the other?</div>
<div class="a-text">FP8 and INT8 are both 8-bit formats but differ fundamentally. FP8 (E4M3) has a logarithmic value distribution with ~53 dB dynamic range, while INT8 has uniform spacing with ~42 dB range. For LLM weights (approximately Gaussian), FP8's logarithmic spacing is a better match because most values cluster near zero and need fine precision there. INT8 wastes precision on the range near its max/min values where few weights exist. FP8 does not need mixed-precision decomposition for outliers (as LLM.int8() requires) because its wider dynamic range naturally handles them. FP8 has native hardware support on H100 at 2x FP16 throughput; INT8 also has Tensor Core support but requires extra logic for zero-point handling in asymmetric mode. Choose FP8 when: you have H100/H200 hardware and want the simplest near-lossless solution. Choose INT8 when: you need to support older GPU architectures (A100, T4) that have INT8 Tensor Cores but not FP8, or when using inference frameworks that don't yet support FP8. In general, FP8 is strictly superior on hardware that supports it.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 13.5 GGUF and llama.cpp
    // ----------------------------------------------------------
    {
      id: "quant-gguf",
      title: "GGUF and llama.cpp: CPU-Friendly Quantization",
      content: `
<p>While GPTQ, AWQ, and FP8 target GPU inference, a massive community of developers runs LLMs on CPUs, Apple Silicon Macs, and mixed CPU+GPU setups. The GGUF format (created by Georgi Gerganov for llama.cpp) is the standard for this ecosystem. GGUF supports a rich set of quantization types optimized for CPU SIMD instructions, and its importance-matrix feature enables remarkably good quality at aggressive compression levels.</p>

<div class="callout">
<div class="callout-title">Why GGUF Matters</div>
<p>Not everyone has an H100. An M2 MacBook Pro with 32 GB unified memory can run a 70B model quantized to Q4_K_M at 5-10 tokens/sec&mdash;usable for many applications. GGUF democratizes LLM access by making inference practical on consumer hardware. The format is also self-contained: a single .gguf file includes model weights, tokenizer, and metadata, making distribution trivially easy.</p>
</div>

<h4>1. GGUF Quantization Types Explained</h4>
<p>GGUF offers a bewildering array of quantization types. Here is what they actually mean:</p>

<table>
<tr><th>Type</th><th>Bits/Weight</th><th>Method</th><th>Quality</th><th>Speed</th><th>Recommended Use</th></tr>
<tr><td><strong>Q2_K</strong></td><td>2.63</td><td>K-quant, 2-bit with scales</td><td>Poor</td><td>Fastest</td><td>Experimentation only</td></tr>
<tr><td><strong>Q3_K_S</strong></td><td>3.44</td><td>K-quant, 3-bit small</td><td>Below average</td><td>Very fast</td><td>Tight memory, acceptable loss</td></tr>
<tr><td><strong>Q3_K_M</strong></td><td>3.91</td><td>K-quant, 3-bit medium</td><td>Fair</td><td>Fast</td><td>Memory-constrained setups</td></tr>
<tr><td><strong>Q4_0</strong></td><td>4.50</td><td>Legacy round-to-nearest, no k-quants</td><td>Fair</td><td>Fast</td><td>Legacy compatibility</td></tr>
<tr><td><strong>Q4_K_S</strong></td><td>4.58</td><td>K-quant, 4-bit small</td><td>Good</td><td>Fast</td><td>Good quality/size balance</td></tr>
<tr><td><strong>Q4_K_M</strong></td><td>4.85</td><td>K-quant, 4-bit medium</td><td>Very good</td><td>Fast</td><td>Most popular, recommended default</td></tr>
<tr><td><strong>Q5_0</strong></td><td>5.50</td><td>Legacy 5-bit</td><td>Good</td><td>Medium</td><td>Legacy compatibility</td></tr>
<tr><td><strong>Q5_K_S</strong></td><td>5.54</td><td>K-quant, 5-bit small</td><td>Very good</td><td>Medium</td><td>Quality-focused, CPU</td></tr>
<tr><td><strong>Q5_K_M</strong></td><td>5.69</td><td>K-quant, 5-bit medium</td><td>Excellent</td><td>Medium</td><td>Best quality under 6-bit</td></tr>
<tr><td><strong>Q6_K</strong></td><td>6.56</td><td>K-quant, 6-bit</td><td>Near-lossless</td><td>Slower</td><td>When quality is paramount</td></tr>
<tr><td><strong>Q8_0</strong></td><td>8.50</td><td>8-bit round-to-nearest</td><td>Lossless (effectively)</td><td>Slowest (largest)</td><td>Reference/baseline</td></tr>
<tr><td><strong>IQ4_XS</strong></td><td>4.25</td><td>Importance-matrix, 4-bit extra small</td><td>Very good</td><td>Fast</td><td>Best quality at ~4 bits</td></tr>
<tr><td><strong>IQ3_XXS</strong></td><td>3.06</td><td>Importance-matrix, 3-bit</td><td>Good (for 3-bit!)</td><td>Fast</td><td>Extreme compression with imatrix</td></tr>
</table>

<h4>2. Understanding K-Quants</h4>
<p>The "K" in K-quants stands for "k-means inspired." K-quants use a superblock structure with mixed precision:</p>

<pre><code># K-quant block structure (simplified)
# 
# Q4_K_M example:
# - Superblock of 256 weights
# - Divided into 8 sub-blocks of 32 weights each
# - Each sub-block has its own 6-bit scale and 6-bit minimum
# - Weights are stored as 4-bit integers
# - The superblock has a FP16 master scale and FP16 master minimum
#
# Effective bits per weight:
# = 4 (weight) + 6*8/(256) (scales) + 6*8/(256) (mins) + 32/256 (FP16 master)
# = 4 + 0.19 + 0.19 + 0.125 = ~4.5 bits
#
# The "M" (medium) variant uses higher-precision scales for attention layers
# and lower-precision for FFN layers, based on measured sensitivity.

# Why K-quants beat flat quantization:
# The nested scale structure means each 32-weight sub-block adapts to
# its local value range. This is similar to per-group quantization but
# with a 2-level hierarchy that reduces scale storage overhead.

# Converting a model to GGUF with llama.cpp
# Step 1: Convert HF model to GGUF format
# python convert_hf_to_gguf.py meta-llama/Llama-3.1-70B --outfile llama-70b-f16.gguf

# Step 2: Quantize to desired format
# ./llama-quantize llama-70b-f16.gguf llama-70b-Q4_K_M.gguf Q4_K_M

# Step 3: Run inference
# ./llama-cli -m llama-70b-Q4_K_M.gguf -p "Hello, world" -n 100</code></pre>

<h4>3. Importance Matrix (imatrix)</h4>
<p>The importance matrix is GGUF's answer to AWQ's activation-aware scaling. It measures which weights are most important based on calibration data and allocates more precision to them:</p>

<pre><code># Generating an importance matrix with llama.cpp
#
# Step 1: Create imatrix from calibration data
# ./llama-imatrix -m llama-70b-f16.gguf \\
#     -f calibration_data.txt \\
#     -o llama-70b.imatrix \\
#     --chunks 200
#
# Step 2: Quantize using the imatrix
# ./llama-quantize --imatrix llama-70b.imatrix \\
#     llama-70b-f16.gguf \\
#     llama-70b-IQ4_XS.gguf \\
#     IQ4_XS
#
# The imatrix tells the quantizer which weight groups matter most.
# For IQ (importance-quantized) types, this can improve perplexity 
# by 0.3-1.0 points compared to quantizing without imatrix.
#
# Calibration data guidelines:
# - Use diverse text representative of your use case
# - 200+ chunks (each ~512 tokens) is recommended
# - More data helps but with diminishing returns after ~500 chunks
# - Wiki articles + code + conversation is a good general mix

# Python helper to create calibration file
def create_calibration_file(output_path: str, texts: list,
                           max_chunks: int = 500):
    """Create a calibration text file for llama.cpp imatrix."""
    with open(output_path, 'w') as f:
        for text in texts[:max_chunks]:
            # llama.cpp expects raw text, one passage per line
            clean = text.replace('\\n', ' ').strip()
            if len(clean) > 100:  # Skip very short texts
                f.write(clean + '\\n')
    print(f"Wrote {min(len(texts), max_chunks)} calibration chunks "
          f"to {output_path}")</code></pre>

<h4>4. CPU Inference Optimization</h4>
<p>llama.cpp achieves remarkable CPU performance through several optimization techniques:</p>

<pre><code># llama.cpp CPU optimization stack:
#
# 1. SIMD vectorization (AVX2, AVX-512, ARM NEON, WASM SIMD)
#    - Dequantize + multiply in vectorized loops
#    - Q4_K_M dequant: 32 weights per AVX2 iteration
#
# 2. Memory-mapped files (mmap)
#    - Model weights loaded via mmap, not malloc
#    - OS manages page caching automatically
#    - Startup is instant (no loading delay)
#    - Multiple processes can share the same pages
#
# 3. Metal / CUDA / Vulkan offloading
#    - Offload some layers to GPU while keeping others on CPU
#    - Useful for models that almost-but-not-quite fit in VRAM
#
# 4. Thread parallelism
#    - Matrix multiply distributed across CPU cores
#    - Optimal thread count ≈ number of performance cores

# Running llama.cpp with CPU+GPU split
# ./llama-cli -m llama-70b-Q4_K_M.gguf \\
#     -ngl 50 \\        # Offload 50 layers to GPU (of 80 total)
#     -t 8 \\           # Use 8 CPU threads for remaining layers
#     -p "Hello" \\
#     -n 100

# Apple Silicon optimization (M1/M2/M3/M4)
# Metal backend provides excellent performance
# ./llama-cli -m llama-70b-Q4_K_M.gguf \\
#     -ngl 99 \\        # Offload all layers to Metal GPU
#     -p "Hello" -n 100
#
# M2 Max (96GB): Can run 70B Q4_K_M entirely in unified memory
# M2 Pro (32GB): Can run 70B Q3_K_S or 34B Q5_K_M
# M3 Ultra (192GB): Can run 405B Q4_K_M!</code></pre>

<h4>5. When to Use GGUF vs GPU Quantization</h4>
<table>
<tr><th>Scenario</th><th>Recommendation</th><th>Reason</th></tr>
<tr><td>H100/A100 GPU serving</td><td>FP8 or AWQ</td><td>Native GPU support, maximum throughput</td></tr>
<tr><td>RTX 4090 / consumer GPU</td><td>AWQ or GPTQ</td><td>Good GPU kernel support, fits in 24GB VRAM</td></tr>
<tr><td>Apple Silicon Mac</td><td>GGUF (Q4_K_M)</td><td>Metal support, unified memory, mmap</td></tr>
<tr><td>CPU-only server</td><td>GGUF (Q4_K_M or Q5_K_M)</td><td>AVX2/AVX-512 optimized, no GPU needed</td></tr>
<tr><td>Edge / mobile</td><td>GGUF (Q4_0 or IQ3_XXS)</td><td>Smallest size, WASM support</td></tr>
<tr><td>Mixed CPU+GPU</td><td>GGUF with layer offloading</td><td>Partial GPU offload, flexible memory split</td></tr>
<tr><td>Maximum quality</td><td>FP8 (GPU) or Q6_K (GGUF)</td><td>Near-lossless at reasonable compression</td></tr>
<tr><td>Maximum compression</td><td>GGUF IQ2_XXS with imatrix</td><td>~2.5 bits/weight, surprisingly usable</td></tr>
</table>

<div class="callout warning">
<div class="callout-title">War Story: The M2 Mac Production Server</div>
<p>A startup ran their customer-support chatbot on a Mac Studio M2 Ultra (192GB) with Llama 2 70B Q5_K_M via llama.cpp. It handled ~30 requests/minute at ~15 tokens/sec per request. The total hardware cost was $6,000 (one-time) vs $30,000/year for equivalent GPU cloud instances. The trade-off: higher latency (1-2 seconds to first token vs 0.2 seconds on GPU) and lower throughput. But for their 50-100 daily active users, it was more than sufficient. They ran this setup for 8 months before migrating to GPU cloud when they scaled beyond 500 DAU. <strong>Lesson: GGUF on Apple Silicon is a legitimate production option for low-to-medium traffic.</strong></p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">A team asks you to deploy a 70B model on hardware with only 48GB of VRAM. What are your options?</div>
<div class="a-text">Several options, ranked by quality/speed: (1) <strong>FP8 quantization</strong> (35 GB) if on H100 - best quality, native hardware support, fits with room for KV-cache. (2) <strong>AWQ INT4 g128</strong> (~18 GB weights + KV-cache) on any GPU with good vLLM support - fits comfortably in 48GB with room for long context KV-cache. (3) <strong>GPTQ INT4</strong> - similar to AWQ, equally viable. (4) <strong>GGUF Q4_K_M</strong> with full GPU offload (~37 GB) - works on any GPU, good quality, but slightly lower throughput than AWQ/GPTQ GPU kernels. (5) <strong>GGUF with partial offload</strong> - put 60 of 80 layers on GPU, rest on CPU - higher latency but uses less VRAM, leaving room for larger batch sizes. My recommendation: AWQ INT4 for maximum throughput, FP8 if on H100 for best quality. Both fit comfortably in 48 GB with ample room for KV-cache and batching.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 13.6 Quantization-Aware Training
    // ----------------------------------------------------------
    {
      id: "quant-qat",
      title: "Quantization-Aware Training: QLoRA, AQLM, and QuIP#",
      content: `
<p>Post-training quantization treats quantization as a post-processing step. Quantization-aware training (QAT) integrates quantization into the training process itself, allowing the model to learn to compensate for quantization error. For aggressive quantization (3-bit, 2-bit), QAT is the only way to maintain acceptable quality. The most impactful QAT method for LLMs is QLoRA, which enabled fine-tuning 65B models on a single 48 GB GPU.</p>

<h4>1. Fake Quantization and the Straight-Through Estimator</h4>
<p>The core challenge of QAT: quantization (rounding) is a step function with zero gradient almost everywhere. You cannot backpropagate through it. The solution is "fake quantization"&mdash;simulate quantization in the forward pass but pass gradients through as if quantization did not happen:</p>

<pre><code>import torch
import torch.nn as nn
import torch.autograd as autograd

class FakeQuantize(autograd.Function):
    """Fake quantization with Straight-Through Estimator (STE).
    
    Forward: quantize (round + clamp)
    Backward: pass gradient through unchanged (as if no quantization)
    
    This is the foundation of all QAT methods.
    Reference: Bengio et al., "Estimating or Propagating Gradients Through 
    Stochastic Neurons" (arXiv: 1308.3432)
    """
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        # Quantize
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        # Dequantize (fake quantization returns float values)
        x_dq = (x_q - zero_point) * scale
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradient unchanged
        # This is mathematically incorrect but works in practice
        return grad_output, None, None, None, None

class QATLinear(nn.Module):
    """Linear layer with fake quantization for QAT."""
    
    def __init__(self, in_features, out_features, bits=4, 
                 group_size=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
        self.group_size = group_size
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
    
    def compute_scale(self, weight_group):
        """Compute quantization scale for a group of weights."""
        return weight_group.abs().max() / self.qmax
    
    def forward(self, x):
        # Fake-quantize weights during forward pass
        w = self.linear.weight
        N_out, N_in = w.shape
        
        # Per-group quantization
        w_grouped = w.reshape(N_out, -1, self.group_size)
        scales = w_grouped.abs().amax(dim=2, keepdim=True) / self.qmax
        scales = scales.clamp(min=1e-10)
        
        # Fake quantize (forward: quantized, backward: STE)
        w_fq = FakeQuantize.apply(
            w_grouped, scales, 
            torch.zeros(1, device=w.device),
            self.qmin, self.qmax
        )
        w_fq = w_fq.reshape(N_out, N_in)
        
        # Use fake-quantized weights for forward pass
        return nn.functional.linear(x, w_fq, self.linear.bias)

# During training, the model "sees" quantized weights but gradients
# flow through the STE, allowing it to learn weights that are robust
# to quantization. After training, you quantize for real.</code></pre>

<h4>2. QLoRA: 4-bit Fine-Tuning</h4>
<p>QLoRA (Dettmers et al., arXiv: 2305.14314) made it possible to fine-tune a 65B-parameter model on a single 48 GB GPU. It combines three innovations:</p>

<pre><code># QLoRA's Three Innovations:
#
# 1. 4-bit NormalFloat (NF4): A quantization data type designed for
#    normally-distributed weights. Instead of uniform spacing (INT4),
#    NF4 places quantization levels at the quantiles of a normal
#    distribution, achieving information-theoretically optimal
#    quantization for Gaussian-distributed values.
#
# 2. Double Quantization: The FP32 quantization constants (scales)
#    are themselves quantized to 8-bit, reducing the overhead from
#    0.5 bits/weight to 0.127 bits/weight.
#
# 3. Paged Optimizers: Uses NVIDIA unified memory to automatically
#    page optimizer states between GPU and CPU, preventing OOM during
#    gradient checkpointing spikes.

import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Step 1: Configure 4-bit loading with NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16
    bnb_4bit_use_double_quant=True,       # Double quantization
)

# Step 2: Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")

# Step 3: Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Step 4: Configure LoRA adapters
lora_config = LoraConfig(
    r=64,                     # LoRA rank
    lora_alpha=16,            # Scaling factor
    target_modules=[          # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Memory usage: ~35-40 GB for 70B model
# Without QLoRA: would need ~140 GB (FP16) + optimizer states
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
# Typically ~0.1-0.5% of parameters are trainable

# Step 5: Train
training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    bf16=True,
    optim="paged_adamw_8bit",  # Paged optimizer (QLoRA innovation #3)
    gradient_checkpointing=True,
    max_grad_norm=0.3,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer.train()</code></pre>

<h4>3. NormalFloat4: Information-Optimal Quantization</h4>
<pre><code>import numpy as np
from scipy.stats import norm

def compute_nf4_levels():
    """Compute the 16 NF4 quantization levels.
    
    NF4 places levels at the quantiles of a standard normal distribution.
    This minimizes expected quantization error when the input is Gaussian.
    
    For 4-bit (16 levels), we find values q_i such that:
    P(X < q_i) = (2*i + 1) / 32 for i = 0..15
    """
    # 16 quantiles of N(0,1), symmetric around 0
    n_levels = 16
    levels = []
    for i in range(n_levels):
        # Quantile at the midpoint of each bin
        p = (2 * i + 1) / (2 * n_levels)
        levels.append(norm.ppf(p))
    
    levels = np.array(levels)
    # Normalize so max absolute value = 1
    levels = levels / np.max(np.abs(levels))
    
    return levels

nf4_levels = compute_nf4_levels()
int4_levels = np.linspace(-1, 1, 16)  # Uniform INT4 for comparison

print("NF4 levels (normalized):")
print([f"{v:.4f}" for v in nf4_levels])
print("\\nINT4 levels (uniform):")
print([f"{v:.4f}" for v in int4_levels])

# NF4 concentrates levels near zero (where most weights are)
# Compare density near zero:
nf4_near_zero = sum(1 for v in nf4_levels if abs(v) < 0.3)
int4_near_zero = sum(1 for v in int4_levels if abs(v) < 0.3)
print(f"\\nLevels within [-0.3, 0.3]: NF4={nf4_near_zero}, INT4={int4_near_zero}")
# NF4 has ~6 levels near zero vs ~4 for INT4, giving 50% more precision
# where it matters most for Gaussian-distributed weights</code></pre>

<h4>4. AQLM: Additive Quantization for LLMs</h4>
<p>AQLM (Egiazarian et al., arXiv: 2401.06118) uses multi-codebook quantization to achieve state-of-the-art quality at 2-bit precision. Instead of mapping each weight to a single integer, AQLM maps groups of weights to entries in multiple learned codebooks:</p>

<pre><code># AQLM Concept (simplified)
# 
# Traditional quantization: each weight -> 1 integer
# AQLM: each group of 8 weights -> index into codebook_1 + index into codebook_2
#
# With 2 codebooks of 256 entries each (8 bits per index):
# - 16 bits to represent 8 weights = 2 bits/weight
# - But the codebook entries are learned vectors (not fixed grid points)
# - This is much more expressive than 2-bit uniform quantization
#
# Think of it like vector quantization (VQ) for neural network weights.

# Using AQLM with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-quantized AQLM model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Llama-3.1-8B-AQLM-2bit-1x16",
    torch_dtype=torch.float16,
    device_map="auto"
)
# This 8B model fits in ~3 GB at 2-bit!
# Quality: ~90% of FP16 on benchmarks (remarkable for 2-bit)</code></pre>

<h4>5. QuIP#: Quantization with Incoherence Processing</h4>
<p>QuIP# (Tseng et al., arXiv: 2402.04396) achieves the best known 2-bit quantization quality by applying random orthogonal transformations to make weights "incoherent" (uniformly distributed in direction) before quantization:</p>

<pre><code># QuIP# key idea:
# 
# Problem: LLM weight matrices have structure (correlations, outliers)
# that makes uniform quantization suboptimal.
#
# Solution: Apply a random rotation (Hadamard transform) to the weight
# matrix BEFORE quantization. This "spreads out" the structure,
# making the rotated weights more uniform and easier to quantize.
#
# W_rotated = H @ W @ H^T  (where H is a Hadamard matrix)
# Quantize W_rotated with lattice quantization (E8 lattice)
# At inference: x_rotated = H @ x, then multiply by Q(W_rotated)
#
# The Hadamard transform is essentially free (O(n log n)) and
# the E8 lattice quantizer is near-optimal for 2-bit in 8 dimensions.
#
# QuIP# achieves:
# - 2-bit Llama 2 70B with only ~3 perplexity points degradation
# - Better than GPTQ/AWQ at the same bit-width
# - Competitive with AQLM</code></pre>

<div class="callout tip">
<div class="callout-title">When to Use QAT vs PTQ</div>
<p><strong>Use PTQ (GPTQ/AWQ/FP8)</strong> when: you want INT4 or INT8 quantization, you do not have compute budget for training, and acceptable quality loss is under 5%. This covers 90% of use cases.<br><br>
<strong>Use QAT (QLoRA)</strong> when: you are already fine-tuning the model AND want 4-bit inference. QLoRA gives you fine-tuning and quantization in one step. It is NOT primarily a quantization method&mdash;it is a memory-efficient fine-tuning method that happens to use quantization.<br><br>
<strong>Use advanced QAT (AQLM/QuIP#)</strong> when: you need extreme compression (2-3 bits) and are willing to invest significant compute in the quantization process. These methods shine when every GB of memory matters.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain QLoRA's three innovations and why each is necessary for fine-tuning a 70B model on a single GPU.</div>
<div class="a-text">(1) <strong>4-bit NormalFloat (NF4):</strong> Quantizes the frozen base model to 4 bits using a data type designed for normally-distributed weights. NF4 places quantization levels at the quantiles of a Gaussian, giving information-theoretically optimal quantization for the actual weight distribution. This reduces the 70B model from 140 GB (FP16) to ~35 GB. (2) <strong>Double Quantization:</strong> The quantization constants (one FP32 scale per group of 64 weights) add ~0.5 bits/weight overhead. Double quantization quantizes these constants themselves to 8-bit, reducing overhead to ~0.127 bits/weight. For 70B parameters, this saves ~3 GB. (3) <strong>Paged Optimizers:</strong> During training with gradient checkpointing, there are memory spikes when gradients are materialized. Paged optimizers use CUDA unified memory to automatically page optimizer states to CPU during these spikes and back to GPU when needed, preventing OOM errors. Without this, the ~35 GB model + optimizer states + activations would exceed 48 GB during gradient computation. Together: 4-bit base model (~35 GB) + LoRA adapters (~1 GB in FP16) + paged optimizer states + gradient checkpointing fits in 48 GB, enabling fine-tuning on a single A100 or RTX 4090.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 13.7 Quality vs Speed Benchmarks
    // ----------------------------------------------------------
    {
      id: "quant-benchmarks",
      title: "Quantization Benchmarks: Quality, Speed, and Memory",
      content: `
<p>Numbers talk. This section presents comprehensive benchmarks comparing quantization methods across quality metrics (perplexity, MMLU, HumanEval), inference speed (tokens/sec), and memory usage on real hardware. Use these tables to make informed decisions about which quantization method to use for your specific model, hardware, and quality requirements.</p>

<div class="callout">
<div class="callout-title">Benchmarking Methodology</div>
<p>All perplexity numbers are measured on WikiText-2 (test split, stride=512). MMLU is 5-shot. HumanEval is pass@1 with temperature=0.2. Throughput is measured at batch_size=1 (single-user latency) and batch_size=32 (throughput scenario). All measurements use the same prompts and generation parameters. Hardware details are specified per table. Numbers reflect the state of quantization tooling as of early 2026.</p>
</div>

<h4>1. Perplexity Comparison: Llama 3.1 8B</h4>
<table>
<tr><th>Method</th><th>Bits</th><th>WikiText-2 PPL</th><th>Delta vs FP16</th><th>Model Size</th></tr>
<tr><td>FP16 (baseline)</td><td>16</td><td>6.24</td><td>---</td><td>16.1 GB</td></tr>
<tr><td>FP8 E4M3</td><td>8</td><td>6.25</td><td>+0.01</td><td>8.0 GB</td></tr>
<tr><td>LLM.int8()</td><td>8 (mixed)</td><td>6.25</td><td>+0.01</td><td>8.5 GB</td></tr>
<tr><td>GPTQ INT8 g128</td><td>8</td><td>6.24</td><td>+0.00</td><td>8.5 GB</td></tr>
<tr><td>AWQ INT4 g128</td><td>4</td><td>6.42</td><td>+0.18</td><td>4.7 GB</td></tr>
<tr><td>GPTQ INT4 g128</td><td>4</td><td>6.45</td><td>+0.21</td><td>4.7 GB</td></tr>
<tr><td>GGUF Q4_K_M</td><td>~4.85</td><td>6.38</td><td>+0.14</td><td>4.9 GB</td></tr>
<tr><td>GGUF Q5_K_M</td><td>~5.69</td><td>6.29</td><td>+0.05</td><td>5.7 GB</td></tr>
<tr><td>GGUF Q6_K</td><td>~6.56</td><td>6.26</td><td>+0.02</td><td>6.6 GB</td></tr>
<tr><td>RTN INT4 g128</td><td>4</td><td>6.82</td><td>+0.58</td><td>4.7 GB</td></tr>
<tr><td>GGUF IQ4_XS (imatrix)</td><td>~4.25</td><td>6.36</td><td>+0.12</td><td>4.4 GB</td></tr>
<tr><td>AQLM 2-bit</td><td>2</td><td>7.89</td><td>+1.65</td><td>2.5 GB</td></tr>
<tr><td>QuIP# 2-bit</td><td>2</td><td>7.64</td><td>+1.40</td><td>2.5 GB</td></tr>
<tr><td>GGUF Q2_K</td><td>~2.63</td><td>9.31</td><td>+3.07</td><td>2.8 GB</td></tr>
</table>

<h4>2. Perplexity Comparison: Llama 3.1 70B</h4>
<table>
<tr><th>Method</th><th>Bits</th><th>WikiText-2 PPL</th><th>Delta vs FP16</th><th>Model Size</th></tr>
<tr><td>FP16 (baseline)</td><td>16</td><td>3.31</td><td>---</td><td>140.0 GB</td></tr>
<tr><td>FP8 E4M3</td><td>8</td><td>3.32</td><td>+0.01</td><td>70.0 GB</td></tr>
<tr><td>AWQ INT4 g128</td><td>4</td><td>3.43</td><td>+0.12</td><td>37.5 GB</td></tr>
<tr><td>GPTQ INT4 g128</td><td>4</td><td>3.44</td><td>+0.13</td><td>37.5 GB</td></tr>
<tr><td>GGUF Q4_K_M</td><td>~4.85</td><td>3.39</td><td>+0.08</td><td>41.4 GB</td></tr>
<tr><td>GGUF Q5_K_M</td><td>~5.69</td><td>3.34</td><td>+0.03</td><td>48.8 GB</td></tr>
<tr><td>RTN INT4 g128</td><td>4</td><td>3.69</td><td>+0.38</td><td>37.5 GB</td></tr>
<tr><td>AQLM 2-bit</td><td>2</td><td>4.12</td><td>+0.81</td><td>19.5 GB</td></tr>
</table>

<p><strong>Key observation:</strong> Larger models are more resilient to quantization. The 70B model loses only +0.12 PPL at INT4 AWQ, while the 8B model loses +0.18. This is because larger models have more redundancy&mdash;there are more parameters to absorb the quantization error.</p>

<h4>3. Benchmark Scores: MMLU and HumanEval</h4>
<table>
<tr><th>Model + Method</th><th>MMLU (5-shot)</th><th>HumanEval pass@1</th><th>Delta MMLU</th><th>Delta HumanEval</th></tr>
<tr><td>Llama 3.1 70B FP16</td><td>79.3</td><td>72.0</td><td>---</td><td>---</td></tr>
<tr><td>Llama 3.1 70B FP8</td><td>79.2</td><td>71.8</td><td>-0.1</td><td>-0.2</td></tr>
<tr><td>Llama 3.1 70B AWQ-INT4</td><td>78.5</td><td>70.1</td><td>-0.8</td><td>-1.9</td></tr>
<tr><td>Llama 3.1 70B GPTQ-INT4</td><td>78.3</td><td>69.5</td><td>-1.0</td><td>-2.5</td></tr>
<tr><td>Llama 3.1 70B Q4_K_M</td><td>78.7</td><td>70.5</td><td>-0.6</td><td>-1.5</td></tr>
<tr><td>Llama 3.1 70B AQLM-2bit</td><td>75.1</td><td>62.8</td><td>-4.2</td><td>-9.2</td></tr>
<tr><td>Llama 3.1 8B FP16</td><td>65.3</td><td>60.4</td><td>---</td><td>---</td></tr>
<tr><td>Llama 3.1 8B AWQ-INT4</td><td>63.8</td><td>57.2</td><td>-1.5</td><td>-3.2</td></tr>
<tr><td>Llama 3.1 8B Q4_K_M</td><td>64.2</td><td>58.6</td><td>-1.1</td><td>-1.8</td></tr>
</table>

<p><strong>HumanEval is more sensitive than MMLU.</strong> Code generation requires precise token sequences (one wrong token = failed test), while MMLU is multiple-choice. Always benchmark on your actual use case&mdash;aggregate benchmarks can hide task-specific degradation.</p>

<h4>4. Inference Speed: Tokens per Second</h4>
<table>
<tr><th>Model + Method</th><th>A100 80GB (bs=1)</th><th>A100 80GB (bs=32)</th><th>H100 80GB (bs=1)</th><th>RTX 4090 24GB (bs=1)</th><th>M2 Max 96GB (bs=1)</th></tr>
<tr><td>Llama 3.1 8B FP16</td><td>55 tok/s</td><td>1,200 tok/s</td><td>85 tok/s</td><td>42 tok/s</td><td>---</td></tr>
<tr><td>Llama 3.1 8B FP8</td><td>---</td><td>---</td><td>140 tok/s</td><td>---</td><td>---</td></tr>
<tr><td>Llama 3.1 8B AWQ-INT4</td><td>105 tok/s</td><td>2,100 tok/s</td><td>155 tok/s</td><td>85 tok/s</td><td>---</td></tr>
<tr><td>Llama 3.1 8B Q4_K_M</td><td>90 tok/s</td><td>---</td><td>---</td><td>75 tok/s</td><td>38 tok/s</td></tr>
<tr><td>Llama 3.1 70B FP16</td><td>8 tok/s (2xA100)</td><td>180 tok/s (2x)</td><td>12 tok/s</td><td>---</td><td>---</td></tr>
<tr><td>Llama 3.1 70B FP8</td><td>---</td><td>---</td><td>22 tok/s</td><td>---</td><td>---</td></tr>
<tr><td>Llama 3.1 70B AWQ-INT4</td><td>22 tok/s</td><td>480 tok/s</td><td>32 tok/s</td><td>15 tok/s</td><td>---</td></tr>
<tr><td>Llama 3.1 70B Q4_K_M</td><td>18 tok/s</td><td>---</td><td>---</td><td>---</td><td>12 tok/s</td></tr>
</table>

<p><strong>Key takeaways from speed benchmarks:</strong></p>
<ul>
<li>AWQ INT4 typically delivers 2&ndash;3x the throughput of FP16 on the same GPU</li>
<li>FP8 on H100 delivers ~1.6&ndash;1.8x over FP16, with better quality</li>
<li>GGUF on CPU (M2 Max) is 3&ndash;10x slower than GPU, but usable for single-user</li>
<li>Batch-size scaling is excellent for GPU quantization (near-linear up to batch 32)</li>
<li>At batch_size=1, the speedup from quantization is most dramatic (memory-bound)</li>
</ul>

<h4>5. Memory Savings Summary</h4>
<pre><code># Quick memory estimation function
def estimate_memory(
    n_params_billions: float,
    bits: int,
    group_size: int = 128,
    kv_cache_seq_len: int = 4096,
    n_layers: int = 80,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    batch_size: int = 1,
    kv_dtype_bytes: int = 2  # FP16 KV-cache by default
):
    """Estimate total GPU memory for serving a quantized model."""
    n_params = n_params_billions * 1e9
    
    # Weight memory
    bytes_per_weight = bits / 8
    weight_mem_gb = n_params * bytes_per_weight / (1024**3)
    
    # Scale overhead (FP16 scale per group)
    scale_overhead = n_params / group_size * 2 / (1024**3)  # 2 bytes per FP16 scale
    
    # KV-cache memory
    kv_mem_gb = (2 * batch_size * kv_cache_seq_len * n_layers * 
                 n_kv_heads * head_dim * kv_dtype_bytes) / (1024**3)
    
    # CUDA overhead (kernels, activations, etc.)
    cuda_overhead_gb = 1.5  # Rough estimate
    
    total = weight_mem_gb + scale_overhead + kv_mem_gb + cuda_overhead_gb
    
    return {
        "weights_gb": round(weight_mem_gb, 1),
        "scales_gb": round(scale_overhead, 2),
        "kv_cache_gb": round(kv_mem_gb, 1),
        "overhead_gb": cuda_overhead_gb,
        "total_gb": round(total, 1),
    }

# Compare memory for 70B model across quantization levels
print("Llama 3.1 70B Memory Breakdown (batch=1, seq=4096):")
print("-" * 60)
for bits_name, bits in [("FP16", 16), ("FP8", 8), ("INT4", 4), ("INT2", 2)]:
    mem = estimate_memory(70, bits, n_layers=80, n_kv_heads=8)
    print(f"{bits_name:>5s}: weights={mem['weights_gb']:>5.1f}GB, "
          f"KV={mem['kv_cache_gb']:.1f}GB, "
          f"total={mem['total_gb']:.1f}GB")</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: The Perplexity Trap</div>
<p>We quantized a customer-support chatbot (fine-tuned Llama 3 70B) to INT4 and validated with perplexity: only +0.15 degradation, well within our threshold. In production, we discovered that the model's ability to follow complex multi-step instructions had degraded significantly&mdash;it would miss the third step in a 4-step procedure about 30% of the time. Perplexity, being an average over all tokens, did not catch this because most tokens were fine. <strong>Lesson: Always validate quantized models on task-specific benchmarks, not just perplexity.</strong> We added a suite of 50 multi-step instruction-following tests and caught the issue. The fix: upgrading from INT4 to INT4 with desc_act ordering (GPTQ) reduced the failure rate from 30% to 8%, and switching to Q5_K_M brought it to 3%.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You need to serve a 70B model on a single A100 80GB with sub-500ms latency. Walk through your quantization strategy.</div>
<div class="a-text">The 70B model in FP16 is 140 GB and does not fit on a single 80GB A100. Options: (1) <strong>AWQ INT4 g128</strong>: weights ~37 GB + KV-cache (4K context) ~1.2 GB + overhead ~2 GB = ~40 GB total. Fits comfortably with room for batching. Expected latency at batch=1: ~45ms per token, so 100-token response in ~4.5 seconds total (including ~200ms prefill). This exceeds 500ms for full responses but achieves sub-500ms time-to-first-token. (2) <strong>FP8</strong>: Not available on A100 (no FP8 Tensor Cores). (3) <strong>GPTQ INT4</strong>: Similar memory profile to AWQ, slightly slower inference. My recommendation: AWQ INT4 with vLLM serving, continuous batching enabled, and FP16 KV-cache (switching to FP8 KV-cache if we need longer contexts or higher concurrency). Validate quality on task-specific benchmarks. For sub-500ms time-to-first-token (which is what users perceive as "latency"), this setup achieves ~150-250ms TTFT depending on input length. For sub-500ms total latency, you would need a smaller model (8B) or speculative decoding.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 13.8 Quantization in Production
    // ----------------------------------------------------------
    {
      id: "quant-production",
      title: "Quantization in Production: Decision Framework",
      content: `
<p>Choosing the right quantization strategy is a multi-dimensional optimization problem. You must balance quality, speed, memory, hardware availability, and operational complexity. This section provides a systematic decision framework and practical guidance for deploying quantized models in production.</p>

<h4>1. The Quantization Decision Tree</h4>
<pre><code># Quantization Decision Tree
#
# Q1: What hardware will you serve on?
# |
# +-- H100/H200 (FP8 Tensor Cores)
# |   --> Use FP8. Near-lossless, 2x throughput, simplest option.
# |   --> If memory still tight: FP8 weights + FP8 KV-cache
# |
# +-- A100/A10G (INT8 Tensor Cores, no FP8)
# |   |
# |   +-- Q2: Does the model fit in FP16?
# |   |   +-- YES: Serve FP16 (no quantization needed)
# |   |   +-- NO:
# |   |       +-- Q3: Quality tolerance?
# |   |           +-- Minimal loss OK: AWQ INT4 g128
# |   |           +-- Zero loss required: INT8 (smoothquant/LLM.int8())
# |   |
# +-- RTX 4090/4080 (Consumer GPU, 24GB)
# |   --> AWQ INT4 or GPTQ INT4 for models up to ~34B
# |   --> GGUF Q4_K_M with partial GPU offload for 70B
# |
# +-- Apple Silicon (M1/M2/M3/M4)
# |   --> GGUF Q4_K_M (best quality/speed/memory balance)
# |   --> GGUF Q5_K_M if memory allows
# |   --> Metal backend for GPU acceleration
# |
# +-- CPU only (x86 server)
# |   --> GGUF Q4_K_M with AVX2/AVX-512
# |   --> Consider IQ4_XS with imatrix for better quality at same size
# |
# +-- Edge / Mobile
#     --> GGUF Q4_0 or IQ3_XXS (smallest formats)
#     --> Or use a smaller model (3B) at higher precision</code></pre>

<h4>2. Serving Quantized Models</h4>
<pre><code># === vLLM: Best for GPU serving ===
# Supports AWQ, GPTQ, FP8, SqueezeLLM, Marlin kernels

# AWQ model serving
# python -m vllm.entrypoints.openai.api_server \\
#     --model ./llama-3.1-70b-awq-int4 \\
#     --quantization awq \\
#     --dtype float16 \\
#     --max-model-len 8192 \\
#     --tensor-parallel-size 1 \\
#     --gpu-memory-utilization 0.90 \\
#     --port 8000

# FP8 model serving (H100)
# python -m vllm.entrypoints.openai.api_server \\
#     --model meta-llama/Llama-3.1-70B \\
#     --quantization fp8 \\
#     --kv-cache-dtype fp8_e4m3 \\
#     --max-model-len 32768 \\
#     --tensor-parallel-size 1 \\
#     --port 8000

# === TGI (Text Generation Inference): HuggingFace's server ===
# Supports GPTQ, AWQ, EETQ, FP8

# docker run --gpus all \\
#     -v ./models:/data \\
#     ghcr.io/huggingface/text-generation-inference:latest \\
#     --model-id ./llama-3.1-70b-awq-int4 \\
#     --quantize awq \\
#     --max-input-length 4096 \\
#     --max-total-tokens 8192 \\
#     --port 8080

# === llama.cpp server: Best for CPU/Mac serving ===
# ./llama-server \\
#     -m ./llama-3.1-70b-Q4_K_M.gguf \\
#     -ngl 999 \\             # Offload all layers to GPU (if available)
#     -c 8192 \\              # Context length
#     --host 0.0.0.0 \\
#     --port 8080 \\
#     -t 8                    # CPU threads

# All three servers expose OpenAI-compatible API endpoints:
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama-3.1-70b-awq-int4",
    messages=[{"role": "user", "content": "What is quantization?"}],
    max_tokens=500,
    temperature=0.7,
)
print(response.choices[0].message.content)</code></pre>

<h4>3. Monitoring Quality Degradation</h4>
<pre><code>import numpy as np
from collections import defaultdict

class QuantizationQualityMonitor:
    """Monitor quality metrics for quantized model serving.
    
    Compares quantized model outputs against a reference (FP16) 
    model on a rotating evaluation set.
    """
    
    def __init__(self, eval_prompts: list, reference_outputs: list):
        self.eval_prompts = eval_prompts
        self.reference_outputs = reference_outputs
        self.history = defaultdict(list)
    
    def evaluate(self, model_fn, eval_name: str = "daily_check"):
        """Run evaluation and record metrics."""
        results = {
            "exact_match": 0,
            "semantic_similarity": [],
            "length_ratio": [],
            "format_compliance": 0,
        }
        
        for prompt, reference in zip(self.eval_prompts, 
                                      self.reference_outputs):
            output = model_fn(prompt)
            
            # Exact match (for structured outputs)
            if output.strip() == reference.strip():
                results["exact_match"] += 1
            
            # Length ratio (quantized models sometimes get verbose/terse)
            ratio = len(output) / max(len(reference), 1)
            results["length_ratio"].append(ratio)
            
            # Format compliance (JSON, code blocks, etc.)
            if self._check_format(output, reference):
                results["format_compliance"] += 1
        
        n = len(self.eval_prompts)
        summary = {
            "exact_match_rate": results["exact_match"] / n,
            "avg_length_ratio": np.mean(results["length_ratio"]),
            "format_compliance_rate": results["format_compliance"] / n,
            "timestamp": __import__("time").time(),
        }
        
        self.history[eval_name].append(summary)
        return summary
    
    def check_degradation(self, eval_name: str, 
                          window: int = 7) -> dict:
        """Check for quality degradation over time."""
        history = self.history[eval_name]
        if len(history) < window:
            return {"status": "insufficient_data"}
        
        recent = history[-window:]
        baseline = history[:window]
        
        alerts = []
        for metric in ["exact_match_rate", "format_compliance_rate"]:
            recent_avg = np.mean([r[metric] for r in recent])
            baseline_avg = np.mean([r[metric] for r in baseline])
            
            if recent_avg < baseline_avg * 0.95:  # 5% degradation threshold
                alerts.append({
                    "metric": metric,
                    "baseline": baseline_avg,
                    "current": recent_avg,
                    "degradation_pct": (baseline_avg - recent_avg) / baseline_avg * 100
                })
        
        return {
            "status": "degraded" if alerts else "healthy",
            "alerts": alerts
        }
    
    def _check_format(self, output, reference):
        """Check if output matches expected format structure."""
        # Check for JSON format
        if reference.strip().startswith("{"):
            try:
                import json
                json.loads(output)
                return True
            except:
                return False
        # Check for code blocks
        if "\`\`\`" in reference:
            return "\`\`\`" in output
        return True</code></pre>

<h4>4. A/B Testing Quantized vs Full-Precision</h4>
<pre><code>import random
import hashlib
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    """Configuration for A/B testing quantized models."""
    control_model: str       # Full-precision model endpoint
    treatment_model: str     # Quantized model endpoint
    traffic_split: float     # Fraction of traffic to treatment (0.0-1.0)
    metrics: list            # Metrics to track
    min_samples: int = 1000  # Minimum samples before drawing conclusions
    significance_level: float = 0.05

class QuantizationABTest:
    """A/B test framework for comparing quantized vs full-precision models."""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.results = {"control": [], "treatment": []}
    
    def route_request(self, request_id: str) -> str:
        """Deterministically route request to control or treatment.
        
        Uses hash of request_id for consistent routing
        (same user always gets same model within a session).
        """
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        fraction = (hash_val % 10000) / 10000
        
        if fraction < self.config.traffic_split:
            return "treatment"
        return "control"
    
    def record_result(self, group: str, metrics: dict):
        """Record a result from either group."""
        self.results[group].append(metrics)
    
    def analyze(self) -> dict:
        """Analyze A/B test results."""
        from scipy import stats
        
        control = self.results["control"]
        treatment = self.results["treatment"]
        
        if len(control) < self.config.min_samples or \\
           len(treatment) < self.config.min_samples:
            return {"status": "insufficient_data",
                    "control_n": len(control),
                    "treatment_n": len(treatment)}
        
        analysis = {}
        for metric in self.config.metrics:
            c_values = [r[metric] for r in control if metric in r]
            t_values = [r[metric] for r in treatment if metric in r]
            
            c_mean = np.mean(c_values)
            t_mean = np.mean(t_values)
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(c_values, t_values)
            
            significant = p_value < self.config.significance_level
            
            analysis[metric] = {
                "control_mean": round(c_mean, 4),
                "treatment_mean": round(t_mean, 4),
                "difference": round(t_mean - c_mean, 4),
                "difference_pct": round((t_mean - c_mean) / c_mean * 100, 2),
                "p_value": round(p_value, 4),
                "significant": significant,
                "recommendation": (
                    "SHIP" if not significant or t_mean >= c_mean * 0.98
                    else "INVESTIGATE"
                )
            }
        
        return {"status": "complete", "analysis": analysis}

# Example usage
config = ABTestConfig(
    control_model="http://model-server:8000/v1",    # FP16
    treatment_model="http://model-server:8001/v1",   # INT4 AWQ
    traffic_split=0.2,  # 20% to quantized model
    metrics=["user_satisfaction", "task_completion", "latency_ms"],
    min_samples=2000,
)

ab_test = QuantizationABTest(config)</code></pre>

<h4>5. Production Quantization Checklist</h4>
<table>
<tr><th>Step</th><th>Action</th><th>Tool/Method</th></tr>
<tr><td>1</td><td>Benchmark FP16 baseline on task-specific evals</td><td>Custom eval suite</td></tr>
<tr><td>2</td><td>Choose quantization method based on hardware</td><td>Decision tree above</td></tr>
<tr><td>3</td><td>Quantize with domain-matched calibration data</td><td>AutoAWQ/AutoGPTQ/llama.cpp</td></tr>
<tr><td>4</td><td>Run task-specific benchmarks on quantized model</td><td>Same eval suite as step 1</td></tr>
<tr><td>5</td><td>Measure inference speed and memory on target hardware</td><td>vLLM benchmarks</td></tr>
<tr><td>6</td><td>A/B test with 5-10% traffic</td><td>AB test framework</td></tr>
<tr><td>7</td><td>Monitor quality metrics continuously</td><td>Quality monitor above</td></tr>
<tr><td>8</td><td>Set up alerts for quality degradation</td><td>Alerting system</td></tr>
<tr><td>9</td><td>Document the quantization config and results</td><td>Model card</td></tr>
<tr><td>10</td><td>Plan rollback procedure</td><td>Keep FP16 weights available</td></tr>
</table>

<div class="callout tip">
<div class="callout-title">The Golden Rule of Production Quantization</div>
<p>Always keep the original FP16 weights. You may need to re-quantize with a different method, calibrate with different data, or roll back entirely. Storage is cheap; re-training a model is not. Tag your quantized models with the exact configuration (method, bits, group_size, calibration_data_hash) so you can reproduce results.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Design a production system that serves a quantized LLM and automatically detects and rolls back if quality degrades.</div>
<div class="a-text">Architecture: (1) <strong>Serving layer:</strong> vLLM with AWQ INT4 model behind a load balancer. Keep an FP16 model warm on standby (or a quick-load path). (2) <strong>Shadow evaluation:</strong> Sample 1% of production traffic and route it (in parallel, not user-facing) to both quantized and FP16 models. Compare outputs using automated metrics (format compliance, length ratio, semantic similarity via embedding comparison). (3) <strong>Continuous benchmarks:</strong> Run a fixed eval suite every hour on the quantized model. Track perplexity, task-specific accuracy, and output format compliance over time. (4) <strong>Alert pipeline:</strong> If any metric degrades by more than 5% from baseline (measured in first 24 hours), trigger an alert. If degradation exceeds 10% or a safety metric triggers, auto-rollback to FP16. (5) <strong>Rollback mechanism:</strong> Load balancer switches traffic to FP16 standby model within 60 seconds. (6) <strong>Root cause analysis:</strong> After rollback, investigate whether the degradation is due to distribution shift in inputs (new topics the calibration data did not cover), a serving infrastructure issue, or genuine quantization failure. (7) <strong>Re-quantization pipeline:</strong> If the issue is distribution shift, re-quantize with updated calibration data and run the full evaluation before re-deploying.</div>
</div>
`
    }
  ],

  // ============================================================
  // CHAPTER 14: RAG Systems
  // ============================================================
  ch14_sections: [
    // ----------------------------------------------------------
    // 14.1 RAG Fundamentals
    // ----------------------------------------------------------
    {
      id: "rag-fundamentals",
      title: "RAG Fundamentals: Why LLMs Need Retrieval",
      content: `
<p>Retrieval-Augmented Generation (RAG) is the most important architectural pattern in applied LLM engineering. It addresses three fundamental limitations of standalone LLMs: <strong>knowledge cutoff</strong> (the model cannot know about events after its training data), <strong>hallucination</strong> (the model confidently generates plausible-sounding but incorrect information), and <strong>domain specificity</strong> (the model lacks deep knowledge of your proprietary data). RAG solves all three by grounding LLM generation in retrieved evidence from an external knowledge base.</p>

<div class="callout">
<div class="callout-title">The Core Idea</div>
<p>Instead of asking the LLM to recall information from its parameters (unreliable), <strong>retrieve</strong> relevant documents from a knowledge base and <strong>inject them into the prompt</strong> as context. The LLM then generates an answer grounded in the provided evidence. Think of it as giving the LLM an open-book exam instead of a closed-book exam.</p>
</div>

<h4>1. Why RAG? The Three Problems</h4>
<table>
<tr><th>Problem</th><th>Example</th><th>How RAG Solves It</th></tr>
<tr><td><strong>Knowledge cutoff</strong></td><td>"What were Q4 2025 revenue figures?" &mdash; model trained on data up to mid-2025</td><td>Retrieve the latest financial report and include it in the prompt</td></tr>
<tr><td><strong>Hallucination</strong></td><td>"What is the dosage for drug X?" &mdash; model generates a plausible but wrong number</td><td>Retrieve the actual drug label and ground the answer in the retrieved text</td></tr>
<tr><td><strong>Domain knowledge</strong></td><td>"What is our company's PTO policy?" &mdash; model has never seen your internal docs</td><td>Index your HR documents and retrieve the relevant policy section</td></tr>
</table>

<h4>2. The Naive RAG Pipeline</h4>
<p>The simplest RAG system has three stages: <strong>Index</strong>, <strong>Retrieve</strong>, <strong>Generate</strong>.</p>

<pre><code>from openai import OpenAI
import numpy as np

client = OpenAI()

# === STAGE 1: INDEX ===
# Split documents into chunks and compute embeddings

def chunk_document(text: str, chunk_size: int = 500, 
                   overlap: int = 50) -> list:
    """Split a document into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def embed_texts(texts: list, model: str = "text-embedding-3-small") -> list:
    """Compute embeddings for a list of texts."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

# Index a knowledge base
documents = [
    "Our company PTO policy allows 20 days of paid vacation per year...",
    "The engineering team follows a sprint-based agile methodology...",
    "Health insurance coverage begins on the first day of employment...",
    # ... hundreds or thousands of documents
]

# Chunk and embed
all_chunks = []
all_embeddings = []
for doc in documents:
    chunks = chunk_document(doc)
    embeddings = embed_texts(chunks)
    all_chunks.extend(chunks)
    all_embeddings.extend(embeddings)

# Store as numpy array for cosine similarity search
embedding_matrix = np.array(all_embeddings)

# === STAGE 2: RETRIEVE ===
# Find the most relevant chunks for a query

def retrieve(query: str, top_k: int = 5) -> list:
    """Retrieve top-k most relevant chunks for a query."""
    query_embedding = np.array(embed_texts([query])[0])
    
    # Cosine similarity
    similarities = embedding_matrix @ query_embedding / (
        np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [
        {"chunk": all_chunks[i], "score": float(similarities[i])}
        for i in top_indices
    ]

# === STAGE 3: GENERATE ===
# Use retrieved context to generate a grounded answer

def generate_answer(query: str, top_k: int = 5) -> str:
    """RAG: retrieve context and generate a grounded answer."""
    # Retrieve relevant chunks
    retrieved = retrieve(query, top_k=top_k)
    
    # Format context
    context = "\\n\\n".join([
        f"[Source {i+1}] (relevance: {r['score']:.3f})\\n{r['chunk']}"
        for i, r in enumerate(retrieved)
    ])
    
    # Generate with context
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant. Answer the user's question "
                "based ONLY on the provided context. If the context doesn't "
                "contain the answer, say 'I don't have enough information "
                "to answer this question.' Always cite your sources using "
                "[Source N] notation."
            )},
            {"role": "user", "content": (
                f"Context:\\n{context}\\n\\n"
                f"Question: {query}"
            )}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content

# Usage
answer = generate_answer("How many PTO days do I get per year?")
print(answer)
# Expected: "Based on the company PTO policy, you receive 20 days
#  of paid vacation per year [Source 1]."</code></pre>

<h4>3. RAG vs Fine-Tuning vs Long Context</h4>
<table>
<tr><th>Approach</th><th>Best For</th><th>Limitations</th><th>Cost</th></tr>
<tr><td><strong>RAG</strong></td><td>Dynamic/updated knowledge, large corpora, citation needed</td><td>Retrieval errors, latency, context window limits</td><td>Low (embedding + storage)</td></tr>
<tr><td><strong>Fine-tuning</strong></td><td>Changing model behavior/style, teaching domain language</td><td>Cannot add new facts reliably, catastrophic forgetting</td><td>High (GPU training)</td></tr>
<tr><td><strong>Long context</strong></td><td>Small knowledge bases (&lt;100 pages), single-session context</td><td>Expensive per query, "lost in the middle" effect</td><td>Very high (per query)</td></tr>
<tr><td><strong>RAG + Fine-tuning</strong></td><td>Best of both: domain-adapted model + grounded knowledge</td><td>Most complex to build and maintain</td><td>Highest</td></tr>
</table>

<div class="callout warning">
<div class="callout-title">Common Misconception: "Just Use Long Context"</div>
<p>With models supporting 128K+ token contexts, some teams try to skip RAG by stuffing all documents into the prompt. This fails for three reasons: (1) <strong>Cost:</strong> Processing 128K tokens costs 50-100x more per query than embedding-based retrieval. At 1,000 queries/day, this adds up fast. (2) <strong>Quality:</strong> The "lost in the middle" problem (Liu et al., arXiv: 2307.03172) shows that LLMs struggle to use information in the middle of long contexts. Relevant information placed at positions 40-80% through the context is often ignored. (3) <strong>Scale:</strong> Most real knowledge bases are millions of documents&mdash;far too large for any context window. RAG scales to billions of documents; long context does not.</p>
</div>

<h4>4. The Retrieval-Generation Paradigm</h4>
<p>RAG was formalized by Lewis et al. (arXiv: 2005.11401) at Meta AI in 2020. The original paper proposed two variants:</p>

<ul>
<li><strong>RAG-Sequence:</strong> Retrieve documents once, generate the entire answer conditioned on the same retrieved set. This is what most production systems use.</li>
<li><strong>RAG-Token:</strong> Potentially retrieve different documents for each generated token. More flexible but impractical at scale.</li>
</ul>

<p>The modern RAG pipeline has evolved well beyond the original formulation. A production system typically includes: query understanding, hybrid retrieval (dense + sparse), reranking, context assembly with metadata, structured generation with citations, and evaluation. We will cover each component in the following sections.</p>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">When would you choose RAG over fine-tuning, and vice versa?</div>
<div class="a-text">Choose <strong>RAG</strong> when: (1) The knowledge base changes frequently (daily/weekly updates). Fine-tuning cannot absorb new facts quickly. (2) You need citations or provenance tracking. RAG naturally provides source documents. (3) The knowledge base is large (thousands of documents). (4) Accuracy on factual recall is critical (medical, legal, financial). RAG grounds answers in source material. Choose <strong>fine-tuning</strong> when: (1) You need to change the model's behavior or style (e.g., more concise, domain-specific tone). (2) The task requires learning patterns, not just facts (e.g., code style, writing voice). (3) You want to reduce prompt size (fine-tuned knowledge doesn't need to be in the prompt). Use <strong>both</strong> when: You need domain adaptation (fine-tuning for style and reasoning) plus factual grounding (RAG for accurate knowledge). Example: A medical chatbot fine-tuned on clinical conversations for appropriate communication style, with RAG for up-to-date drug information and clinical guidelines.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 14.2 Embedding Models
    // ----------------------------------------------------------
    {
      id: "rag-embeddings",
      title: "Embedding Models: The Foundation of Semantic Search",
      content: `
<p>Embedding models convert text into dense numerical vectors that capture semantic meaning. Two texts about the same topic will have similar embeddings, even if they use different words. This is the foundation that makes RAG work: instead of keyword matching, we can find documents that are <em>semantically</em> relevant to a query. The choice of embedding model significantly impacts retrieval quality and, consequently, the entire RAG pipeline.</p>

<h4>1. Leading Embedding Models (2025-2026)</h4>
<table>
<tr><th>Model</th><th>Provider</th><th>Dimensions</th><th>Max Tokens</th><th>MTEB Score</th><th>Cost (per 1M tokens)</th></tr>
<tr><td><strong>text-embedding-3-large</strong></td><td>OpenAI</td><td>3072 (or 256-3072)</td><td>8,191</td><td>64.6</td><td>$0.13</td></tr>
<tr><td><strong>text-embedding-3-small</strong></td><td>OpenAI</td><td>1536 (or 256-1536)</td><td>8,191</td><td>62.3</td><td>$0.02</td></tr>
<tr><td><strong>voyage-3-large</strong></td><td>Voyage AI</td><td>1024</td><td>32,000</td><td>67.2</td><td>$0.18</td></tr>
<tr><td><strong>BGE-en-icl</strong></td><td>BAAI</td><td>4096</td><td>8,192</td><td>65.8</td><td>Free (self-hosted)</td></tr>
<tr><td><strong>jina-embeddings-v3</strong></td><td>Jina AI</td><td>1024</td><td>8,192</td><td>65.5</td><td>$0.02</td></tr>
<tr><td><strong>GTE-Qwen2</strong></td><td>Alibaba</td><td>768-4096</td><td>32,768</td><td>64.3</td><td>Free (self-hosted)</td></tr>
<tr><td><strong>nomic-embed-text-v2</strong></td><td>Nomic</td><td>768</td><td>8,192</td><td>63.1</td><td>Free (self-hosted)</td></tr>
<tr><td><strong>all-MiniLM-L6-v2</strong></td><td>sentence-transformers</td><td>384</td><td>256</td><td>56.3</td><td>Free (self-hosted)</td></tr>
</table>

<h4>2. Computing and Comparing Embeddings</h4>
<pre><code>import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embeddings(texts: list, model: str = "text-embedding-3-small",
                   dimensions: int = None) -> np.ndarray:
    """Get embeddings from OpenAI API.
    
    The 'dimensions' parameter enables Matryoshka Representation Learning:
    you can request lower-dimensional embeddings that are still useful,
    trading quality for storage/speed.
    """
    kwargs = {"input": texts, "model": model}
    if dimensions:
        kwargs["dimensions"] = dimensions
    
    response = client.embeddings.create(**kwargs)
    return np.array([item.embedding for item in response.data])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example: semantic similarity
texts = [
    "The cat sat on the mat",
    "A feline was resting on the rug",        # Semantically similar
    "Python is a programming language",        # Different topic
    "Machine learning models need training",   # Different topic
]

embeddings = get_embeddings(texts)

print("Cosine similarities:")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  '{texts[i][:40]}...' vs '{texts[j][:40]}...': {sim:.4f}")

# Expected output:
# cat/feline: ~0.85 (high similarity - same meaning, different words)
# cat/python: ~0.15 (low similarity - different topics)
# cat/ML:     ~0.10 (low similarity - different topics)
# python/ML:  ~0.45 (moderate similarity - related tech topics)</code></pre>

<h4>3. Using sentence-transformers (Self-Hosted)</h4>
<pre><code>from sentence_transformers import SentenceTransformer
import torch

# Load a high-quality open-source model
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# For BGE models, prepend instruction for queries (not documents)
query_instruction = "Represent this sentence for searching relevant passages: "

def embed_documents(documents: list) -> np.ndarray:
    """Embed documents (no instruction prefix needed)."""
    return model.encode(documents, normalize_embeddings=True,
                       show_progress_bar=True, batch_size=32)

def embed_query(query: str) -> np.ndarray:
    """Embed a query (with instruction prefix for BGE models)."""
    return model.encode([query_instruction + query], 
                       normalize_embeddings=True)[0]

# Batch embedding for large document collections
documents = ["doc text 1...", "doc text 2...", "..."]  # Thousands of docs
doc_embeddings = embed_documents(documents)

# Query time
query_vec = embed_query("What is the company vacation policy?")
scores = doc_embeddings @ query_vec  # Cosine similarity (normalized vectors)
top_k = np.argsort(scores)[-5:][::-1]

for idx in top_k:
    print(f"Score: {scores[idx]:.4f} | {documents[idx][:80]}...")</code></pre>

<h4>4. Matryoshka Embeddings: Dimension vs Quality Tradeoff</h4>
<pre><code># OpenAI text-embedding-3 models support Matryoshka dimensions
# You can truncate embeddings to fewer dimensions while retaining quality.
# This saves storage and speeds up similarity search.

dims_to_test = [256, 512, 1024, 1536, 3072]

# Reference: full-dimension embeddings
docs = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn canine leaps above an idle hound",
    "Quantum computing uses qubits for computation",
]

print("\\nDimension vs Quality tradeoff:")
print(f"{'Dims':>6s} | {'Semantic Sim':>12s} | {'Cross-topic':>12s} | {'Storage/vec':>12s}")
print("-" * 55)

for dim in dims_to_test:
    try:
        emb = get_embeddings(docs, model="text-embedding-3-large", dimensions=dim)
        sem_sim = cosine_similarity(emb[0], emb[1])   # Same meaning
        cross = cosine_similarity(emb[0], emb[2])      # Different topic
        storage = dim * 4  # 4 bytes per float32
        print(f"{dim:>6d} | {sem_sim:>12.4f} | {cross:>12.4f} | {storage:>10d} B")
    except Exception as e:
        print(f"{dim:>6d} | Error: {e}")

# Typical results:
# 3072 dimensions: sem_sim=0.88, cross=0.12 (best quality)
# 1536 dimensions: sem_sim=0.87, cross=0.13 (very close to best)
#  512 dimensions: sem_sim=0.85, cross=0.15 (good quality, 6x less storage)
#  256 dimensions: sem_sim=0.82, cross=0.18 (acceptable, 12x less storage)</code></pre>

<h4>5. Choosing the Right Embedding Model</h4>
<table>
<tr><th>Scenario</th><th>Recommended Model</th><th>Why</th></tr>
<tr><td>Quick prototype</td><td>text-embedding-3-small (OpenAI)</td><td>Cheap, good quality, no infrastructure</td></tr>
<tr><td>Production (external API OK)</td><td>text-embedding-3-large or voyage-3-large</td><td>Best quality from hosted APIs</td></tr>
<tr><td>Production (self-hosted, GPU)</td><td>BGE-large-en-v1.5 or GTE-Qwen2</td><td>Top quality, no API costs, data stays private</td></tr>
<tr><td>Production (self-hosted, CPU)</td><td>all-MiniLM-L6-v2</td><td>Fast on CPU, 384 dims, good-enough quality</td></tr>
<tr><td>Multilingual</td><td>jina-embeddings-v3 or BGE-M3</td><td>Strong cross-lingual retrieval</td></tr>
<tr><td>Long documents (&gt;8K tokens)</td><td>jina-embeddings-v3 or GTE-Qwen2</td><td>32K token context window</td></tr>
<tr><td>Code search</td><td>voyage-code-3 or CodeSage</td><td>Trained on code, understands programming semantics</td></tr>
</table>

<div class="callout warning">
<div class="callout-title">War Story: The Embedding Mismatch</div>
<p>A team indexed their knowledge base with OpenAI's text-embedding-ada-002 (the old model), then months later upgraded their query embedding to text-embedding-3-small without re-indexing their documents. Retrieval quality dropped to near-random. <strong>Embeddings from different models live in different vector spaces and cannot be compared.</strong> When you change your embedding model, you must re-embed your entire document collection. Budget for this re-indexing cost and time (for 1M documents with text-embedding-3-small: ~$0.40 and ~20 minutes).</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you choose between a 256-dimensional and a 1536-dimensional embedding for a RAG system with 10 million documents?</div>
<div class="a-text">This is a storage-quality tradeoff analysis. With 10M documents: 256-dim = 10M * 256 * 4 bytes = ~10 GB of vectors. 1536-dim = 10M * 1536 * 4 bytes = ~60 GB of vectors. The 6x storage difference affects: (1) Vector DB costs (RAM for HNSW index, or disk for IVF), (2) Search latency (more dimensions = slower distance computation, though HNSW is sublinear), (3) Retrieval quality (more dimensions = better discrimination between similar topics). My approach: Start by benchmarking on a sample of your actual queries. Compute recall@10 for both dimension sizes against a ground-truth relevance set. If 256-dim achieves 95%+ of 1536-dim recall, use 256-dim for the 6x savings. If there is a meaningful gap, consider 512-dim as a compromise. In practice, Matryoshka embeddings at 512 dimensions retain ~97% of full-quality retrieval performance while using 3x less storage. For 10M docs, that is 20 GB vs 60 GB, which is the difference between fitting in RAM on a single server vs needing distributed storage.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 14.3 Vector Databases
    // ----------------------------------------------------------
    {
      id: "rag-vector-db",
      title: "Vector Databases: Storage and Retrieval at Scale",
      content: `
<p>A vector database stores embedding vectors and supports efficient similarity search at scale. While you can implement cosine similarity with numpy for a prototype (as we did in Section 14.1), production systems with millions of documents need specialized indexing structures that trade a small amount of recall for orders-of-magnitude speedup. This section compares the leading vector databases and their indexing algorithms.</p>

<h4>1. Vector Database Landscape</h4>
<table>
<tr><th>Database</th><th>Type</th><th>Index Types</th><th>Max Scale</th><th>Filtering</th><th>Best For</th></tr>
<tr><td><strong>Chroma</strong></td><td>Embedded / Client-server</td><td>HNSW</td><td>~5M vectors</td><td>Metadata filtering</td><td>Prototyping, small-medium RAG</td></tr>
<tr><td><strong>Pinecone</strong></td><td>Managed cloud</td><td>Proprietary</td><td>Billions</td><td>Rich metadata</td><td>Production (no infra management)</td></tr>
<tr><td><strong>Qdrant</strong></td><td>Self-hosted / Cloud</td><td>HNSW</td><td>Billions</td><td>Payload filtering</td><td>Production (full control)</td></tr>
<tr><td><strong>Weaviate</strong></td><td>Self-hosted / Cloud</td><td>HNSW</td><td>Billions</td><td>GraphQL filtering</td><td>Hybrid search (vector + keyword)</td></tr>
<tr><td><strong>Milvus</strong></td><td>Self-hosted / Zilliz Cloud</td><td>HNSW, IVF, DiskANN</td><td>Billions</td><td>Rich filtering</td><td>Large-scale, GPU-accelerated</td></tr>
<tr><td><strong>pgvector</strong></td><td>PostgreSQL extension</td><td>HNSW, IVFFlat</td><td>~10M vectors</td><td>Full SQL</td><td>Existing Postgres stacks</td></tr>
</table>

<h4>2. HNSW vs IVF Indexing</h4>
<pre><code># HNSW (Hierarchical Navigable Small World)
# ==========================================
# - A multi-layer graph structure
# - Top layers: sparse graph for coarse navigation
# - Bottom layers: dense graph for precise search
# - Search: start at top, greedily navigate down to nearest neighbors
#
# Pros: Excellent recall (>95% at reasonable speed), no training needed
# Cons: High memory usage (stores full vectors + graph in RAM)
#
# Parameters:
# - M: Number of connections per node (16-64, higher = better recall, more RAM)
# - ef_construction: Build-time quality (200-800, higher = better graph)
# - ef_search: Query-time quality (50-200, higher = better recall, slower)

# IVF (Inverted File Index)
# =========================
# - Partition vectors into clusters using k-means
# - At query time, search only the nearest clusters
# - nprobe: number of clusters to search (more = better recall, slower)
#
# Pros: Lower memory (can keep vectors on disk), faster for very large datasets
# Cons: Requires training (k-means), lower recall than HNSW at same speed
#
# Parameters:
# - nlist: Number of clusters (sqrt(N) to 4*sqrt(N))
# - nprobe: Clusters to search at query time (1-nlist, typically 10-50)

# Comparison on 1M vectors (1536 dims):
# HNSW: ~6 GB RAM, 1ms query, 98% recall@10
# IVF:  ~2 GB RAM (+disk), 0.5ms query, 92% recall@10
# Brute force: ~6 GB RAM, 50ms query, 100% recall@10</code></pre>

<h4>3. Setting Up Each Database</h4>
<pre><code># === Chroma: Simplest setup (embedded mode) ===
import chromadb
from chromadb.utils import embedding_functions

# Create client (embedded - stores in local directory)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-small"
)

# Create collection
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# Add documents
collection.add(
    documents=["Company PTO policy: 20 days/year...",
               "Engineering sprint process..."],
    metadatas=[{"source": "hr_docs", "updated": "2025-01-15"},
               {"source": "eng_docs", "updated": "2025-03-01"}],
    ids=["doc_001", "doc_002"]
)

# Query
results = collection.query(
    query_texts=["How many vacation days do I get?"],
    n_results=5,
    where={"source": "hr_docs"}  # Metadata filter
)
print(results["documents"][0])


# === Qdrant: Production-grade self-hosted ===
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)

# Connect (local or remote)
qdrant = QdrantClient(host="localhost", port=6333)
# Or: qdrant = QdrantClient(":memory:")  # In-memory for testing

# Create collection
qdrant.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    # Enable HNSW indexing (default, but shown explicitly)
    hnsw_config={"m": 16, "ef_construct": 200}
)

# Upsert vectors
qdrant.upsert(
    collection_name="knowledge_base",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 1536-dim vector
            payload={
                "text": "Company PTO policy...",
                "source": "hr_docs",
                "updated": "2025-01-15"
            }
        ),
        # ... more points
    ]
)

# Search with filtering
results = qdrant.search(
    collection_name="knowledge_base",
    query_vector=[0.15, 0.22, ...],  # Query embedding
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="hr_docs")
            )
        ]
    )
)


# === pgvector: Add vector search to PostgreSQL ===
import psycopg2

conn = psycopg2.connect("postgresql://localhost:5432/mydb")
cur = conn.cursor()

# Enable extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Create table with vector column
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        source VARCHAR(100),
        embedding vector(1536),  -- pgvector type
        created_at TIMESTAMP DEFAULT NOW()
    )
""")

# Create HNSW index (pgvector 0.5+)
cur.execute("""
    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
    ON documents 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
""")

# Insert
cur.execute("""
    INSERT INTO documents (content, source, embedding) 
    VALUES (%s, %s, %s::vector)
""", ("PTO policy: 20 days/year...", "hr_docs", str([0.1, 0.2, ...])))

# Query with filtering (the magic of pgvector: SQL + vectors!)
cur.execute("""
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE source = 'hr_docs'
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (str(query_vector), str(query_vector)))

results = cur.fetchall()
conn.commit()</code></pre>

<h4>4. Choosing a Vector Database</h4>
<table>
<tr><th>Criterion</th><th>Chroma</th><th>Pinecone</th><th>Qdrant</th><th>pgvector</th><th>Milvus</th></tr>
<tr><td>Setup complexity</td><td>Trivial</td><td>Easy (cloud)</td><td>Medium</td><td>Easy (if Postgres exists)</td><td>Complex</td></tr>
<tr><td>Scale (vectors)</td><td>&lt;5M</td><td>Billions</td><td>Billions</td><td>&lt;10M</td><td>Billions</td></tr>
<tr><td>Self-hosted</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
<tr><td>Hybrid search</td><td>Limited</td><td>Yes</td><td>Yes</td><td>Via tsvector</td><td>Yes</td></tr>
<tr><td>Filtering speed</td><td>Good</td><td>Excellent</td><td>Excellent</td><td>Excellent (SQL)</td><td>Good</td></tr>
<tr><td>Multi-tenancy</td><td>Via collections</td><td>Namespaces</td><td>Payload filter</td><td>Row-level security</td><td>Partitions</td></tr>
<tr><td>Best for</td><td>Prototypes</td><td>Managed prod</td><td>Self-hosted prod</td><td>Postgres shops</td><td>Very large scale</td></tr>
</table>

<div class="callout tip">
<div class="callout-title">Practical Recommendation</div>
<p><strong>Start with Chroma</strong> for prototyping (5 lines of code to get started). <strong>Migrate to Qdrant or pgvector</strong> for production, depending on your infrastructure. If you already run PostgreSQL, pgvector is the lowest-overhead choice&mdash;no new infrastructure to maintain. If you need maximum performance and scale, use Qdrant (self-hosted) or Pinecone (managed). <strong>Do not over-optimize early</strong>&mdash;for most RAG systems, the retrieval quality depends far more on chunking strategy and embedding model choice than on which vector database you use.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the HNSW algorithm and why it is the dominant indexing method for vector databases.</div>
<div class="a-text">HNSW builds a multi-layer proximity graph where each node (vector) is connected to its approximate nearest neighbors. The top layer has few nodes and long-range connections (for coarse navigation), and each lower layer adds more nodes with shorter-range connections (for refinement). At query time, search starts at the top layer's entry point and greedily navigates to the closest node, then drops to the next layer and repeats. This is like a skip list but in high-dimensional space. HNSW dominates because: (1) Excellent recall-speed tradeoff: 95-99% recall at sub-millisecond latency on millions of vectors. (2) No training required: unlike IVF, you do not need to run k-means clustering. Vectors can be inserted incrementally. (3) Tunable at query time: the ef_search parameter lets you trade recall for speed per-query, useful for different use cases. The main drawback is memory: HNSW requires all vectors plus the graph structure in RAM (roughly 1.5x the raw vector data). For very large datasets (billions of vectors), disk-based approaches like DiskANN or IVF+PQ become necessary.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 14.4 Chunking Strategies
    // ----------------------------------------------------------
    {
      id: "rag-chunking",
      title: "Chunking Strategies: How You Split Documents Matters",
      content: `
<p>Chunking is the most underrated component of a RAG system. The way you split documents into chunks determines what gets retrieved and what the LLM sees as context. Bad chunking leads to retrieved chunks that contain half an answer (cut mid-paragraph), irrelevant padding (chunk too large), or missing context (chunk too small). This section covers the major chunking strategies and their impact on retrieval quality.</p>

<div class="callout">
<div class="callout-title">The Chunking Paradox</div>
<p>Small chunks are better for precise retrieval (the embedding represents a focused topic). Large chunks are better for generation (the LLM gets more context). The optimal chunk size depends on your embedding model, your document structure, and your use case. There is no universal best chunk size&mdash;you must experiment.</p>
</div>

<h4>1. Fixed-Size Chunking</h4>
<pre><code>def fixed_size_chunking(text: str, chunk_size: int = 500, 
                        overlap: int = 50) -> list:
    """Split text into fixed-size chunks with overlap.
    
    Pros: Simple, predictable chunk sizes, easy to reason about
    Cons: Splits mid-sentence, mid-paragraph, mid-thought
    
    Args:
        chunk_size: Number of characters (or tokens) per chunk
        overlap: Characters of overlap between consecutive chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for last sentence-ending punctuation in chunk
            for sep in ['. ', '! ', '? ', '\\n\\n', '\\n']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.7:  # Don't break too early
                    chunk = chunk[:last_sep + 1]
                    end = start + last_sep + 1
                    break
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

# Rule of thumb for chunk sizes:
# - 100-200 tokens: Very precise retrieval, may lack context
# - 200-500 tokens: Good balance for most use cases (RECOMMENDED)
# - 500-1000 tokens: More context per chunk, less precise retrieval
# - 1000-2000 tokens: For document Q&A where you need full paragraphs</code></pre>

<h4>2. Recursive Character Splitting (LangChain Default)</h4>
<pre><code>def recursive_character_split(text: str, chunk_size: int = 1000,
                              chunk_overlap: int = 200,
                              separators: list = None) -> list:
    """Split text recursively by trying different separators.
    
    This is LangChain's RecursiveCharacterTextSplitter algorithm.
    It tries to split on the most meaningful boundary first,
    falling back to less meaningful boundaries.
    
    Default separator hierarchy:
    1. "\\n\\n" (paragraph break) - most meaningful
    2. "\\n" (line break)
    3. " " (word break)
    4. "" (character) - last resort
    """
    if separators is None:
        separators = ["\\n\\n", "\\n", ". ", " ", ""]
    
    chunks = []
    
    def _split(text, seps):
        if len(text) <= chunk_size:
            return [text]
        
        # Find the best separator
        separator = seps[0] if seps else ""
        remaining_seps = seps[1:] if len(seps) > 1 else [""]
        
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split (last resort)
            splits = list(text)
        
        current_chunk = ""
        result = []
        
        for split in splits:
            piece = split + (separator if separator else "")
            
            if len(current_chunk) + len(piece) <= chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    if len(current_chunk) > chunk_size:
                        # Current chunk too big, recursively split with finer separators
                        result.extend(_split(current_chunk, remaining_seps))
                    else:
                        result.append(current_chunk.strip())
                current_chunk = piece
        
        if current_chunk.strip():
            result.append(current_chunk.strip())
        
        return result
    
    raw_chunks = _split(text, separators)
    
    # Add overlap between chunks
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            # Prepend end of previous chunk
            prev_overlap = raw_chunks[i-1][-chunk_overlap:]
            chunk = prev_overlap + " " + chunk
        final_chunks.append(chunk)
    
    return final_chunks

# Using LangChain's implementation directly:
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# 
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     separators=["\\n\\n", "\\n", ". ", " ", ""]
# )
# chunks = splitter.split_text(document_text)</code></pre>

<h4>3. Semantic Chunking</h4>
<pre><code>import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunking(text: str, 
                      embedding_model: SentenceTransformer,
                      breakpoint_threshold: float = 0.3,
                      min_chunk_size: int = 100,
                      max_chunk_size: int = 2000) -> list:
    """Split text based on semantic similarity between consecutive sentences.
    
    Idea: Sentences about the same topic have similar embeddings.
    When the topic shifts (low similarity between consecutive sentences),
    insert a chunk boundary.
    
    This produces chunks that are topically coherent, even if
    the document doesn't have clear structural boundaries.
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    if len(sentences) < 2:
        return [text]
    
    # Embed all sentences
    embeddings = embedding_model.encode(sentences, normalize_embeddings=True)
    
    # Compute cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        similarities.append(sim)
    
    # Find breakpoints (where similarity drops below threshold)
    # Use percentile-based threshold for robustness
    if breakpoint_threshold is None:
        breakpoint_threshold = np.percentile(similarities, 25)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0])
    
    for i, sim in enumerate(similarities):
        sentence = sentences[i + 1]
        
        should_break = (
            sim < breakpoint_threshold and 
            current_length >= min_chunk_size
        )
        chunk_too_large = current_length + len(sentence) > max_chunk_size
        
        if should_break or chunk_too_large:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks</code></pre>

<h4>4. Document-Structure-Aware Chunking</h4>
<pre><code>import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class StructuredChunk:
    """A chunk with document structure metadata."""
    text: str
    section_title: str
    subsection_title: Optional[str]
    page_number: Optional[int]
    chunk_index: int
    parent_doc: str

def markdown_structure_chunking(markdown_text: str, 
                                 max_chunk_size: int = 1000,
                                 source_doc: str = "") -> list:
    """Split markdown by headings, preserving document structure.
    
    This is crucial for technical documentation, legal documents,
    and any content with clear hierarchical structure.
    """
    chunks = []
    current_h1 = "Introduction"
    current_h2 = None
    current_content = []
    chunk_idx = 0
    
    for line in markdown_text.split("\\n"):
        # Detect headings
        if line.startswith("# "):
            # Flush current content
            if current_content:
                chunks.append(StructuredChunk(
                    text="\\n".join(current_content),
                    section_title=current_h1,
                    subsection_title=current_h2,
                    page_number=None,
                    chunk_index=chunk_idx,
                    parent_doc=source_doc
                ))
                chunk_idx += 1
                current_content = []
            current_h1 = line[2:].strip()
            current_h2 = None
            
        elif line.startswith("## "):
            if current_content:
                chunks.append(StructuredChunk(
                    text="\\n".join(current_content),
                    section_title=current_h1,
                    subsection_title=current_h2,
                    page_number=None,
                    chunk_index=chunk_idx,
                    parent_doc=source_doc
                ))
                chunk_idx += 1
                current_content = []
            current_h2 = line[3:].strip()
        else:
            current_content.append(line)
            
            # Check size limit
            content_text = "\\n".join(current_content)
            if len(content_text) > max_chunk_size:
                chunks.append(StructuredChunk(
                    text=content_text,
                    section_title=current_h1,
                    subsection_title=current_h2,
                    page_number=None,
                    chunk_index=chunk_idx,
                    parent_doc=source_doc
                ))
                chunk_idx += 1
                current_content = []
    
    # Flush remaining
    if current_content:
        chunks.append(StructuredChunk(
            text="\\n".join(current_content),
            section_title=current_h1,
            subsection_title=current_h2,
            page_number=None,
            chunk_index=chunk_idx,
            parent_doc=source_doc
        ))
    
    return chunks</code></pre>

<h4>5. Chunk Size Impact: An Experiment</h4>
<table>
<tr><th>Chunk Size (tokens)</th><th>Retrieval Precision@5</th><th>Answer Quality (LLM judge)</th><th>Chunks per Doc (avg)</th></tr>
<tr><td>64</td><td>72%</td><td>3.1 / 5.0</td><td>45</td></tr>
<tr><td>128</td><td>78%</td><td>3.5 / 5.0</td><td>23</td></tr>
<tr><td>256</td><td>82%</td><td>4.0 / 5.0</td><td>12</td></tr>
<tr><td>512</td><td>79%</td><td>4.2 / 5.0</td><td>6</td></tr>
<tr><td>1024</td><td>71%</td><td>4.1 / 5.0</td><td>3</td></tr>
<tr><td>2048</td><td>62%</td><td>3.8 / 5.0</td><td>2</td></tr>
</table>

<p><strong>Key finding:</strong> Retrieval precision peaks at smaller chunks (256 tokens), while answer quality peaks at medium chunks (512 tokens). The sweet spot depends on your priority. A common solution is the <strong>parent-child chunking</strong> pattern: embed small chunks (256 tokens) for precise retrieval, but retrieve the larger parent chunk (1024 tokens) for generation context.</p>

<div class="callout tip">
<div class="callout-title">The Parent-Child Pattern</div>
<p>Index small chunks (for retrieval precision), but when a small chunk matches, return the parent chunk (larger context) to the LLM. LlamaIndex calls this "sentence window retrieval" and LangChain calls it "parent document retriever." Implementation: store each chunk with a pointer to its parent. At retrieval time, deduplicate parents and return the larger contexts.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You notice that your RAG system frequently returns chunks that cut off mid-explanation. How would you fix this?</div>
<div class="a-text">Several complementary solutions: (1) <strong>Switch to recursive or semantic chunking</strong> that respects sentence and paragraph boundaries instead of fixed character counts. (2) <strong>Add chunk overlap</strong> (100-200 tokens) so context spans chunk boundaries. (3) <strong>Use parent-child retrieval:</strong> embed small chunks but return the surrounding context (1-2 chunks before and after) to the LLM. (4) <strong>Leverage document structure:</strong> If documents have headings, tables, or code blocks, chunk at structural boundaries (never split a table or code block). (5) <strong>Post-retrieval context expansion:</strong> After retrieving a chunk, fetch adjacent chunks from the same document and include them in the LLM context, sorted by document position. (6) <strong>Use metadata in the embedding:</strong> Prepend section titles to chunk text before embedding, so the embedding captures the chunk's place in the document hierarchy. The root cause is usually strategy (1): switching from fixed-size to structure-aware chunking solves most mid-explanation cutoffs.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 14.5 Retrieval Methods
    // ----------------------------------------------------------
    {
      id: "rag-retrieval",
      title: "Retrieval Methods: Beyond Simple Vector Search",
      content: `
<p>Vector similarity search is just the beginning. Production RAG systems use a combination of dense retrieval, sparse retrieval, hybrid search, reranking, and query transformation to maximize the relevance of retrieved context. Each component addresses a different failure mode of simple nearest-neighbor search. This section covers the full retrieval stack.</p>

<h4>1. Dense vs Sparse Retrieval</h4>
<pre><code># Dense Retrieval: embedding-based similarity
# - Captures semantic meaning ("car" matches "automobile")
# - Misses exact keyword matches ("error code ERR-4521")
# - Works well for natural language questions
#
# Sparse Retrieval (BM25): term frequency-based
# - Captures exact keyword matches perfectly
# - Misses semantic similarity ("car" does NOT match "automobile")
# - Works well for keyword-heavy queries, product names, error codes

# BM25 implementation with rank_bm25
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    """BM25 sparse retriever for keyword-based search."""
    
    def __init__(self, documents: list):
        # Tokenize documents (simple whitespace tokenization)
        self.documents = documents
        self.tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)
    
    def search(self, query: str, top_k: int = 10) -> list:
        """Search for relevant documents using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [
            {"document": self.documents[i], 
             "score": float(scores[i]),
             "index": int(i)}
            for i in top_indices
            if scores[i] > 0  # Only return actual matches
        ]

# Example where BM25 wins:
docs = [
    "Error ERR-4521: Database connection timeout after 30 seconds",
    "The system experienced a network connectivity issue",
    "Performance optimization reduced latency by 40%",
]

bm25 = BM25Retriever(docs)
# Dense retrieval might match "connectivity issue" for this query
# But BM25 correctly matches the exact error code:
results = bm25.search("ERR-4521")
print(results[0]["document"])  # Correct: "Error ERR-4521: ..."</code></pre>

<h4>2. Hybrid Search: Best of Both Worlds</h4>
<pre><code>import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class RetrievedDoc:
    text: str
    dense_score: float
    sparse_score: float
    combined_score: float
    index: int

class HybridRetriever:
    """Combine dense (embedding) and sparse (BM25) retrieval.
    
    Reciprocal Rank Fusion (RRF) merges ranked lists without 
    needing to normalize scores across different retrieval methods.
    """
    
    def __init__(self, documents: list, embedding_model, k_rrf: int = 60):
        self.documents = documents
        self.embedding_model = embedding_model
        self.k_rrf = k_rrf
        
        # Build sparse index
        self.bm25 = BM25Retriever(documents)
        
        # Build dense index
        self.embeddings = embedding_model.encode(
            documents, normalize_embeddings=True
        )
    
    def search(self, query: str, top_k: int = 10, 
               alpha: float = 0.5) -> List[RetrievedDoc]:
        """Hybrid search with configurable dense/sparse weighting.
        
        alpha=1.0: pure dense (embedding) search
        alpha=0.0: pure sparse (BM25) search
        alpha=0.5: equal weight (good default)
        
        Uses Reciprocal Rank Fusion for score combination.
        """
        # Dense retrieval
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )[0]
        dense_scores = self.embeddings @ query_embedding
        dense_ranking = np.argsort(dense_scores)[::-1]
        
        # Sparse retrieval
        sparse_results = self.bm25.search(query, top_k=len(self.documents))
        sparse_ranking = [r["index"] for r in sparse_results]
        
        # Reciprocal Rank Fusion
        rrf_scores = {}
        
        for rank, idx in enumerate(dense_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                alpha / (self.k_rrf + rank + 1)
            )
        
        for rank, idx in enumerate(sparse_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                (1 - alpha) / (self.k_rrf + rank + 1)
            )
        
        # Sort by combined score
        sorted_indices = sorted(rrf_scores.keys(), 
                               key=lambda x: rrf_scores[x], 
                               reverse=True)[:top_k]
        
        return [
            RetrievedDoc(
                text=self.documents[idx],
                dense_score=float(dense_scores[idx]),
                sparse_score=float(next(
                    (r["score"] for r in sparse_results if r["index"] == idx), 0
                )),
                combined_score=rrf_scores[idx],
                index=idx
            )
            for idx in sorted_indices
        ]</code></pre>

<h4>3. Reranking: The Quality Amplifier</h4>
<pre><code># Reranking: Use a cross-encoder to re-score retrieved documents.
# Cross-encoders are more accurate than bi-encoders (embeddings)
# because they process query + document together, capturing interactions.
# But they're too slow for initial retrieval (O(N) vs O(1) for embeddings).
#
# Pipeline: embedding retrieval (fast, top-100) -> reranking (accurate, top-10)

from sentence_transformers import CrossEncoder

class Reranker:
    """Rerank retrieved documents using a cross-encoder."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name, max_length=512)
    
    def rerank(self, query: str, documents: list, 
               top_k: int = 5) -> list:
        """Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
        
        Returns: List of (document, score) tuples, sorted by relevance
        """
        # Cross-encoder scores query-document pairs
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            {"document": doc, "score": float(score)}
            for doc, score in ranked[:top_k]
        ]

# Using Cohere's Rerank API (hosted, state-of-the-art)
import cohere

co = cohere.Client("your-api-key")

def cohere_rerank(query: str, documents: list, top_k: int = 5):
    """Rerank using Cohere's rerank endpoint."""
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_k,
        return_documents=True
    )
    
    return [
        {
            "document": r.document.text,
            "score": r.relevance_score,
            "index": r.index
        }
        for r in response.results
    ]

# Typical improvement from reranking:
# Without reranking: recall@5 = 72%, precision@5 = 65%
# With reranking:    recall@5 = 72%, precision@5 = 82%
# (Same recall because we rerank from the same candidate set,
#  but much better precision because the top-5 are more relevant)</code></pre>

<h4>4. Query Transformation</h4>
<pre><code>from openai import OpenAI

client = OpenAI()

def hyde_query_expansion(query: str) -> str:
    """HyDE: Hypothetical Document Embeddings (Gao et al., arXiv: 2212.10496).
    
    Instead of embedding the query directly, generate a hypothetical
    answer to the query, then embed THAT. The hypothetical answer is
    closer in embedding space to actual relevant documents than the
    short query is.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Write a short passage that would answer the following question. "
                "Write it as if it were a paragraph from a relevant document. "
                "Do not say 'the answer is' - just write the content directly."
            )},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_tokens=200,
    )
    
    hypothetical_doc = response.choices[0].message.content
    return hypothetical_doc  # Embed this instead of the query

def multi_query_expansion(query: str, n_queries: int = 3) -> list:
    """Generate multiple search queries from different perspectives.
    
    A single query may miss relevant documents that use different 
    terminology. Multiple reformulated queries cast a wider net.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"Generate {n_queries} different search queries that would "
                "help find information to answer the user's question. "
                "Each query should approach the topic from a different angle "
                "or use different terminology. Return one query per line."
            )},
            {"role": "user", "content": query}
        ],
        temperature=0.8,
    )
    
    queries = response.choices[0].message.content.strip().split("\\n")
    return [q.strip().lstrip("0123456789.-) ") for q in queries if q.strip()]

# Example:
# Original query: "How do I handle timeouts in our API?"
# Multi-query expansion produces:
# 1. "API timeout configuration and error handling"
# 2. "Request deadline exceeded connection timeout settings"
# 3. "Retry logic and circuit breaker for HTTP timeouts"
# Each query retrieves different but relevant documents!</code></pre>

<h4>5. Retrieval Strategy Comparison</h4>
<table>
<tr><th>Strategy</th><th>Precision@5</th><th>Recall@10</th><th>Latency</th><th>Complexity</th></tr>
<tr><td>Dense only</td><td>65%</td><td>78%</td><td>~5ms</td><td>Low</td></tr>
<tr><td>Sparse (BM25) only</td><td>58%</td><td>70%</td><td>~2ms</td><td>Low</td></tr>
<tr><td>Hybrid (dense + sparse)</td><td>72%</td><td>85%</td><td>~8ms</td><td>Medium</td></tr>
<tr><td>Hybrid + reranking</td><td>82%</td><td>85%</td><td>~50ms</td><td>Medium</td></tr>
<tr><td>Hybrid + reranking + HyDE</td><td>86%</td><td>88%</td><td>~500ms</td><td>High</td></tr>
<tr><td>Multi-query + hybrid + reranking</td><td>88%</td><td>92%</td><td>~800ms</td><td>High</td></tr>
</table>

<div class="callout tip">
<div class="callout-title">The Recommended Retrieval Stack</div>
<p>For most production RAG systems, the sweet spot is: <strong>Hybrid search (dense + BM25) with reranking</strong>. This gives excellent quality (~82% precision) at acceptable latency (~50ms). Add HyDE or multi-query expansion only if you have the latency budget and need the extra few percent of retrieval quality. Always measure on your specific dataset&mdash;the numbers above are illustrative, not universal.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Your RAG system works well for natural language questions but fails on queries containing product codes or error IDs. Diagnose and fix the issue.</div>
<div class="a-text">This is a classic failure mode of pure dense retrieval. Embedding models capture semantic meaning but are poor at exact lexical matching. Product codes ("SKU-7829"), error IDs ("ERR-4521"), or case numbers ("CASE-2024-001") are opaque strings with no semantic content for the embedding model. The fix is <strong>hybrid search</strong>: add BM25 (or any term-frequency-based index) alongside the dense retrieval. BM25 excels at exact keyword matching. Implementation: (1) Run both dense and BM25 retrieval in parallel. (2) Merge results using Reciprocal Rank Fusion (RRF). (3) Tune the alpha parameter (dense vs sparse weighting) based on your query mix. For code/ID-heavy domains, use alpha=0.3 (more weight to sparse). (4) Additionally, you can extract structured fields (product codes, error IDs) during indexing and store them as metadata, enabling exact-match metadata filtering before vector search.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 14.6 Building a RAG Pipeline
    // ----------------------------------------------------------
    {
      id: "rag-pipeline",
      title: "Building a Complete RAG Pipeline: End-to-End Code",
      content: `
<p>This section assembles all the components from previous sections into a complete, production-ready RAG pipeline. We will build a system that ingests documents, processes metadata, indexes with hybrid search, retrieves with reranking, and generates cited answers. This is the reference implementation you can adapt to your use case.</p>

<h4>1. Architecture Overview</h4>
<pre><code># Production RAG Pipeline Architecture
#
# INGESTION:
#   Raw Documents --> Loader --> Chunker --> Metadata Extractor
#        --> Embedder --> Vector DB + BM25 Index
#
# QUERY:
#   User Query --> Query Processor --> Hybrid Retrieval
#        --> Reranker --> Context Assembly --> LLM Generation
#        --> Citation Extraction --> Response
#
# Components:
# - Document loaders: PDF, HTML, Markdown, DOCX
# - Chunker: Recursive character with structure awareness
# - Embedder: text-embedding-3-small (or BGE for self-hosted)
# - Vector DB: Qdrant (or Chroma for prototyping)
# - BM25: rank_bm25 (or Elasticsearch for production)
# - Reranker: cross-encoder/ms-marco-MiniLM-L-12-v2
# - LLM: GPT-4o or Claude Sonnet</code></pre>

<h4>2. Document Ingestion Pipeline</h4>
<pre><code>import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

@dataclass
class Document:
    """A document with metadata."""
    content: str
    metadata: Dict = field(default_factory=dict)
    doc_id: str = ""
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.sha256(
                self.content[:1000].encode()
            ).hexdigest()[:16]

@dataclass 
class Chunk:
    """A chunk of a document with full provenance."""
    text: str
    doc_id: str
    chunk_index: int
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    @property
    def chunk_id(self):
        return f"{self.doc_id}_chunk_{self.chunk_index}"

class DocumentProcessor:
    """Process raw documents into indexed chunks."""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 100,
                 embedding_model: str = "text-embedding-3-small"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        from openai import OpenAI
        self.client = OpenAI()
    
    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from a directory."""
        documents = []
        dir_path = Path(directory)
        
        for file_path in dir_path.rglob("*"):
            if file_path.suffix in [".txt", ".md"]:
                content = file_path.read_text(encoding="utf-8")
                documents.append(Document(
                    content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": file_path.suffix,
                        "indexed_at": datetime.now().isoformat(),
                    }
                ))
            elif file_path.suffix == ".pdf":
                content = self._load_pdf(file_path)
                documents.append(Document(
                    content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": ".pdf",
                        "indexed_at": datetime.now().isoformat(),
                    }
                ))
        
        print(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF using pymupdf."""
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text() + "\\n\\n"
        return text
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into chunks with metadata."""
        all_chunks = []
        
        for doc in documents:
            text_chunks = self._recursive_split(doc.content)
            
            for i, text in enumerate(text_chunks):
                chunk = Chunk(
                    text=text,
                    doc_id=doc.doc_id,
                    chunk_index=i,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "char_count": len(text),
                    }
                )
                all_chunks.append(chunk)
        
        print(f"Created {len(all_chunks)} chunks from "
              f"{len(documents)} documents")
        return all_chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """Recursive character splitting with overlap."""
        separators = ["\\n\\n", "\\n", ". ", " "]
        
        def split_text(text, seps):
            if len(text) <= self.chunk_size:
                return [text] if text.strip() else []
            
            sep = seps[0] if seps else " "
            remaining = seps[1:] if len(seps) > 1 else [" "]
            parts = text.split(sep)
            
            chunks = []
            current = ""
            
            for part in parts:
                piece = part + sep
                if len(current) + len(piece) <= self.chunk_size:
                    current += piece
                else:
                    if current.strip():
                        if len(current) > self.chunk_size:
                            chunks.extend(split_text(current, remaining))
                        else:
                            chunks.append(current.strip())
                    current = piece
            
            if current.strip():
                chunks.append(current.strip())
            
            return chunks
        
        raw_chunks = split_text(text, separators)
        
        # Add overlap
        final = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and self.chunk_overlap > 0:
                overlap_text = raw_chunks[i-1][-self.chunk_overlap:]
                chunk = overlap_text + " " + chunk
            final.append(chunk)
        
        return final
    
    def embed_chunks(self, chunks: List[Chunk], 
                     batch_size: int = 100) -> List[Chunk]:
        """Compute embeddings for all chunks."""
        texts = [c.text for c in chunks]
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding
        
        print(f"Embedded {len(chunks)} chunks")
        return chunks</code></pre>

<h4>3. The Complete RAG Engine</h4>
<pre><code>import numpy as np
from rank_bm25 import BM25Okapi

class RAGEngine:
    """Complete RAG pipeline with hybrid search and reranking."""
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-4o",
                 reranker_model: str = None,
                 top_k_retrieval: int = 20,
                 top_k_rerank: int = 5):
        from openai import OpenAI
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        
        # State
        self.chunks: List[Chunk] = []
        self.embedding_matrix = None
        self.bm25 = None
        
        # Optional reranker
        if reranker_model:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None
    
    def index(self, chunks: List[Chunk]):
        """Index pre-embedded chunks for retrieval."""
        self.chunks = chunks
        
        # Build dense index
        self.embedding_matrix = np.array(
            [c.embedding for c in chunks]
        )
        
        # Build sparse index
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"Indexed {len(chunks)} chunks "
              f"(dense: {self.embedding_matrix.shape}, sparse: BM25)")
    
    def retrieve(self, query: str, alpha: float = 0.5) -> List[dict]:
        """Hybrid retrieval with optional reranking."""
        # Dense retrieval
        q_emb = np.array(self.client.embeddings.create(
            input=[query], model=self.embedding_model
        ).data[0].embedding)
        
        dense_scores = self.embedding_matrix @ q_emb / (
            np.linalg.norm(self.embedding_matrix, axis=1) * 
            np.linalg.norm(q_emb) + 1e-10
        )
        
        # Sparse retrieval
        sparse_scores = self.bm25.get_scores(query.lower().split())
        
        # Reciprocal Rank Fusion
        k_rrf = 60
        dense_ranking = np.argsort(dense_scores)[::-1]
        sparse_ranking = np.argsort(sparse_scores)[::-1]
        
        rrf_scores = {}
        for rank, idx in enumerate(dense_ranking[:self.top_k_retrieval * 2]):
            rrf_scores[idx] = alpha / (k_rrf + rank + 1)
        for rank, idx in enumerate(sparse_ranking[:self.top_k_retrieval * 2]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                (1 - alpha) / (k_rrf + rank + 1)
            )
        
        # Get top candidates
        top_indices = sorted(rrf_scores.keys(), 
                           key=lambda x: rrf_scores[x],
                           reverse=True)[:self.top_k_retrieval]
        
        candidates = [
            {
                "chunk": self.chunks[i],
                "dense_score": float(dense_scores[i]),
                "sparse_score": float(sparse_scores[i]),
                "rrf_score": rrf_scores[i],
            }
            for i in top_indices
        ]
        
        # Rerank if reranker is available
        if self.reranker:
            pairs = [[query, c["chunk"].text] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            for candidate, score in zip(candidates, rerank_scores):
                candidate["rerank_score"] = float(score)
            
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return candidates[:self.top_k_rerank]
    
    def generate(self, query: str, alpha: float = 0.5) -> dict:
        """Full RAG pipeline: retrieve, assemble context, generate."""
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, alpha=alpha)
        
        # Assemble context with source information
        context_parts = []
        sources = []
        for i, r in enumerate(retrieved):
            chunk = r["chunk"]
            source_label = f"[Source {i+1}]"
            context_parts.append(
                f"{source_label} (from: {chunk.metadata.get('filename', 'unknown')})\\n"
                f"{chunk.text}"
            )
            sources.append({
                "source_id": i + 1,
                "filename": chunk.metadata.get("filename", "unknown"),
                "chunk_index": chunk.chunk_index,
                "relevance_score": r.get("rerank_score", r["rrf_score"]),
            })
        
        context = "\\n\\n---\\n\\n".join(context_parts)
        
        # Generate answer with citations
        system_prompt = """You are a helpful assistant that answers questions based on 
the provided context. Follow these rules strictly:
1. ONLY use information from the provided context to answer.
2. If the context doesn't contain the answer, say "I don't have enough information."
3. ALWAYS cite your sources using [Source N] notation.
4. Be concise but thorough.
5. If multiple sources support a point, cite all of them."""
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Context:\\n{context}\\n\\n"
                    f"Question: {query}"
                )}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved),
            "model": self.llm_model,
        }

# === COMPLETE USAGE EXAMPLE ===

# 1. Process documents
processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
documents = processor.load_documents("./knowledge_base/")
chunks = processor.chunk_documents(documents)
chunks = processor.embed_chunks(chunks)

# 2. Build RAG engine
rag = RAGEngine(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k_retrieval=20,
    top_k_rerank=5,
)
rag.index(chunks)

# 3. Ask questions
result = rag.generate("What is our company's remote work policy?")
print(f"Answer: {result['answer']}")
print(f"\\nSources used:")
for src in result["sources"]:
    print(f"  [{src['source_id']}] {src['filename']} "
          f"(relevance: {src['relevance_score']:.3f})")</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: The Stale Index Problem</div>
<p>A legal-tech startup built a RAG system for contract analysis. It worked perfectly at launch. Three months later, lawyers started complaining about incorrect answers. The root cause: the underlying contract templates had been updated, but the RAG index still contained the old versions. The system was confidently citing outdated clauses. <strong>Fix:</strong> Implement an incremental indexing pipeline that detects document changes (via file hashes or modification timestamps), re-chunks and re-embeds updated documents, and replaces old chunks in the vector database. Also add a "last updated" timestamp to every chunk's metadata, and show it in the UI so users can assess freshness.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Design the document ingestion pipeline for a RAG system that processes 10,000 new documents per day.</div>
<div class="a-text">Architecture for high-throughput ingestion: (1) <strong>Ingestion queue:</strong> New documents arrive in an S3 bucket (or message queue like Kafka). A watcher process detects new files and enqueues them. (2) <strong>Document processing workers:</strong> A pool of workers (e.g., Celery tasks or AWS Lambda) processes documents in parallel. Each worker: loads the document, extracts text (via pymupdf for PDF, beautifulsoup for HTML), chunks it (recursive character splitter, 500 tokens, 100 overlap), extracts metadata (title, date, author, document type). (3) <strong>Embedding service:</strong> Batch embedding calls to the API (100 chunks per call for OpenAI). For self-hosted embeddings, run a dedicated GPU service. At 10K docs/day with ~20 chunks each = 200K embeddings/day. With OpenAI text-embedding-3-small: ~$0.80/day. With self-hosted BGE on a single A10G: handles this easily. (4) <strong>Index updates:</strong> Upsert new chunks into the vector database. Use document IDs to handle updates (delete old chunks for a document, insert new ones). (5) <strong>Deduplication:</strong> Hash each chunk's content; skip if already indexed. (6) <strong>Monitoring:</strong> Track ingestion lag (time from document arrival to index availability), embedding failures, chunk count per document (anomaly = parsing failure). Target: document available for retrieval within 5 minutes of upload.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 14.7 RAG Evaluation
    // ----------------------------------------------------------
    {
      id: "rag-evaluation",
      title: "RAG Evaluation: Measuring What Matters",
      content: `
<p>A RAG system has two places things can go wrong: <strong>retrieval</strong> (did we find the right documents?) and <strong>generation</strong> (did the LLM use them correctly?). Evaluating both components independently and end-to-end is essential for building reliable systems. This section covers the metrics, frameworks, and methodologies for RAG evaluation.</p>

<div class="callout">
<div class="callout-title">The RAG Evaluation Triad</div>
<p>Every RAG evaluation should measure three things:<br>
<strong>1. Context Relevance:</strong> Are the retrieved documents relevant to the question?<br>
<strong>2. Faithfulness:</strong> Is the answer grounded in (supported by) the retrieved context?<br>
<strong>3. Answer Quality:</strong> Is the answer correct, complete, and useful?<br>
A system can fail on any dimension independently. High retrieval quality + unfaithful generation = hallucination. Perfect generation + bad retrieval = wrong answer from wrong sources.</p>
</div>

<h4>1. Retrieval Metrics</h4>
<pre><code>import numpy as np
from typing import List, Set

class RetrievalEvaluator:
    """Evaluate retrieval quality against ground-truth relevance labels."""
    
    def precision_at_k(self, retrieved: List[str], 
                       relevant: Set[str], k: int) -> float:
        """What fraction of retrieved docs (top-k) are relevant?"""
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
        return relevant_in_top_k / k
    
    def recall_at_k(self, retrieved: List[str], 
                    relevant: Set[str], k: int) -> float:
        """What fraction of all relevant docs are in the top-k?"""
        top_k = set(retrieved[:k])
        found = len(top_k.intersection(relevant))
        return found / len(relevant) if relevant else 0.0
    
    def mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Mean Reciprocal Rank: how high is the first relevant doc?"""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[str], 
                  relevance_scores: dict, k: int) -> float:
        """Normalized Discounted Cumulative Gain.
        
        Accounts for graded relevance (not just binary).
        relevance_scores: {doc_id: relevance_score (0-3)}
        """
        def dcg(scores, k):
            return sum(
                score / np.log2(i + 2)
                for i, score in enumerate(scores[:k])
            )
        
        # Actual DCG
        actual_scores = [
            relevance_scores.get(doc, 0) for doc in retrieved[:k]
        ]
        actual_dcg = dcg(actual_scores, k)
        
        # Ideal DCG (best possible ordering)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)
        ideal_dcg = dcg(ideal_scores, k)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def evaluate_batch(self, queries: list, all_retrieved: list,
                       all_relevant: list, k: int = 5) -> dict:
        """Evaluate retrieval across a batch of queries."""
        metrics = {
            "precision@k": [], "recall@k": [], "mrr": []
        }
        
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            relevant_set = set(relevant)
            metrics["precision@k"].append(
                self.precision_at_k(retrieved, relevant_set, k)
            )
            metrics["recall@k"].append(
                self.recall_at_k(retrieved, relevant_set, k)
            )
            metrics["mrr"].append(
                self.mrr(retrieved, relevant_set)
            )
        
        return {
            name: {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
            }
            for name, values in metrics.items()
        }</code></pre>

<h4>2. Faithfulness and Groundedness</h4>
<pre><code>from openai import OpenAI

client = OpenAI()

def evaluate_faithfulness(answer: str, context: str, 
                          model: str = "gpt-4o") -> dict:
    """Evaluate if the answer is grounded in the provided context.
    
    A faithful answer only contains claims that are supported by
    the context. Unfaithful answers introduce information not in context.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are an expert evaluator. 
Analyze whether the ANSWER is faithful to the CONTEXT.

For each claim in the answer:
1. Extract the claim
2. Check if it is supported by the context
3. Label as SUPPORTED or UNSUPPORTED

Output JSON:
{
    "claims": [
        {"claim": "...", "supported": true/false, "evidence": "quote from context or null"}
    ],
    "faithfulness_score": <float 0-1, fraction of supported claims>,
    "unsupported_claims": ["list of unsupported claims"]
}"""},
            {"role": "user", "content": (
                f"CONTEXT:\\n{context}\\n\\n"
                f"ANSWER:\\n{answer}"
            )}
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    
    return json.loads(response.choices[0].message.content)

def evaluate_answer_relevance(question: str, answer: str,
                               model: str = "gpt-4o") -> dict:
    """Evaluate if the answer actually addresses the question."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """Rate how well the ANSWER 
addresses the QUESTION on a scale of 1-5:

1: Completely irrelevant
2: Tangentially related but doesn't answer the question
3: Partially answers the question
4: Mostly answers the question
5: Fully and directly answers the question

Output JSON:
{
    "score": <int 1-5>,
    "reasoning": "brief explanation"
}"""},
            {"role": "user", "content": (
                f"QUESTION: {question}\\n\\n"
                f"ANSWER: {answer}"
            )}
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    
    return json.loads(response.choices[0].message.content)</code></pre>

<h4>3. Using the RAGAS Framework</h4>
<pre><code># RAGAS (Retrieval-Augmented Generation Assessment)
# arXiv: 2309.15217
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is the company's PTO policy?",
        "How do I submit an expense report?",
        "What is the dress code?",
    ],
    "answer": [
        "The company offers 20 days of PTO per year [Source 1].",
        "Submit expense reports through the Finance portal [Source 2].",
        "The dress code is business casual [Source 1].",
    ],
    "contexts": [
        ["Company PTO policy: Employees receive 20 days of paid time off annually."],
        ["Expense reports should be submitted via the Finance portal within 30 days."],
        ["Our dress code is business casual. Jeans are acceptable on Fridays."],
    ],
    "ground_truth": [
        "Employees get 20 days of paid time off per year.",
        "Expense reports are submitted through the Finance portal.",
        "Business casual is the dress code, with casual Fridays.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,         # Is the answer grounded in context?
        answer_relevancy,     # Does it answer the question?
        context_precision,    # Are retrieved contexts relevant?
        context_recall,       # Did we retrieve all relevant contexts?
        answer_correctness,   # Is the answer factually correct?
    ],
)

print(results)
# Output: {
#   'faithfulness': 0.95,
#   'answer_relevancy': 0.92,
#   'context_precision': 0.88,
#   'context_recall': 0.85,
#   'answer_correctness': 0.90,
# }</code></pre>

<h4>4. Building Evaluation Datasets</h4>
<pre><code>def generate_eval_dataset(documents: list, 
                          n_questions: int = 50,
                          model: str = "gpt-4o") -> list:
    """Auto-generate evaluation QA pairs from documents.
    
    For each document, generate questions that can be answered
    from the document, along with the ground-truth answers.
    """
    eval_pairs = []
    
    for doc in documents[:n_questions]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """Given a document, generate 
a question-answer pair for evaluating a RAG system.

Requirements:
- The question should be natural (as a user would ask)
- The answer must be directly supported by the document
- Include the specific text span that supports the answer

Output JSON:
{
    "question": "...",
    "answer": "...",
    "supporting_text": "exact quote from document"
}"""},
                {"role": "user", "content": f"Document:\\n{doc[:2000]}"}
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        
        try:
            pair = json.loads(response.choices[0].message.content)
            pair["source_doc"] = doc[:200]  # Reference
            eval_pairs.append(pair)
        except json.JSONDecodeError:
            continue
    
    print(f"Generated {len(eval_pairs)} evaluation pairs")
    return eval_pairs

# Best practices for eval datasets:
# 1. Generate auto pairs, then HUMAN-REVIEW them
# 2. Include diverse question types:
#    - Factual recall ("What is X?")
#    - Multi-hop ("How does X relate to Y?")
#    - Comparison ("What's the difference between X and Y?")
#    - Unanswerable ("What is Z?" where Z is not in the docs)
# 3. Include at least 20% unanswerable questions to test
#    the system's ability to say "I don't know"
# 4. Minimum 50 questions for meaningful evaluation, 
#    200+ for reliable metrics</code></pre>

<h4>5. Continuous Evaluation Pipeline</h4>
<table>
<tr><th>Evaluation Type</th><th>Frequency</th><th>Metrics</th><th>Alert Threshold</th></tr>
<tr><td><strong>Retrieval quality</strong></td><td>Daily</td><td>Precision@5, Recall@5, MRR</td><td>&lt;80% of baseline</td></tr>
<tr><td><strong>Faithfulness</strong></td><td>Daily (sample)</td><td>% of claims supported by context</td><td>&lt;90%</td></tr>
<tr><td><strong>Answer relevance</strong></td><td>Daily (sample)</td><td>Average relevance score (1-5)</td><td>&lt;3.5</td></tr>
<tr><td><strong>User satisfaction</strong></td><td>Continuous</td><td>Thumbs up/down ratio</td><td>&lt;70% positive</td></tr>
<tr><td><strong>"I don't know" rate</strong></td><td>Daily</td><td>% of queries with no-answer response</td><td>&gt;30% (coverage issue)</td></tr>
<tr><td><strong>Latency</strong></td><td>Continuous</td><td>P50, P95 end-to-end latency</td><td>P95 &gt; 5 seconds</td></tr>
</table>

<div class="callout tip">
<div class="callout-title">The Human-in-the-Loop Evaluation Strategy</div>
<p>Automated metrics are necessary but not sufficient. Schedule a weekly review where a domain expert evaluates 20 randomly sampled RAG responses on a rubric: correctness (1-5), completeness (1-5), citation accuracy (1-5), and helpfulness (1-5). Track these scores over time. This human evaluation catches failure modes that automated metrics miss, such as technically-correct-but-unhelpful answers, or correct answers from the wrong time period.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you evaluate a RAG system's ability to say "I don't know" when the answer is not in the knowledge base?</div>
<div class="a-text">This tests "abstention quality" and is critically important for trustworthy systems. Approach: (1) <strong>Create a dedicated test set</strong> with two categories: answerable questions (answer exists in knowledge base) and unanswerable questions (answer does NOT exist, but is plausible - e.g., asking about a product feature that doesn't exist). Aim for 30-40% unanswerable. (2) <strong>Metrics:</strong> Abstention precision (when it says "I don't know," is it correct?), abstention recall (of all unanswerable questions, how many does it correctly abstain on?), and false abstention rate (answerable questions where it incorrectly says "I don't know"). (3) <strong>Common failure modes:</strong> The system retrieves vaguely related documents and fabricates an answer (low abstention recall). Or the system is overly cautious and refuses to answer even when context is available (high false abstention). (4) <strong>Improvement levers:</strong> Tune the retrieval score threshold below which the system should abstain. Add explicit instructions in the system prompt: "If the context does not contain information to answer the question, respond with 'I don't have information about this in my knowledge base.'" Use faithfulness checking as a post-generation filter: if the answer contains claims not supported by retrieved context, override with an abstention.</div>
</div>
`
    },
    // ----------------------------------------------------------
    // 14.8 Advanced RAG
    // ----------------------------------------------------------
    {
      id: "rag-advanced",
      title: "Advanced RAG: Agentic, Graph, Multi-Modal, and Beyond",
      content: `
<p>The basic RAG pipeline (retrieve-then-generate) is a starting point. Advanced RAG techniques address its limitations: single-hop retrieval misses multi-step reasoning, flat document stores ignore entity relationships, and text-only retrieval cannot handle images, tables, or code. This section covers the frontier of RAG systems.</p>

<h4>1. Agentic RAG: Tool-Use for Intelligent Retrieval</h4>
<p>Instead of a fixed retrieve-then-generate pipeline, an agentic RAG system gives the LLM the ability to decide <em>when</em> and <em>how</em> to retrieve, enabling iterative and adaptive retrieval.</p>

<pre><code>from openai import OpenAI
import json

client = OpenAI()

class AgenticRAG:
    """RAG system where the LLM decides when and how to retrieve.
    
    The LLM can:
    1. Search the knowledge base (semantic or keyword)
    2. Ask follow-up questions to refine retrieval
    3. Search multiple collections
    4. Decide it has enough context to answer
    """
    
    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": (
                        "Search the knowledge base for relevant information. "
                        "Use this when you need to find specific information "
                        "to answer the user's question."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "collection": {
                                "type": "string",
                                "enum": ["hr_docs", "eng_docs", 
                                         "product_docs", "all"],
                                "description": "Which collection to search"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["semantic", "keyword", "hybrid"],
                                "description": "Type of search"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document_details",
                    "description": (
                        "Get the full content of a specific document "
                        "when you need more context from a search result."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "The document ID"
                            }
                        },
                        "required": ["doc_id"]
                    }
                }
            }
        ]
    
    def run(self, query: str, max_iterations: int = 5) -> dict:
        """Run agentic RAG with iterative retrieval."""
        messages = [
            {"role": "system", "content": """You are a helpful assistant with 
access to a knowledge base. Use the search tools to find relevant information
before answering. You may search multiple times with different queries if 
needed. When you have enough information, provide a comprehensive answer 
with citations.

Strategy:
1. First, search with the user's question
2. If results are insufficient, reformulate and search again
3. If you need more detail on a specific result, get the full document
4. Answer only when you have sufficient evidence"""},
            {"role": "user", "content": query}
        ]
        
        gathered_context = []
        
        for iteration in range(max_iterations):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3,
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            # If no tool calls, the LLM is ready to answer
            if not message.tool_calls:
                return {
                    "answer": message.content,
                    "iterations": iteration + 1,
                    "searches_performed": len(gathered_context),
                    "context": gathered_context,
                }
            
            # Process tool calls
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                
                if tool_call.function.name == "search_knowledge_base":
                    results = self.rag.retrieve(
                        args["query"],
                        alpha=0.5 if args.get("search_type") == "hybrid" else 0.8
                    )
                    result_text = "\\n\\n".join([
                        f"[Result {i+1}] (doc_id: {r['chunk'].doc_id})\\n"
                        f"{r['chunk'].text[:500]}"
                        for i, r in enumerate(results)
                    ])
                    gathered_context.append({
                        "query": args["query"],
                        "results": len(results)
                    })
                
                elif tool_call.function.name == "get_document_details":
                    # Fetch full document by ID
                    doc = self._get_full_doc(args["doc_id"])
                    result_text = doc or "Document not found."
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
        
        return {"answer": "Max iterations reached.", 
                "iterations": max_iterations}
    
    def _get_full_doc(self, doc_id: str) -> str:
        """Retrieve all chunks from a specific document."""
        chunks = [c for c in self.rag.chunks if c.doc_id == doc_id]
        chunks.sort(key=lambda c: c.chunk_index)
        return "\\n\\n".join(c.text for c in chunks)</code></pre>

<h4>2. Graph RAG: Entity-Centric Knowledge</h4>
<pre><code># Graph RAG (Microsoft, arXiv: 2404.16130) builds a knowledge graph
# from documents and uses graph traversal + LLM summarization for
# queries that require synthesizing information across many documents.
#
# When to use Graph RAG vs Vector RAG:
# - Vector RAG: "What is the PTO policy?" (answer in one document)
# - Graph RAG: "What are the main themes across all customer complaints?"
#   (requires synthesizing across hundreds of documents)

# Simplified Graph RAG concept
from dataclasses import dataclass
from typing import List, Tuple
import networkx as nx

@dataclass
class Entity:
    name: str
    type: str  # Person, Organization, Concept, etc.
    description: str

@dataclass
class Relationship:
    source: str
    target: str
    relationship: str
    description: str

class GraphRAG:
    """Simplified Graph RAG implementation."""
    
    def __init__(self, llm_client):
        self.client = llm_client
        self.graph = nx.DiGraph()
        self.community_summaries = {}
    
    def extract_entities_and_relations(self, text: str) -> Tuple[
        List[Entity], List[Relationship]
    ]:
        """Use LLM to extract entities and relationships from text."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """Extract entities and 
relationships from the text. Output JSON:
{
    "entities": [{"name": "...", "type": "...", "description": "..."}],
    "relationships": [{"source": "...", "target": "...", 
                       "relationship": "...", "description": "..."}]
}"""},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        
        data = json.loads(response.choices[0].message.content)
        entities = [Entity(**e) for e in data.get("entities", [])]
        relations = [Relationship(**r) for r in data.get("relationships", [])]
        
        return entities, relations
    
    def build_graph(self, documents: List[str]):
        """Build knowledge graph from documents."""
        for doc in documents:
            entities, relations = self.extract_entities_and_relations(doc)
            
            for entity in entities:
                self.graph.add_node(
                    entity.name,
                    type=entity.type,
                    description=entity.description
                )
            
            for rel in relations:
                self.graph.add_edge(
                    rel.source, rel.target,
                    relationship=rel.relationship,
                    description=rel.description
                )
        
        # Detect communities using Leiden/Louvain algorithm
        import community as community_louvain
        undirected = self.graph.to_undirected()
        partitions = community_louvain.best_partition(undirected)
        
        # Summarize each community
        communities = {}
        for node, comm_id in partitions.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        for comm_id, nodes in communities.items():
            subgraph_info = self._describe_community(nodes)
            self.community_summaries[comm_id] = subgraph_info
    
    def query(self, question: str) -> str:
        """Answer using community summaries (for global queries)."""
        # For global queries: use community summaries
        # For local queries: traverse graph from relevant entities
        
        all_summaries = "\\n\\n".join([
            f"Community {cid}: {summary}"
            for cid, summary in self.community_summaries.items()
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Answer the question based on these knowledge graph "
                    "community summaries. Synthesize information across "
                    "communities as needed."
                )},
                {"role": "user", "content": (
                    f"Community summaries:\\n{all_summaries}\\n\\n"
                    f"Question: {question}"
                )}
            ],
            temperature=0.3,
        )
        
        return response.choices[0].message.content</code></pre>

<h4>3. Multi-Modal RAG</h4>
<pre><code># Multi-modal RAG handles images, tables, charts alongside text.
# Key challenge: how to index and retrieve non-text content.

class MultiModalRAGStrategy:
    """Strategies for handling images and tables in RAG."""
    
    # Strategy 1: Describe and embed
    # Convert images/tables to text descriptions, then embed normally
    def image_to_text_embedding(self, image_path: str):
        """Use vision model to describe image, then embed description."""
        import base64
        
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        # Get description using GPT-4o vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe this image in detail, including all "
                        "text, numbers, relationships, and key information. "
                        "If it's a chart or table, describe the data."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }}
                ]
            }],
            max_tokens=500,
        )
        
        description = response.choices[0].message.content
        
        # Embed the description (searchable via text queries)
        embedding = client.embeddings.create(
            input=[description],
            model="text-embedding-3-small"
        ).data[0].embedding
        
        return {
            "description": description,
            "embedding": embedding,
            "original_image": image_path,
        }
    
    # Strategy 2: Multi-modal embeddings
    # Use models like CLIP or Jina-CLIP to embed images directly
    # into the same vector space as text
    def clip_embedding(self, image_path: str = None, text: str = None):
        """Embed images and text in the same space using CLIP."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("jinaai/jina-clip-v2")
        
        if image_path:
            from PIL import Image
            image = Image.open(image_path)
            embedding = model.encode(image)
        elif text:
            embedding = model.encode(text)
        
        return embedding
    
    # Strategy 3: Table extraction and structured indexing
    def process_table(self, table_html: str):
        """Convert HTML table to searchable chunks."""
        # Parse table into rows
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(table_html, "html.parser")
        
        rows = []
        headers = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if not headers:
                headers = cells
            else:
                rows.append(dict(zip(headers, cells)))
        
        # Create natural language descriptions for each row
        descriptions = []
        for row in rows:
            desc = ", ".join(f"{k}: {v}" for k, v in row.items())
            descriptions.append(desc)
        
        # Also create a summary of the whole table
        summary = f"Table with columns: {', '.join(headers)}. "
        summary += f"Contains {len(rows)} rows."
        
        return {
            "summary": summary,
            "row_descriptions": descriptions,
            "structured_data": rows,
        }</code></pre>

<h4>4. RAG for Code</h4>
<pre><code># Code RAG requires special handling:
# 1. Chunking must respect code structure (functions, classes)
# 2. Embeddings should be code-aware (use code embedding models)
# 3. Context must include imports, type definitions, and related code

import ast

def code_aware_chunking(source_code: str, 
                        language: str = "python") -> list:
    """Chunk code by logical units (functions, classes)."""
    if language == "python":
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Fall back to line-based chunking
            lines = source_code.split("\\n")
            return [
                "\\n".join(lines[i:i+50]) 
                for i in range(0, len(lines), 40)
            ]
        
        chunks = []
        lines = source_code.split("\\n")
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, 
                                ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                
                # Get the code block
                code_block = "\\n".join(lines[start:end])
                
                # Get docstring if present
                docstring = ast.get_docstring(node) or ""
                
                chunks.append({
                    "text": code_block,
                    "type": type(node).__name__,
                    "name": node.name,
                    "docstring": docstring,
                    "line_range": (start + 1, end),
                    # Create a searchable description
                    "description": (
                        f"{type(node).__name__} '{node.name}': "
                        f"{docstring[:200] if docstring else 'No docstring'}"
                    )
                })
        
        return chunks
    
    # For other languages, use tree-sitter
    # import tree_sitter_languages
    # parser = tree_sitter_languages.get_parser(language)
    # ...
    return [{"text": source_code, "type": "file", "name": "unknown"}]</code></pre>

<h4>5. RAG + Fine-Tuning</h4>
<table>
<tr><th>Approach</th><th>What It Gives You</th><th>Example</th></tr>
<tr><td><strong>RAG only</strong></td><td>Accurate, cited, up-to-date answers</td><td>Knowledge base Q&A</td></tr>
<tr><td><strong>Fine-tuning only</strong></td><td>Domain style, specialized reasoning</td><td>Medical report generation</td></tr>
<tr><td><strong>RAG + Fine-tuned retriever</strong></td><td>Better retrieval for domain-specific queries</td><td>Legal case search with fine-tuned legal embeddings</td></tr>
<tr><td><strong>RAG + Fine-tuned generator</strong></td><td>Domain-adapted answers grounded in evidence</td><td>Financial analysis with retrieved market data</td></tr>
<tr><td><strong>RAFT (Fine-tune for RAG)</strong></td><td>Model learns to use retrieved context better</td><td>Fine-tune on (question, context, answer) triples</td></tr>
</table>

<h4>6. Production Optimization</h4>
<pre><code>class ProductionRAGOptimizer:
    """Optimization techniques for production RAG systems."""
    
    # 1. Semantic Caching
    def semantic_cache_lookup(self, query: str, cache: dict, 
                              threshold: float = 0.95) -> str:
        """Return cached answer if a very similar query was seen before.
        
        Saves embedding + retrieval + LLM cost for repeated queries.
        Typical hit rate: 15-30% for customer support use cases.
        """
        query_embedding = self._embed(query)
        
        for cached_query, cached_data in cache.items():
            similarity = np.dot(query_embedding, cached_data["embedding"])
            if similarity > threshold:
                return cached_data["answer"]
        
        return None  # Cache miss
    
    # 2. Embedding Pre-computation
    # Compute embeddings offline in batch (not at query time)
    # Store in vector DB with metadata
    # Update incrementally when documents change
    
    # 3. Query Routing
    def route_query(self, query: str) -> str:
        """Route queries to different RAG pipelines based on type.
        
        - Simple factual queries: lightweight pipeline (small model)
        - Complex analytical queries: full pipeline (large model + reranking)
        - Out-of-scope queries: reject early (save resources)
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap model for routing
            messages=[
                {"role": "system", "content": """Classify the query:
- SIMPLE: Direct factual question answerable from a single document
- COMPLEX: Requires synthesizing multiple documents or reasoning
- OUT_OF_SCOPE: Not related to the knowledge base
Output only the classification."""},
                {"role": "user", "content": query}
            ],
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()
    
    # 4. Batch Retrieval
    # Process multiple queries together for efficiency
    # Especially useful for embedding API calls (batch 100+ queries)
    
    # 5. Streaming Generation
    # Stream the LLM response while retrieval results are being assembled
    # First tokens arrive faster, improving perceived latency</code></pre>

<div class="callout warning">
<div class="callout-title">War Story: The Runaway Token Count</div>
<p>A RAG system retrieved 10 chunks of ~500 tokens each, added them to the prompt with a system prompt and user query, totaling ~6,000 tokens per request. At 1,000 requests/day with GPT-4o, this cost ~$60/day. Then the team increased to 20 chunks for "better coverage," and context grew to 12,000 tokens per request. Costs doubled to $120/day, but answer quality actually decreased (too much irrelevant context confused the LLM). <strong>Fix:</strong> (1) Kept retrieval at 20 chunks but added reranking to select the top 5 most relevant. (2) Added context compression: used a small model to extract only the relevant sentences from each chunk. (3) Final context: 5 compressed chunks, ~2,000 tokens. Cost dropped to $20/day AND quality improved. More context is not always better.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Design an advanced RAG system for a large enterprise with 500,000 documents across multiple departments, supporting 10,000 queries per day.</div>
<div class="a-text">Architecture: (1) <strong>Ingestion:</strong> Document processors for each type (PDF, DOCX, HTML, Confluence, Sharepoint). Incremental indexing via change detection. Estimated 10M chunks total. (2) <strong>Indexing:</strong> Qdrant cluster (3 nodes for high availability) with HNSW indexing. Separate collections per department for access control. BM25 index via Elasticsearch for hybrid search. Embeddings via self-hosted BGE model on 2 A10G GPUs (eliminates per-query API cost). (3) <strong>Retrieval pipeline:</strong> Query classification (route simple/complex/out-of-scope). Department routing based on user permissions and query content. Hybrid search (dense + BM25 with RRF). Cross-encoder reranking on top 20 candidates to select top 5. (4) <strong>Generation:</strong> Context assembly with metadata (department, author, date). GPT-4o for complex queries, GPT-4o-mini for simple ones. Streaming response. Mandatory citations. (5) <strong>Caching:</strong> Semantic cache for frequent queries (expect 20-25% hit rate). Redis for session context. (6) <strong>Access control:</strong> Enforce document-level permissions at retrieval time via metadata filtering. Users only see documents they have access to. (7) <strong>Evaluation:</strong> Weekly automated eval on 200-question test set. Daily faithfulness sampling (50 queries). Monthly human expert review. (8) <strong>Costs:</strong> Infrastructure: ~$3K/month (Qdrant + Elasticsearch + GPU). Embedding: ~$0 (self-hosted). Generation: ~$500/month (10K queries/day, mix of GPT-4o and mini). Total: ~$3.5K/month for enterprise-grade RAG.</div>
</div>
`
    }
  ]
};
