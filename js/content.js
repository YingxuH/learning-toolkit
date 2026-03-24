// Textbook content organized by parts and chapters
const TEXTBOOK = {
  lastUpdated: "2026-03-24",
  changelog: [
    { date: "2026-03-24", text: "Added 6 Production War Stories: SD batch-size backfire, vLLM OOM debugging, Whisper hallucination fix, LoRA data corruption, NCCL timeout, Agent infinite loop" },
    { date: "2026-03-24", text: "Fixed KV-cache calculations with GQA formula; updated diffusion step comparisons" },
    { date: "2026-03-24", text: "Initial release: 10 chapters covering Audio AI, LLM Inference, ML Training, and Career Prep" }
  ],
  readingGoals: [
    { id: "week1", label: "Week 1: Audio AI Foundations", chapters: ["audio-llm-landscape", "speech-to-speech", "tts-technology"] },
    { id: "week2", label: "Week 2: LLM Inference", chapters: ["speculative-decoding", "vllm-serving"] },
    { id: "week3", label: "Week 3: Training & Engineering", chapters: ["rl-training", "ml-engineering"] },
    { id: "week4", label: "Week 4: Applications & Interview", chapters: ["agent-development", "system-design", "interview-prep"] }
  ],
  parts: [
    {
      title: "Foundations of Audio AI",
      chapters: [
        {
          id: "audio-llm-landscape",
          title: "Audio LLM Research Landscape",
          sections: [
            {
              id: "audio-llm-overview",
              title: "Overview & Architecture Evolution",
              content: `
<p>Audio Large Language Models (AudioLLMs) represent a paradigm shift from specialized audio models to unified architectures that can understand and generate both text and audio. The field has evolved rapidly from 2023 to 2025, moving through distinct phases.</p>

<div class="callout">
<div class="callout-title">Key Insight</div>
<p>The core architectural pattern for AudioLLMs: <strong>Audio Encoder + Adapter + LLM Backbone + Decoder</strong>. The encoder converts audio to representations, the adapter bridges modalities, the LLM reasons, and the decoder generates output.</p>
</div>

<h4>Foundational Papers (2023-2024)</h4>
<table>
<tr><th>Paper</th><th>Key Innovation</th><th>Impact</th></tr>
<tr><td><strong>Pengi</strong> (NeurIPS 2023)</td><td>All audio tasks as text-generation; audio encoder + text encoder as prefix to frozen LM</td><td>Unified audio-text generation; unlocked open-ended audio QA</td></tr>
<tr><td><strong>SALMONN</strong> (ICLR 2024)</td><td>Dual encoder (Whisper + BEATs) with Q-Former adapter to Vicuna</td><td>First to study cross-modal emergent capabilities</td></tr>
<tr><td><strong>Qwen-Audio</strong></td><td>30+ tasks, hierarchical tag conditioning to solve multi-task interference</td><td>Proved scale + task taxonomy beats hand-crafted models</td></tr>
<tr><td><strong>AudioPaLM</strong></td><td>Joint audio-text vocabulary; first to generate audio tokens directly from LLM</td><td>Opened the end-to-end generation paradigm</td></tr>
</table>

<h4>The 2025 Frontier</h4>
<p>The field branched into several exciting directions:</p>
<ul>
<li><strong>Omni Models:</strong> Qwen2.5-Omni, Qwen3-Omni, and Kimi-Audio achieved all-modal input with text/speech output, including streaming capabilities</li>
<li><strong>Reasoning in Audio:</strong> Audio Flamingo Sound-CoT introduced systematic audio chain-of-thought; AudSemThinker grounded reasoning in structured auditory semantics</li>
<li><strong>Long Context:</strong> CALM uses continuous audio tokens (VAE) instead of discrete codecs; YaRN + VLAT extended context windows substantially</li>
<li><strong>Domain Specialization:</strong> SeaLLMs-Audio for Southeast Asian languages; FinAudio for financial audio analysis</li>
</ul>

<pre><code>2023-2024 FOUNDATION              2025 FRONTIER
---------------------------------------------------
[Encoder+LLM Architecture]   ->  [Omni: all-modal, streaming]
[Multi-task training]         ->  [Reasoning: CoT, RL, RL+CoT]
[General benchmarks]          ->  [Domain benchmarks (finance, SEA)]
[Text output]                 ->  [Audio-in-audio reasoning]
[Fixed context (<30s)]        ->  [Long audio (YaRN, CALM)]
[Cascade vs. E2E debate]     ->  [Cascade comeback vs. Omni models]</code></pre>
`
            },
            {
              id: "audio-neglect",
              title: "The Audio Neglect Problem",
              content: `
<p><strong>Audio Neglect</strong> is a critical finding from 2025: AudioLLMs systematically under-utilize audio evidence. The text-pretrained LLM backbone is so powerful that it answers from language priors, effectively ignoring the actual audio signal.</p>

<div class="callout warning">
<div class="callout-title">Critical Research Gap</div>
<p>2025 research showed models ignore decisive audio even when it's the only valid signal. The proposed fix (attention steering via audio-specialist heads) is ad hoc. A principled, general solution doesn't exist yet.</p>
</div>

<p>This finding calls into question the validity of many published AudioLLM results. If models are primarily using text priors rather than actually processing audio, benchmark numbers may be misleading.</p>

<h4>The X-Talk Counter-Narrative</h4>
<p>Meanwhile, X-Talk demonstrated that modular ASR to LLM to TTS cascades remain competitive with end-to-end systems, challenging the "omni is always better" narrative. The key insight: <strong>deployment robustness does not equal benchmark performance</strong>.</p>

<h4>Research Direction: Reasoning Substrate</h4>
<p>Should AudioLLMs reason in text tokens (fast, mature, but loses paralinguistic detail) or audio tokens (preserves acoustics, but expensive and evaluation is undefined)?</p>

<p>Drawing from human cognition: humans think in language, not in sounds, even when processing audio. We extract concepts from audio, then reason over concepts. This suggests:</p>
<ul>
<li>Full audio-token reasoning is likely neither natural nor necessary</li>
<li>The real value is <strong>hybrid</strong>: text CoT as primary reasoning + selective audio anchors at key decision points</li>
</ul>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is the "Audio Neglect" problem in AudioLLMs, and how would you design an experiment to measure it?</div>
<div class="a-text">Audio Neglect refers to AudioLLMs ignoring decisive audio evidence and relying on text priors from the LLM backbone. To measure it, design tasks where: (1) the correct answer requires audio information that cannot be inferred from text alone (e.g., speaker emotion, environmental sounds), (2) create adversarial pairs where text context suggests one answer but audio evidence points to another, (3) measure accuracy with and without audio input - if performance barely changes, the model is neglecting audio.</div>
</div>
`
            },
            {
              id: "research-taste",
              title: "Building Research Taste in Audio AI",
              content: `
<p><strong>Research taste</strong> is the compass that tells you <em>which</em> problem is worth solving before you touch data. It's distinct from research skill (how to execute).</p>

<h4>The 10 Questions Framework</h4>
<p>Apply these to every paper you read. Taste is trained by running this drill consistently:</p>
<ol>
<li><strong>Core claim?</strong> One sentence. If you can't write it, you haven't understood the paper.</li>
<li><strong>What was previously broken?</strong> Not "it achieves SOTA" - what was actually broken before this?</li>
<li><strong>Key architectural/methodological choice - why not the obvious alternative?</strong></li>
<li><strong>What would Reviewer 2 say?</strong> Weak baselines? Artificial tasks? Vague contributions?</li>
<li><strong>Who cites this - and who conspicuously doesn't?</strong> Tells you if it started a lineage or ended one.</li>
<li><strong>What does it NOT solve?</strong> Read Limitations. The biggest clues to future papers live there.</li>
<li><strong>What becomes possible if the claim is true?</strong> The "unlock" question.</li>
<li><strong>Is the eval metric measuring what actually matters?</strong></li>
<li><strong>Simplest baseline that could undermine the paper?</strong></li>
<li><strong>If this paper disappeared from history, what wouldn't exist today?</strong> The "leverage" question.</li>
</ol>

<div class="callout tip">
<div class="callout-title">Taste Development Plan</div>
<p><strong>Weeks 1-4 (Map):</strong> Read 30 papers; build concept map; ask "if this disappeared, what wouldn't exist?"<br>
<strong>Weeks 5-8 (Filter):</strong> Reverse-engineer 5 accepted papers; read 5 borderline rejections; weekly idea triage<br>
<strong>Weeks 9-16 (Engage):</strong> Follow key researchers; attend talks; write monthly Reviewer 2 critiques<br>
<strong>Weeks 17-24 (Test):</strong> 2-week prototype drills; submit workshop paper; write intro before experiments</p>
</div>
`
            }
          ]
        },
        {
          id: "speech-to-speech",
          title: "Speech-to-Speech Models",
          sections: [
            {
              id: "s2s-taxonomy",
              title: "Architecture Taxonomy",
              content: `
<p>Speech-to-Speech (S2S) models can be classified by their architecture type and interaction capabilities:</p>

<table>
<tr><th>Type</th><th>Description</th><th>Examples</th></tr>
<tr><td><strong>Half-Duplex E2E</strong></td><td>Listen-then-speak; single LLM backbone; no simultaneous I/O</td><td>LLaMA-Omni, Mini-Omni</td></tr>
<tr><td><strong>Full-Duplex</strong></td><td>Simultaneous listening and speaking; continuous stream processing</td><td>Moshi, OmniFlatten, LSLM</td></tr>
<tr><td><strong>Infrastructure</strong></td><td>Datasets, evaluation frameworks, TTS components</td><td>VoiceAssistant-400K</td></tr>
<tr><td><strong>Alignment/Safety</strong></td><td>Preference alignment, adversarial robustness</td><td>SpeechAlign</td></tr>
</table>

<h4>Moshi: The Full-Duplex Pioneer</h4>
<p>Moshi (Kyutai, 2024) is the first genuinely real-time full-duplex spoken language model. Key innovations:</p>
<ul>
<li><strong>Dual-stream architecture:</strong> One stream for model's speech, one for user's</li>
<li><strong>RQ-Transformer:</strong> Operates on Mimi codec tokens with 160ms theoretical latency</li>
<li><strong>Inner Monologue:</strong> Text reasoning tokens generated in parallel with audio tokens, letting the model "think while speaking"</li>
</ul>

<div class="callout">
<div class="callout-title">Key Concept: Inner Monologue</div>
<p>The Inner Monologue mechanism allows the model to maintain a text-based reasoning stream alongside audio generation. This is analogous to how humans can think about what to say next while still speaking.</p>
</div>

<h4>The Stream Flattening Approach (OmniFlatten)</h4>
<p>OmniFlatten solved a critical training challenge: how to go from a half-duplex pretrained LLM to full-duplex without catastrophic forgetting. The solution is a 3-stage pipeline:</p>
<ol>
<li>Text-only half-duplex training</li>
<li>Speech half-duplex training</li>
<li>Speech full-duplex training with stream flattening (interleaving user and model audio chunks into a single causal sequence)</li>
</ol>
`
            },
            {
              id: "s2s-latency",
              title: "Latency & Real-Time Considerations",
              content: `
<p>For natural conversation, response latency must be under ~300ms. Here's how key systems compare:</p>

<table>
<tr><th>System</th><th>Latency</th><th>Architecture</th><th>Trade-off</th></tr>
<tr><td>Moshi</td><td>200ms</td><td>Full-duplex, RQ-Transformer</td><td>Content quality lags text LLMs</td></tr>
<tr><td>LLaMA-Omni</td><td>226ms</td><td>Half-duplex, Whisper+LLaMA</td><td>No interruption handling</td></tr>
<tr><td>Mini-Omni</td><td>~300ms</td><td>Think-while-speaking</td><td>0.5B too small for knowledge tasks</td></tr>
<tr><td>SyncLLM</td><td>Variable</td><td>Time-slotted FD</td><td>Fixed-chunk limits prosody</td></tr>
</table>

<h4>Key Design Decisions for Low-Latency S2S</h4>
<ul>
<li><strong>Codec choice:</strong> Discrete codecs (EnCodec, Mimi) enable fast token-by-token generation but quantization loses acoustic detail. Continuous representations (SALMONN-Omni) preserve quality but are harder to stream.</li>
<li><strong>Streaming architecture:</strong> Chunk-based processing with overlapping windows. The chunk size creates a latency floor.</li>
<li><strong>KV-cache management:</strong> For long conversations, efficient KV-cache pruning or compression is essential.</li>
</ul>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What are the key trade-offs between half-duplex and full-duplex speech-to-speech systems?</div>
<div class="a-text">Half-duplex systems (like LLaMA-Omni) are simpler - they listen, then speak. Benefits: easier to train, can leverage existing LLMs, more stable. Drawbacks: no interruption handling, unnatural conversation flow. Full-duplex systems (like Moshi) can listen and speak simultaneously. Benefits: natural conversation, barge-in support, backchanneling. Drawbacks: much harder to train (need dual-stream architecture), higher compute cost, doubled sequence length, and content quality often suffers due to the added complexity.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you reduce end-to-end latency in a speech dialogue system?</div>
<div class="a-text">Key strategies: (1) Use streaming ASR instead of waiting for complete utterances, (2) Employ speculative decoding for faster LLM inference, (3) Use streaming TTS that starts speaking before the full response is generated, (4) Implement endpoint detection to know when the user stops speaking, (5) Pre-compute likely responses (speculative generation), (6) Optimize audio codec for low-latency encoding/decoding, (7) Use model distillation for smaller, faster models.</div>
</div>
`
            }
          ]
        },
        {
          id: "tts-technology",
          title: "Text-to-Speech Technology",
          sections: [
            {
              id: "tts-evolution",
              title: "TTS Evolution: From Pipelines to Neural Codecs",
              content: `
<h4>Traditional TTS Pipeline</h4>
<p>Traditional TTS systems follow three stages:</p>
<ol>
<li><strong>Text Frontend:</strong> Text to phonemes</li>
<li><strong>Acoustic Model:</strong> Phonemes to mel-spectrogram (e.g., Tacotron 2)</li>
<li><strong>Vocoder:</strong> Mel-spectrogram to waveform (e.g., HiFi-GAN, WaveGlow)</li>
</ol>

<p><strong>Limitations:</strong> Requires 20-50 hours of single-speaker studio recordings, poor generalization across speakers, near-zero zero-shot capability.</p>

<h4>The Neural Codec Language Model Revolution</h4>
<p>VALL-E (Microsoft, 2023) fundamentally changed TTS by reframing it as a language modeling problem:</p>

<div class="callout">
<div class="callout-title">Core Insight</div>
<p>Treat TTS as language modeling, not signal regression. Use an audio codec (EnCodec) to convert waveforms to discrete tokens, then model these tokens with a Transformer, just like GPT models text.</p>
</div>

<pre><code>Text -> Phoneme Sequence -> [AR Model] -> Coarse codec tokens
                         -> [NAR Model] -> Fine codec tokens -> EnCodec Decode -> Waveform</code></pre>

<p><strong>VALL-E's key insight:</strong> Data scale matters more than model design. Training on 60,000 hours of multi-speaker data (LibriLight, 7000+ speakers) produced better results than carefully designed models trained on small datasets. The model learned to separate "what is said" from "who is speaking" automatically.</p>

<h4>Modern TTS Systems</h4>
<table>
<tr><th>System</th><th>Architecture</th><th>Key Feature</th></tr>
<tr><td><strong>CosyVoice</strong> (Alibaba)</td><td>LLM backbone + Flow Matching decoder</td><td>Streaming, multi-lingual, zero-shot cloning</td></tr>
<tr><td><strong>F5-TTS</strong></td><td>Flow-matching based, non-autoregressive</td><td>Fast inference, high quality</td></tr>
<tr><td><strong>VALL-E X</strong></td><td>Cross-lingual extension of VALL-E</td><td>Cross-language voice cloning with accent control</td></tr>
<tr><td><strong>Parler-TTS</strong></td><td>Text-described voice control</td><td>Natural language voice description</td></tr>
</table>
`
            },
            {
              id: "tts-flow-matching",
              title: "Flow Matching & Modern Generation",
              content: `
<h4>Flow Matching vs. Diffusion</h4>
<p>Flow Matching has emerged as the preferred generation mechanism in modern TTS:</p>

<p><strong>Diffusion models</strong> gradually add noise to data, then learn to reverse the process. Early diffusion required many iterations (50-1000 steps), though modern solvers (DPM-Solver, consistency distillation) can reduce this to 1-4 steps.</p>

<p><strong>Flow Matching</strong> learns a direct velocity field that transforms a simple distribution (noise) into the data distribution. It's like drawing a straight path between noise and clean audio, rather than taking many small random steps.</p>

<pre><code># Flow Matching conceptual pseudocode
# Instead of learning noise prediction (diffusion), learn velocity
def flow_matching_loss(model, x_clean, x_noise, t):
    # Interpolate between noise and clean signal
    x_t = (1 - t) * x_noise + t * x_clean
    # The "velocity" - direction from noise to clean
    target_velocity = x_clean - x_noise
    # Model predicts the velocity at time t
    predicted_velocity = model(x_t, t)
    return mse_loss(predicted_velocity, target_velocity)</code></pre>

<p><strong>Advantages of Flow Matching:</strong></p>
<ul>
<li>Fewer inference steps needed (10-50 vs 50-1000 for vanilla diffusion; though modern diffusion solvers close this gap)</li>
<li>More stable training</li>
<li>Straighter generation paths = faster convergence</li>
<li>Compatible with Optimal Transport for even more efficient paths</li>
</ul>

<h4>Audio Codecs: The Foundation Layer</h4>
<p>Modern TTS relies on neural audio codecs that compress audio into discrete tokens:</p>
<table>
<tr><th>Codec</th><th>Bitrate</th><th>Key Feature</th></tr>
<tr><td>EnCodec (Meta)</td><td>1.5-24 kbps</td><td>RVQ with 8 codebook layers</td></tr>
<tr><td>Mimi (Kyutai)</td><td>1.1 kbps</td><td>Used in Moshi, semantic + acoustic split</td></tr>
<tr><td>DAC</td><td>8 kbps</td><td>Higher quality, larger tokens</td></tr>
<tr><td>SpeechTokenizer</td><td>Variable</td><td>Separates semantic and acoustic info</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain Residual Vector Quantization (RVQ) and why it's used in audio codecs.</div>
<div class="a-text">RVQ quantizes audio representations in multiple passes. The first codebook captures the coarsest features (content, speaker identity). Each subsequent codebook quantizes the <em>residual</em> - the error left by previous codebooks - capturing progressively finer acoustic details. This hierarchical structure is key for TTS: autoregressive models can generate just the first codebook (capturing "what" and "who") while non-autoregressive models fill in the finer details in parallel. This separation enables both quality and speed.</div>
</div>
`
            }
          ]
        }
      ]
    },
    {
      title: "LLM Inference & Optimization",
      chapters: [
        {
          id: "speculative-decoding",
          title: "Speculative Decoding",
          sections: [
            {
              id: "sd-fundamentals",
              title: "Fundamentals & Core Protocol",
              content: `
<p>LLM inference is bottlenecked by <strong>memory bandwidth, not compute</strong>. Speculative decoding (SD) solves the latency problem by making one expensive forward pass verify many tokens at once.</p>

<h4>Core Protocol</h4>
<ol>
<li>A cheap <strong>draft model</strong> proposes gamma tokens autoregressively</li>
<li>The <strong>target model</strong> runs one parallel forward pass over all gamma+1 positions</li>
<li><strong>Rejection sampling</strong> accepts/rejects each token while preserving the target distribution</li>
<li>Expected tokens per step = gamma * alpha (alpha = acceptance rate), giving 2-5x speedups</li>
</ol>

<div class="callout">
<div class="callout-title">Why It Works</div>
<p>GPU utilization during autoregressive decoding is typically very low (memory-bound). SD fills those idle compute cycles by verifying multiple draft tokens in parallel. The key mathematical guarantee: rejection sampling ensures the output distribution is <em>identical</em> to the target model - SD is lossless.</p>
</div>

<h4>The Field's Branches (2024 onwards)</h4>
<ul>
<li><strong>Better Drafters:</strong> Medusa, Hydra, EAGLE series</li>
<li><strong>Better Verification:</strong> Tree attention, dynamic tree building</li>
<li><strong>Draft-free / Self-speculative:</strong> Models that speculate from their own layers</li>
<li><strong>System-level Serving:</strong> Integration with vLLM, SGLang for production</li>
<li><strong>Non-text Modalities:</strong> SD for image, audio, and video generation</li>
</ul>
`
            },
            {
              id: "sd-eagle",
              title: "The EAGLE Series: State of the Art",
              content: `
<p>The EAGLE series represents the evolution of draft-head based speculative decoding:</p>

<h4>EAGLE (ICML 2024)</h4>
<p><strong>Key insight:</strong> Autoregression at the feature level (second-to-top-layer) is simpler than at the token level. EAGLE uses both the current token AND current top-layer feature as input to a single extra transformer decoder layer.</p>
<p><strong>Results:</strong> LLaMA2-Chat 70B: 2.7-3.5x latency speedup. Outperforms Medusa by 1.5-1.6x. Lossless.</p>

<h4>EAGLE-2 (EMNLP 2024)</h4>
<p><strong>Key insight:</strong> EAGLE's draft model is well-calibrated - confidence scores approximate acceptance rates. EAGLE-2 builds <strong>context-aware dynamic draft trees</strong>, expanding high-confidence branches and pruning low-confidence ones.</p>
<p><strong>Results:</strong> 3.05-4.26x speedup. 20-40% faster than EAGLE-1.</p>

<h4>EAGLE-3 (2025)</h4>
<p><strong>Key insight:</strong> Abandons feature prediction (the ceiling of EAGLE-1/2). Directly predicts next tokens using <strong>multi-layer feature fusion</strong>. A "training-time test" technique exposes the draft to diverse contexts simulating inference.</p>
<p><strong>Results:</strong> Up to 6.5x speedup. First EAGLE variant to hold up at batch=64. SOTA for single-model SD.</p>

<pre><code># EAGLE conceptual architecture
class EAGLEDraftHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        self.feature_proj = nn.Linear(hidden_size, hidden_size)
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.decoder_layer = TransformerDecoderLayer(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, features):
        # Combine token embeddings with feature-level info
        x = self.token_embed(token_ids) + self.feature_proj(features)
        x = self.decoder_layer(x)
        return self.lm_head(x)</code></pre>

<h4>Other Important Draft Methods</h4>
<table>
<tr><th>Method</th><th>Approach</th><th>Speedup</th></tr>
<tr><td>Medusa</td><td>k independent extra LM-heads, tree attention verification</td><td>2.2-3.6x</td></tr>
<tr><td>Hydra</td><td>Sequential draft heads (each conditions on previous)</td><td>2.7x</td></tr>
<tr><td>HASS</td><td>Harmonized representation alignment + consistency training</td><td>Improved acceptance rate</td></tr>
<tr><td>Lookahead</td><td>N-gram based drafting from Jacobi iteration</td><td>~1.5-2x</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Why does speculative decoding not change the output distribution of the target model?</div>
<div class="a-text">SD uses rejection sampling to decide whether to accept each draft token. For each position, the target model computes the true probability p(x). If the draft probability q(x) <= p(x), the token is always accepted. If q(x) > p(x), it's accepted with probability p(x)/q(x). Rejected tokens are resampled from an adjusted distribution (p - q, normalized). This mathematically guarantees the final output follows exactly the target distribution. The draft model only affects speed, never quality.</div>
</div>
`
            },
            {
              id: "sd-production",
              title: "SD in Production: Serving Systems",
              content: `
<h4>The Batch Size Problem</h4>
<p>The biggest deployment blocker for SD: speedup collapses at large batch sizes. At batch=1, SD gives 3-5x speedup. At batch=64+, gains shrink dramatically because:</p>
<ul>
<li>Verification becomes compute-bound (not memory-bound) at large batches</li>
<li>Tree attention memory overhead scales with batch size</li>
<li>Draft model and target model compete for GPU resources</li>
</ul>

<h4>Integration with Serving Frameworks</h4>
<p>Modern serving systems (vLLM, SGLang, TensorRT-LLM) have built-in SD support:</p>
<ul>
<li><strong>SGLang:</strong> EAGLE-2 is the default SD method. Supports dynamic tree building with RadixAttention for prefix sharing.</li>
<li><strong>vLLM:</strong> Supports Medusa, EAGLE, and external draft models. Integrated with continuous batching.</li>
</ul>

<pre><code># Using EAGLE with SGLang
python -m sglang.launch_server \\
    --model meta-llama/Llama-3-70B-Instruct \\
    --speculative-algorithm EAGLE \\
    --speculative-draft-model eagle-llama3-70b \\
    --speculative-num-steps 5 \\
    --speculative-eagle-topk 8</code></pre>

<div class="callout tip">
<div class="callout-title">Production Tip</div>
<p>For production deployments: use EAGLE-2/3 at low concurrency (batch 1-8) for maximum latency reduction. At high concurrency (batch 32+), the baseline throughput of continuous batching often exceeds SD's benefits. Profile your specific workload.</p>
</div>

<div class="callout warning">
<div class="callout-title">Production War Story: When Speculative Decoding Backfired</div>
<p>We deployed EAGLE-2 on a Qwen-2.5-72B service expecting 3x latency reduction. At batch=1 during testing: 3.2x improvement. In production with 20 concurrent users: only 1.1x, barely worth the complexity. The draft model consumed GPU memory that could have served 30% more concurrent requests via vanilla continuous batching. <strong>Lesson:</strong> Always benchmark SD under your actual concurrency patterns, not batch=1. We switched to SD only for our low-traffic high-priority API tier and removed it from the main serving path. Throughput improved 25% after removing it.</p>
</div>
`
            }
          ]
        },
        {
          id: "vllm-serving",
          title: "LLM Serving with vLLM",
          sections: [
            {
              id: "vllm-architecture",
              title: "vLLM Architecture & PagedAttention",
              content: `
<p>vLLM is the most widely-deployed open-source LLM serving engine. Its core innovation is <strong>PagedAttention</strong>, which manages KV-cache memory like virtual memory in operating systems.</p>

<h4>The KV-Cache Problem</h4>
<p>During autoregressive generation, each token requires attention to all previous tokens. The key-value pairs (KV-cache) grow linearly with sequence length and must be stored in GPU memory. With naive allocation:</p>
<ul>
<li>A 13B model (MHA, 40 layers, 40 heads) serving 2048-token sequences needs ~1.6GB of KV-cache per request</li>
<li>Memory fragmentation wastes 60-80% of available KV-cache memory</li>
<li>Internal fragmentation: pre-allocating max sequence length wastes memory for shorter sequences</li>
<li>External fragmentation: gaps between allocated blocks can't be used</li>
</ul>

<h4>PagedAttention Solution</h4>
<p>PagedAttention divides the KV-cache into fixed-size blocks (pages). Blocks are allocated on demand and can be non-contiguous in physical memory:</p>

<pre><code># PagedAttention concept
# Instead of: one contiguous buffer per sequence
# Use: a page table mapping logical blocks to physical blocks

class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        # Physical KV-cache blocks (shared pool)
        self.k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        # Per-sequence page tables (logical -> physical mapping)
        self.page_tables = {}

    def allocate_block(self, seq_id):
        physical_block = self.free_blocks.pop()
        self.page_tables[seq_id].append(physical_block)
        return physical_block</code></pre>

<div class="callout">
<div class="callout-title">Key Benefit</div>
<p>PagedAttention reduces KV-cache memory waste from 60-80% to near zero, enabling 2-4x more concurrent requests on the same hardware.</p>
</div>

<h4>Continuous Batching</h4>
<p>Unlike static batching (wait for a batch to complete), continuous batching inserts new requests as soon as any request finishes a step. This maximizes GPU utilization and reduces time-to-first-token (TTFT).</p>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain PagedAttention and why it's important for LLM serving.</div>
<div class="a-text">PagedAttention manages KV-cache memory using a paging system inspired by OS virtual memory. Instead of pre-allocating contiguous memory for each sequence's maximum possible length, it allocates fixed-size blocks on demand. A page table maps each sequence's logical blocks to physical memory locations. Benefits: (1) near-zero memory waste from fragmentation, (2) 2-4x more concurrent requests, (3) enables memory sharing across sequences with common prefixes (e.g., system prompts), and (4) supports dynamic sequence lengths without over-allocation.</div>
</div>
`
            },
            {
              id: "vllm-optimization",
              title: "Optimization Techniques for LLM Serving",
              content: `
<h4>Quantization for Inference</h4>
<table>
<tr><th>Method</th><th>Bits</th><th>Quality Impact</th><th>Speedup</th></tr>
<tr><td>FP16</td><td>16</td><td>Baseline</td><td>1x</td></tr>
<tr><td>GPTQ</td><td>4</td><td>Minimal on most tasks</td><td>~2-3x memory reduction</td></tr>
<tr><td>AWQ</td><td>4</td><td>Slightly better than GPTQ</td><td>~2-3x memory, faster kernels</td></tr>
<tr><td>GGUF (llama.cpp)</td><td>2-8</td><td>Varies by method</td><td>CPU-friendly</td></tr>
<tr><td>FP8 (H100)</td><td>8</td><td>Near-lossless</td><td>~1.5x compute speedup</td></tr>
</table>

<h4>Tensor Parallelism vs Pipeline Parallelism</h4>
<ul>
<li><strong>Tensor Parallelism (TP):</strong> Split each layer's weight matrices across GPUs. All GPUs compute every token. Best for latency-sensitive serving with fast interconnects (NVLink).</li>
<li><strong>Pipeline Parallelism (PP):</strong> Split layers across GPUs. Each GPU handles a subset of layers. Better for throughput with slower interconnects.</li>
</ul>

<pre><code># vLLM deployment example
# 4-GPU TP for a 70B model
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3-70B-Instruct \\
    --tensor-parallel-size 4 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.9 \\
    --enable-prefix-caching</code></pre>

<h4>Prefix Caching</h4>
<p>When many requests share the same system prompt, prefix caching reuses the KV-cache for the shared prefix. This can reduce TTFT by 80%+ for chat applications with long system prompts.</p>

<div class="callout warning">
<div class="callout-title">Production War Story: The OOM That Wasn't</div>
<p>Our vLLM deployment kept OOMing at ~200 concurrent requests despite having 80GB A100s. <code>nvidia-smi</code> showed 71GB used, well under the 80GB limit. The culprit: <code>gpu-memory-utilization</code> was set to 0.9 (72GB), but vLLM reserves memory for KV-cache blocks upfront. With our 4096 max_model_len and the model weights taking 28GB, only 44GB was left for KV-cache - enough for ~180 concurrent requests at average sequence length. <strong>Fix:</strong> Reduced <code>max_model_len</code> to 2048 (our actual P99 was 1200 tokens) and increased <code>gpu-memory-utilization</code> to 0.95. Concurrent capacity jumped to 350+. <strong>Lesson:</strong> Always set <code>max_model_len</code> based on your actual traffic distribution, not the model's maximum capability.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you optimize an LLM serving deployment to handle 1000 concurrent users?</div>
<div class="a-text">Key strategies: (1) Use vLLM/SGLang with continuous batching for maximum throughput, (2) Enable prefix caching if requests share system prompts, (3) Quantize to FP8 or INT4 (AWQ) to fit more concurrent requests in GPU memory, (4) Use tensor parallelism across GPUs for the model, (5) Deploy multiple replicas behind a load balancer, (6) Implement request queuing with priority levels, (7) Set appropriate max_tokens limits, (8) Consider speculative decoding at low-medium concurrency, (9) Monitor GPU utilization, queue depth, and P95 latency, (10) Use streaming responses to reduce perceived latency.</div>
</div>
`
            }
          ]
        }
      ]
    },
    {
      title: "ML Training & Infrastructure",
      chapters: [
        {
          id: "rl-training",
          title: "RL Training for LLMs (RLHF/RLVR)",
          sections: [
            {
              id: "rlvr-fundamentals",
              title: "GRPO & Verifiable Rewards",
              content: `
<p>RLVR (Reinforcement Learning with Verifiable Rewards) has emerged as the dominant paradigm for training reasoning models. The key algorithm is <strong>GRPO (Group Relative Policy Optimization)</strong>.</p>

<h4>GRPO at Each Step</h4>
<ol>
<li><strong>Rollout:</strong> Given a prompt, sample G candidate responses from the current policy</li>
<li><strong>Reward:</strong> Score each response with a verifiable reward function (e.g., code execution, math verification)</li>
<li><strong>Update:</strong> Compute normalized advantages within the group, update policy via PPO-clip loss + KL penalty</li>
</ol>

<div class="callout">
<div class="callout-title">Key Engineering Challenge</div>
<p>Rollout is inference (benefits from vLLM/SGLang, continuous batching, KV-cache). Update is training (needs backprop, TP/PP, gradient checkpointing). The two have very different memory/compute profiles. The state-of-the-art solution: <strong>weight resharding</strong> between inference and training layouts.</p>
</div>

<h4>verl: The Production RLVR Framework</h4>
<p>verl (Volcano Engine RL) provides a 3D-HybridEngine that automatically reshards weights between vLLM (inference) and Megatron (training):</p>

<table>
<tr><th>Dimension</th><th>verl</th><th>DIY Approach</th></tr>
<tr><td>HybridEngine</td><td>Auto resharding between vLLM and Megatron</td><td>4-6 weeks to implement</td></tr>
<tr><td>Rollout throughput</td><td>vLLM/SGLang backend, 3-5x faster</td><td>HF generate only</td></tr>
<tr><td>RL algorithms</td><td>GRPO, PPO, DAPO, REINFORCE++ out of box</td><td>Manual implementation</td></tr>
<tr><td>Setup cost</td><td>~1 week to integrate</td><td>~3-4 weeks from scratch</td></tr>
</table>

<h4>Training Pipeline: Audio Agent Example</h4>
<pre><code># Stage 1: Cold-Start SFT
# Teach the model the think/answer format
model: Qwen3-8B
audio_encoder: whisper-large-v3 (frozen)
projection: 2-layer MLP (trainable)
llm: LoRA rank=64, alpha=128
batch_size: 32, lr: 2e-4, epochs: 3
hardware: 4x H100

# Stage 2: RLVR with GRPO (verl)
# Optimize for verifiable task completion
rollout: vLLM backend with custom audio worker
reward: task completion verification
training: Megatron-LM backend with TP=4

# Stage 3: Evaluation
# AudioAgentBench + out-of-distribution sets</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is GRPO and how does it differ from PPO for LLM training?</div>
<div class="a-text">GRPO (Group Relative Policy Optimization) simplifies PPO by eliminating the need for a separate critic/value model. Instead of estimating advantages using a learned value function, GRPO samples a group of G responses for each prompt and computes advantages relative to the group's mean reward. This is simpler to implement, requires less memory (no value model), and works well for tasks with verifiable rewards. The key trade-off: GRPO requires more samples per prompt (the group) but avoids the instability and complexity of training a separate critic.</div>
</div>
`
            },
            {
              id: "distributed-training",
              title: "Distributed Training Fundamentals",
              content: `
<h4>Parallelism Strategies</h4>

<p><strong>Data Parallelism (DP):</strong> Each GPU has a full model copy, processes different data. Gradients are averaged via all-reduce. Simple but memory-limited.</p>

<p><strong>Tensor Parallelism (TP):</strong> Split weight matrices across GPUs within each layer. Requires fast interconnect (NVLink). Best for intra-node parallelism.</p>

<p><strong>Pipeline Parallelism (PP):</strong> Split layers across GPU groups. Micro-batching reduces bubble overhead. Good for inter-node scaling.</p>

<p><strong>Sequence Parallelism (SP):</strong> Split the sequence dimension across GPUs. Essential for long sequences that don't fit in single-GPU memory.</p>

<p><strong>ZeRO (Zero Redundancy Optimizer):</strong></p>
<ul>
<li>ZeRO-1: Partition optimizer states across GPUs</li>
<li>ZeRO-2: + partition gradients</li>
<li>ZeRO-3: + partition parameters (most memory efficient, highest communication)</li>
</ul>

<h4>Memory Optimization</h4>
<pre><code># Gradient checkpointing - trade compute for memory
# Instead of storing all activations for backprop,
# recompute them during the backward pass
model.gradient_checkpointing_enable()

# Mixed precision training
# Use FP16/BF16 for forward/backward, FP32 for optimizer
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    loss = model(input_ids, labels=labels).loss
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()</code></pre>

<h4>Common Training Failures & Debugging</h4>
<table>
<tr><th>Symptom</th><th>Likely Cause</th><th>Fix</th></tr>
<tr><td>Loss NaN/Inf</td><td>Learning rate too high, bad data</td><td>Reduce LR, check data pipeline, use BF16</td></tr>
<tr><td>Loss plateau early</td><td>LR too low, data repetition</td><td>Increase LR, check data shuffling</td></tr>
<tr><td>OOM at step N>0</td><td>Dynamic shapes, memory leak</td><td>Fix padding, check gradient accumulation</td></tr>
<tr><td>Slow all-reduce</td><td>Network bottleneck</td><td>Check NCCL, use gradient compression</td></tr>
<tr><td>Gradient norm spikes</td><td>Bad data batch, model instability</td><td>Gradient clipping, data filtering</td></tr>
</table>

<div class="callout warning">
<div class="callout-title">Production War Story: The Silent Data Corruption</div>
<p>Our LoRA fine-tune of Qwen2.5-7B showed great validation metrics but produced gibberish in production. Root cause: our data pipeline had a race condition in multi-worker data loading that corrupted ~2% of training examples by truncating them mid-sentence. The model learned to generate truncated outputs. Validation didn't catch it because we used BLEU/ROUGE on the remaining 98% of clean data. <strong>Fix:</strong> Added data integrity checks (hash verification per batch), logged sample outputs during training (not just loss), and added a "coherence score" to validation that measures output completeness. <strong>Lesson:</strong> Always inspect actual model outputs during training, not just aggregate metrics. A low loss number can hide catastrophic failure modes.</p>
</div>

<div class="callout warning">
<div class="callout-title">Production War Story: NCCL Timeout on 8-Node Training</div>
<p>Training on 64 H100s across 8 nodes kept hanging at step ~500 with NCCL timeout errors. Single-node training worked fine. The issue: one node had a flaky InfiniBand cable that dropped packets under sustained all-reduce load. <code>ibstat</code> showed the link was "Active" but <code>perfquery</code> revealed 0.1% packet loss. <strong>Fix:</strong> Replaced the cable, added <code>NCCL_IB_TIMEOUT=23</code> and <code>NCCL_IB_RETRY_CNT=7</code> as environment variables, and set up a pre-training health check script that runs <code>all_reduce_bench</code> across all nodes before every training job. <strong>Lesson:</strong> Network issues in distributed training manifest as random hangs, not error messages. Always benchmark inter-node communication before starting a multi-day training run.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You need to train a 70B model. You have 64 H100 GPUs across 8 nodes. What parallelism strategy would you use?</div>
<div class="a-text">Recommended strategy: TP=8 (within each node, using NVLink), PP=2 (across 2 node groups), DP=4 (4 data-parallel replicas). Within each node, 8 GPUs use tensor parallelism for minimum latency. Pipeline parallelism of 2 across nodes handles the model size with reasonable pipeline bubble. 4-way data parallelism gives good throughput. Also use: gradient checkpointing to reduce activation memory, BF16 mixed precision, sequence parallelism for long sequences, ZeRO-1 for optimizer state sharding within DP groups.</div>
</div>
`
            }
          ]
        },
        {
          id: "ml-engineering",
          title: "ML Engineering Best Practices",
          sections: [
            {
              id: "benchmark-hygiene",
              title: "ML Benchmark Hygiene",
              content: `
<p>Rigorous benchmarking is the foundation of trustworthy ML research and engineering. Without it, you can't tell if your model is actually better or if you're fooling yourself.</p>

<h4>Core Principles</h4>
<ol>
<li><strong>Fix your random seeds</strong> - but run multiple seeds and report variance</li>
<li><strong>Version everything</strong> - data, code, model checkpoints, environment</li>
<li><strong>Never tune on your test set</strong> - use a separate validation set for hyperparameter selection</li>
<li><strong>Report the right metrics</strong> - accuracy alone is often misleading; include confidence intervals</li>
<li><strong>Compare fairly</strong> - same data, same compute budget, same preprocessing</li>
</ol>

<div class="callout warning">
<div class="callout-title">Common Pitfalls</div>
<p><strong>Data contamination:</strong> Your test data leaked into training. Especially common with web-scraped data.<br>
<strong>Selective reporting:</strong> Running many experiments, reporting only the best. Use Bonferroni correction or similar.<br>
<strong>Unfair baselines:</strong> Comparing your tuned model against an out-of-the-box baseline.<br>
<strong>Metric gaming:</strong> Optimizing for a metric that doesn't reflect real-world performance.</p>
</div>

<h4>Practical Checklist</h4>
<pre><code># Before running experiments:
- [ ] Reproducibility: seeds, versions, environment logged
- [ ] Data splits: train/val/test properly separated, no leakage
- [ ] Baselines: fair comparison with same compute/data budget
- [ ] Metrics: primary metric chosen before experiments

# During experiments:
- [ ] Track all runs (W&B, MLflow, or at minimum: git + logs)
- [ ] Monitor for overfitting on validation set
- [ ] Log system metrics (GPU util, memory, throughput)

# Reporting results:
- [ ] Mean +/- std over multiple runs (minimum 3)
- [ ] Statistical significance tests where appropriate
- [ ] Ablation studies for each component
- [ ] Failure cases and limitations documented</code></pre>
`
            },
            {
              id: "asr-pipeline",
              title: "ASR Pipeline Engineering",
              content: `
<p>Automatic Speech Recognition (ASR) pipelines in production involve much more than just the model. Here's a comprehensive guide to building robust ASR systems.</p>

<h4>Architecture Overview</h4>
<pre><code>Audio Input -> VAD -> Preprocessing -> ASR Model -> Post-processing -> Output
              |                          |
              v                          v
         Silence removal          Punctuation restoration
         Chunk segmentation       Inverse text normalization
         Noise detection          Speaker diarization</code></pre>

<h4>Key Components</h4>
<ul>
<li><strong>Voice Activity Detection (VAD):</strong> Silero VAD or WebRTC VAD for detecting speech segments. Critical for reducing compute and avoiding hallucination on silence.</li>
<li><strong>Preprocessing:</strong> Resampling to model's expected rate (usually 16kHz), normalization, noise reduction if needed.</li>
<li><strong>ASR Model:</strong> Whisper (OpenAI), Canary (NVIDIA), Parakeet, or specialized models for target languages.</li>
<li><strong>Post-processing:</strong> Punctuation restoration, inverse text normalization (ITN), disfluency removal.</li>
</ul>

<h4>Common Failure Modes</h4>
<table>
<tr><th>Issue</th><th>Cause</th><th>Solution</th></tr>
<tr><td>Hallucinated text on silence</td><td>No VAD, model generates text for any input</td><td>Add VAD preprocessing</td></tr>
<tr><td>Wrong language output</td><td>Language detection failure</td><td>Force language parameter or add language ID</td></tr>
<tr><td>Truncated transcription</td><td>Audio longer than model's max length</td><td>Chunk with overlap, stitch results</td></tr>
<tr><td>High WER on accented speech</td><td>Training data mismatch</td><td>Fine-tune on target accent data</td></tr>
<tr><td>Repeated phrases</td><td>Attention alignment failure</td><td>Adjust beam search, add repetition penalty</td></tr>
</table>

<div class="callout warning">
<div class="callout-title">Production War Story: Whisper Hallucinating on Silence</div>
<p>Our Singlish ASR pipeline using Whisper large-v3 produced garbage transcriptions for ~8% of audio files. Investigation revealed these files contained long silence segments (>5s) where Whisper hallucinated repeated phrases like "Thank you for watching" or random Chinese text. The root cause: no Voice Activity Detection (VAD) preprocessing. <strong>Fix:</strong> Added Silero VAD as a preprocessing step to trim silence and segment audio. Hallucination rate dropped from 8% to 0.3%. Additional improvement: set <code>no_speech_threshold=0.6</code> and <code>condition_on_previous_text=False</code> to prevent hallucination cascading. <strong>Lesson:</strong> VAD is not optional in production ASR - it's your first line of defense against the model's tendency to generate text from noise.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you build a production ASR system that handles multiple languages and accents?</div>
<div class="a-text">Architecture: (1) Language/accent identification module at input (either from metadata or a classifier on the first few seconds), (2) Route to the appropriate ASR model or use a multilingual model like Whisper with forced language tokens, (3) For accented speech, fine-tune on accent-specific data using LoRA for efficient multi-adapter serving, (4) Implement VAD to handle silence and cross-talk, (5) Post-processing pipeline for language-specific punctuation and ITN rules, (6) Confidence scoring to flag low-quality transcriptions for human review, (7) Continuous monitoring of WER by language/accent segment with automatic alerts on degradation.</div>
</div>
`
            },
            {
              id: "pytorch-gpu",
              title: "PyTorch GPU Service Patterns",
              content: `
<h4>GPU Memory Management</h4>
<pre><code>import torch

# Check GPU memory
def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Clear cache when switching between models
torch.cuda.empty_cache()

# Use memory-efficient attention (PyTorch 2.0+)
from torch.nn.functional import scaled_dot_product_attention
# Automatically selects FlashAttention, Memory-Efficient, or Math backend

# Inference optimization
with torch.inference_mode():  # Faster than no_grad()
    output = model(input_ids)</code></pre>

<h4>Common GPU Service Patterns</h4>
<ol>
<li><strong>Model Loading:</strong> Load model to CPU first, then move to GPU. For multi-GPU, use device_map="auto".</li>
<li><strong>Batched Inference:</strong> Dynamic batching with timeout to balance latency and throughput.</li>
<li><strong>Streaming:</strong> Use generators and Server-Sent Events for token-by-token output.</li>
<li><strong>Health Checks:</strong> Monitor GPU temperature, memory, and model responsiveness.</li>
</ol>

<h4>Debugging GPU Issues</h4>
<table>
<tr><th>Tool</th><th>Use Case</th></tr>
<tr><td><code>nvidia-smi</code></td><td>GPU utilization, memory, temperature</td></tr>
<tr><td><code>torch.cuda.memory_summary()</code></td><td>Detailed memory breakdown</td></tr>
<tr><td><code>CUDA_LAUNCH_BLOCKING=1</code></td><td>Synchronize for accurate error locations</td></tr>
<tr><td><code>torch.autograd.set_detect_anomaly(True)</code></td><td>Find NaN/Inf sources in backward pass</td></tr>
<tr><td>PyTorch Profiler</td><td>Kernel-level timing and memory analysis</td></tr>
</table>
`
            }
          ]
        }
      ]
    },
    {
      title: "Software Engineering for AI",
      chapters: [
        {
          id: "agent-development",
          title: "AI Agent Development",
          sections: [
            {
              id: "agent-patterns",
              title: "Agent Architecture Patterns",
              content: `
<p>AI agents extend LLMs with the ability to take actions, use tools, and maintain state across interactions. Key architectural patterns:</p>

<h4>ReAct Pattern (Reasoning + Acting)</h4>
<pre><code># The ReAct loop
while not done:
    # Reason: LLM thinks about what to do
    thought = llm.generate(prompt + observations)
    # Act: Execute the chosen tool
    action = parse_action(thought)
    observation = execute_tool(action)
    # Update: Add observation to context
    prompt += f"Thought: {thought}\\nAction: {action}\\nObservation: {observation}"</code></pre>

<h4>Tool-Use Patterns</h4>
<ul>
<li><strong>Function Calling:</strong> LLM outputs structured JSON for tool invocation. Used by OpenAI, Anthropic, Google APIs.</li>
<li><strong>Code Generation:</strong> LLM writes and executes code. More flexible but requires sandboxing.</li>
<li><strong>Multi-step Planning:</strong> LLM creates a plan, then executes steps sequentially or in parallel.</li>
</ul>

<h4>Memory Systems for Agents</h4>
<table>
<tr><th>Type</th><th>Implementation</th><th>Use Case</th></tr>
<tr><td>Short-term</td><td>Conversation context window</td><td>Current task state</td></tr>
<tr><td>Working memory</td><td>Scratchpad / structured state</td><td>Multi-step reasoning</td></tr>
<tr><td>Long-term</td><td>Vector DB + retrieval</td><td>Past interactions, knowledge</td></tr>
<tr><td>Episodic</td><td>Logged interaction history</td><td>Learning from past experiences</td></tr>
</table>

<div class="callout tip">
<div class="callout-title">Self-Improving Agent Pattern</div>
<p>A self-improving agent learns from its own execution: (1) Execute task, (2) Evaluate outcome, (3) Extract learnings, (4) Update memory/prompts. Key: separate the "execution" from the "reflection" phase. The reflection should happen with full context of what went wrong and why.</p>
</div>

<div class="callout warning">
<div class="callout-title">Production War Story: Agent Infinite Loop in Production</div>
<p>Our code-generation agent entered an infinite loop in production: it would write code, the test would fail, it would "fix" the code by reverting to the original, the test would fail again, repeat. Cost us $400 in API calls before the timeout hit. Root cause: the agent's context window filled up with repeated failed attempts, pushing out the original error message that explained the actual bug. <strong>Fix:</strong> (1) Added a hard limit of 5 retries per sub-task, (2) Implemented "error deduplication" - if the same error appears 3x, escalate to a different strategy instead of retrying, (3) Added a sliding window that always preserves the first error message. <strong>Lesson:</strong> Agents need circuit breakers just like microservices. Without explicit loop detection, LLMs will confidently repeat the same mistake forever.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you design a multi-agent system where agents can collaborate on complex tasks?</div>
<div class="a-text">Design: (1) Define agent roles with clear responsibilities (e.g., Planner, Researcher, Coder, Reviewer), (2) Use a message-passing architecture where agents communicate through a shared message queue, (3) Implement a Planner agent that decomposes tasks and assigns sub-tasks, (4) Each agent has its own tools and memory but can access shared context, (5) Use structured output formats for inter-agent communication, (6) Implement conflict resolution (e.g., Reviewer can send back to Coder), (7) Add human-in-the-loop checkpoints for critical decisions, (8) Monitor agent interactions for loops or deadlocks.</div>
</div>
`
            },
            {
              id: "agent-dev-practices",
              title: "Agent-Assisted Development",
              content: `
<p>Using AI agents for software development requires specific practices to maintain code quality and developer productivity.</p>

<h4>Iterative Development with AI</h4>
<ol>
<li><strong>Specification First:</strong> Write clear requirements before asking AI to code. Ambiguous specs lead to wasted iterations.</li>
<li><strong>Small Increments:</strong> Break tasks into small, verifiable pieces. Each piece should be testable independently.</li>
<li><strong>Review Rigorously:</strong> AI-generated code needs the same (or more) review as human code. Watch for: hallucinated APIs, security issues, over-engineering.</li>
<li><strong>Test-Driven:</strong> Write tests first, then use AI to implement. Tests serve as an executable specification.</li>
</ol>

<h4>Effective Prompting for Code Generation</h4>
<pre><code># Bad prompt:
"Write a function to process audio"

# Good prompt:
"Write a Python function process_audio(file_path: str) -> np.ndarray that:
1. Loads a WAV or MP3 file using librosa
2. Resamples to 16kHz mono
3. Normalizes amplitude to [-1, 1]
4. Applies VAD to trim silence (energy threshold -40dB)
5. Returns the processed numpy array
Include type hints and handle FileNotFoundError."</code></pre>

<h4>Common Anti-Patterns</h4>
<ul>
<li><strong>Prompt-and-pray:</strong> Dumping a vague description and hoping for the best. Always be specific.</li>
<li><strong>Blind trust:</strong> Accepting AI code without understanding it. You own the code.</li>
<li><strong>Context overload:</strong> Feeding the entire codebase. Focus on relevant files and interfaces.</li>
<li><strong>Skipping tests:</strong> "It looks right" is not a test. AI code needs automated verification.</li>
</ul>
`
            }
          ]
        },
        {
          id: "system-design",
          title: "System Design for AI Applications",
          sections: [
            {
              id: "ml-system-design",
              title: "ML System Design Patterns",
              content: `
<h4>The ML System Design Interview Framework</h4>
<ol>
<li><strong>Clarify Requirements</strong> (2-3 min): Functional requirements, scale, latency, accuracy targets</li>
<li><strong>High-Level Design</strong> (5 min): Data pipeline, model architecture, serving infrastructure</li>
<li><strong>Deep Dive</strong> (15-20 min): The core ML components - feature engineering, model selection, training pipeline, evaluation</li>
<li><strong>Deployment & Monitoring</strong> (5 min): A/B testing, monitoring, feedback loops</li>
</ol>

<h4>Common ML System Design Questions</h4>

<div class="interview-q">
<div class="q-label">System Design</div>
<div class="q-text">Design a real-time speech translation system.</div>
<div class="a-text"><strong>Architecture:</strong> Streaming ASR (Whisper with chunked processing) -> Machine Translation (NLLB or fine-tuned mBART) -> Streaming TTS (CosyVoice or VITS). <strong>Key challenges:</strong> (1) End-to-end latency budget: aim for < 2s. Allocate: ASR 500ms, MT 300ms, TTS 500ms, network 200ms. (2) Handle partial utterances - ASR outputs partial results, MT must handle incomplete sentences. (3) Speaker diarization for multi-speaker scenarios. (4) Preserve prosody and emphasis across translation. <strong>Scaling:</strong> Stateless microservices, GPU auto-scaling per component, WebSocket connections for streaming.</div>
</div>

<div class="interview-q">
<div class="q-label">System Design</div>
<div class="q-text">Design a content moderation system for audio/video uploads.</div>
<div class="a-text"><strong>Pipeline:</strong> (1) Extract audio track, (2) ASR for speech content, (3) Audio classification for non-speech (music, gunshots, explicit sounds), (4) Text content analysis (toxicity, PII, policy violations), (5) Confidence-based routing: high-confidence violations auto-remove, medium-confidence to human review queue. <strong>Scale:</strong> Async processing via message queue, GPU pool for model inference, separate queues by priority. <strong>Monitoring:</strong> False positive/negative rates, reviewer agreement scores, latency percentiles. <strong>Key decisions:</strong> Threshold tuning (precision vs recall trade-off depends on content type), multi-language support, appeal process.</div>
</div>

<h4>Infrastructure Patterns</h4>
<pre><code># Typical AI service architecture
Client -> API Gateway -> Load Balancer
                              |
                    +---------+---------+
                    |         |         |
                 Replica1  Replica2  Replica3
                    |         |         |
                    +----+----+----+----+
                         |         |
                    GPU Pool   Model Registry
                         |
                    Object Storage (models, data)

# Key metrics to monitor:
# - P50/P95/P99 latency
# - GPU utilization (aim for >80%)
# - Queue depth (requests waiting)
# - Error rate by error type
# - Model prediction distribution shift</code></pre>
`
            },
            {
              id: "networking-deploy",
              title: "Deployment in Restricted Networks",
              content: `
<p>Deploying AI services in enterprise or government environments often involves restricted networks, air-gapped systems, and strict security requirements.</p>

<h4>Common Challenges</h4>
<ul>
<li><strong>No internet access:</strong> Can't pull packages, models, or Docker images at runtime</li>
<li><strong>Firewall restrictions:</strong> Only specific ports/protocols allowed</li>
<li><strong>Proxy requirements:</strong> All traffic must go through corporate proxies</li>
<li><strong>Compliance:</strong> Data must not leave the network boundary</li>
</ul>

<h4>Solutions</h4>
<pre><code># Pre-package everything into Docker images
# Include model weights, all dependencies
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Copy pre-downloaded model weights
COPY ./models/whisper-large-v3 /app/models/whisper-large-v3

# Install from local wheels (no pip install from internet)
COPY ./wheels /tmp/wheels
RUN pip install --no-index --find-links=/tmp/wheels -r requirements.txt

# Pre-download NLTK data, tokenizers, etc.
COPY ./data/tokenizers /app/data/tokenizers</code></pre>

<h4>Reverse Tunneling for Development</h4>
<p>When you need to access services running in restricted networks from your development machine:</p>
<pre><code># SSH reverse tunnel: access remote GPU server's port 8000 locally
ssh -R 8000:localhost:8000 user@jump-server

# Cloudflare Tunnel for more permanent setups
cloudflared tunnel --url http://localhost:8000

# For WebSocket-based services (like Discord bots)
# Use a relay server in the DMZ</code></pre>
`
            }
          ]
        }
      ]
    },
    {
      title: "Career & Interview Preparation",
      chapters: [
        {
          id: "interview-prep",
          title: "AI Engineer Interview Guide",
          sections: [
            {
              id: "ml-fundamentals-interview",
              title: "ML Fundamentals",
              content: `
<h4>Core Concepts to Master</h4>

<div class="interview-q">
<div class="q-label">Fundamentals</div>
<div class="q-text">Explain the bias-variance tradeoff and how it relates to model selection.</div>
<div class="a-text"><strong>Bias</strong> is error from overly simplistic assumptions (underfitting). <strong>Variance</strong> is error from sensitivity to training data fluctuations (overfitting). The tradeoff: as model complexity increases, bias decreases but variance increases. <strong>Practical implications:</strong> (1) Use cross-validation to estimate the sweet spot, (2) Regularization (L1/L2, dropout) reduces variance without increasing bias much, (3) Ensemble methods (bagging reduces variance, boosting reduces bias), (4) For deep learning: early stopping and learning rate schedules are key controls.</div>
</div>

<div class="interview-q">
<div class="q-label">Fundamentals</div>
<div class="q-text">What is the difference between BatchNorm and LayerNorm, and when would you use each?</div>
<div class="a-text"><strong>BatchNorm:</strong> Normalizes across the batch dimension for each feature. Statistics depend on batch composition, so behavior differs between training (batch stats) and inference (running stats). Great for CNNs with fixed-size inputs. <strong>LayerNorm:</strong> Normalizes across the feature dimension for each sample. Statistics are per-sample, so behavior is identical in training and inference. Essential for Transformers because: (1) sequence lengths vary, (2) batch size may be 1 during inference, (3) autoregressive generation processes one token at a time. <strong>RMSNorm</strong> is a simplified LayerNorm (no mean subtraction) used in LLaMA and modern LLMs for slightly faster computation.</div>
</div>

<div class="interview-q">
<div class="q-label">Fundamentals</div>
<div class="q-text">Explain the Transformer attention mechanism. What is the purpose of the scaling factor?</div>
<div class="a-text">Attention computes: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. Q (queries), K (keys), V (values) are linear projections of the input. QK^T computes similarity scores between all pairs of positions. The <strong>scaling factor sqrt(d_k)</strong> prevents the dot products from growing large as dimension increases (they scale as O(d_k)), which would push softmax into regions with extremely small gradients. <strong>Multi-head attention</strong> runs h parallel attention operations with different projections, allowing the model to attend to information from different representation subspaces at different positions. <strong>Grouped Query Attention (GQA)</strong> in modern LLMs shares K,V heads across multiple Q heads to reduce KV-cache memory.</div>
</div>

<div class="interview-q">
<div class="q-label">Fundamentals</div>
<div class="q-text">What is KV-cache in LLM inference, and why is it necessary?</div>
<div class="a-text">During autoregressive generation, each new token needs to attend to all previous tokens. Without caching, generating token N requires recomputing K and V projections for all N-1 previous tokens - O(N^2) total computation for a sequence. The KV-cache stores previously computed K and V tensors so each step only computes K,V for the new token and appends to the cache. This reduces generation from O(N^2) to O(N). <strong>The cost:</strong> Memory. Formula: <code>2 * n_layers * n_kv_heads * head_dim * seq_len * bytes_per_param</code>. For LLaMA-2 70B (GQA with 8 KV heads, 80 layers, head_dim=128, FP16): KV-cache for 4096 tokens is ~1.25GB per request. For MHA models (all heads are KV heads), the cost is much higher. This is why PagedAttention (vLLM) and KV-cache compression are critical for production serving.</div>
</div>
`
            },
            {
              id: "coding-interview",
              title: "Coding Patterns for ML Interviews",
              content: `
<h4>Common ML Coding Patterns</h4>

<pre><code># Pattern 1: Implementing Softmax from scratch
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Pattern 2: Multi-head attention
def multi_head_attention(Q, K, V, num_heads):
    """Simple multi-head attention implementation."""
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Reshape to (batch, num_heads, seq_len, d_k)
    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V

    # Reshape back
    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return output

# Pattern 3: Beam search
def beam_search(model, input_ids, beam_width=5, max_length=100):
    """Basic beam search decoding."""
    # Each beam: (log_probability, token_sequence)
    beams = [(0.0, input_ids)]

    for _ in range(max_length):
        all_candidates = []
        for score, seq in beams:
            if seq[-1] == EOS_TOKEN:
                all_candidates.append((score, seq))
                continue
            logits = model(seq)
            log_probs = log_softmax(logits[-1])
            top_k = np.argsort(log_probs)[-beam_width:]
            for token in top_k:
                new_seq = seq + [token]
                new_score = score + log_probs[token]
                all_candidates.append((new_score, new_seq))

        # Keep top beam_width candidates
        beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        if all(b[1][-1] == EOS_TOKEN for b in beams):
            break

    return beams[0][1]  # Return best sequence</code></pre>

<h4>System Design Coding Patterns</h4>
<pre><code># Pattern: Async batched inference server
import asyncio
from collections import deque

class BatchedInferenceServer:
    def __init__(self, model, max_batch_size=32, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()

    async def predict(self, input_data):
        """Single prediction request - gets batched automatically."""
        future = asyncio.Future()
        self.queue.append((input_data, future))

        if len(self.queue) >= self.max_batch_size:
            await self._process_batch()
        else:
            # Wait briefly for more requests
            await asyncio.sleep(self.max_wait_ms / 1000)
            if not future.done():
                await self._process_batch()

        return await future

    async def _process_batch(self):
        batch_items = []
        while self.queue and len(batch_items) < self.max_batch_size:
            batch_items.append(self.queue.popleft())

        inputs = [item[0] for item in batch_items]
        outputs = self.model.batch_predict(inputs)

        for (_, future), output in zip(batch_items, outputs):
            if not future.done():
                future.set_result(output)</code></pre>
`
            },
            {
              id: "behavioral-interview",
              title: "Behavioral & Research Discussion",
              content: `
<h4>Research Discussion Framework</h4>
<p>When asked about a paper or research direction:</p>
<ol>
<li><strong>Summarize the core contribution</strong> in 2-3 sentences</li>
<li><strong>Explain the technical approach</strong> at a level appropriate for the audience</li>
<li><strong>Identify limitations</strong> that you would address</li>
<li><strong>Connect to broader trends</strong> in the field</li>
<li><strong>Propose concrete next steps</strong> if you were to build on this work</li>
</ol>

<h4>Common Behavioral Questions for AI Roles</h4>

<div class="interview-q">
<div class="q-label">Behavioral</div>
<div class="q-text">Tell me about a time you had to debug a complex ML issue in production.</div>
<div class="a-text"><strong>Framework (STAR):</strong> Describe the Situation (model performance degraded in production), Task (identify root cause and fix), Action (systematic debugging: check data pipeline, compare distributions, analyze error patterns, identify data drift or preprocessing bug), Result (found and fixed the issue, implemented monitoring to catch similar issues). <strong>Key signals interviewers look for:</strong> systematic approach, use of monitoring/logging, ability to prioritize hypotheses, communication with team during incident.</div>
</div>

<div class="interview-q">
<div class="q-label">Behavioral</div>
<div class="q-text">How do you stay current with the rapidly evolving AI field?</div>
<div class="a-text">Concrete actions: (1) Follow key researchers and labs on social media/arXiv, (2) Read 3-5 papers per week using the 10-question framework (core claim, what's broken, key choice, etc.), (3) Reproduce interesting results - reading isn't enough, you need to build, (4) Participate in reading groups or discussion forums, (5) Build side projects that apply new techniques, (6) Maintain a knowledge base of papers organized by topic with personal annotations.</div>
</div>

<h4>Questions to Ask Your Interviewer</h4>
<ul>
<li>What does the model development lifecycle look like here? (SFT, RLHF, evaluation)</li>
<li>What's the biggest technical challenge the team is currently facing?</li>
<li>How do you evaluate model quality beyond standard benchmarks?</li>
<li>What's the team's approach to research vs. production trade-offs?</li>
<li>How much ownership does an individual engineer have over projects?</li>
</ul>
`
            }
          ]
        }
      ]
    }
  ]
};
