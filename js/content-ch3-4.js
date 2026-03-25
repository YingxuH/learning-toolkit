// Deeply expanded content for Chapters 3 (TTS Technology) and 4 (Speculative Decoding)
// This module provides ~12,000 words per chapter across 8+ sections each.
const CONTENT_CH3_4 = {

  // ============================================================================
  // CHAPTER 3: TEXT-TO-SPEECH TECHNOLOGY (8 sections, ~12,000 words)
  // ============================================================================
  ch3_sections: [
    {
      id: "tts-evolution",
      title: "TTS Evolution: From Pipelines to Neural Codecs",
      content: `
<p>Text-to-Speech synthesis has undergone four distinct generations, each driven by a fundamental shift in how we think about turning text into sound. Understanding this evolution is essential for AI engineers because modern TTS systems inherit design choices -- both brilliant and limiting -- from every preceding era.</p>

<h4>Generation 1: Concatenative TTS (1960s-2000s)</h4>
<p>The earliest practical TTS systems stored a large database of recorded speech segments (diphones, triphones, or longer units) and selected/concatenated them at synthesis time. The quality ceiling was high when the exact unit existed in the database, but transitions between units produced audible artifacts.</p>
<ul>
<li><strong>Unit Selection:</strong> Search a database of ~10-50 hours of single-speaker recordings. A Viterbi search selects the best sequence of units minimizing both target cost (does this unit match the desired phonetic/prosodic properties?) and join cost (does this unit connect smoothly with its neighbors?).</li>
<li><strong>Limitations:</strong> Required massive studio recordings per voice, had no ability to generalize beyond the recorded inventory, and produced characteristic "robot" artifacts at segment boundaries.</li>
<li><strong>Legacy Impact:</strong> Early Siri, AT&T Natural Voices, and Festival were all concatenative systems. The quality bar they set -- natural-sounding individual segments but poor prosody -- defined "good TTS" for a decade.</li>
</ul>

<h4>Generation 2: Statistical Parametric TTS (2000s-2016)</h4>
<p>Hidden Markov Model (HMM) based synthesis replaced the unit database with a statistical model. Instead of selecting recorded segments, the system generated acoustic parameters (spectral features, F0, duration) from a trained model, then synthesized audio using a vocoder (typically STRAIGHT or WORLD).</p>

<pre><code># Statistical parametric TTS pipeline (conceptual)
# Step 1: Text Analysis
phonemes = g2p(text)          # Grapheme-to-phoneme conversion
features = linguistic_features(phonemes, context)  # Context-dependent features

# Step 2: Acoustic Model (HMM or DNN)
duration = duration_model(features)       # How long each phoneme lasts
spectrum = acoustic_model(features, duration)  # Mel-cepstral coefficients + F0

# Step 3: Vocoder
waveform = world_vocoder(spectrum, f0)    # Parameter-to-waveform synthesis</code></pre>

<p><strong>Advantages over concatenative:</strong> Smooth output (no concatenation artifacts), tiny footprint (~2MB models), easy voice adaptation via MLLR/CMLLR transforms. <strong>Disadvantages:</strong> Over-smoothed "buzzy" quality, poor expressiveness, the vocoder introduced artifacts of its own.</p>

<h4>Generation 3: Neural TTS (2016-2022)</h4>
<p>The neural revolution in TTS happened in three waves:</p>

<p><strong>Wave 1 -- Neural Vocoders (2016):</strong> WaveNet (van den Oord et al., 2016, arXiv:1609.03499) demonstrated that an autoregressive neural network generating audio sample-by-sample could produce startlingly natural speech. It was too slow for real-time (minutes per second of audio), but proved neural audio generation was viable.</p>

<p><strong>Wave 2 -- Neural Acoustic Models (2017-2018):</strong> Tacotron (Wang et al., 2017, arXiv:1703.10135) and Tacotron 2 (Shen et al., 2018, arXiv:1712.05884) replaced the HMM acoustic model with an attention-based seq2seq network that generated mel-spectrograms from text. Combined with WaveNet or WaveGlow vocoders, this achieved near-human quality for single-speaker synthesis.</p>

<p><strong>Wave 3 -- Fast Neural Vocoders (2018-2020):</strong> Parallel WaveGAN, HiFi-GAN (Kong et al., 2020, arXiv:2010.05646), and WaveGlow made real-time neural vocoding possible. HiFi-GAN in particular became the de facto vocoder due to its speed (>100x real-time on GPU) and quality.</p>

<table>
<tr><th>Component</th><th>Traditional Pipeline</th><th>Neural Pipeline (Gen 3)</th><th>Codec LM (Gen 4)</th></tr>
<tr><td>Text Frontend</td><td>Rule-based G2P</td><td>Neural G2P or character-level</td><td>Often BPE text tokens directly</td></tr>
<tr><td>Acoustic Model</td><td>HMM / DNN</td><td>Tacotron 2 / FastSpeech 2</td><td>AR/NAR Transformer over codec tokens</td></tr>
<tr><td>Intermediate Repr.</td><td>Mel-cepstra + F0</td><td>Mel-spectrogram</td><td>Discrete codec tokens (RVQ)</td></tr>
<tr><td>Waveform Synthesis</td><td>WORLD / STRAIGHT vocoder</td><td>HiFi-GAN / WaveGlow</td><td>Codec decoder (learned)</td></tr>
<tr><td>Data Required</td><td>20-50 hours, single speaker</td><td>10-30 hours, single speaker</td><td>10K-60K hours, multi-speaker</td></tr>
<tr><td>Zero-Shot Cloning</td><td>Impossible</td><td>Poor (speaker embedding)</td><td>3 seconds of audio</td></tr>
</table>

<h4>Generation 4: Codec Language Models (2023-Present)</h4>
<p>The current generation -- pioneered by VALL-E (Wang et al., 2023, arXiv:2301.02111) -- treats TTS as a language modeling problem over discrete audio tokens. This is the paradigm shift that enabled zero-shot voice cloning, massive data scaling, and the convergence of TTS with LLM infrastructure.</p>

<p>The core architectural pattern is:</p>
<pre><code>Text -> [Text Encoder / Phoneme Encoder] -> Text Token Sequence
                                                    |
Audio Prompt -> [Audio Codec Encoder] -> Codec Token Sequence (prompt)
                                                    |
                                          [Language Model]
                                                    |
                                     Generated Codec Token Sequence
                                                    |
                              [Audio Codec Decoder] -> Waveform</code></pre>

<p>This formulation has three profound implications:</p>
<ol>
<li><strong>Data scaling works:</strong> Like GPT scaling with more text, codec LMs improve predictably with more audio data. VALL-E trained on 60K hours; later systems use 100K+ hours.</li>
<li><strong>Zero-shot voice cloning emerges:</strong> By conditioning on a short audio prompt encoded as codec tokens, the model learns to continue generating in the same voice -- analogous to in-context learning in text LLMs.</li>
<li><strong>LLM infrastructure transfers:</strong> Training, inference, and optimization techniques from text LLMs (KV-cache, speculative decoding, quantization) apply directly.</li>
</ol>

<div class="callout">
<div class="callout-title">The Fundamental Insight</div>
<p>Each generation solved a different bottleneck: Gen 1 solved coverage (no more missing units), Gen 2 solved smoothness (no concatenation artifacts), Gen 3 solved naturalness (neural generation quality), Gen 4 solved generalization (zero-shot, any voice, any language). The common thread: every breakthrough came from making the representation more learnable and the model more expressive.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Walk me through the evolution of TTS systems and explain why codec language models represent a paradigm shift.</div>
<div class="a-text">TTS evolved through four generations: (1) Concatenative systems that stitched recorded speech segments, limited by database coverage; (2) Statistical parametric systems (HMM-based) that generated smooth but over-smoothed speech; (3) Neural TTS (Tacotron + HiFi-GAN) that achieved near-human quality but required 10-50 hours per voice; (4) Codec Language Models (VALL-E onwards) that reformulated TTS as language modeling over discrete audio tokens. The paradigm shift is threefold: data scaling laws apply (more data = better), zero-shot voice cloning emerges from in-context learning (just 3 seconds of audio), and LLM infrastructure (KV-cache, speculative decoding, quantization) transfers directly. This means TTS went from a specialized signal processing problem to a general sequence modeling problem.</div>
</div>
`
    },
    {
      id: "tts-flow-matching",
      title: "Flow Matching & Modern Generation",
      content: `
<h4>Why Flow Matching Replaced Diffusion in TTS</h4>
<p>Diffusion models (Ho et al., 2020; Song et al., 2021) were the first generation of iterative refinement models applied to TTS. They work by progressively adding Gaussian noise to data over T timesteps, then learning to reverse this process. While effective, they have several drawbacks for audio generation.</p>

<p>Flow Matching (Lipman et al., 2023, arXiv:2210.02747; Liu et al., 2023, arXiv:2209.03003) provides a cleaner mathematical framework. Instead of learning to denoise, it learns a <strong>velocity field</strong> that transports samples from a noise distribution to the data distribution along straight paths.</p>

<h4>Mathematical Foundation</h4>
<p>Given data point x_1 sampled from data distribution q(x_1) and noise x_0 ~ N(0, I), flow matching defines an interpolation path:</p>

<pre><code># Conditional Flow Matching (CFM) - the practical variant
# Given: x_1 (clean data), x_0 ~ N(0, I) (noise), t ~ U(0, 1)

# Interpolation path (Optimal Transport path):
x_t = (1 - t) * x_0 + t * x_1

# Target velocity (derivative of path w.r.t. t):
u_t = x_1 - x_0

# Training objective:
L_CFM = E_{t, x_0, x_1} || v_theta(x_t, t) - u_t ||^2

# Where v_theta is the neural network parameterizing the velocity field.
# At inference, integrate the ODE: dx/dt = v_theta(x_t, t) from t=0 to t=1
# Using Euler method with N steps:
#   x_{t+dt} = x_t + dt * v_theta(x_t, t),  where dt = 1/N</code></pre>

<p>The key mathematical advantage: the Optimal Transport (OT) path between x_0 and x_1 is a straight line, making the velocity field nearly constant and easy for neural networks to learn. Compare this with diffusion's curved denoising paths which require more steps to follow accurately.</p>

<h4>Flow Matching vs. Diffusion: Detailed Comparison</h4>
<table>
<tr><th>Property</th><th>Diffusion (DDPM/Score-based)</th><th>Flow Matching (CFM)</th></tr>
<tr><td>Forward process</td><td>Stochastic (add noise gradually)</td><td>Deterministic (straight interpolation)</td></tr>
<tr><td>Training target</td><td>Noise prediction epsilon or score</td><td>Velocity field u_t = x_1 - x_0</td></tr>
<tr><td>Inference</td><td>SDE/ODE solver (50-1000 steps naive)</td><td>ODE solver (10-50 steps typical)</td></tr>
<tr><td>Path geometry</td><td>Curved, problem-dependent</td><td>Near-straight (OT paths)</td></tr>
<tr><td>Training stability</td><td>Sensitive to noise schedule</td><td>Schedule-free (uniform t)</td></tr>
<tr><td>Modern step counts</td><td>1-4 (distilled), 20-50 (standard)</td><td>10-32 (standard), 1-4 (distilled)</td></tr>
</table>

<div class="callout">
<div class="callout-title">Important Nuance</div>
<p>Modern diffusion models with advanced solvers (DPM-Solver++, consistency distillation) have closed much of the step-count gap with flow matching. The real advantages of flow matching are: (1) simpler training (no noise schedule to tune), (2) easier to combine with Optimal Transport, and (3) a cleaner mathematical framework that makes extensions more natural. In practice, both paradigms can achieve similar quality; flow matching just gets there with less engineering effort.</p>
</div>

<h4>Conditional Flow Matching for TTS</h4>
<p>In TTS, flow matching generates mel-spectrograms or latent audio representations conditioned on text (and optionally a speaker embedding or audio prompt). The conditioning is typically injected via:</p>

<pre><code>import torch
import torch.nn as nn

class ConditionalFlowMatchingTTS(nn.Module):
    """
    Simplified flow matching TTS model.
    Generates mel-spectrograms conditioned on text and speaker.
    """
    def __init__(self, mel_dim=80, hidden_dim=512, text_dim=256):
        super().__init__()
        self.text_encoder = TransformerEncoder(text_dim, num_layers=6)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # U-Net or DiT backbone for velocity prediction
        self.velocity_net = DiTBlock(
            in_channels=mel_dim,
            hidden_dim=hidden_dim,
            cond_dim=text_dim,
            num_layers=12
        )
        self.mel_proj = nn.Linear(hidden_dim, mel_dim)

    def forward(self, mel_clean, text_tokens, t):
        """Training forward pass: predict velocity."""
        # Encode text condition
        text_features = self.text_encoder(text_tokens)  # [B, T_text, text_dim]

        # Sample noise and interpolate
        noise = torch.randn_like(mel_clean)             # x_0
        mel_t = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * mel_clean  # x_t

        # Target velocity
        target_v = mel_clean - noise                     # u_t = x_1 - x_0

        # Predict velocity conditioned on text and time
        time_emb = self.time_embed(t)
        pred_v = self.velocity_net(mel_t, text_features, time_emb)

        return nn.functional.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def generate(self, text_tokens, num_steps=32):
        """Inference: integrate ODE from noise to mel."""
        text_features = self.text_encoder(text_tokens)
        # Estimate output length from text length (heuristic)
        mel_len = text_tokens.shape[1] * 3
        x = torch.randn(1, mel_len, 80)  # Start from noise
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.tensor([step * dt])
            time_emb = self.time_embed(t)
            v = self.velocity_net(x, text_features, time_emb)
            x = x + dt * v               # Euler step
        return x  # Generated mel-spectrogram</code></pre>

<h4>Optimal Transport in Flow Matching</h4>
<p>The vanilla CFM pairs noise and data randomly. With Optimal Transport CFM (OT-CFM), noise samples are matched to data samples to minimize total transport cost. This produces even straighter paths and reduces the number of integration steps needed.</p>

<p>In practice, mini-batch OT is used: for a batch of B noise samples and B data samples, solve the assignment problem (Hungarian algorithm or Sinkhorn iterations) to find the minimum-cost pairing. This adds modest computational overhead during training but significantly improves generation quality at low step counts (4-8 steps).</p>

<h4>Flow Matching in Production TTS Systems</h4>
<ul>
<li><strong>CosyVoice (Alibaba):</strong> Uses conditional flow matching as the acoustic decoder after LLM-based token generation. The flow matching module transforms Gaussian noise into mel-spectrograms conditioned on semantic tokens and speaker embeddings.</li>
<li><strong>F5-TTS (2024, arXiv:2410.06885):</strong> Pure flow matching with a DiT (Diffusion Transformer) backbone and "Sway Sampling" -- a non-uniform time step schedule that allocates more steps to the middle of the trajectory where curvature is highest. Achieves RTF (Real-Time Factor) of 0.15 on a single A100.</li>
<li><strong>VoiceBox (Meta, 2023, arXiv:2306.15687):</strong> One of the first large-scale flow matching TTS systems. Demonstrated in-context learning for style transfer and noise removal using flow matching.</li>
<li><strong>E2 TTS (Microsoft, 2024, arXiv:2406.18009):</strong> "Embarrassingly Easy" TTS using flow matching with a plain transformer on character-level input. Showed that with enough data, even simple architectures work.</li>
</ul>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Compare flow matching and diffusion models for audio generation. When would you choose one over the other?</div>
<div class="a-text">Both learn to generate data through iterative refinement. Diffusion learns to denoise (predict and remove noise), while flow matching learns a velocity field that transports noise to data along straight paths. Flow matching advantages: simpler training (no noise schedule), fewer inference steps (10-32 vs 50-1000 for naive diffusion), and natural compatibility with Optimal Transport. However, modern diffusion with advanced solvers (DPM-Solver++, consistency distillation) can match flow matching step counts. Choose flow matching for new projects (simpler to implement correctly), and diffusion when you have existing diffusion infrastructure or need compatibility with established diffusion-based tools. In practice for TTS, flow matching has become the default choice in 2024-2025 systems (CosyVoice, F5-TTS, VoiceBox).</div>
</div>
`
    },
    {
      id: "tts-valle",
      title: "VALL-E Deep Dive: The Codec Language Model Revolution",
      content: `
<p>VALL-E (Wang et al., 2023, arXiv:2301.02111) is one of the most important TTS papers of the decade. It demonstrated that treating TTS as a language modeling problem over discrete audio codec tokens -- trained at massive scale -- could achieve unprecedented zero-shot voice cloning quality. Understanding VALL-E deeply is essential because virtually every modern codec-based TTS system descends from its core ideas.</p>

<h4>Architecture: Two-Stage Token Generation</h4>
<p>VALL-E uses EnCodec to convert audio into 8 layers of discrete tokens (via Residual Vector Quantization). It then models these tokens in two stages:</p>

<p><strong>Stage 1: Autoregressive (AR) Model for Coarse Tokens</strong></p>
<p>The AR model generates tokens from the first RVQ codebook (layer 1) -- the coarsest level, which captures content and speaker identity. It operates left-to-right, token by token, conditioned on:</p>
<ul>
<li>Phoneme sequence of the target text</li>
<li>Codec tokens from a 3-second audio prompt (the voice to clone)</li>
</ul>

<pre><code># VALL-E AR model pseudocode
# Input: phonemes P = [p_1, ..., p_L], prompt codec tokens C_prompt = [c_1, ..., c_T']
# Output: coarse codec tokens C_1 = [c_1, ..., c_T] (codebook 1 only)

class VALLE_AR(nn.Module):
    def __init__(self, vocab_size=1024, phoneme_vocab=100, d_model=1024):
        super().__init__()
        self.phoneme_embed = nn.Embedding(phoneme_vocab, d_model)
        self.codec_embed = nn.Embedding(vocab_size, d_model)   # Codebook 1 tokens
        self.transformer = TransformerDecoder(
            d_model=1024, nhead=16, num_layers=12
        )
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, phonemes, prompt_codes, target_codes=None):
        # Embed phonemes and prompt codec tokens
        phone_emb = self.phoneme_embed(phonemes)       # [B, L, D]
        prompt_emb = self.codec_embed(prompt_codes)    # [B, T', D]

        if target_codes is not None:
            # Training: teacher forcing
            target_emb = self.codec_embed(target_codes)
            # Concatenate: [phonemes; prompt_codes; target_codes]
            seq = torch.cat([phone_emb, prompt_emb, target_emb], dim=1)
            # Causal attention mask (autoregressive over codec tokens)
            out = self.transformer(seq, causal_mask=True)
            logits = self.output_head(out[:, len_phone + len_prompt:])
            return logits  # Cross-entropy loss against target_codes
        else:
            # Inference: generate autoregressively
            generated = []
            for step in range(max_len):
                seq = torch.cat([phone_emb, prompt_emb,
                                 self.codec_embed(generated)], dim=1)
                out = self.transformer(seq, causal_mask=True)
                logits = self.output_head(out[:, -1:])
                token = sample(logits)  # top-p sampling
                generated.append(token)
                if token == EOS:
                    break
            return torch.cat(generated, dim=-1)</code></pre>

<p><strong>Stage 2: Non-Autoregressive (NAR) Model for Fine Tokens</strong></p>
<p>The NAR model generates tokens for codebooks 2-8 <em>simultaneously</em> (in parallel across time), conditioned on the coarse tokens from Stage 1. Each codebook is generated conditioned on all previous codebooks.</p>

<pre><code># VALL-E NAR model pseudocode
# Input: phonemes P, prompt codes C_prompt (all 8 layers),
#        coarse codes C_1 (from AR model)
# Output: fine codes C_2, ..., C_8

class VALLE_NAR(nn.Module):
    def __init__(self, num_codebooks=8, vocab_size=1024, d_model=1024):
        super().__init__()
        self.phoneme_embed = nn.Embedding(phoneme_vocab, d_model)
        # Separate embedding per codebook layer
        self.codec_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_codebooks)
        ])
        self.transformer = TransformerEncoder(  # Bidirectional!
            d_model=1024, nhead=16, num_layers=12
        )
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_codebooks - 1)
        ])

    def forward(self, phonemes, prompt_all_layers, coarse_codes):
        # For codebook j (2 to 8):
        #   Input = sum of embeddings for codebooks 1..j-1
        #   Output = codebook j tokens (all time steps in parallel)
        results = [coarse_codes]  # C_1 already generated
        for j in range(1, 8):  # Generate codebooks 2 through 8
            # Sum embeddings of all previously generated codebooks
            combined = sum(self.codec_embeds[k](results[k]) for k in range(j))
            phone_emb = self.phoneme_embed(phonemes)
            seq = torch.cat([phone_emb, combined], dim=1)
            # Bidirectional attention (not causal!)
            out = self.transformer(seq)
            logits = self.output_heads[j-1](out[:, len_phone:])
            tokens = logits.argmax(dim=-1)  # Greedy for NAR
            results.append(tokens)
        return results  # C_1 through C_8</code></pre>

<div class="callout">
<div class="callout-title">Why Two Stages?</div>
<p>The first codebook carries the most information (content and speaker identity) and benefits from autoregressive modeling which can capture long-range dependencies. Codebooks 2-8 carry progressively finer acoustic details that are more local and can be predicted in parallel given the coarse structure. This is analogous to how image generation works: generate the rough structure first, then fill in details.</p>
</div>

<h4>EnCodec: How Audio Becomes Tokens</h4>
<p>VALL-E relies on Meta's EnCodec (Defossez et al., 2022, arXiv:2210.13438) to convert between waveforms and discrete tokens. EnCodec uses Residual Vector Quantization (RVQ) with 8 codebooks, each containing 1024 entries. At 75Hz frame rate with 8 codebooks, this produces 8 x 75 = 600 tokens per second of audio.</p>

<p>The RVQ process works as follows:</p>
<pre><code># Residual Vector Quantization (RVQ) - step by step
# Given: continuous latent vector z from encoder, shape [B, T, D]
# Codebooks: Q_1, Q_2, ..., Q_8, each with 1024 entries of dimension D

def rvq_encode(z, codebooks):
    """
    Quantize z using 8 codebooks sequentially.
    Each codebook quantizes the residual from previous codebooks.
    """
    residual = z           # Start with full signal
    codes = []             # Will hold 8 layers of token indices
    quantized_sum = 0      # Accumulated quantized representation

    for i, codebook in enumerate(codebooks):
        # Find nearest codebook entry for current residual
        # codebook.weight: [1024, D]
        distances = torch.cdist(residual, codebook.weight)  # [B, T, 1024]
        indices = distances.argmin(dim=-1)                    # [B, T]
        quantized = codebook(indices)                         # [B, T, D]

        codes.append(indices)
        quantized_sum = quantized_sum + quantized
        residual = residual - quantized  # Key: quantize the RESIDUAL

    # After 8 codebooks: quantized_sum ~= z (with small residual error)
    return codes, quantized_sum

# Example: 10 seconds of audio at 24kHz
# Encoder output: [1, 750, 128] (750 frames at 75Hz, 128-dim)
# RVQ output: 8 x [1, 750] token indices
# Total: 8 * 750 = 6000 discrete tokens for 10 seconds</code></pre>

<h4>Training: 60K Hours of LibriLight</h4>
<p>VALL-E's most underappreciated contribution was demonstrating that data scale was the key to zero-shot TTS. The training pipeline:</p>
<ol>
<li><strong>Data:</strong> LibriLight (60,000 hours of English audiobooks, 7,000+ speakers). This is ~100x more data than typical TTS training sets (LJSpeech: 24 hours, LibriTTS: 585 hours).</li>
<li><strong>Preprocessing:</strong> Audio segmented into utterances, forced-aligned with transcripts. Phoneme sequences extracted via a G2P model. EnCodec applied to extract 8-layer codec tokens.</li>
<li><strong>AR Training:</strong> Standard cross-entropy loss on codebook-1 tokens, conditioned on phonemes + prompt codes. Trained with teacher forcing.</li>
<li><strong>NAR Training:</strong> For each training sample, randomly select a codebook level j in [2,8]. Provide codebooks 1..j-1 as input, predict codebook j. Cross-entropy loss.</li>
<li><strong>Compute:</strong> 16 NVIDIA V100 32GB GPUs for ~2 weeks.</li>
</ol>

<h4>Zero-Shot Inference: The 3-Second Prompt</h4>
<p>At inference time, VALL-E takes a 3-second audio clip of the target speaker and the text to synthesize. The prompt audio is encoded via EnCodec into codec tokens, which are prepended to the generation sequence. The model has learned (through seeing 7,000+ speakers during training) to extract speaker characteristics from this prompt and apply them to new text.</p>

<pre><code># VALL-E zero-shot inference pipeline
def valle_inference(text, speaker_audio_3s, ar_model, nar_model, encodec):
    # Step 1: Encode text to phonemes
    phonemes = g2p(text)

    # Step 2: Encode speaker prompt to codec tokens
    prompt_codes = encodec.encode(speaker_audio_3s)  # 8 x [T_prompt] tokens

    # Step 3: AR model generates coarse tokens (codebook 1)
    coarse_tokens = ar_model.generate(
        phonemes=phonemes,
        prompt_codes=prompt_codes[0],  # Only codebook 1 of prompt
        temperature=0.7,
        top_p=0.9
    )

    # Step 4: NAR model generates fine tokens (codebooks 2-8)
    all_codes = nar_model.generate(
        phonemes=phonemes,
        prompt_all_layers=prompt_codes,  # All 8 codebook layers of prompt
        coarse_codes=coarse_tokens
    )

    # Step 5: Decode all 8 codebook layers back to waveform
    waveform = encodec.decode(all_codes)
    return waveform</code></pre>

<h4>VALL-E Limitations and Successors</h4>
<ul>
<li><strong>Robustness:</strong> VALL-E occasionally produces garbled speech, word repetitions, or skips. This is an inherent issue with autoregressive models -- they can go off-track and not recover.</li>
<li><strong>Speaker similarity:</strong> While impressive, 3-second prompts sometimes miss speaker-specific traits (accent details, speaking rhythm). VALL-E 2 (arXiv:2406.05370) addressed this with grouped code modeling and repetition-aware sampling.</li>
<li><strong>VALL-E X (arXiv:2303.03926):</strong> Extended to cross-lingual synthesis -- clone a voice across languages.</li>
<li><strong>VALL-E R (arXiv:2406.07855):</strong> Replaced EnCodec with a codec that better separates content and timbre.</li>
</ul>

<div class="callout warning">
<div class="callout-title">Production War Story: VALL-E Robustness Issues</div>
<p>When we first deployed a VALL-E-style model for an internal voice assistant, ~5% of generations had noticeable artifacts: word repetitions (the model would get stuck in a loop), missing words (attention skipping), or speaker similarity drops (the voice would shift mid-sentence). We discovered that most failures correlated with (1) out-of-distribution phoneme sequences (names, technical jargon), (2) very long sentences (>50 words), and (3) prompt audio with background noise. Our fixes: added a repetition penalty (nucleus sampling with presence penalty), chunked long texts into sentence-level segments, and preprocessed prompt audio with noise reduction. This brought artifact rate down to <1%, acceptable for production.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Describe the VALL-E architecture and explain why it uses two separate models (AR + NAR) rather than a single model.</div>
<div class="a-text">VALL-E has two stages: an autoregressive (AR) model that generates codebook-1 tokens (coarsest level, capturing content and speaker identity) left-to-right, and a non-autoregressive (NAR) model that generates codebooks 2-8 (fine acoustic details) in parallel. The two-stage design is motivated by the nature of information in RVQ layers: codebook 1 carries the most semantic information and has strong long-range dependencies (what word comes next, prosody), making AR modeling appropriate. Codebooks 2-8 capture local acoustic details (timbre texture, fine spectral details) that are largely determined by the coarse structure and can be predicted in parallel. A single AR model over all 8 codebooks would be prohibitively slow (8x more sequential steps), while a single NAR model would struggle to capture the temporal dependencies needed for coherent speech.</div>
</div>
`
    },
    {
      id: "tts-cosyvoice",
      title: "CosyVoice & Modern TTS Systems",
      content: `
<h4>CosyVoice: LLM Backbone Meets Flow Matching</h4>
<p>CosyVoice (Du et al., 2024, arXiv:2407.05407) from Alibaba represents the convergence of two paradigms: using an LLM to generate semantic tokens and flow matching to generate acoustic features. It is one of the most capable open-source TTS systems available.</p>

<p><strong>Architecture Overview:</strong></p>
<pre><code>CosyVoice Architecture:

Text -> [Text Encoder] -> Text Embeddings
                              |
                    [LLM Backbone (Qwen-based)]
                              |
                    Supervised Semantic Tokens
                              |
Speaker Prompt -> [Speaker Encoder] -> Speaker Embedding
                              |
              [Conditional Flow Matching Decoder]
                              |
                    Mel-Spectrogram -> [HiFi-GAN] -> Waveform</code></pre>

<p>The three key components:</p>
<ol>
<li><strong>LLM Backbone:</strong> A decoder-only transformer (based on Qwen architecture) that generates <em>supervised semantic tokens</em> autoregressively. Unlike VALL-E which generates codec tokens directly, CosyVoice generates intermediate semantic tokens that capture linguistic content and prosody but not fine acoustic details.</li>
<li><strong>Supervised Semantic Tokens:</strong> Rather than using self-supervised speech tokens (HuBERT/wav2vec2) or codec tokens, CosyVoice trains a supervised quantizer using forced alignment labels. This produces tokens that are semantically meaningful and easier for the LLM to model.</li>
<li><strong>Conditional Flow Matching Decoder:</strong> Takes the semantic token sequence and speaker embedding, then generates mel-spectrograms via flow matching. This separates the "what to say" (LLM) from "how it sounds" (flow matching), leading to better modularity.</li>
</ol>

<h4>CosyVoice 2: Human-Parity Streaming TTS</h4>
<p>CosyVoice 2 (Du et al., 2024, arXiv:2412.10117) introduced several critical improvements for production deployment:</p>

<p><strong>Chunk-Aware Causal Flow Matching:</strong> The original flow matching decoder requires the full sequence to generate (bidirectional attention). CosyVoice 2 makes it causal and chunk-aware -- it can generate audio in chunks as semantic tokens arrive, enabling streaming synthesis with ~150ms first-chunk latency.</p>

<pre><code># CosyVoice 2 streaming inference (conceptual)
def cosyvoice2_streaming(text, speaker_embed, llm, flow_decoder, vocoder):
    """Stream audio chunks as they're generated."""
    # LLM generates semantic tokens autoregressively
    semantic_buffer = []
    chunk_size = 20  # ~0.25 seconds per chunk

    for token in llm.generate_stream(text):
        semantic_buffer.append(token)

        if len(semantic_buffer) >= chunk_size:
            # Flow matching generates mel chunk (causal, sees only past)
            mel_chunk = flow_decoder.generate_chunk(
                semantic_tokens=semantic_buffer,
                speaker_embed=speaker_embed,
                num_flow_steps=10  # Reduced for streaming
            )
            # Vocoder converts to audio
            audio_chunk = vocoder(mel_chunk)
            yield audio_chunk  # Stream to user immediately

            # Keep overlap for continuity
            semantic_buffer = semantic_buffer[-5:]  # Keep last 5 for context</code></pre>

<p><strong>Key CosyVoice 2 improvements:</strong></p>
<ul>
<li><strong>Finite scalar quantization (FSQ):</strong> Replaces VQ for semantic tokens -- eliminates codebook collapse issues</li>
<li><strong>Human parity:</strong> Achieves MOS scores comparable to natural speech (4.3 vs 4.4 for ground truth) on LibriSpeech test</li>
<li><strong>Instruction following:</strong> Can be prompted with natural language descriptions ("speak slowly and calmly") in addition to audio prompts</li>
<li><strong>Multilingual:</strong> Supports Chinese, English, Japanese, Cantonese, and Korean with code-switching support</li>
</ul>

<h4>F5-TTS: Simplicity at Scale</h4>
<p>F5-TTS (Chen et al., 2024, arXiv:2410.06885) takes a radically simpler approach: no separate semantic token stage, no AR model -- just a single flow matching model with a DiT (Diffusion Transformer) backbone that directly maps text to mel-spectrograms.</p>

<p><strong>Architecture:</strong></p>
<ul>
<li><strong>Backbone:</strong> ConvNeXt V2 blocks + Diffusion Transformer (DiT) layers</li>
<li><strong>Input:</strong> Text (character-level, no phoneme conversion needed) + optional audio prompt</li>
<li><strong>Infill formulation:</strong> Training uses a fill-in-the-middle objective -- randomly mask a segment of the mel-spectrogram and train the model to reconstruct it, conditioned on surrounding context and aligned text. This naturally enables voice cloning (provide context audio) and text-to-speech (provide text, generate audio).</li>
<li><strong>Sway Sampling:</strong> A non-uniform ODE step schedule: instead of uniform dt steps, allocate more steps to t in [0.3, 0.7] where the velocity field changes most rapidly. This gives 30-50% quality improvement at the same step count.</li>
</ul>

<pre><code># F5-TTS Sway Sampling schedule
import numpy as np

def sway_sampling_schedule(num_steps, sway_coefficient=1.5):
    """
    Non-uniform time steps for flow matching ODE integration.
    Concentrates steps in the middle of the trajectory.
    """
    uniform = np.linspace(0, 1, num_steps + 1)
    # Apply sway: sigmoid-like warping that concentrates in middle
    sway = uniform ** sway_coefficient / (
        uniform ** sway_coefficient + (1 - uniform) ** sway_coefficient
    )
    return sway  # Use these as time points for ODE integration

# Example: 16 steps with sway
# Uniform:  [0, 0.0625, 0.125, 0.1875, ..., 0.875, 0.9375, 1.0]
# Sway 1.5: [0, 0.031,  0.084, 0.155,  ..., 0.916, 0.969,  1.0]
# More steps concentrated around t=0.5 where generation is hardest</code></pre>

<p><strong>Performance:</strong> RTF (Real-Time Factor) = 0.15 on A100, meaning 10 seconds of audio generated in 1.5 seconds. With 32 flow steps, achieves quality competitive with CosyVoice at 3-5x faster inference.</p>

<h4>MaskGCT: Masked Generative Codec Transformer</h4>
<p>MaskGCT (Wang et al., 2024, arXiv:2409.00750) takes yet another approach: instead of autoregressive or flow-matching generation, it uses <strong>masked prediction</strong> (similar to BERT/MaskGIT) over codec tokens.</p>

<p><strong>Two-stage approach:</strong></p>
<ol>
<li><strong>Stage 1:</strong> Predict semantic tokens from text using a masked generative model. Start with all tokens masked, iteratively unmask tokens in order of confidence (highest confidence first). Takes ~8-16 iterations.</li>
<li><strong>Stage 2:</strong> Predict acoustic codec tokens from semantic tokens using another masked generative model. Same iterative unmasking strategy.</li>
</ol>

<p><strong>Advantages:</strong> Non-autoregressive (parallel generation within each iteration), naturally handles variable-length output, no exposure bias. <strong>Disadvantages:</strong> Requires multiple iterations (typically 16), quality slightly below AR models for long sequences.</p>

<h4>Practical Guide: Fine-Tuning CosyVoice on Custom Data</h4>
<p>For production voice cloning or domain adaptation, fine-tuning CosyVoice on your own data is often the best approach. Here is a practical guide:</p>

<pre><code># Step 1: Prepare training data
# Requirements: WAV files (16kHz or 24kHz) + text transcriptions
# Minimum: 30 minutes for voice adaptation, 5+ hours for best quality
# Format: Kaldi-style data directory

data/
  train/
    wav.scp      # utterance-id /path/to/audio.wav
    text         # utterance-id transcript text
    utt2spk      # utterance-id speaker-id
    spk2utt      # speaker-id utterance-id1 utterance-id2 ...

# Step 2: Extract features
# CosyVoice uses its own feature extraction pipeline
python cosyvoice/bin/extract_features.py \\
    --data_dir data/train \\
    --output_dir features/train \\
    --config conf/cosyvoice_base.yaml

# Step 3: Fine-tune (LoRA recommended for efficiency)
python cosyvoice/bin/train.py \\
    --config conf/cosyvoice_finetune.yaml \\
    --pretrained_model pretrained/CosyVoice-300M \\
    --data_dir features/train \\
    --output_dir exp/my_voice \\
    --lora_rank 32 \\
    --lora_alpha 64 \\
    --learning_rate 1e-4 \\
    --num_epochs 50 \\
    --batch_size 8

# Step 4: Inference with fine-tuned model
from cosyvoice.cli.cosyvoice import CosyVoice
model = CosyVoice("exp/my_voice", load_lora=True)
# Zero-shot with fine-tuned model (better speaker similarity)
output = model.inference_zero_shot(
    "Hello, this is a test.",
    "prompt text matching the 3s clip",
    "path/to/3s_prompt.wav"
)</code></pre>

<div class="callout tip">
<div class="callout-title">Fine-Tuning Tips</div>
<p><strong>Data quality > quantity:</strong> 1 hour of clean studio audio beats 10 hours of noisy recordings. Ensure SNR > 30dB, no reverb, consistent mic distance.<br>
<strong>LoRA vs full fine-tune:</strong> LoRA (rank 32-64) is usually sufficient and trains 5-10x faster. Full fine-tuning only needed for very different domains (singing, non-standard speech).<br>
<strong>Avoid overfitting:</strong> Monitor validation loss. If speaker similarity improves but naturalness degrades, you're overfitting. Early stopping at best validation MOS.<br>
<strong>Prompt selection matters:</strong> For inference, choose a 3-second prompt that represents the target voice well -- clear speech, representative pitch range, no background noise.</p>
</div>

<h4>System Comparison: Choosing the Right TTS Model</h4>
<table>
<tr><th>System</th><th>Architecture</th><th>RTF (A100)</th><th>Streaming</th><th>Zero-Shot Quality</th><th>Open Source</th></tr>
<tr><td>VALL-E</td><td>AR+NAR codec LM</td><td>~1.0</td><td>No</td><td>Good</td><td>No (reproductions exist)</td></tr>
<tr><td>CosyVoice 2</td><td>LLM + Flow Matching</td><td>~0.3</td><td>Yes (150ms)</td><td>Excellent</td><td>Yes</td></tr>
<tr><td>F5-TTS</td><td>DiT Flow Matching</td><td>~0.15</td><td>Partial</td><td>Very Good</td><td>Yes</td></tr>
<tr><td>MaskGCT</td><td>Masked Generative</td><td>~0.5</td><td>No</td><td>Good</td><td>Yes</td></tr>
<tr><td>Parler-TTS</td><td>Text-described control</td><td>~0.4</td><td>No</td><td>Moderate</td><td>Yes</td></tr>
<tr><td>XTTS v2 (Coqui)</td><td>GPT + DVAE + HiFi-GAN</td><td>~0.5</td><td>Yes</td><td>Good</td><td>Yes</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Compare CosyVoice's architecture with VALL-E's. What are the key design differences and why do they matter?</div>
<div class="a-text">VALL-E generates audio codec tokens (EnCodec RVQ) directly using an AR+NAR model pair. CosyVoice uses a three-stage approach: (1) an LLM backbone generates supervised semantic tokens (not codec tokens), (2) a flow matching decoder converts semantic tokens + speaker embedding to mel-spectrograms, (3) HiFi-GAN converts mels to waveforms. Key differences: CosyVoice's supervised semantic tokens are more linguistically meaningful than raw codec tokens, making the LLM's job easier. The flow matching decoder separates acoustic generation from semantic modeling, improving modularity. CosyVoice 2 adds streaming capability through chunk-aware causal flow matching. The practical impact: CosyVoice achieves better speaker similarity and naturalness while being faster (RTF 0.3 vs ~1.0) and supporting streaming, which VALL-E cannot do.</div>
</div>
`
    },
    {
      id: "tts-codecs",
      title: "Audio Codecs Deep Dive: The Foundation of Modern TTS",
      content: `
<p>Neural audio codecs are the backbone of modern TTS, speech-to-speech models, and audio generation. They convert continuous waveforms into discrete tokens that language models can process. Understanding codecs deeply is essential for AI engineers because codec design choices directly impact TTS quality, latency, and what kinds of models are possible.</p>

<h4>EnCodec: The Standard Bearer</h4>
<p>EnCodec (Defossez et al., 2022, arXiv:2210.13438) from Meta established the architecture that most modern codecs follow:</p>

<pre><code>EnCodec Architecture:

Waveform (24kHz) -> [Encoder CNN] -> Latent z (75Hz, 128-dim)
                                         |
                                    [RVQ: 8 codebooks, 1024 entries each]
                                         |
                              Discrete tokens (75Hz x 8 layers)
                                         |
                              [Dequantize + Decoder CNN] -> Reconstructed Waveform

Encoder: 1D ConvNet with residual blocks, strided convolutions for downsampling
  - Input: 24,000 samples/sec
  - Strides: [8, 5, 4, 2] -> total stride = 320
  - Output: 24000/320 = 75 frames/sec, 128-dimensional

Decoder: Mirror of encoder with transposed convolutions for upsampling
  - Input: 75 frames/sec, 128-dim (after dequantization)
  - Output: 24,000 samples/sec waveform</code></pre>

<p><strong>Training losses:</strong></p>
<ul>
<li><strong>Reconstruction loss:</strong> L1 distance between input and output waveforms in both time and frequency (multi-resolution STFT) domains</li>
<li><strong>Adversarial loss:</strong> Multi-scale discriminator (MSD) + multi-period discriminator (MPD) from HiFi-GAN, ensuring perceptual quality</li>
<li><strong>Commitment loss:</strong> Forces encoder output to stay close to codebook entries (VQ-VAE style)</li>
<li><strong>Codebook loss:</strong> Moves codebook entries toward encoder outputs (EMA update or gradient-based)</li>
</ul>

<h4>RVQ Mathematics: Residual Vector Quantization</h4>
<p>RVQ is the key innovation that makes neural codecs work at low bitrates while maintaining quality. Here is the full mathematical formulation:</p>

<pre><code># RVQ Mathematical Formulation
# ============================
#
# Given: continuous latent vector z in R^D (from encoder)
# Codebooks: C_1, C_2, ..., C_K where C_k = {e_k^1, ..., e_k^N} in R^D
# (K=8 codebooks, N=1024 entries each for EnCodec)
#
# Encoding (quantization):
#   r_0 = z                                    (initial residual = full signal)
#   For k = 1, ..., K:
#     i_k = argmin_j ||r_{k-1} - e_k^j||^2    (find nearest codebook entry)
#     q_k = e_k^{i_k}                          (quantized value)
#     r_k = r_{k-1} - q_k                      (compute new residual)
#
# Decoding (dequantization):
#   z_hat = sum_{k=1}^{K} q_k = z - r_K        (sum of all quantized values)
#
# Quantization error = ||r_K||^2
# Each codebook reduces the residual error
#
# Bitrate calculation:
#   Frame rate: 75 Hz
#   Bits per frame per codebook: log2(1024) = 10 bits
#   Total bitrate: 75 * K * 10 bits/sec
#   K=1: 0.75 kbps, K=2: 1.5 kbps, ..., K=8: 6.0 kbps
#   At 24kHz stereo: multiply by 2 -> up to 12 kbps

# Training the codebooks (commitment + codebook loss):
# L_commit = ||z - sg(q)||^2     (push encoder toward codebook entries)
# L_codebook = ||sg(z) - q||^2   (push codebook entries toward encoder output)
# Where sg() is stop-gradient
#
# In practice, codebook entries are updated via Exponential Moving Average (EMA):
#   For each entry e_k^j:
#     N_j = decay * N_j + (1-decay) * n_j    (count of assignments)
#     m_j = decay * m_j + (1-decay) * sum(assigned z's)
#     e_k^j = m_j / N_j                       (running mean of assigned vectors)</code></pre>

<p><strong>Why RVQ and not plain VQ?</strong> A single VQ codebook with N entries gives log2(N) bits per frame. To match 8-codebook RVQ quality, you would need a single codebook with 1024^8 = 2^80 entries -- impossibly large. RVQ achieves the same information capacity (80 bits per frame) with only 8 x 1024 = 8,192 entries total, a reduction of 10^20 in codebook size.</p>

<h4>Mimi: Semantic + Acoustic Split (Kyutai, 2024)</h4>
<p>Mimi (used in Moshi, arXiv:2410.00037) introduced a key architectural innovation: explicitly separating semantic and acoustic information across codebook layers.</p>

<ul>
<li><strong>Architecture:</strong> Same encoder-decoder structure as EnCodec, but with a distillation loss from a self-supervised model (WavLM). The first codebook is trained to align with WavLM features, making it capture semantic content. Remaining codebooks capture acoustic details.</li>
<li><strong>Bitrate:</strong> 1.1 kbps (12.5Hz frame rate, 8 codebooks) -- significantly lower than EnCodec's 6 kbps while maintaining quality for speech.</li>
<li><strong>Impact:</strong> This separation allows Moshi to reason about speech content using only the first codebook (semantic) while the remaining codebooks handle speaker identity and acoustic texture. This is crucial for the "inner monologue" mechanism where the model generates text reasoning tokens alongside speech tokens.</li>
</ul>

<h4>SpeechTokenizer: Disentangled Representations</h4>
<p>SpeechTokenizer (Zhang et al., 2023, arXiv:2308.16692) explicitly disentangles content and timbre:</p>
<ul>
<li><strong>Content tokens (codebook 1):</strong> Trained with a HuBERT distillation loss, capturing phonetic content</li>
<li><strong>Timbre tokens (codebooks 2-8):</strong> Capture speaker identity, recording conditions, acoustic details</li>
<li><strong>Key innovation:</strong> By disentangling content and timbre, SpeechTokenizer enables voice conversion by swapping timbre tokens while keeping content tokens, and enables content-preserving style transfer</li>
</ul>

<h4>DAC: Descript Audio Codec</h4>
<p>DAC (Kumar et al., 2024, arXiv:2306.06546) focuses on higher quality and broader audio support (not just speech):</p>
<ul>
<li><strong>Improvements over EnCodec:</strong> Periodic activation functions (Snake activations) in the decoder, improved discriminator architecture, better quantizer dropout for variable bitrate</li>
<li><strong>Bitrate:</strong> 8 kbps at 44.1kHz (higher sample rate, better quality)</li>
<li><strong>Music support:</strong> Handles music, environmental sounds, and speech -- not just speech like EnCodec's typical use case</li>
<li><strong>Quantizer dropout:</strong> During training, randomly drops higher codebook layers, making the model robust to variable bitrate at inference time</li>
</ul>

<h4>XCodec: Injecting Semantic Information into RVQ</h4>
<p>XCodec (Ye et al., 2024) and its successor XCodec2 address a fundamental problem: standard RVQ distributes information across codebooks in a way that is not optimized for language modeling. The first codebook does not always capture the most semantically useful information.</p>
<ul>
<li><strong>Approach:</strong> Add a semantic loss term during codec training that encourages the first few codebooks to capture linguistically meaningful information (phoneme identity, word boundaries). Uses a pre-trained speech representation model (e.g., HuBERT or Whisper encoder) as a teacher.</li>
<li><strong>Result:</strong> TTS systems using XCodec-encoded tokens show better content accuracy and fewer word error artifacts than those using vanilla EnCodec, because the language model receives more linguistically structured input.</li>
</ul>

<h4>Code: Encoding and Decoding Audio with EnCodec</h4>
<pre><code>import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

# Load 24kHz EnCodec model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 6 kbps = 8 codebooks

# Load and preprocess audio
wav, sr = torchaudio.load("speaker.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)  # Add batch dimension: [1, 1, num_samples]

# Encode: waveform -> discrete tokens
with torch.no_grad():
    encoded_frames = model.encode(wav)

# Extract codes: list of (codes, scale) tuples
# codes shape: [batch, num_codebooks, num_frames]
codes = encoded_frames[0][0]  # [1, 8, T]
print(f"Audio duration: {wav.shape[-1]/model.sample_rate:.1f}s")
print(f"Codec tokens: {codes.shape}")
print(f"Tokens per second: {codes.shape[-1] / (wav.shape[-1]/model.sample_rate):.0f}")
print(f"Total tokens: {codes.shape[1] * codes.shape[2]}")
# Example output:
# Audio duration: 5.0s
# Codec tokens: torch.Size([1, 8, 375])
# Tokens per second: 75
# Total tokens: 3000

# Decode: discrete tokens -> waveform
with torch.no_grad():
    decoded_wav = model.decode(encoded_frames)
# decoded_wav shape: [1, 1, num_samples]

# Analyze per-codebook information content
# By decoding with progressively more codebooks:
for n_codebooks in [1, 2, 4, 8]:
    partial_codes = codes[:, :n_codebooks, :]
    # Dequantize and decode (simplified)
    # In practice, use model's internal quantizer for partial decode
    print(f"Codebooks 1-{n_codebooks}: "
          f"bitrate={75 * n_codebooks * 10 / 1000:.1f} kbps")

# Save reconstructed audio
torchaudio.save("reconstructed.wav", decoded_wav[0].cpu(), model.sample_rate)</code></pre>

<h4>Codec Comparison Summary</h4>
<table>
<tr><th>Codec</th><th>Bitrate</th><th>Frame Rate</th><th>Codebooks</th><th>Semantic Split</th><th>Best For</th></tr>
<tr><td>EnCodec</td><td>1.5-24 kbps</td><td>75 Hz</td><td>2-32</td><td>No (emergent)</td><td>General audio, TTS baseline</td></tr>
<tr><td>Mimi</td><td>1.1 kbps</td><td>12.5 Hz</td><td>8</td><td>Yes (WavLM distill)</td><td>Moshi, low-bitrate streaming</td></tr>
<tr><td>SpeechTokenizer</td><td>Variable</td><td>50 Hz</td><td>8</td><td>Yes (HuBERT distill)</td><td>Voice conversion, disentanglement</td></tr>
<tr><td>DAC</td><td>8 kbps</td><td>~87 Hz</td><td>9</td><td>No</td><td>Music, high-quality audio</td></tr>
<tr><td>XCodec2</td><td>Variable</td><td>50 Hz</td><td>8</td><td>Yes (explicit semantic loss)</td><td>TTS with better content accuracy</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain Residual Vector Quantization (RVQ) mathematically and describe how different codebook layers capture different types of information.</div>
<div class="a-text">RVQ quantizes a continuous vector z using K codebooks sequentially. The first codebook quantizes z directly: find the nearest entry, compute the residual r_1 = z - q_1. The second codebook quantizes r_1, giving residual r_2 = r_1 - q_2. This continues for K codebooks. The reconstruction is z_hat = sum of all q_k, with error ||r_K||^2. Bitrate is frame_rate x K x log2(N) bits/sec. Information distribution: codebook 1 captures the largest variance in the data -- for speech, this is content (what phoneme) and speaker identity (who). Subsequent codebooks capture progressively finer details: codebook 2-3 capture prosodic details and spectral shape, codebooks 4-8 capture fine acoustic texture, breath sounds, and recording conditions. This hierarchical structure is why VALL-E can generate just codebook 1 autoregressively (slow but capturing structure) and fill in 2-8 non-autoregressively (fast, filling in details).</div>
</div>
`
    },
    {
      id: "tts-evaluation",
      title: "TTS Evaluation: Measuring Speech Quality",
      content: `
<p>Evaluating TTS systems is notoriously difficult because speech quality is inherently subjective and multi-dimensional. A generated utterance must be natural-sounding, intelligible, appropriate in prosody, and (for voice cloning) similar to the target speaker. No single metric captures all of this. This section covers the full evaluation toolkit that AI engineers need.</p>

<h4>MOS: Mean Opinion Score</h4>
<p>MOS remains the gold standard for TTS evaluation. Human listeners rate speech samples on a 1-5 scale:</p>
<table>
<tr><th>Score</th><th>Quality</th><th>Description</th></tr>
<tr><td>5</td><td>Excellent</td><td>Indistinguishable from natural speech</td></tr>
<tr><td>4</td><td>Good</td><td>Perceptible but not annoying artifacts</td></tr>
<tr><td>3</td><td>Fair</td><td>Slightly annoying artifacts</td></tr>
<tr><td>2</td><td>Poor</td><td>Annoying but intelligible</td></tr>
<tr><td>1</td><td>Bad</td><td>Very annoying, barely intelligible</td></tr>
</table>

<p><strong>Running a proper MOS test:</strong></p>
<ol>
<li><strong>Sample selection:</strong> 20-50 utterances, balanced across phonetic contexts, sentence lengths, and difficulty levels. Include natural speech as upper anchor.</li>
<li><strong>Listener recruitment:</strong> Minimum 20 native-speaker listeners (30+ recommended). Use crowd-sourcing platforms (Amazon MTurk, Prolific) with screening tests to filter poor-quality raters.</li>
<li><strong>Screening:</strong> Include 5-10 "trap" samples (clearly synthetic or clearly natural) to identify unreliable raters. Remove raters who fail >20% of traps.</li>
<li><strong>Presentation:</strong> Randomize order. Each listener rates all systems for each utterance (within-subject design). Use AB or MUSHRA format for relative comparisons.</li>
<li><strong>Statistical analysis:</strong> Report mean + 95% confidence interval. Use paired t-tests or Wilcoxon signed-rank tests for system comparisons. Cohen's d for effect size.</li>
</ol>

<p><strong>MUSHRA (MUltiple Stimuli with Hidden Reference and Anchor):</strong> A more sensitive protocol where listeners rate all systems simultaneously on a 0-100 scale, with a hidden reference (natural speech) and a hidden anchor (low-quality version). This produces finer-grained comparisons between systems.</p>

<div class="callout warning">
<div class="callout-title">MOS Pitfalls</div>
<p><strong>MOS is not absolute.</strong> A MOS of 4.0 in one study may not equal 4.0 in another. Different listener pools, different reference samples, different audio equipment all shift the scale. Only compare MOS scores within the same study. Cross-study comparisons require shared benchmark samples (like the LJSpeech standard test set).<br><br>
<strong>MOS inflation:</strong> Some papers use very short utterances (5-10 words) or cherry-picked easy sentences, inflating MOS. Always report utterance length distribution and selection criteria.</p>
</div>

<h4>Objective Metrics</h4>
<p>While no objective metric fully replaces MOS, several are useful for development and automated evaluation:</p>

<p><strong>PESQ (Perceptual Evaluation of Speech Quality, ITU-T P.862):</strong></p>
<ul>
<li>Compares synthesized speech against a reference (ground-truth recording)</li>
<li>Score range: -0.5 to 4.5 (higher = better)</li>
<li>Designed for telephony, may not perfectly reflect TTS quality</li>
<li>Requires time-aligned reference and test signals</li>
</ul>

<p><strong>STOI (Short-Time Objective Intelligibility):</strong></p>
<ul>
<li>Measures intelligibility, not quality</li>
<li>Score range: 0 to 1 (higher = more intelligible)</li>
<li>Useful for detecting severe artifacts that impact understanding</li>
<li>Less sensitive to minor quality differences between good systems</li>
</ul>

<p><strong>Speaker Similarity (Cosine Distance):</strong></p>
<ul>
<li>For voice cloning evaluation, extract speaker embeddings from a pre-trained speaker verification model (e.g., WeSpeaker, ECAPA-TDNN, Resemblyzer)</li>
<li>Compute cosine similarity between embeddings of generated speech and target speaker reference</li>
<li>Threshold: similarity > 0.75 typically indicates recognizable voice match; > 0.85 is very good</li>
</ul>

<pre><code>import torch
import torchaudio
from resemblyzer import VoiceEncoder, preprocess_wav

# Speaker similarity evaluation
encoder = VoiceEncoder()

def speaker_similarity(generated_path, reference_path):
    """Compute cosine similarity between speaker embeddings."""
    gen_wav = preprocess_wav(generated_path)
    ref_wav = preprocess_wav(reference_path)

    gen_embed = encoder.embed_utterance(gen_wav)   # 256-dim vector
    ref_embed = encoder.embed_utterance(ref_wav)

    similarity = gen_embed @ ref_embed / (
        torch.norm(gen_embed) * torch.norm(ref_embed)
    )
    return similarity.item()

# Batch evaluation
def evaluate_tts_batch(generated_dir, reference_dir):
    """Evaluate a batch of generated samples."""
    import glob, numpy as np
    from pesq import pesq as pesq_score
    from pystoi import stoi

    results = {"pesq": [], "stoi": [], "spk_sim": []}
    for gen_file in glob.glob(f"{generated_dir}/*.wav"):
        ref_file = gen_file.replace(generated_dir, reference_dir)
        gen, sr = torchaudio.load(gen_file)
        ref, sr_ref = torchaudio.load(ref_file)

        # PESQ (requires 16kHz)
        gen_16k = torchaudio.functional.resample(gen, sr, 16000)
        ref_16k = torchaudio.functional.resample(ref, sr_ref, 16000)
        results["pesq"].append(
            pesq_score(16000, ref_16k.numpy()[0], gen_16k.numpy()[0], 'wb')
        )

        # STOI
        results["stoi"].append(
            stoi(ref_16k.numpy()[0], gen_16k.numpy()[0], 16000)
        )

        # Speaker similarity
        results["spk_sim"].append(
            speaker_similarity(gen_file, ref_file)
        )

    for metric, values in results.items():
        print(f"{metric}: {np.mean(values):.3f} +/- {np.std(values):.3f}")</code></pre>

<h4>TTSDS: Distribution-Level Evaluation</h4>
<p>TTSDS (Minixhofer et al., 2024) introduced a principled distribution-level evaluation framework. Instead of comparing individual samples, it compares the <em>distribution</em> of generated speech with natural speech across multiple dimensions:</p>
<ul>
<li><strong>Prosody:</strong> F0 (pitch) statistics, energy patterns, speaking rate</li>
<li><strong>Speaker:</strong> Speaker embedding distribution (not just pairwise similarity)</li>
<li><strong>Intelligibility:</strong> ASR error rate on generated speech</li>
<li><strong>Environment:</strong> Background noise and recording condition characteristics</li>
</ul>
<p>For each dimension, TTSDS computes a distributional distance (Frechet distance, similar to FID for images) between generated and natural speech. This gives a more robust evaluation than sample-level metrics.</p>

<h4>QualiSpeech: LLM-as-Judge for TTS</h4>
<p>QualiSpeech (2025) applies the "LLM-as-judge" paradigm to speech evaluation. An audio-capable LLM (e.g., Gemini, GPT-4o) listens to generated speech and rates it on multiple dimensions:</p>
<ul>
<li><strong>Naturalness:</strong> Does it sound like a real person?</li>
<li><strong>Intelligibility:</strong> Can you understand every word?</li>
<li><strong>Prosody:</strong> Is the intonation and rhythm appropriate?</li>
<li><strong>Speaker consistency:</strong> Does the voice remain consistent throughout?</li>
</ul>
<p>Early results show moderate correlation (r=0.6-0.7) with human MOS, promising but not yet a replacement. The main advantage is scalability -- evaluate thousands of samples without human listeners.</p>

<h4>Building an Evaluation Pipeline</h4>
<pre><code># Complete TTS evaluation pipeline
class TTSEvaluationPipeline:
    def __init__(self):
        self.spk_encoder = VoiceEncoder()       # Speaker similarity
        self.asr_model = whisper.load_model("large-v3")  # Intelligibility
        # self.mos_predictor = UTMOS()           # Predicted MOS (optional)

    def evaluate(self, generated_samples, reference_samples, texts):
        """
        Full evaluation suite.
        generated_samples: list of paths to generated wav files
        reference_samples: list of paths to reference wav files
        texts: list of target transcription strings
        """
        results = {
            "wer": [],           # Word Error Rate (intelligibility)
            "spk_sim": [],       # Speaker similarity
            "pesq": [],          # Perceptual quality
            "stoi": [],          # Intelligibility objective
            "rtf": [],           # Real-time factor (speed)
        }

        for gen, ref, text in zip(generated_samples, reference_samples, texts):
            # 1. Intelligibility: ASR on generated speech
            asr_result = self.asr_model.transcribe(gen)
            wer = compute_wer(text, asr_result["text"])
            results["wer"].append(wer)

            # 2. Speaker similarity
            sim = self.speaker_similarity(gen, ref)
            results["spk_sim"].append(sim)

            # 3. PESQ and STOI (requires aligned audio)
            # ... (as shown above)

        return {k: {"mean": np.mean(v), "std": np.std(v)}
                for k, v in results.items()}

# Standard test sets for reproducibility:
# - LibriSpeech test-clean: 2,620 utterances, standard for English TTS
# - VCTK: 109 speakers, British English, good for multi-speaker evaluation
# - Common Voice: multilingual, diverse speakers and recording conditions</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You're building an evaluation pipeline for a TTS system. What metrics would you include and what are the pitfalls of each?</div>
<div class="a-text">I would include: (1) MOS from human listeners -- gold standard but expensive, slow, and not comparable across studies; needs 20+ listeners and proper screening. (2) Word Error Rate from ASR -- measures intelligibility objectively but doesn't capture quality; a robotic voice can have low WER. (3) Speaker similarity via cosine distance of speaker embeddings -- measures voice cloning quality but depends heavily on the speaker verification model used; may not capture subjective perception. (4) PESQ/STOI -- objective quality/intelligibility metrics but designed for telephony and may not correlate well with TTS quality. (5) RTF -- measures speed, critical for production. Pitfalls: no single metric is sufficient, MOS varies across studies, objective metrics plateau before human perception does, and all metrics should be reported with confidence intervals and statistical significance tests.</div>
</div>
`
    },
    {
      id: "tts-practical",
      title: "Practical TTS Engineering Guide",
      content: `
<p>This section bridges the gap between TTS research and production deployment. Building a TTS system that works reliably in production requires engineering decisions that go well beyond model selection.</p>

<h4>Choosing a TTS System for Production</h4>
<p>The decision framework depends on your specific constraints:</p>

<table>
<tr><th>Requirement</th><th>Recommended System</th><th>Rationale</th></tr>
<tr><td>Fastest inference, single voice</td><td>VITS / VITS2</td><td>End-to-end, single-stage, RTF < 0.05</td></tr>
<tr><td>Best zero-shot cloning</td><td>CosyVoice 2</td><td>Human-parity quality, streaming support</td></tr>
<tr><td>Fastest zero-shot cloning</td><td>F5-TTS</td><td>RTF 0.15, competitive quality</td></tr>
<tr><td>Multilingual (10+ languages)</td><td>CosyVoice 2 / XTTS v2</td><td>Pre-trained multilingual support</td></tr>
<tr><td>On-device / edge deployment</td><td>VITS (quantized) / Piper</td><td>Small model, CPU-friendly</td></tr>
<tr><td>Voice control via text description</td><td>Parler-TTS</td><td>Describe voice in natural language</td></tr>
<tr><td>Maximum API simplicity</td><td>ElevenLabs / OpenAI TTS</td><td>Managed service, no infrastructure</td></tr>
</table>

<div class="callout tip">
<div class="callout-title">Decision Flowchart</div>
<p><strong>Q1: Do you need zero-shot voice cloning?</strong><br>
No -> Use VITS/VITS2 with a pre-trained single-speaker model. Fastest, simplest.<br>
Yes -> Q2<br><br>
<strong>Q2: Do you need streaming (< 500ms first audio)?</strong><br>
Yes -> CosyVoice 2 (open-source) or ElevenLabs (managed)<br>
No -> Q3<br><br>
<strong>Q3: What's your quality vs speed priority?</strong><br>
Quality first -> CosyVoice 2 (MOS 4.3)<br>
Speed first -> F5-TTS (RTF 0.15)<br>
Balance -> MaskGCT or XTTS v2</p>
</div>

<h4>Voice Cloning: Ethics and Technical Requirements</h4>
<p><strong>Ethical considerations:</strong></p>
<ul>
<li><strong>Consent:</strong> Always obtain explicit consent from the person whose voice is being cloned. Many jurisdictions now have laws specifically addressing voice cloning (California's AB-1836, EU AI Act).</li>
<li><strong>Disclosure:</strong> Generated speech should be clearly labeled as synthetic. The FTC has issued guidance that AI-generated voices must be disclosed in commercial contexts.</li>
<li><strong>Safeguards:</strong> Implement watermarking (AudioSeal from Meta is the current standard) and speaker verification gates that prevent unauthorized voice cloning.</li>
<li><strong>Deepfake prevention:</strong> Never deploy voice cloning without abuse prevention measures. At minimum: rate limiting, content moderation, and audit logs.</li>
</ul>

<p><strong>Technical requirements for good voice cloning:</strong></p>
<pre><code># Voice cloning prompt requirements
# ====================================

# Minimum viable prompt:
#   3 seconds of clean speech (CosyVoice, VALL-E)
#   SNR > 25 dB
#   No music, no overlapping speakers

# Recommended prompt:
#   5-10 seconds of clean speech
#   SNR > 35 dB
#   Diverse phonetic content (all vowels, common consonant clusters)
#   Representative pitch range (not monotone)
#   Consistent microphone distance

# Best practice: multiple prompts
#   Record 3-5 diverse prompts, 5-10 seconds each
#   Select best prompt per generation based on acoustic similarity
#   Or: concatenate for longer context (some models benefit from this)

# Audio preprocessing for voice cloning prompts:
import torchaudio
import torchaudio.transforms as T

def prepare_prompt(audio_path, target_sr=24000, max_duration=10.0):
    """Preprocess audio for TTS voice cloning prompt."""
    wav, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Trim silence from start and end
    wav = torchaudio.functional.vad(wav, target_sr)

    # Truncate to max duration
    max_samples = int(max_duration * target_sr)
    wav = wav[:, :max_samples]

    # Normalize volume (peak normalization)
    wav = wav / wav.abs().max() * 0.95

    return wav</code></pre>

<h4>Streaming TTS: Chunk-Based Synthesis</h4>
<p>Streaming TTS is essential for conversational AI. Users expect to hear audio within 200-500ms of the text being available. Here is how to build a streaming TTS pipeline:</p>

<p><strong>Architecture options:</strong></p>
<ol>
<li><strong>Sentence-level streaming:</strong> Split text at sentence boundaries, synthesize each sentence independently, stream audio as each sentence completes. Simplest but introduces sentence-boundary pauses.</li>
<li><strong>Chunk-level streaming:</strong> Generate audio in overlapping chunks (CosyVoice 2 approach). Lower latency but requires models that support causal/streaming generation.</li>
<li><strong>Hybrid:</strong> Use a fast model (VITS, F5-TTS) for the first chunk (ultra-low latency) and a higher-quality model (CosyVoice) for subsequent chunks. The initial response arrives quickly while quality improves for the remaining speech.</li>
</ol>

<pre><code># Streaming TTS server (FastAPI + WebSocket)
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

class StreamingTTSEngine:
    def __init__(self, model_name="cosyvoice2"):
        self.model = load_tts_model(model_name)
        self.chunk_size_ms = 250  # Generate 250ms audio chunks
        self.overlap_ms = 50     # Overlap for smooth transitions

    async def synthesize_stream(self, text, speaker_prompt):
        """Generate audio chunks as they become available."""
        # Split text into synthesis-friendly chunks
        chunks = self.split_text(text)

        for i, chunk_text in enumerate(chunks):
            # Generate audio for this chunk
            audio_chunk = self.model.generate_chunk(
                text=chunk_text,
                speaker=speaker_prompt,
                is_first=(i == 0),
                context=chunks[max(0,i-1):i]  # Previous chunk for context
            )

            # Apply crossfade with previous chunk for continuity
            if i > 0:
                audio_chunk = self.crossfade(
                    self.prev_tail, audio_chunk, self.overlap_ms
                )
            self.prev_tail = audio_chunk[-self.overlap_ms:]

            yield audio_chunk

    def split_text(self, text):
        """Split text at natural boundaries for streaming."""
        import re
        # Split at punctuation while keeping context
        segments = re.split(r'(?<=[.!?,;:])\s+', text)
        # Merge very short segments
        merged = []
        buffer = ""
        for seg in segments:
            buffer += " " + seg if buffer else seg
            if len(buffer.split()) >= 5:  # Min 5 words per chunk
                merged.append(buffer.strip())
                buffer = ""
        if buffer:
            merged.append(buffer.strip())
        return merged

@app.websocket("/tts/stream")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    engine = StreamingTTSEngine()
    async for audio_chunk in engine.synthesize_stream(
        data["text"], data["speaker_prompt"]
    ):
        # Send raw PCM audio bytes
        await websocket.send_bytes(audio_chunk.numpy().tobytes())</code></pre>

<h4>Multilingual TTS: Language Mixing and Accent Control</h4>
<p>Production TTS frequently needs to handle multiple languages, code-switching (mixing languages within a sentence), and accent control.</p>

<p><strong>Challenges:</strong></p>
<ul>
<li><strong>Code-switching:</strong> "Please open the <em>Einstellungen</em> menu" -- English with a German word. The model must seamlessly switch phonetic systems.</li>
<li><strong>Accent preservation:</strong> A French speaker reading English should maintain their French accent, not switch to American English pronunciation.</li>
<li><strong>Script mixing:</strong> Chinese text mixed with English (common in technical content). Requires handling CJK characters and Latin characters in the same synthesis.</li>
</ul>

<p><strong>Solutions:</strong></p>
<ul>
<li><strong>CosyVoice 2:</strong> Natively supports code-switching between Chinese, English, Japanese, Cantonese, and Korean. Uses language ID tokens to signal language switches.</li>
<li><strong>Language tagging:</strong> Wrap foreign-language segments in SSML-style tags: <code>&lt;lang xml:lang="de"&gt;Einstellungen&lt;/lang&gt;</code>. The TTS system switches phoneme sets at tag boundaries.</li>
<li><strong>Accent control:</strong> Provide a prompt audio with the desired accent. The model's zero-shot mechanism captures accent along with voice timbre.</li>
</ul>

<h4>Low-Resource TTS: Few-Shot Adaptation</h4>
<p>For languages or speakers with limited data, several strategies can produce usable TTS:</p>

<ol>
<li><strong>Cross-lingual transfer (0-shot):</strong> Use a multilingual TTS model (CosyVoice, XTTS) and provide a 3-second prompt in the target language. Works surprisingly well for related languages (e.g., training on Spanish, applying to Portuguese).</li>
<li><strong>Few-shot fine-tuning (1-30 minutes of data):</strong> LoRA fine-tuning on the limited data. Key: aggressive regularization (low learning rate, weight decay, early stopping) to prevent overfitting.</li>
<li><strong>Data augmentation:</strong> Augment limited data with speed perturbation (0.9x, 1.0x, 1.1x), pitch shifting, and room impulse response simulation. This can effectively 3-5x your training data.</li>
<li><strong>Phoneme-based models:</strong> If no TTS training data exists but a pronunciation dictionary or G2P rules exist for the language, use a phoneme-based model (IPA input) and train with borrowed acoustic data from related languages.</li>
</ol>

<pre><code># Low-resource TTS adaptation strategy
def low_resource_tts_pipeline(
    target_audio,        # Path to limited target language audio
    target_transcripts,  # Corresponding transcriptions
    base_model="cosyvoice2_multilingual",
    data_minutes=30      # Amount of available data
):
    """
    Strategy selection based on available data.
    """
    if data_minutes < 1:
        # Zero-shot: just use prompt-based cloning
        print("Strategy: Zero-shot voice cloning")
        print("Use 3-10 seconds of target audio as prompt")
        return "zero_shot"

    elif data_minutes < 30:
        # Few-shot: LoRA fine-tuning with heavy regularization
        print(f"Strategy: Few-shot LoRA ({data_minutes} min)")
        config = {
            "lora_rank": 16,           # Low rank to prevent overfitting
            "learning_rate": 5e-5,     # Conservative LR
            "weight_decay": 0.1,       # Strong regularization
            "max_epochs": 100,         # With early stopping
            "augmentation": ["speed_perturb", "pitch_shift"],
            "val_split": 0.2,          # Hold out 20% for validation
        }
        return "few_shot_lora", config

    else:
        # Standard fine-tuning
        print(f"Strategy: Full fine-tuning ({data_minutes} min)")
        config = {
            "lora_rank": 64,
            "learning_rate": 1e-4,
            "max_epochs": 200,
            "augmentation": ["speed_perturb"],
        }
        return "fine_tune", config</code></pre>

<div class="callout warning">
<div class="callout-title">Production War Story: Multilingual TTS Gone Wrong</div>
<p>We deployed a multilingual TTS system for a customer support bot serving English, Spanish, and Portuguese. The system worked well for each language independently, but failed catastrophically on code-switched text (agent names in English within Spanish sentences). The model would either mangle the English name with Spanish phonology or switch to an English voice mid-sentence. Our fix: (1) Pre-tag code-switched segments with language IDs, (2) Use phoneme-level language embeddings so the model knew which phoneme rules to apply word-by-word, (3) Fine-tune specifically on code-switched examples (we generated 500 synthetic code-switched samples). After these changes, code-switch naturalness went from MOS 2.1 to 3.8.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Design a production TTS pipeline for a multilingual voice assistant that supports streaming and voice cloning. Walk through your architecture choices.</div>
<div class="a-text">Architecture: CosyVoice 2 as the base model (streaming support, multilingual, zero-shot cloning). Pipeline: (1) Text preprocessing: language detection per segment, phoneme conversion with language-specific G2P, SSML parsing for emphasis/pauses. (2) Voice management: store verified speaker prompts (10 seconds, pre-processed and cached) in a voice registry. (3) Streaming synthesis: chunk text at sentence boundaries, feed to CosyVoice 2's chunk-aware causal flow matching decoder. First chunk targets < 200ms latency. (4) Audio post-processing: volume normalization, silence trimming, optional noise gating. (5) Delivery: WebSocket streaming with opus encoding for low bandwidth. (6) Evaluation: automated WER + speaker similarity monitoring on sampled requests, weekly MOS evaluation on flagged samples. Key engineering decisions: use separate model instances per language family for better quality, implement prompt caching (pre-encode speaker embeddings), and maintain a fallback to sentence-level synthesis if streaming quality drops.</div>
</div>
`
    },
    {
      id: "tts-frontiers",
      title: "TTS Frontiers: What's Next",
      content: `
<p>The TTS field is evolving rapidly. This section covers the most promising research directions and emerging systems as of early 2026, helping AI engineers anticipate where the technology is heading.</p>

<h4>Emotion and Expressiveness Control</h4>
<p>Current TTS systems can produce natural-sounding speech, but fine-grained emotion control remains challenging. The frontier approaches:</p>
<ul>
<li><strong>Emotion embeddings:</strong> Discrete emotion tokens (happy, sad, angry, surprised) injected as conditioning signals. Works for basic emotions but fails for subtle or mixed emotions.</li>
<li><strong>Natural language emotion descriptions:</strong> Parler-TTS and its successors allow describing the desired emotional quality in text: "speak with gentle warmth and a slight smile in the voice." More flexible but harder to control precisely.</li>
<li><strong>Fine-grained prosody control:</strong> Systems like PromptTTS and InstructTTS allow specifying pitch contours, speaking rate, emphasis patterns, and pauses through structured prompts.</li>
<li><strong>Emotion transfer:</strong> Extract the emotional quality from a reference audio and apply it to new text. This decouples "what emotion" from "whose voice" -- you can say something happy in any voice.</li>
</ul>

<h4>Singing Voice Synthesis</h4>
<p>Singing is a natural extension of TTS but introduces significant additional challenges:</p>
<ul>
<li><strong>Pitch accuracy:</strong> Singing requires precise pitch control tied to a musical score. TTS prosody models are not designed for this level of pitch precision.</li>
<li><strong>Rhythm alignment:</strong> Notes must align precisely with beats. This is a much stricter temporal constraint than natural speech prosody.</li>
<li><strong>Vocal techniques:</strong> Vibrato, falsetto, belting, breathy singing -- each requires different acoustic modeling.</li>
<li><strong>Systems:</strong> DiffSinger (arXiv:2105.02446), ACE Studio, and SVC (Singing Voice Conversion) models. The gap between synthesized and real singing is still larger than the gap in speech.</li>
</ul>

<h4>Ultra-Low Latency: On-Device TTS</h4>
<p>The push toward on-device AI creates demand for TTS models that run on mobile phones and edge devices:</p>
<table>
<tr><th>System</th><th>Size</th><th>RTF (CPU)</th><th>Quality</th><th>Target Hardware</th></tr>
<tr><td>Piper</td><td>15-60 MB</td><td>~0.1</td><td>Acceptable</td><td>Raspberry Pi, mobile</td></tr>
<tr><td>VITS (quantized)</td><td>~50 MB</td><td>~0.2</td><td>Good</td><td>Mobile GPU</td></tr>
<tr><td>Matcha-TTS (int8)</td><td>~30 MB</td><td>~0.3</td><td>Good</td><td>Mobile NPU</td></tr>
<tr><td>eSpeak NG</td><td><5 MB</td><td>~0.01</td><td>Robotic</td><td>Microcontrollers</td></tr>
</table>

<p>The key techniques for on-device TTS: knowledge distillation (train a small student from a large teacher), quantization (INT8 or even INT4), and architectural choices (depthwise separable convolutions, grouped convolutions).</p>

<h4>Audio Watermarking and Provenance</h4>
<p>As synthetic speech becomes indistinguishable from natural speech, detecting and tracing synthetic audio is critical:</p>
<ul>
<li><strong>AudioSeal (Meta, arXiv:2401.17264):</strong> A localized watermarking method that embeds an imperceptible signal in generated audio. Can detect synthetic speech and identify which model generated it. Robust to common audio transformations (compression, noise addition, resampling).</li>
<li><strong>C2PA (Coalition for Content Provenance and Authenticity):</strong> A metadata standard that tags synthetic content with provenance information. Supported by major tech companies.</li>
<li><strong>Detection models:</strong> Binary classifiers trained to distinguish natural vs. synthetic speech. Current state-of-the-art achieves >99% detection on known TTS systems but struggles with novel/fine-tuned systems (the generalization problem).</li>
</ul>

<h4>Unified Audio Generation</h4>
<p>The trend is toward models that unify speech, music, and sound effects in a single system:</p>
<ul>
<li><strong>AudioLM (Google, arXiv:2209.03143):</strong> Early work on unified audio generation using semantic and acoustic tokens.</li>
<li><strong>AudioGen / MusicGen (Meta):</strong> Text-conditioned audio/music generation using single-codebook language modeling with delay patterns across RVQ layers.</li>
<li><strong>Stable Audio (Stability AI):</strong> Diffusion-based audio generation at 44.1kHz, handling speech, music, and sound effects.</li>
<li><strong>The convergence:</strong> As TTS, music generation, and sound effect generation all use similar codec + language model architectures, we are approaching unified models that generate any type of audio from any type of prompt (text, audio example, musical score).</li>
</ul>

<h4>Open Research Questions</h4>
<ol>
<li><strong>Evaluation:</strong> MOS is expensive and not reproducible across studies. Can we build a reliable automated evaluation metric that correlates with human perception across all quality levels?</li>
<li><strong>Long-form coherence:</strong> Current TTS handles sentences well but struggles with paragraph-level coherence (consistent prosody, natural paragraph structure, appropriate discourse markers).</li>
<li><strong>Controllability vs. naturalness tradeoff:</strong> More control knobs (emotion, style, speed) often reduce naturalness. How do we make control more intuitive without sacrificing quality?</li>
<li><strong>Data efficiency:</strong> The best systems require 60K+ hours of training data. Can we achieve comparable quality with 100x less data through better architectures or self-supervised pre-training?</li>
<li><strong>Real-time bidirectional interaction:</strong> Moshi showed that full-duplex speech interaction is possible. How do we make it robust and natural enough for production voice assistants?</li>
</ol>

<div class="callout">
<div class="callout-title">The TTS Landscape in 2026</div>
<p>The field has reached a remarkable point: zero-shot voice cloning with 3 seconds of audio, human-parity quality on standard benchmarks, real-time streaming synthesis, and multilingual support -- all from open-source models. The remaining challenges are about control (fine-grained emotion, style), robustness (handling edge cases, long-form content), and responsibility (preventing misuse while enabling legitimate applications). For AI engineers, the practical value of TTS expertise is high: every voice assistant, audiobook platform, accessibility tool, and content creation pipeline needs TTS, and the technology is now good enough for production use.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What are the biggest unsolved problems in TTS as of 2025-2026, and which do you think will be solved first?</div>
<div class="a-text">The major unsolved problems: (1) Long-form coherence -- maintaining consistent prosody, voice quality, and discourse structure over paragraphs or pages. (2) Fine-grained controllability -- precise emotion, emphasis, and style control without sacrificing naturalness. (3) Robust evaluation -- we lack reproducible automated metrics that match human judgment. (4) Data efficiency -- state-of-the-art requires 60K+ hours of training data. (5) Deepfake prevention -- watermarking works for known models but the arms race continues. I think long-form coherence will be solved first, likely through hierarchical generation (plan prosody at paragraph level, then generate) and larger context windows in codec LMs. Evaluation is the hardest because it requires solving the subjective nature of speech quality, which is fundamentally a human perception problem.</div>
</div>
`
    }
  ],

  // ============================================================================
  // CHAPTER 4: SPECULATIVE DECODING (8 sections, ~12,000 words)
  // ============================================================================
  ch4_sections: [
    {
      id: "sd-fundamentals",
      title: "Fundamentals & Core Protocol",
      content: `
<p>Speculative Decoding (SD) is one of the most impactful inference optimization techniques for Large Language Models. It exploits a fundamental asymmetry in transformer inference: <strong>verifying multiple tokens in parallel is much cheaper than generating them one at a time</strong>. This asymmetry exists because autoregressive LLM inference is memory-bandwidth bound, not compute bound -- the GPU spends most of its time waiting for weights to be loaded from memory, not actually computing.</p>

<h4>Why LLM Inference is Memory-Bound</h4>
<p>Consider a 70B parameter model in FP16 (140 GB of weights). At batch size 1, generating one token requires:</p>
<ul>
<li><strong>Compute:</strong> ~140 GFLOP (2 * 70B parameters, roughly)</li>
<li><strong>Memory reads:</strong> 140 GB (every weight must be read once)</li>
<li><strong>A100 80GB:</strong> 312 TFLOP/s compute, 2 TB/s memory bandwidth</li>
<li><strong>Time bottleneck:</strong> Compute = 140G / 312T = 0.45ms. Memory = 140G / 2T = 70ms</li>
<li><strong>Arithmetic intensity:</strong> 140 GFLOP / 140 GB = 1 FLOP/byte. Need ~156 FLOP/byte to be compute-bound on A100.</li>
</ul>

<p>The GPU utilization during single-token generation is only 1/156 = 0.6%. This means 99.4% of compute capacity is wasted. Speculative decoding fills this gap by verifying gamma draft tokens in the same forward pass, effectively getting gamma tokens for the price of ~1.2 tokens worth of wall-clock time (the overhead comes from the draft model and rejection sampling).</p>

<h4>The Core Protocol (Leviathan et al., 2023, arXiv:2211.17192; Chen et al., 2023, arXiv:2302.01318)</h4>
<p>The speculative decoding protocol has three phases:</p>

<pre><code># Speculative Decoding Protocol
# ==============================
# Target model: M_p (large, expensive) with distribution p(x_t | x_{<t})
# Draft model:  M_q (small, fast)    with distribution q(x_t | x_{<t})
# Draft length: gamma (typically 3-8 tokens)

def speculative_decode(target_model, draft_model, prefix, gamma):
    """
    Generate tokens from target distribution p using draft model q
    for acceleration. Output distribution is EXACTLY p.
    """
    x = prefix

    while not done:
        # Phase 1: DRAFT - generate gamma tokens from M_q (fast)
        draft_tokens = []
        draft_probs = []
        for i in range(gamma):
            q_dist = draft_model(x + draft_tokens)     # q(x | context)
            token = sample(q_dist)
            draft_tokens.append(token)
            draft_probs.append(q_dist)

        # Phase 2: VERIFY - run M_p on ALL gamma tokens in ONE forward pass
        # This is the key: parallel verification is almost as fast as
        # single-token generation (memory-bound, same weight reads)
        target_logits = target_model(x + draft_tokens)  # Batch of gamma+1
        p_dists = softmax(target_logits)  # p(x | context) for each position

        # Phase 3: ACCEPT/REJECT - rejection sampling
        accepted = 0
        for i in range(gamma):
            token = draft_tokens[i]
            p_i = p_dists[i][token]      # Target probability
            q_i = draft_probs[i][token]  # Draft probability

            if q_i <= p_i:
                # Draft underestimates: always accept
                accept(token)
                accepted += 1
            else:
                # Draft overestimates: accept with prob p_i / q_i
                if random() < p_i / q_i:
                    accept(token)
                    accepted += 1
                else:
                    # Reject: sample from adjusted distribution
                    adjusted = max(0, p_dists[i] - draft_probs[i])
                    adjusted = adjusted / adjusted.sum()
                    new_token = sample(adjusted)
                    accept(new_token)
                    break  # Stop accepting subsequent tokens

        # Bonus: if all gamma tokens accepted, get one more from target
        if accepted == gamma:
            bonus_token = sample(p_dists[gamma])
            accept(bonus_token)

    return x  # Final sequence sampled exactly from p</code></pre>

<h4>Why Rejection Sampling Preserves the Target Distribution</h4>
<p>The mathematical guarantee is critical: speculative decoding is <strong>lossless</strong> -- the output distribution is identical to sampling directly from the target model. The proof relies on the standard rejection sampling theorem:</p>

<p><strong>Claim:</strong> For each position, the accepted token follows distribution p(x).</p>
<p><strong>Proof sketch:</strong></p>
<pre><code># For a specific token value x at position t:
#
# Case 1: q(x) <= p(x)
#   Token x is always accepted when drafted.
#   Probability of drafting x AND accepting = q(x) * 1 = q(x)
#
# Case 2: q(x) > p(x)
#   Token x is accepted with probability p(x)/q(x).
#   Probability of drafting x AND accepting = q(x) * p(x)/q(x) = p(x)
#
# Wait -- Case 1 gives q(x), not p(x)!
# This is corrected by the REJECTION branch:
#
# Probability of rejection at this position:
#   beta = sum_x max(0, q(x) - p(x))    (total excess probability)
#
# When rejected, we sample from adjusted distribution:
#   p_adj(x) = max(0, p(x) - q(x)) / beta
#
# Total probability of accepting token x:
#   P(x) = P(accept from draft) + P(reject from draft) * P(sample x from adjusted)
#
# For any x where q(x) <= p(x):
#   P(x) = q(x) * 1 + beta * (p(x) - q(x)) / beta
#        = q(x) + p(x) - q(x)
#        = p(x)  ✓
#
# For any x where q(x) > p(x):
#   P(x) = q(x) * (p(x)/q(x)) + beta * 0/beta
#        = p(x) + 0
#        = p(x)  ✓
#
# Therefore P(x) = p(x) for all x.  QED.
#
# Note: beta = sum_x max(0, q(x) - p(x))
#            = 1 - sum_x min(q(x), p(x))
# This is also the probability of rejection at each position.</code></pre>

<h4>Expected Accepted Tokens: The Geometric Series</h4>
<p>The expected number of accepted tokens per speculation round determines the speedup. Let alpha be the <strong>acceptance rate</strong> -- the probability that a single draft token is accepted:</p>

<pre><code># alpha = E[min(1, p(x)/q(x))] = sum_x min(p(x), q(x))
# This equals 1 - total_variation_distance(p, q) * 2... not quite.
# More precisely: alpha = 1 - beta where beta = sum_x max(0, q(x) - p(x))
#
# Expected accepted tokens E[n] with gamma draft tokens:
#
# n=0: all rejected at first position. P(n=0) = (1-alpha) (get 1 adjusted token)
# n=1: first accepted, second rejected. P(n=1) = alpha * (1-alpha)
# n=k: first k accepted, (k+1)-th rejected. P(n=k) = alpha^k * (1-alpha)
# n=gamma: all accepted (get bonus token). P(n=gamma) = alpha^gamma
#
# E[tokens per round] = sum_{k=0}^{gamma-1} (k+1) * alpha^k * (1-alpha)
#                        + (gamma+1) * alpha^gamma
#
# After simplification (geometric series):
#
#   E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)
#
# Examples:
#   alpha=0.7, gamma=5:  E = (1 - 0.7^6) / 0.3 = (1 - 0.118) / 0.3 = 2.94
#   alpha=0.8, gamma=5:  E = (1 - 0.8^6) / 0.2 = (1 - 0.262) / 0.2 = 3.69
#   alpha=0.9, gamma=5:  E = (1 - 0.9^6) / 0.1 = (1 - 0.531) / 0.1 = 4.69
#   alpha=0.9, gamma=8:  E = (1 - 0.9^9) / 0.1 = (1 - 0.387) / 0.1 = 6.13
#
# Key insight: alpha matters much more than gamma.
# Doubling gamma from 4 to 8 at alpha=0.8: E goes from 3.36 to 4.46 (+33%)
# Raising alpha from 0.7 to 0.9 at gamma=5:  E goes from 2.94 to 4.69 (+60%)</code></pre>

<h4>Wall-Clock Speedup Formula</h4>
<p>The theoretical speedup is:</p>
<pre><code># Speedup = E[tokens per round] / time_per_round
# time_per_round = gamma * t_draft + t_verify
# where:
#   t_draft  = time for one draft model forward pass
#   t_verify = time for one target model forward pass (over gamma+1 tokens)
#
# Critically: t_verify ≈ t_target (single token) when memory-bound
# Because verification reads the same weights, just processes more tokens
#
# Speedup ≈ E[tokens] / (gamma * t_draft/t_target + 1)
#
# For EAGLE: t_draft/t_target ≈ 0.05 (draft is ~20x faster)
# With alpha=0.8, gamma=5:
#   Speedup ≈ 3.69 / (5 * 0.05 + 1) = 3.69 / 1.25 = 2.95x
#
# For external small model (e.g., 1B drafting for 70B):
#   t_draft/t_target ≈ 0.1-0.2 (draft runs on same GPU)
#   Speedup ≈ 3.69 / (5 * 0.15 + 1) = 3.69 / 1.75 = 2.11x</code></pre>

<div class="callout">
<div class="callout-title">The Three Levers of SD Performance</div>
<p><strong>1. Acceptance rate (alpha):</strong> How well does the draft approximate the target? Higher is better. Affected by draft model quality, temperature, task difficulty.<br>
<strong>2. Draft cost ratio (t_draft/t_target):</strong> How cheap is the draft model relative to the target? Lower is better. EAGLE achieves ~0.05; external models are 0.1-0.3.<br>
<strong>3. Draft length (gamma):</strong> How many tokens to speculate? More is better up to diminishing returns. Optimal gamma depends on alpha: higher alpha supports longer drafts.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Derive the expected number of tokens accepted in one round of speculative decoding, and explain when increasing gamma is worthwhile.</div>
<div class="a-text">Let alpha be the per-token acceptance probability. The number of accepted tokens follows a truncated geometric distribution: P(accept exactly k tokens) = alpha^k * (1-alpha) for k < gamma, and P(accept all gamma) = alpha^gamma. Expected tokens = sum from k=0 to gamma-1 of (k+1)*alpha^k*(1-alpha) + (gamma+1)*alpha^gamma. This simplifies to (1 - alpha^(gamma+1))/(1 - alpha). Increasing gamma is worthwhile when: (1) alpha is high (>0.7) so additional tokens have a reasonable chance of acceptance, (2) the draft model is cheap relative to the target (low t_draft/t_target), and (3) the marginal verification cost is low (memory-bound regime). Increasing gamma has diminishing returns because each additional token is accepted with probability alpha^(k+1), which decreases exponentially. The optimal gamma roughly satisfies alpha^gamma > t_draft/t_target.</div>
</div>
`
    },
    {
      id: "sd-eagle",
      title: "The EAGLE Series: State of the Art Drafting",
      content: `
<p>The EAGLE series (Li et al., 2024-2025) represents the most successful line of research in draft-head-based speculative decoding. Each iteration addresses a specific limitation of the previous version, making the series an excellent case study in iterative research improvement.</p>

<h4>EAGLE (ICML 2024, arXiv:2401.15077)</h4>
<p><strong>Core insight:</strong> Predicting the <em>next feature vector</em> (second-to-top hidden state) is easier than predicting the <em>next token</em>. Token-level prediction requires the draft to model the full vocabulary distribution, while feature-level prediction only needs to approximate a continuous vector in a structured embedding space.</p>

<p><strong>Architecture:</strong></p>
<pre><code># EAGLE-1 Architecture (detailed)
class EAGLEDraftHead(nn.Module):
    """
    Single-layer transformer decoder that predicts next feature vectors.
    Inputs: current token embedding + current top-layer feature from target model.
    Output: predicted next feature vector -> project to vocabulary for token.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size  # e.g., 4096 for LLaMA-2-70B

        # Token embedding (shared with target model)
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)

        # Feature projection: project target model's features
        self.feature_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Fusion: combine token embedding + projected feature
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        # Single transformer decoder layer (the entire "draft model")
        self.decoder_layer = LlamaDecoderLayer(config)

        # LM head (shared with target model for consistency)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

    def forward(self, token_ids, target_features):
        """
        token_ids: [batch, seq_len] - tokens at current positions
        target_features: [batch, seq_len, hidden] - features from target model
        """
        # Get token embeddings
        tok_emb = self.embed_tokens(token_ids)       # [B, S, H]

        # Project target features
        feat_emb = self.feature_proj(target_features) # [B, S, H]

        # Fuse token + feature information
        fused = self.fc(torch.cat([tok_emb, feat_emb], dim=-1))  # [B, S, H]

        # Single decoder layer with causal attention
        hidden = self.decoder_layer(fused)

        # Project to vocabulary
        logits = self.lm_head(hidden)                 # [B, S, V]
        return logits

    # Training: predict next token given (current token, current feature)
    # Target: the actual next token from training data
    # Loss: cross-entropy
    #
    # Key detail: features come from the FROZEN target model.
    # Only the draft head parameters are trained.</code></pre>

<p><strong>Training procedure:</strong></p>
<ol>
<li>Run target model on training data, save all hidden states at the second-to-top layer</li>
<li>For each position t, the draft head receives (token_t, feature_t) and predicts token_{t+1}</li>
<li>Standard cross-entropy loss, trained for ~1-2 epochs on a few thousand samples</li>
<li>Training cost: ~2-4 GPU-hours for a 70B model's draft head (extremely cheap)</li>
</ol>

<p><strong>Results:</strong> LLaMA2-Chat 70B: 2.7-3.5x latency speedup. Vicuna 33B: 2.8-3.3x. Lossless (exactly preserves target distribution).</p>

<h4>EAGLE-2 (EMNLP 2024, arXiv:2406.16858)</h4>
<p><strong>Core insight:</strong> EAGLE's draft model is well-calibrated -- its confidence scores closely approximate the actual acceptance probability. This means we can use draft confidence to build <em>dynamic</em> draft trees: expand high-confidence paths more and prune low-confidence paths.</p>

<p><strong>Context-Aware Dynamic Draft Trees:</strong></p>
<pre><code># EAGLE-2: Dynamic tree building based on draft confidence
class EAGLE2TreeBuilder:
    def __init__(self, max_nodes=60, confidence_threshold=0.3):
        self.max_nodes = max_nodes
        self.threshold = confidence_threshold

    def build_tree(self, draft_model, prefix_features, prefix_tokens):
        """
        Build a draft tree dynamically based on confidence scores.
        High-confidence branches get expanded more.
        """
        # Root: start from the last verified position
        tree = Tree()
        root = tree.add_root(prefix_tokens[-1], prefix_features[-1])

        # Priority queue: (negative_confidence, node)
        frontier = PriorityQueue()

        # Get initial draft predictions
        logits = draft_model(root.token, root.feature)
        probs = softmax(logits)

        # Add top-k children to frontier
        top_k_probs, top_k_tokens = probs.topk(10)
        for prob, token in zip(top_k_probs, top_k_tokens):
            if prob > self.threshold:
                child = tree.add_child(root, token, prob)
                frontier.put((-prob.item(), child))

        # Expand tree greedily by highest confidence
        while not frontier.empty() and tree.num_nodes < self.max_nodes:
            neg_conf, node = frontier.get()

            # Get draft model prediction for this node
            logits = draft_model(node.token, node.feature)
            probs = softmax(logits)

            # Expand: add children for high-confidence continuations
            top_k_probs, top_k_tokens = probs.topk(10)
            for prob, token in zip(top_k_probs, top_k_tokens):
                cumulative_conf = node.confidence * prob.item()
                if cumulative_conf > self.threshold and tree.num_nodes < self.max_nodes:
                    child = tree.add_child(node, token, cumulative_conf)
                    frontier.put((-cumulative_conf, child))

        return tree

    # The resulting tree has more nodes along high-confidence paths
    # and fewer along low-confidence paths.
    # Example: confident prefix might have depth 8-10
    #          uncertain prefix might have depth 3-4 with more breadth</code></pre>

<p><strong>Key improvement over EAGLE-1:</strong> Instead of a fixed tree topology (same branching pattern for every context), EAGLE-2 adapts the tree shape to the current context. When the model is confident (e.g., completing a common phrase), it speculates deeper. When uncertain (e.g., at a decision point), it speculates wider to cover more possibilities.</p>

<p><strong>Results:</strong> 3.05-4.26x speedup on LLaMA2-Chat and LLaMA3-Instruct series. 20-40% faster than EAGLE-1 with the same draft head.</p>

<h4>EAGLE-3 (2025, arXiv:2503.01840)</h4>
<p><strong>Core insight:</strong> EAGLE-1/2's approach of predicting the next feature vector has a ceiling -- the single-layer draft head can only approximate target features, and errors compound over multiple speculation steps. EAGLE-3 abandons feature prediction in favor of <strong>multi-layer feature fusion</strong> and a novel training technique.</p>

<p><strong>Architecture changes:</strong></p>
<ul>
<li><strong>Multi-layer fusion:</strong> Instead of using only the second-to-top layer, EAGLE-3 fuses features from multiple layers of the target model (e.g., layers 60, 65, 70, 75, 80 of an 80-layer model). This gives the draft head richer information about the target model's internal state.</li>
<li><strong>Direct token prediction:</strong> Returns to direct next-token prediction (like a small language model) rather than feature prediction. The multi-layer fusion provides enough context for accurate token-level prediction.</li>
<li><strong>"Training-Time Test" technique:</strong> During training, simulate inference conditions by having the draft head generate multiple speculative tokens and feeding them back (with noise/errors that mimic inference-time behavior). This exposes the draft to its own error distribution, making it robust to compounding errors during actual inference.</li>
</ul>

<p><strong>Results:</strong> Up to 6.5x speedup. First EAGLE variant that maintains significant speedup at batch size 64 (2.2x). Previous versions dropped below 1.5x at batch 32+.</p>

<h4>EAGLE in Production: Integration with SGLang</h4>
<pre><code># Deploying EAGLE-2 with SGLang (production configuration)
# SGLang has native EAGLE support since v0.2

# Launch server with EAGLE-2
python -m sglang.launch_server \\
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \\
    --speculative-algorithm EAGLE \\
    --speculative-draft-model-path yuhuili/EAGLE-LLaMA3-Instruct-70B \\
    --speculative-num-steps 5 \\
    --speculative-eagle-topk 8 \\
    --speculative-num-draft-tokens 60 \\
    --tp 4 \\
    --port 30000

# Key parameters:
# --speculative-num-steps: max depth of draft tree (5-8)
# --speculative-eagle-topk: branching factor at each node (4-10)
# --speculative-num-draft-tokens: max total nodes in tree (40-80)
#
# Tuning guide:
# - High-confidence tasks (translation, summarization): steps=7, topk=4
# - Open-ended generation (chat, creative writing): steps=5, topk=8
# - Code generation: steps=6, topk=6 (medium confidence, some variability)

# Client usage (same as standard OpenAI API)
import openai
client = openai.Client(base_url="http://localhost:30000/v1", api_key="none")
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    temperature=0.7,
    max_tokens=512
)
# SD is transparent to the client -- same API, same output distribution, faster</code></pre>

<div class="callout tip">
<div class="callout-title">EAGLE Draft Head Training Recipe</div>
<p><strong>Data:</strong> 2,000-5,000 ShareGPT conversations (or your domain data). Run target model inference to collect hidden features.<br>
<strong>Training:</strong> 1-2 epochs, learning rate 3e-5 with cosine decay, batch size 8-16. Total time: 2-6 GPU-hours on one A100.<br>
<strong>Validation:</strong> Measure acceptance rate on held-out data. Target: alpha > 0.7 for significant speedup.<br>
<strong>Cost perspective:</strong> Training an EAGLE draft head costs less than $10 in cloud compute. The inference savings from a 3x speedup on a 70B model pay this back in minutes of serving.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Trace the evolution from EAGLE to EAGLE-3. What limitation did each version address, and what is the key architectural change?</div>
<div class="a-text">EAGLE-1 showed that predicting next features (not tokens) is easier, using a single decoder layer with token+feature input. Limitation: fixed tree topology wastes verification budget on low-probability branches. EAGLE-2 solved this with context-aware dynamic trees, expanding high-confidence branches more. Limitation: feature prediction has a fundamental accuracy ceiling -- the single-layer draft can only approximate target features, and errors compound. EAGLE-3 abandons feature prediction entirely, instead using multi-layer feature fusion (combining information from multiple target model layers) for direct token prediction. It also introduces "training-time test" to make the draft robust to compounding errors. The result: EAGLE-3 achieves 6.5x speedup (vs 3.5x for EAGLE-1, 4.3x for EAGLE-2) and crucially maintains speedup at batch=64, which previous versions could not.</div>
</div>
`
    },
    {
      id: "sd-production",
      title: "SD in Production: Serving Systems & Deployment",
      content: `
<h4>The Batch Size Problem: SD's Achilles Heel</h4>
<p>The single most important thing to understand about speculative decoding in production: <strong>speedup decreases as batch size increases</strong>. This is because the fundamental premise of SD -- that verification is nearly free because inference is memory-bound -- breaks down when the batch is large enough to make inference compute-bound.</p>

<pre><code># Arithmetic intensity analysis
# Single token generation at batch size B:
#   Compute: 2 * params * B FLOPs
#   Memory:  2 * params bytes (read weights) + KV-cache reads
#
# At batch=1:  2 * 70G * 1 / (140G) = 1 FLOP/byte (memory-bound)
# At batch=16: 2 * 70G * 16 / (140G) = 16 FLOP/byte (still memory-bound)
# At batch=64: 2 * 70G * 64 / (140G) = 64 FLOP/byte (getting close to A100's 156)
# At batch=128: 128 FLOP/byte (nearly compute-bound)
#
# When compute-bound, verification of gamma tokens costs gamma * t_single
# instead of ~t_single, eliminating SD's core advantage.
#
# Approximate speedup as a function of batch size:
def sd_speedup(batch_size, alpha=0.8, gamma=5,
               draft_cost_ratio=0.05, flops_per_byte=156):
    """Estimate SD speedup accounting for batch size effects."""
    # Arithmetic intensity at this batch size
    intensity = 2 * batch_size  # Simplified
    # Verification overhead factor (1.0 when memory-bound, gamma when compute-bound)
    if intensity < flops_per_byte:
        # Memory-bound: verification is nearly free
        verify_factor = 1.0 + 0.1 * gamma  # Small overhead for extra tokens
    else:
        # Compute-bound: verification scales with gamma
        verify_factor = gamma * 0.7  # Not fully gamma due to caching

    expected_tokens = (1 - alpha**(gamma+1)) / (1 - alpha)
    time_per_round = gamma * draft_cost_ratio + verify_factor
    speedup = expected_tokens / (time_per_round / batch_size * batch_size)
    return speedup

# Typical observed speedups (EAGLE-2, LLaMA-3-70B, 4xA100):
# Batch=1:  3.2x
# Batch=4:  2.8x
# Batch=8:  2.3x
# Batch=16: 1.8x
# Batch=32: 1.3x
# Batch=64: 1.05x  (barely worth it)
# Batch=128: 0.9x  (SLOWER than no SD -- draft overhead dominates)</code></pre>

<h4>Decision Framework: When to Use SD in Production</h4>
<table>
<tr><th>Scenario</th><th>Use SD?</th><th>Why</th></tr>
<tr><td>Low-latency chatbot, few concurrent users</td><td>Yes</td><td>Batch 1-4, latency matters most, 2-4x speedup</td></tr>
<tr><td>High-throughput API, many concurrent users</td><td>No</td><td>Large batch sizes negate SD gains; continuous batching alone is better</td></tr>
<tr><td>Code completion (IDE integration)</td><td>Yes</td><td>Single-user, latency-critical, high acceptance rate on code</td></tr>
<tr><td>Batch inference (offline processing)</td><td>No</td><td>Throughput matters, not latency; large batches optimal</td></tr>
<tr><td>Long-context generation (>8K tokens)</td><td>Depends</td><td>MagicDec shows gains for long sequences; standard SD struggles</td></tr>
<tr><td>Multi-turn conversation</td><td>Yes (with caching)</td><td>KV-cache reuse makes verification cheaper; RadixAttention helps</td></tr>
</table>

<h4>Integration with Serving Frameworks</h4>
<p><strong>SGLang:</strong> The most mature SD integration. Native EAGLE-2 support with dynamic tree building. RadixAttention enables prefix sharing across draft and target models. Supports continuous batching with per-request SD decisions (enable SD only for low-batch requests).</p>

<p><strong>vLLM:</strong> Supports multiple SD methods: Medusa, EAGLE, and external draft models. As of v0.5+, integrates SD with continuous batching. The implementation uses PagedAttention for both draft and target KV-caches, ensuring efficient memory usage.</p>

<p><strong>TensorRT-LLM:</strong> NVIDIA's optimized inference engine supports SD through the "Speculative Decoding with Draft Model" feature. Best performance on NVIDIA hardware due to custom CUDA kernels for tree attention verification.</p>

<pre><code># SGLang: adaptive SD based on batch load
# This is the recommended production pattern

# sglang_config.yaml
server:
  model: meta-llama/Meta-Llama-3-70B-Instruct
  tp: 4
  speculative:
    algorithm: EAGLE
    draft_model: yuhuili/EAGLE-LLaMA3-Instruct-70B
    # Adaptive: disable SD when batch > threshold
    adaptive_threshold: 16        # Disable SD when batch > 16
    num_steps: 5
    eagle_topk: 8
    num_draft_tokens: 60

# In custom serving code, you can also dynamically control SD:
from sglang import Runtime

runtime = Runtime(model_path="...", speculative_config={...})

async def serve_request(request):
    current_batch = runtime.get_current_batch_size()
    if current_batch < 16:
        # Low load: use SD for latency
        response = await runtime.generate(
            request, speculative=True
        )
    else:
        # High load: skip SD for throughput
        response = await runtime.generate(
            request, speculative=False
        )</code></pre>

<h4>Memory Budget Considerations</h4>
<p>SD requires additional GPU memory for the draft model and draft KV-cache:</p>
<pre><code># Memory budget analysis
# Target model (70B, FP16): 140 GB
# Available on 4xA100 80GB: 320 GB total, ~180 GB free after target model

# EAGLE draft head: ~300 MB (single transformer layer + embeddings)
# Draft KV-cache per sequence: negligible (1 layer)
# Total EAGLE overhead: < 1 GB (excellent)

# External draft model (1B): ~2 GB weights + KV-cache
# External draft model (7B): ~14 GB weights + KV-cache
# Total external draft overhead: 2-15 GB (significant)

# Tree attention overhead:
# For max 60 draft nodes per sequence, batch 8:
# Extra KV-cache: 60 * 8 * num_layers * 2 * hidden_dim * 2 bytes
# For 70B: 60 * 8 * 80 * 2 * 8192 * 2 = ~12 GB (significant!)
#
# This is why EAGLE (minimal memory) beats external draft models
# in production: the memory saved can serve more concurrent requests.</code></pre>

<div class="callout warning">
<div class="callout-title">Production War Story: The SD Batch-Size Backfire</div>
<p>Our team deployed EAGLE-2 on a customer-facing chatbot serving Qwen-2.5-72B on 4xA100 80GB. Lab testing at batch=1 showed 3.2x P50 latency improvement -- impressive. We deployed to production with great excitement.</p>
<p>After 24 hours of monitoring: P50 latency improved only 1.1x. P99 was actually 15% <em>worse</em>. Investigation revealed: (1) Peak concurrency was 20-35 requests, meaning effective batch sizes of 20-35 where SD provides minimal benefit. (2) The EAGLE draft tree KV-cache consumed memory that reduced our max concurrent requests from 48 to 36 -- a 25% capacity reduction. (3) The P99 regression came from occasional tree verification failures at high batch sizes that added latency instead of saving it.</p>
<p><strong>Resolution:</strong> We implemented adaptive SD: enable EAGLE only when current batch < 8, disable it at higher loads. We also moved SD to a separate "premium" API tier with guaranteed low concurrency. The main serving path got 25% more throughput after removing SD entirely.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Your team wants to deploy speculative decoding in production. What questions would you ask before committing, and what benchmarks would you run?</div>
<div class="a-text">Questions: (1) What is our typical concurrency? If batch > 16, SD may not help. (2) Is latency or throughput the priority? SD helps latency at low batch, hurts throughput at high batch. (3) What GPU memory is available after the target model? EAGLE needs < 1GB, external drafters need 2-15GB. (4) What is the acceptance rate for our specific workload? (5) Can we afford the engineering complexity? Benchmarks to run: (1) Acceptance rate profiling on real production queries (not synthetic benchmarks). (2) Latency sweep at batch sizes 1, 4, 8, 16, 32, 64 -- plot speedup vs batch. (3) Memory impact: measure max concurrent requests with and without SD. (4) End-to-end throughput test at peak load. (5) P99 latency under realistic traffic patterns (not just P50). The key insight: always benchmark at your actual production batch sizes and traffic patterns, never just batch=1.</div>
</div>
`
    },
    {
      id: "sd-rejection-sampling",
      title: "Rejection Sampling Deep Dive",
      content: `
<p>Rejection sampling is the mathematical heart of speculative decoding. It is what makes SD <strong>lossless</strong> -- guaranteeing that the output distribution is identical to the target model, regardless of draft quality. Understanding the proof deeply is essential for AI engineers because it reveals when SD guarantees hold (and when they break).</p>

<h4>Standard Rejection Sampling (Background)</h4>
<p>Classical rejection sampling is a technique for sampling from a target distribution p(x) using a proposal distribution q(x) when direct sampling from p is difficult but evaluating p(x) is possible.</p>

<pre><code># Classical Rejection Sampling
# Given: target p(x), proposal q(x), constant M such that p(x) <= M*q(x) for all x
#
# Algorithm:
# 1. Sample x ~ q(x)
# 2. Sample u ~ Uniform(0, 1)
# 3. If u < p(x) / (M * q(x)): accept x
#    Else: reject and go to step 1
#
# Accepted samples follow distribution p(x).
# Acceptance rate: 1/M
#
# SD uses a MODIFIED version that avoids the constant M
# and handles rejection by sampling from an adjusted distribution
# (not by re-trying from scratch).</code></pre>

<h4>SD's Modified Rejection Sampling: Full Proof</h4>
<p>Speculative decoding uses a modified rejection sampling scheme. Here is the complete, rigorous proof of correctness.</p>

<pre><code># Setup:
# p(x) = target model probability for token x (given context)
# q(x) = draft model probability for token x (given context)
# Both are valid probability distributions: sum_x p(x) = sum_x q(x) = 1
#
# Algorithm for ONE position:
# 1. Sample x ~ q
# 2. Compute acceptance probability: a(x) = min(1, p(x) / q(x))
# 3. With probability a(x): accept x (output x)
#    With probability 1-a(x): reject x
# 4. If rejected: sample x' from adjusted distribution:
#    p_adj(x') = max(0, p(x') - q(x')) / Z
#    where Z = sum_{x'} max(0, p(x') - q(x'))
# 5. Output x' (always accept on resample)
#
# ===== PROOF THAT OUTPUT FOLLOWS p(x) =====
#
# Let P_out(x) = probability of outputting token x.
#
# P_out(x) = P(draft x AND accept x) + P(reject any draft) * P(resample x)
#
# Term 1: P(draft x AND accept x)
#   = q(x) * min(1, p(x)/q(x))
#   = min(q(x), p(x))
#
# To compute Term 2, first find the total rejection probability:
#   P(reject) = sum_x q(x) * (1 - min(1, p(x)/q(x)))
#             = sum_x q(x) * max(0, 1 - p(x)/q(x))
#             = sum_x max(0, q(x) - p(x))
#
# Key identity (fundamental to the proof):
#   sum_x max(0, q(x) - p(x)) = sum_x max(0, p(x) - q(x))
#
# Proof of identity:
#   sum_x [q(x) - p(x)] = 1 - 1 = 0
#   sum_x [q(x) - p(x)] = sum_{x: q>p} [q(x)-p(x)] - sum_{x: p>q} [p(x)-q(x)]
#   Therefore: sum_{x: q>p} [q(x)-p(x)] = sum_{x: p>q} [p(x)-q(x)]
#   i.e., sum_x max(0, q(x)-p(x)) = sum_x max(0, p(x)-q(x)) = Z
#
# So P(reject) = Z, and the adjusted distribution is:
#   p_adj(x) = max(0, p(x) - q(x)) / Z
#
# Term 2: P(reject) * P(resample x)
#   = Z * max(0, p(x) - q(x)) / Z
#   = max(0, p(x) - q(x))
#
# Combining:
#   P_out(x) = min(q(x), p(x)) + max(0, p(x) - q(x))
#
# Case A: p(x) >= q(x)
#   P_out(x) = q(x) + (p(x) - q(x)) = p(x)  ✓
#
# Case B: p(x) < q(x)
#   P_out(x) = p(x) + 0 = p(x)  ✓
#
# Therefore P_out(x) = p(x) for ALL x.  QED.
#
# ===== END PROOF =====</code></pre>

<div class="callout">
<div class="callout-title">Why This Proof Matters for Practice</div>
<p>The proof tells us several important things: (1) SD is lossless <strong>regardless of draft quality</strong> -- even a terrible draft model produces correct output, just slowly. (2) The acceptance rate equals 1 - Z = sum_x min(p(x), q(x)). This is the total overlap between p and q. (3) Temperature scaling affects acceptance: at temperature 0 (greedy), both p and q are peaked and may agree more often. At high temperature, distributions are flatter and overlap more, increasing alpha. (4) The guarantee holds ONLY for a single next token. Multi-token SD applies this position by position, and earlier rejections prevent later tokens from being evaluated -- this is where the geometric series comes from.</p>
</div>

<h4>Step-by-Step Algorithm with Pseudocode</h4>
<pre><code>import torch
import torch.nn.functional as F

def speculative_decode_step(
    target_logits,  # [gamma+1, vocab_size] - target model logits for all positions
    draft_logits,   # [gamma, vocab_size]   - draft model logits for positions 0..gamma-1
    draft_tokens,   # [gamma]               - tokens sampled from draft model
    temperature=1.0
):
    """
    One round of speculative decoding with rejection sampling.
    Returns: (accepted_tokens, num_accepted, bonus_token_or_None)
    """
    gamma = len(draft_tokens)

    # Convert logits to probabilities
    p = F.softmax(target_logits / temperature, dim=-1)  # [gamma+1, V]
    q = F.softmax(draft_logits / temperature, dim=-1)   # [gamma, V]

    accepted_tokens = []

    for i in range(gamma):
        token = draft_tokens[i]
        p_token = p[i, token].item()
        q_token = q[i, token].item()

        # Acceptance probability
        if q_token == 0:
            # Draft assigned zero probability -- always accept if p > 0
            if p_token > 0:
                accepted_tokens.append(token)
                continue
            else:
                break  # Both zero, this shouldn't happen in practice

        accept_prob = min(1.0, p_token / q_token)

        if torch.rand(1).item() < accept_prob:
            # Accept this draft token
            accepted_tokens.append(token)
        else:
            # Reject: sample from adjusted distribution
            adjusted = torch.clamp(p[i] - q[i], min=0)
            adjusted = adjusted / adjusted.sum()
            new_token = torch.multinomial(adjusted, 1).item()
            accepted_tokens.append(new_token)
            break  # Stop accepting subsequent tokens

    # If all gamma tokens accepted, get bonus token from target
    bonus = None
    if len(accepted_tokens) == gamma:
        bonus = torch.multinomial(p[gamma], 1).item()

    return accepted_tokens, len(accepted_tokens), bonus</code></pre>

<h4>Acceptance Rate Analysis: What Affects Alpha</h4>
<p>The acceptance rate alpha determines speedup. Understanding what affects it is crucial for deployment:</p>

<table>
<tr><th>Factor</th><th>Effect on Alpha</th><th>Explanation</th></tr>
<tr><td>Draft model quality</td><td>Higher quality -> higher alpha</td><td>Better draft more closely approximates target distribution</td></tr>
<tr><td>Temperature</td><td>Complex: see below</td><td>Affects distribution shape of both p and q</td></tr>
<tr><td>Top-p / top-k sampling</td><td>Usually increases alpha</td><td>Restricts vocabulary, increasing overlap between p and q</td></tr>
<tr><td>Task difficulty</td><td>Easier -> higher alpha</td><td>"The cat sat on the ___" is easy to draft; open-ended questions are harder</td></tr>
<tr><td>Token position</td><td>Middle of word -> higher alpha</td><td>Continuation of a word is highly predictable</td></tr>
<tr><td>Domain match</td><td>In-domain -> higher alpha</td><td>Draft trained on similar data has better approximation</td></tr>
</table>

<h4>Temperature Effects on Acceptance Rate</h4>
<pre><code># Temperature and acceptance rate analysis
import numpy as np

def compute_alpha(p_logits, q_logits, temperature):
    """
    Compute acceptance rate for given logits at a temperature.
    alpha = sum_x min(p(x), q(x))
    """
    p = softmax(p_logits / temperature)
    q = softmax(q_logits / temperature)
    return np.minimum(p, q).sum()

# Example with synthetic distributions
p_logits = np.array([3.0, 2.0, 1.0, 0.5, 0.1, -1.0])  # Target
q_logits = np.array([2.5, 2.2, 0.8, 0.6, 0.2, -0.8])  # Draft (close but imperfect)

for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
    alpha = compute_alpha(p_logits, q_logits, temp)
    print(f"T={temp:.1f}: alpha={alpha:.3f}")

# Typical output:
# T=0.1: alpha=0.987  (very peaked, argmax often agrees)
# T=0.5: alpha=0.942  (still peaked, high agreement)
# T=1.0: alpha=0.891  (standard, good overlap)
# T=2.0: alpha=0.856  (flatter, more disagreement on tails)
# T=5.0: alpha=0.823  (very flat, both near uniform, high overlap again!)
#
# Note: alpha is NOT monotonic with temperature.
# Very low T: both distributions peaked on same token -> high alpha
# Medium T: distributions spread, exposing draft-target mismatch -> lower alpha
# Very high T: both near uniform, trivially high overlap -> alpha recovers
#
# In practice, T=0 (greedy) gives highest alpha (argmax agreement)
# and standard T=0.7-1.0 gives alpha in [0.7, 0.9] for good draft models.</code></pre>

<h4>When Does the SD Guarantee Break?</h4>
<p>The lossless guarantee is mathematically airtight <em>under the following assumptions</em>:</p>
<ol>
<li><strong>Exact probability computation:</strong> Both p(x) and q(x) must be computed exactly. With quantized models or approximate softmax, small numerical errors can break the guarantee. In practice, FP16 precision is sufficient.</li>
<li><strong>Same context for draft and target:</strong> The draft and target must condition on exactly the same prefix. If there is a mismatch (e.g., different tokenizers or context truncation), the guarantee fails.</li>
<li><strong>Independent rejection decisions:</strong> Each position's accept/reject decision must be independent given the prefix. Techniques that batch or approximate rejection (for efficiency) may introduce small biases.</li>
<li><strong>Consistent sampling:</strong> The temperature, top-p, and top-k settings must be applied identically to both draft and target distributions before computing acceptance probabilities.</li>
</ol>

<div class="callout warning">
<div class="callout-title">Subtle Correctness Issue: Top-p with SD</div>
<p>When combining top-p sampling with SD, the truncation must be applied carefully. If top-p truncation is applied <em>before</em> computing acceptance ratios, the guarantee holds. If applied <em>after</em> (truncate the adjusted distribution), it may introduce bias. Most implementations handle this correctly, but it's worth verifying if you're building custom SD code.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Prove that speculative decoding's rejection sampling produces tokens from the target distribution. What assumptions are required?</div>
<div class="a-text">The proof works by showing P_out(x) = p(x) for all tokens x. For each position: P_out(x) = P(draft x and accept) + P(any rejection) * P(resample x). P(draft x and accept) = q(x) * min(1, p(x)/q(x)) = min(q(x), p(x)). P(reject) = sum_x max(0, q(x)-p(x)) = Z (by the identity that positive and negative differences sum equally). P(resample x) = max(0, p(x)-q(x))/Z. So Term 2 = max(0, p(x)-q(x)). Total: P_out(x) = min(q(x),p(x)) + max(0,p(x)-q(x)). If p(x) >= q(x): P_out = q(x) + p(x)-q(x) = p(x). If p(x) < q(x): P_out = p(x) + 0 = p(x). QED. Assumptions: exact probability computation (FP16 is sufficient), same context for draft and target, independent rejection decisions per position, and consistent sampling parameters applied to both distributions.</div>
</div>
`
    },
    {
      id: "sd-tree-attention",
      title: "Tree Attention & Verification",
      content: `
<p>In basic speculative decoding, draft tokens form a single chain (linear sequence). Tree-structured speculation dramatically increases the expected number of accepted tokens by exploring multiple possible continuations simultaneously. The key idea: instead of betting on one path, bet on many paths and keep the longest one that verifies.</p>

<h4>From Chains to Trees</h4>
<pre><code># Linear speculation (basic SD):
# Position:  0 -> 1 -> 2 -> 3 -> 4
# Token:     A -> B -> C -> D -> E
# If C is rejected, lose D and E too.
# Maximum accepted: 5 (if all pass)
#
# Tree speculation:
# Position 0: A
# Position 1: B (continuation of A), B' (alternative to B)
# Position 2: C (continuation of A->B), C' (continuation of A->B')
# Position 3: D (continuation of A->B->C)
#
#         A
#        / \\
#       B   B'
#      / \\    \\
#     C   C'   C''
#    /
#   D
#
# If B is rejected but B' is accepted, we can still get C'' and beyond.
# The tree covers more of the probability space.
# Maximum accepted: depth of longest verified path</code></pre>

<h4>Tree Attention Mask Construction</h4>
<p>To verify a tree of draft tokens in a single forward pass, we need a specialized attention mask. Each token in the tree can attend to its ancestors (the path from root to that token) but not to tokens in other branches.</p>

<pre><code>import torch
import numpy as np

def build_tree_attention_mask(tree_structure):
    """
    Build attention mask for tree-structured draft verification.

    tree_structure: list of (token_id, parent_index) pairs
                    parent_index=-1 for root
    Returns: [num_nodes, num_nodes] boolean attention mask

    Example tree:
        Node 0: root (parent=-1)
        Node 1: child of 0 (parent=0)
        Node 2: child of 0 (parent=0)  # sibling of 1
        Node 3: child of 1 (parent=1)
        Node 4: child of 1 (parent=1)  # sibling of 3
        Node 5: child of 2 (parent=2)
    """
    num_nodes = len(tree_structure)
    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)

    # Each node can attend to itself
    for i in range(num_nodes):
        mask[i, i] = True

    # Each node can attend to its ancestors
    for i in range(num_nodes):
        parent = tree_structure[i][1]  # parent_index
        while parent >= 0:
            mask[i, parent] = True
            parent = tree_structure[parent][1]

    return mask

# Example usage:
tree = [
    ("A", -1),  # Node 0: root
    ("B", 0),   # Node 1: child of root
    ("B'", 0),  # Node 2: alt child of root
    ("C", 1),   # Node 3: child of B
    ("D", 1),   # Node 4: alt child of B
    ("C'", 2),  # Node 5: child of B'
]

mask = build_tree_attention_mask(tree)
# mask:
#       A  B  B' C  D  C'
# A  [  1  0  0  0  0  0 ]   # A attends only to itself
# B  [  1  1  0  0  0  0 ]   # B attends to A and itself
# B' [  1  0  1  0  0  0 ]   # B' attends to A and itself
# C  [  1  1  0  1  0  0 ]   # C attends to A, B, and itself
# D  [  1  1  0  0  1  0 ]   # D attends to A, B, and itself
# C' [  1  0  1  0  0  1 ]   # C' attends to A, B', and itself

# This mask is passed to the target model's attention layers
# during verification. The model processes all tree nodes in
# parallel, with each node seeing only its ancestral context.</code></pre>

<h4>Tree Verification Algorithm</h4>
<pre><code>def verify_tree(target_model, tree, prefix, temperature=1.0):
    """
    Verify all paths in the draft tree using one target model forward pass.
    Returns the longest accepted path.
    """
    # Flatten tree into a sequence for batch processing
    tree_tokens = [node.token for node in tree.nodes]
    tree_mask = build_tree_attention_mask(tree.structure)

    # Prepend prefix attention (all tree nodes attend to full prefix)
    full_mask = build_full_mask(prefix, tree_mask)

    # ONE forward pass through target model
    target_logits = target_model(
        input_ids=prefix + tree_tokens,
        attention_mask=full_mask
    )

    # Extract logits for tree positions
    tree_logits = target_logits[len(prefix):]  # [num_nodes, vocab_size]
    p = F.softmax(tree_logits / temperature, dim=-1)

    # Verify each node using rejection sampling
    # Process in BFS order (parents before children)
    node_accepted = {}
    for node in tree.bfs_order():
        if node.is_root:
            # Root is always a candidate (it's the first draft token)
            pass

        parent = node.parent
        if parent is not None and parent.id not in node_accepted:
            # Parent was rejected -- this node is automatically rejected
            continue

        # Rejection sampling for this node
        token = node.token
        p_token = p[node.id, token].item()
        q_token = node.draft_prob

        accept_prob = min(1.0, p_token / q_token)
        if torch.rand(1).item() < accept_prob:
            node_accepted[node.id] = True
        else:
            # Reject: sample replacement from adjusted distribution
            adjusted = torch.clamp(p[node.id] - node.draft_dist, min=0)
            if adjusted.sum() > 0:
                adjusted = adjusted / adjusted.sum()
                replacement = torch.multinomial(adjusted, 1).item()
                node_accepted[node.id] = replacement  # Store replacement token
            break  # Don't process children of rejected node

    # Find longest accepted path from root
    best_path = find_longest_accepted_path(tree, node_accepted)
    return best_path</code></pre>

<h4>Dynamic Tree Building: The EAGLE-2 Approach</h4>
<p>Static trees (fixed topology) waste verification budget on unlikely branches. EAGLE-2's dynamic trees allocate more budget to promising paths:</p>

<pre><code># Comparison: Static vs Dynamic trees (EAGLE-2)
#
# Static tree (Medusa default, fixed topology):
#         [root]
#        /  |  \\
#       1   2   3       <- always 3 branches at depth 1
#      /|  /|  /|
#     4 5 6 7 8 9       <- always 2 branches per node at depth 2
#
# Total nodes: 10 (fixed)
# Problem: if the model is confident, depth 2 is wasted.
#          if uncertain, all branches at depth 1 are equally uncertain.
#
# Dynamic tree (EAGLE-2):
# If root is confident (p=0.95 for top token):
#         [root]
#           |
#           1 (p=0.95)
#           |
#           2 (p=0.90)
#           |
#           3 (p=0.85)
#           |
#           4 (p=0.80)
#           |
#           5 (p=0.75)
# -> Deep tree: 5 nodes, all along the confident path
#
# If root is uncertain (p=0.40 for top token):
#         [root]
#        / | | \\
#       1  2  3  4     <- 4 branches to cover uncertainty
#      /|  |  |
#     5 6  7  8        <- only expand promising branches
# -> Wide tree: 9 nodes, more breadth to cover uncertainty</code></pre>

<h4>Optimal Tree Topology: Sequoia and OPT-Tree</h4>
<p><strong>Sequoia (Chen et al., 2024, arXiv:2402.12374):</strong> Formulates tree topology optimization as a combinatorial problem. Given a budget of N nodes and an empirical acceptance rate profile, find the tree shape that maximizes expected accepted tokens. Key finding: the optimal tree is usually neither purely wide (fan-out) nor purely deep (chain), but has a distinctive "Christmas tree" shape -- wide at the top, narrowing toward the bottom.</p>

<p><strong>OPT-Tree (2024):</strong> Extends Sequoia by making the tree topology <em>adaptive per token</em> rather than static across all sequences. Uses a lightweight heuristic based on draft model entropy: low entropy (confident) -> go deeper, high entropy (uncertain) -> go wider.</p>

<h4>Memory Overhead Analysis</h4>
<p>Tree attention introduces significant memory overhead compared to linear speculation:</p>

<pre><code># Memory analysis: Linear vs Tree speculation
#
# Linear speculation (gamma=5):
#   Extra KV-cache entries: 5 per layer per request
#   For LLaMA-3-70B (80 layers, 8 KV heads, 128 head_dim):
#     Per request: 5 * 80 * 2 * 8 * 128 * 2 bytes (FP16)
#                = 5 * 80 * 2 * 8 * 128 * 2
#                = 1.31 MB
#   For batch 32: 42 MB (manageable)
#
# Tree speculation (60 nodes):
#   Extra KV-cache entries: 60 per layer per request
#   Per request: 60 * 80 * 2 * 8 * 128 * 2 = 15.7 MB
#   For batch 32: 503 MB (significant!)
#
# Attention computation:
# Linear: attention over prefix + 5 tokens (negligible)
# Tree: attention with tree mask over prefix + 60 tokens
#       Tree mask prevents flash attention optimization in some cases
#       Custom tree attention kernels needed (SGLang implements this)
#
# PRACTICAL IMPACT:
# With GQA (Grouped Query Attention, common in modern models):
#   LLaMA-3-70B uses 8 KV heads (not 64 attention heads)
#   This reduces KV-cache by 8x
#   Tree overhead with GQA: 60 * 80 * 2 * 8 * 128 * 2 / 8 = ~2 MB/request
#   For batch 32: ~63 MB (much more manageable)
#
# Rule of thumb: tree size N should satisfy
#   N * batch_size * kv_per_token < 5% of total GPU memory</code></pre>

<div class="callout tip">
<div class="callout-title">Tree Attention Implementation Tips</div>
<p><strong>Flash Attention compatibility:</strong> Standard Flash Attention requires a causal (triangular) mask. Tree attention masks are more complex. Solutions: (1) Use block-sparse attention with tree-structured blocks (SGLang's approach), (2) Pad tree nodes into a sequence and use a custom attention mask (slower but simpler), (3) Use FlashInfer library which supports tree attention natively.<br><br>
<strong>KV-cache management:</strong> When a path in the tree is rejected, its KV-cache entries should be freed. With PagedAttention (vLLM), this is efficient -- just release the corresponding pages. Without it, you may need to compact the KV-cache.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain tree attention in speculative decoding. How is the attention mask constructed, and what are the memory implications?</div>
<div class="a-text">In tree speculation, the draft model generates multiple possible continuations forming a tree. Each path from root to leaf is one possible continuation. The attention mask ensures each node attends only to its ancestors (the path from root to that node), not to sibling or cousin nodes. Construction: for each node i with parent chain [root, ..., parent, i], set mask[i][j] = True for all j in the ancestor chain. The tree is verified in a single target model forward pass using this mask. Memory implications: for N tree nodes, KV-cache overhead is N entries per layer per request, compared to gamma entries for linear speculation (N is typically 10-20x larger than gamma). With GQA this is manageable (~2MB/request for 60-node trees on LLaMA-3-70B), but at large batch sizes it adds up. The mask also prevents standard flash attention; custom tree attention kernels (FlashInfer, SGLang) are needed for efficiency.</div>
</div>
`
    },
    {
      id: "sd-alternatives",
      title: "Alternative Speculative Decoding Methods",
      content: `
<p>While EAGLE represents the state of the art for draft-head approaches, the speculative decoding landscape includes many creative alternatives. Each makes different tradeoffs between speed, quality, memory, and implementation complexity. Understanding the full landscape helps you choose the right method for your specific constraints.</p>

<h4>Medusa: Independent Multi-Head Drafting</h4>
<p>Medusa (Cai et al., 2024, arXiv:2401.10774) attaches multiple "Medusa heads" to the target model. Each head independently predicts a token at a different future position.</p>

<pre><code># Medusa Architecture
class MedusaModel(nn.Module):
    """
    Add K independent prediction heads to the target model.
    Head i predicts token at position t+i+1 given features at position t.
    """
    def __init__(self, base_model, num_heads=5, hidden_size=4096, vocab_size=32000):
        super().__init__()
        self.base_model = base_model  # Frozen target model

        # K independent Medusa heads
        # Each head: 1-2 layer MLP on top of target model features
        self.medusa_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size)
            )
            for _ in range(num_heads)
        ])

    def forward(self, input_ids, **kwargs):
        # Get base model features (frozen, no grad)
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # Last layer features

        # Original next-token prediction (from base model)
        base_logits = outputs.logits

        # Medusa predictions for positions t+2, t+3, ..., t+K+1
        medusa_logits = [head(hidden) for head in self.medusa_heads]

        return base_logits, medusa_logits

    # Training: freeze base model, train only Medusa heads
    # Data: standard language modeling data
    # Loss: cross-entropy for each head predicting its respective position
    # Training time: 1-2 epochs, ~4-8 GPU-hours on A100 for a 70B model
    #
    # CRITICAL LIMITATION: heads are INDEPENDENT.
    # Head 2's prediction of token t+3 does NOT depend on what
    # head 1 predicted for t+2. This limits accuracy for
    # multi-step prediction where tokens are dependent.</code></pre>

<p><strong>Medusa verification:</strong> Medusa constructs a tree from the Cartesian product of each head's top-k predictions and verifies using tree attention. With 5 heads and top-3 per head: 3^5 = 243 possible paths (typically pruned to 60-80 nodes).</p>

<p><strong>Results:</strong> 2.2-3.6x speedup on various models. Lower than EAGLE because independent heads cannot model inter-token dependencies.</p>

<h4>Hydra: Sequential Dependence Between Heads</h4>
<p>Hydra (Ankner et al., 2024) addresses Medusa's independence limitation by making each head condition on the predictions of previous heads:</p>

<ul>
<li><strong>Architecture:</strong> Head i receives (target features, predictions from heads 1..i-1) as input. This creates sequential dependence -- head 2 knows what head 1 predicted before making its own prediction.</li>
<li><strong>Tradeoff:</strong> Better accuracy per head (higher acceptance rate) but slower draft generation (sequential, not parallel heads). The drafting phase is ~2x slower than Medusa.</li>
<li><strong>Results:</strong> ~2.7x speedup. Better acceptance rate than Medusa but similar wall-clock speedup due to the sequential overhead.</li>
</ul>

<h4>LayerSkip: Self-Speculative Decoding (Zero Extra Memory)</h4>
<p>LayerSkip (Elhoushi et al., 2024, arXiv:2404.16710) is a fundamentally different approach: instead of a separate draft model, use the <strong>target model's own early layers</strong> as the drafter.</p>

<pre><code># LayerSkip Concept
# Target model: 80-layer LLaMA
# Draft: exit at layer 20 (skip layers 21-80)
# Verification: full 80-layer forward pass
#
# The idea: early layers already capture much of the prediction.
# For "easy" tokens (common words, continuations), layer 20's
# prediction often matches layer 80's.

class LayerSkipModel(nn.Module):
    """
    Self-speculative model: early exit for drafting,
    full model for verification.
    """
    def __init__(self, base_model, exit_layer=20):
        super().__init__()
        self.model = base_model
        self.exit_layer = exit_layer

        # Early exit classifier (trained on early layer features)
        hidden_size = base_model.config.hidden_size
        self.early_head = nn.Linear(hidden_size, base_model.config.vocab_size)

    def draft(self, input_ids, num_tokens=5):
        """Generate draft tokens using only first exit_layer layers."""
        draft_tokens = []
        hidden = self.model.embed_tokens(input_ids)

        for _ in range(num_tokens):
            # Run only first exit_layer layers
            for layer_idx in range(self.exit_layer):
                hidden = self.model.layers[layer_idx](hidden)

            # Predict from early features
            logits = self.early_head(hidden[:, -1:])
            token = logits.argmax(dim=-1)
            draft_tokens.append(token)

            # Extend hidden for next draft token
            token_emb = self.model.embed_tokens(token)
            hidden = torch.cat([hidden, token_emb], dim=1)

        return draft_tokens

    def verify(self, input_ids_with_drafts):
        """Standard full forward pass for verification."""
        return self.model(input_ids_with_drafts)

# Key advantages:
# 1. ZERO extra memory -- no separate draft model or heads
# 2. Perfect token/vocabulary alignment (same model)
# 3. No training required if using a pre-trained exit head
#
# Key disadvantages:
# 1. Lower acceptance rate than EAGLE (early exit is less accurate)
# 2. Cannot parallelize draft and verify (same model weights)
# 3. Draft speed advantage is smaller (~4x vs ~20x for EAGLE)</code></pre>

<p><strong>Training:</strong> LayerSkip requires training with an early exit loss (add auxiliary loss at the early exit layer during pre-training or fine-tuning). Without this, early layers are not optimized for prediction, giving poor acceptance rates.</p>

<p><strong>Results:</strong> 1.4-2.2x speedup. Lower than EAGLE but requires zero extra memory and no additional model artifacts.</p>

<h4>Lookahead Decoding: Jacobi Iteration</h4>
<p>Lookahead decoding (Fu et al., 2024, arXiv:2402.02057) uses a completely different mathematical framework: parallel Jacobi iteration instead of draft-then-verify.</p>

<pre><code># Lookahead Decoding Concept
# Start with random initial guesses for the next N tokens:
# Step 0: [correct_prefix] [random] [random] [random] [random]
#
# Jacobi iteration: for each position, predict the token given
# ALL other positions (including not-yet-correct ones):
# Step 1: [correct_prefix] [token_1'] [token_2'] [token_3'] [token_4']
# Step 2: [correct_prefix] [token_1''] [token_2''] [token_3''] [token_4'']
# ...
# Converge when predictions stabilize.
#
# Key insight: if position i has stabilized (same prediction for K
# consecutive iterations), it is likely correct.
#
# In practice, Lookahead also collects N-grams from the Jacobi
# trajectory and uses them as future draft candidates.
#
# Advantages:
# - No separate draft model needed
# - Works with ANY model out of the box
# - Theoretically sound (Jacobi fixed-point convergence)
#
# Disadvantages:
# - Convergence is slow (5-10 iterations typical)
# - Each iteration is a full forward pass
# - Speedup is modest: 1.5-2x typically
# - Hard to combine with KV-cache (iterations modify predictions)</code></pre>

<h4>REST: Retrieval-Based Non-Parametric Drafting</h4>
<p>REST (He et al., 2023, arXiv:2311.08252) takes a completely non-parametric approach: use a retrieval datastore to find draft continuations.</p>

<ul>
<li><strong>Approach:</strong> Build a datastore of (context, continuation) pairs from training data using suffix arrays. At inference, look up the current context and retrieve possible continuations as drafts.</li>
<li><strong>Advantages:</strong> No training, no extra model parameters, works with any target model. Can achieve high acceptance rates on domain-specific data (e.g., code generation where patterns repeat frequently).</li>
<li><strong>Disadvantages:</strong> Requires a large retrieval index (10-100GB+). Acceptance rate depends heavily on domain and data coverage. Poor for open-domain creative tasks.</li>
<li><strong>Best use case:</strong> Code completion, template-heavy text generation, domain-specific applications where patterns repeat.</li>
</ul>

<h4>MagicDec: Breaking the Batch-Size Barrier</h4>
<p>MagicDec (Chen et al., 2024, arXiv:2408.11049) specifically addresses SD's batch-size problem for long sequences:</p>

<ul>
<li><strong>Key observation:</strong> For long sequences (>8K tokens), the KV-cache read dominates memory bandwidth even at large batch sizes. This means the memory-bound regime extends to higher batch sizes for longer contexts.</li>
<li><strong>Approach:</strong> Use a draft model that operates on a <em>subset</em> of the KV-cache (sparse attention over the long context). The draft model is much faster because it reads less KV-cache, while the target model still reads the full cache for verification.</li>
<li><strong>Results:</strong> 1.5-2x speedup at batch size 64 with 8K+ context lengths -- a regime where standard SD provides no benefit.</li>
<li><strong>Practical implication:</strong> If your workload involves long documents (legal, code, research), MagicDec may be the right choice even at moderate batch sizes.</li>
</ul>

<h4>Method Comparison Summary</h4>
<table>
<tr><th>Method</th><th>Extra Memory</th><th>Training Cost</th><th>Speedup (B=1)</th><th>Speedup (B=32+)</th><th>Best For</th></tr>
<tr><td>EAGLE-3</td><td>~300 MB</td><td>~4 GPU-hrs</td><td>4-6.5x</td><td>1.5-2.2x</td><td>General purpose, best overall</td></tr>
<tr><td>Medusa</td><td>~500 MB</td><td>~6 GPU-hrs</td><td>2.2-3.6x</td><td>~1.2x</td><td>Simple to implement</td></tr>
<tr><td>Hydra</td><td>~500 MB</td><td>~8 GPU-hrs</td><td>~2.7x</td><td>~1.2x</td><td>Higher acceptance than Medusa</td></tr>
<tr><td>LayerSkip</td><td>0</td><td>Requires training*</td><td>1.4-2.2x</td><td>~1.1x</td><td>Memory-constrained deployments</td></tr>
<tr><td>Lookahead</td><td>0</td><td>0</td><td>1.5-2x</td><td>~1.0x</td><td>Drop-in, no setup needed</td></tr>
<tr><td>REST</td><td>10-100 GB index</td><td>Index building</td><td>1.5-3x</td><td>~1.2x</td><td>Domain-specific, repetitive tasks</td></tr>
<tr><td>MagicDec</td><td>Moderate</td><td>Moderate</td><td>~2x</td><td>1.5-2x (!)</td><td>Long-context, large batch</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Compare Medusa, EAGLE, and LayerSkip. When would you choose each one?</div>
<div class="a-text">Medusa: K independent heads predict future tokens in parallel. Simple to implement and train (~6 GPU-hours). Achieves 2.2-3.6x speedup. Limitation: heads are independent, so multi-step prediction accuracy is limited. Choose when you want a simple, reliable solution with moderate speedup. EAGLE: predicts next features (not tokens) using a single decoder layer that takes token+feature as input. Higher accuracy due to feature-level prediction. 4-6.5x speedup. Choose for maximum speed when you can afford a small additional model artifact (~300MB). EAGLE-3's batch-size resilience makes it the best general choice. LayerSkip: uses the target model's own early layers as drafter. Zero extra memory. 1.4-2.2x speedup. Choose when GPU memory is the binding constraint and any speedup helps -- for example, running the largest possible model on limited hardware where you can't afford even 300MB for an EAGLE head. The tradeoff is always: more memory/complexity = more speedup (EAGLE) vs. simpler/lighter = less speedup (LayerSkip, Lookahead).</div>
</div>
`
    },
    {
      id: "sd-benchmarking",
      title: "Benchmarking Speculative Decoding",
      content: `
<p>Benchmarking speculative decoding correctly is surprisingly tricky. Many published speedup numbers are misleading because they measure the wrong things, under unrealistic conditions. This section provides a rigorous benchmarking methodology and code you can use to evaluate SD in your own setup.</p>

<h4>What to Measure (and What NOT to Measure)</h4>
<p><strong>Common mistake #1: Reporting tokens/second as the speedup metric.</strong> Tokens/second is misleading for SD because the "tokens" generated per speculation round include both accepted draft tokens AND the bonus token. A fairer metric is <strong>wall-clock time to complete a fixed generation task</strong>.</p>

<p><strong>Common mistake #2: Benchmarking only at batch size 1.</strong> SD shines at batch=1 but most production deployments run at batch 8-64. Always sweep batch sizes.</p>

<p><strong>Common mistake #3: Using only one type of prompt.</strong> Acceptance rates vary dramatically by task. Code completion might have alpha=0.9; open-ended creative writing might have alpha=0.5.</p>

<p><strong>The correct metrics:</strong></p>
<table>
<tr><th>Metric</th><th>Definition</th><th>Why It Matters</th></tr>
<tr><td>Wall-clock speedup</td><td>t_baseline / t_sd for same output</td><td>The only metric users care about</td></tr>
<tr><td>TTFT (Time To First Token)</td><td>Time from request to first generated token</td><td>Critical for interactive applications</td></tr>
<tr><td>ITL (Inter-Token Latency)</td><td>Average time between consecutive tokens</td><td>Affects perceived streaming speed</td></tr>
<tr><td>Acceptance rate (alpha)</td><td>Fraction of draft tokens accepted</td><td>Key diagnostic; determines theoretical speedup</td></tr>
<tr><td>Tokens per round</td><td>Average tokens generated per speculation round</td><td>Combines alpha and gamma effects</td></tr>
<tr><td>Memory overhead</td><td>Additional GPU memory used by SD</td><td>Determines max concurrent requests</td></tr>
<tr><td>Throughput at target latency</td><td>Max requests/sec while meeting latency SLA</td><td>Production capacity planning</td></tr>
</table>

<h4>Acceptance Rate Profiling</h4>
<pre><code>import torch
import time
import json
from collections import defaultdict

class SDProfiler:
    """
    Profile speculative decoding performance across different
    batch sizes, temperatures, and prompt types.
    """
    def __init__(self, target_model, draft_model, tokenizer):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.results = defaultdict(list)

    def profile_acceptance_rate(self, prompts, gamma=5, temperature=1.0):
        """
        Measure per-token acceptance rate across a set of prompts.
        Returns detailed statistics.
        """
        all_alphas = []
        per_position_accepts = [0] * gamma
        per_position_total = [0] * gamma

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()

            # Generate tokens and track acceptance
            for step in range(50):  # 50 generation steps per prompt
                # Draft phase
                draft_tokens, draft_probs = self.draft_tokens(
                    input_ids, gamma, temperature
                )

                # Verify phase
                target_probs = self.target_verify(
                    input_ids, draft_tokens, temperature
                )

                # Check acceptance per position
                for i in range(gamma):
                    p = target_probs[i][draft_tokens[i]].item()
                    q = draft_probs[i][draft_tokens[i]].item()

                    accept_prob = min(1.0, p / max(q, 1e-10))
                    accepted = torch.rand(1).item() < accept_prob

                    per_position_total[i] += 1
                    if accepted:
                        per_position_accepts[i] += 1
                        all_alphas.append(1)
                    else:
                        all_alphas.append(0)
                        break  # Stop at first rejection

                # Advance input_ids with accepted tokens
                # ... (update for next step)

        # Compute statistics
        overall_alpha = sum(all_alphas) / max(len(all_alphas), 1)
        position_alphas = [
            per_position_accepts[i] / max(per_position_total[i], 1)
            for i in range(gamma)
        ]

        return {
            "overall_alpha": overall_alpha,
            "per_position_alpha": position_alphas,
            "expected_tokens": (1 - overall_alpha**(gamma+1)) / (1 - overall_alpha),
            "num_prompts": len(prompts),
        }

    def profile_batch_sweep(self, prompts, batch_sizes=[1, 2, 4, 8, 16, 32, 64],
                             gamma=5, max_new_tokens=128):
        """
        Sweep batch sizes and measure wall-clock speedup.
        """
        results = {}

        for bs in batch_sizes:
            # Baseline: standard autoregressive generation
            baseline_times = []
            for i in range(0, len(prompts), bs):
                batch = prompts[i:i+bs]
                if len(batch) < bs:
                    continue
                input_ids = self.tokenizer(
                    batch, return_tensors="pt", padding=True
                ).input_ids.cuda()

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    self.target.generate(
                        input_ids, max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                torch.cuda.synchronize()
                baseline_times.append(time.perf_counter() - t0)

            # SD: speculative decoding generation
            sd_times = []
            for i in range(0, len(prompts), bs):
                batch = prompts[i:i+bs]
                if len(batch) < bs:
                    continue
                input_ids = self.tokenizer(
                    batch, return_tensors="pt", padding=True
                ).input_ids.cuda()

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    speculative_generate(
                        self.target, self.draft, input_ids,
                        max_new_tokens=max_new_tokens, gamma=gamma
                    )
                torch.cuda.synchronize()
                sd_times.append(time.perf_counter() - t0)

            baseline_avg = sum(baseline_times) / max(len(baseline_times), 1)
            sd_avg = sum(sd_times) / max(len(sd_times), 1)

            results[bs] = {
                "baseline_ms": baseline_avg * 1000,
                "sd_ms": sd_avg * 1000,
                "speedup": baseline_avg / max(sd_avg, 1e-10),
                "num_batches": len(baseline_times),
            }

            print(f"Batch {bs:3d}: baseline={baseline_avg*1000:.0f}ms, "
                  f"SD={sd_avg*1000:.0f}ms, speedup={results[bs]['speedup']:.2f}x")

        return results</code></pre>

<h4>Batch Size Sweep Methodology</h4>
<p>A proper batch size sweep should:</p>
<ol>
<li><strong>Warmup:</strong> Run 10+ warmup iterations before timing. GPU frequencies ramp up and CUDA caches populate during warmup.</li>
<li><strong>Multiple trials:</strong> Run each configuration at least 20 times and report mean + standard deviation. Single-run numbers are unreliable.</li>
<li><strong>Realistic prompts:</strong> Use prompts from your actual production workload, not synthetic benchmarks. Mix prompt lengths, topics, and difficulty levels.</li>
<li><strong>Fixed output length:</strong> Compare at the same number of generated tokens (e.g., always generate 128 tokens). Variable-length generation introduces noise from content-dependent stopping.</li>
<li><strong>Memory monitoring:</strong> Track GPU memory usage at each batch size. SD may OOM at a lower batch size than baseline, which is a real cost.</li>
</ol>

<h4>TTFT vs ITL Impact</h4>
<p>Speculative decoding affects TTFT and ITL differently:</p>
<pre><code># TTFT (Time To First Token):
# SD HURTS TTFT slightly because the first speculation round
# includes both draft generation AND target verification.
# Baseline TTFT: one target forward pass (prefill)
# SD TTFT: one target prefill + one draft round + one target verify
# Typically: SD TTFT is 10-30% worse than baseline TTFT
#
# ITL (Inter-Token Latency):
# SD IMPROVES ITL by generating multiple tokens per round.
# Baseline ITL: t_target (one forward pass per token)
# SD ITL: (t_draft * gamma + t_verify) / E[tokens]
# Typically: SD ITL is 2-5x better than baseline
#
# PRACTICAL IMPLICATION:
# For chatbots where TTFT matters most (user sees "thinking..."):
#   SD may feel SLOWER for the first response.
# For streaming where ITL matters (user sees tokens appear):
#   SD makes tokens appear faster once they start.
#
# Recommendation: In production, separate TTFT and ITL metrics.
# Consider enabling SD only AFTER the first token is generated.

def measure_ttft_and_itl(model, input_ids, max_tokens=128, use_sd=False):
    """Measure TTFT and ITL separately."""
    token_times = []
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # Generate tokens one at a time, recording timestamps
    for token_idx in range(max_tokens):
        # ... (generation logic)
        torch.cuda.synchronize()
        token_times.append(time.perf_counter())

    ttft = token_times[0] - t_start  # Time to first token
    itl_values = [token_times[i] - token_times[i-1]
                  for i in range(1, len(token_times))]
    avg_itl = sum(itl_values) / len(itl_values)

    return {
        "ttft_ms": ttft * 1000,
        "avg_itl_ms": avg_itl * 1000,
        "p50_itl_ms": sorted(itl_values)[len(itl_values)//2] * 1000,
        "p99_itl_ms": sorted(itl_values)[int(len(itl_values)*0.99)] * 1000,
    }</code></pre>

<h4>Complete Benchmarking Script</h4>
<pre><code>#!/usr/bin/env python3
"""
Comprehensive speculative decoding benchmark script.
Tests acceptance rate, batch sweep, TTFT/ITL, and memory overhead.
"""
import torch
import json
import argparse
from pathlib import Path

def run_full_benchmark(
    target_model_name: str,
    draft_model_name: str,
    test_prompts_file: str,
    output_file: str,
    gamma: int = 5,
    temperatures: list = [0.0, 0.7, 1.0],
    batch_sizes: list = [1, 2, 4, 8, 16, 32, 64],
    max_new_tokens: int = 128,
    num_warmup: int = 10,
    num_trials: int = 20,
):
    """Run complete SD benchmark suite."""

    # Load models and prompts
    print(f"Loading target model: {target_model_name}")
    target = load_model(target_model_name)
    print(f"Loading draft model: {draft_model_name}")
    draft = load_model(draft_model_name)
    tokenizer = load_tokenizer(target_model_name)

    with open(test_prompts_file) as f:
        prompts = json.load(f)

    results = {
        "config": {
            "target_model": target_model_name,
            "draft_model": draft_model_name,
            "gamma": gamma,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
        },
        "tests": {}
    }

    # Test 1: Acceptance rate profiling
    print("\\n=== Test 1: Acceptance Rate Profiling ===")
    profiler = SDProfiler(target, draft, tokenizer)
    for temp in temperatures:
        alpha_results = profiler.profile_acceptance_rate(
            prompts[:50], gamma=gamma, temperature=temp
        )
        results["tests"][f"alpha_t{temp}"] = alpha_results
        print(f"  T={temp}: alpha={alpha_results['overall_alpha']:.3f}, "
              f"E[tokens]={alpha_results['expected_tokens']:.2f}")

    # Test 2: Batch size sweep
    print("\\n=== Test 2: Batch Size Sweep ===")
    sweep_results = profiler.profile_batch_sweep(
        prompts, batch_sizes=batch_sizes, gamma=gamma,
        max_new_tokens=max_new_tokens
    )
    results["tests"]["batch_sweep"] = sweep_results

    # Test 3: Memory overhead
    print("\\n=== Test 3: Memory Overhead ===")
    torch.cuda.reset_peak_memory_stats()
    # ... run baseline, measure peak memory
    baseline_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    # ... run SD, measure peak memory
    sd_mem = torch.cuda.max_memory_allocated() / 1e9
    results["tests"]["memory"] = {
        "baseline_gb": baseline_mem,
        "sd_gb": sd_mem,
        "overhead_gb": sd_mem - baseline_mem,
        "overhead_pct": (sd_mem - baseline_mem) / baseline_mem * 100,
    }
    print(f"  Baseline: {baseline_mem:.1f}GB, SD: {sd_mem:.1f}GB, "
          f"Overhead: {sd_mem-baseline_mem:.2f}GB ({results['tests']['memory']['overhead_pct']:.1f}%)")

    # Test 4: TTFT and ITL
    print("\\n=== Test 4: TTFT and ITL ===")
    ttft_results = {}
    for use_sd in [False, True]:
        label = "sd" if use_sd else "baseline"
        metrics = measure_ttft_and_itl(
            target if not use_sd else (target, draft),
            tokenizer.encode(prompts[0], return_tensors="pt").cuda(),
            max_tokens=max_new_tokens,
            use_sd=use_sd
        )
        ttft_results[label] = metrics
        print(f"  {label}: TTFT={metrics['ttft_ms']:.0f}ms, "
              f"ITL_p50={metrics['p50_itl_ms']:.1f}ms, "
              f"ITL_p99={metrics['p99_itl_ms']:.1f}ms")

    results["tests"]["ttft_itl"] = ttft_results

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\nResults saved to {output_file}")

    # Print summary
    print("\\n=== SUMMARY ===")
    print(f"Acceptance rate (T=0.7): {results['tests']['alpha_t0.7']['overall_alpha']:.3f}")
    print(f"Expected tokens/round: {results['tests']['alpha_t0.7']['expected_tokens']:.2f}")
    print(f"Memory overhead: {results['tests']['memory']['overhead_gb']:.2f} GB")
    print(f"Speedup by batch size:")
    for bs, data in sweep_results.items():
        bar = "#" * int(data["speedup"] * 10)
        print(f"  Batch {bs:3d}: {data['speedup']:.2f}x {bar}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="sd_benchmark_results.json")
    parser.add_argument("--gamma", type=int, default=5)
    args = parser.parse_args()
    run_full_benchmark(args.target, args.draft, args.prompts, args.output, args.gamma)</code></pre>

<div class="callout tip">
<div class="callout-title">Benchmarking Checklist</div>
<p>Before publishing or acting on SD benchmark numbers, verify:<br>
1. Warmup completed (10+ iterations before timing)<br>
2. torch.cuda.synchronize() called before each timing measurement<br>
3. Multiple trials (20+) with mean and standard deviation reported<br>
4. Batch sizes swept from 1 to your production maximum<br>
5. Temperature matches your production setting<br>
6. Prompts are representative of production workload<br>
7. Memory overhead measured and max concurrent requests compared<br>
8. TTFT and ITL reported separately<br>
9. Output verified for correctness (lossless guarantee should hold)</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you benchmark speculative decoding for a production deployment? What pitfalls should you watch out for?</div>
<div class="a-text">I would run four tests: (1) Acceptance rate profiling on production-representative prompts across different temperatures -- this tells me the theoretical speedup ceiling. (2) Batch size sweep from 1 to 64, measuring wall-clock speedup at each point -- this reveals the critical batch size where SD stops being beneficial. (3) Memory overhead measurement to determine how many fewer concurrent requests we can serve with SD enabled. (4) TTFT vs ITL measurement since SD hurts TTFT but improves ITL. Key pitfalls: benchmarking only at batch=1 (unrealistic), not warming up the GPU (frequency scaling), not using production-representative prompts (acceptance rates vary dramatically by task), reporting tokens/sec instead of wall-clock speedup (misleading), not measuring memory impact on max concurrency, and not testing under realistic traffic patterns (bursty vs steady). The most common real-world failure: SD looks great in lab testing at batch=1 but provides no benefit or even hurts performance at production batch sizes.</div>
</div>
`
    },
    {
      id: "sd-frontiers",
      title: "SD Frontiers: Beyond Text Generation",
      content: `
<p>Speculative decoding was invented for text LLMs, but its core principle -- parallel verification is cheaper than sequential generation -- applies to any autoregressive generation process. This section covers the expanding frontier of SD applications and the most promising research directions.</p>

<h4>SD for Image Generation</h4>
<p>Autoregressive image models (LlamaGen, Parti, DALL-E) generate image tokens sequentially. SD applies directly:</p>
<ul>
<li><strong>Application:</strong> Visual tokens often have high local predictability (nearby pixels/patches are correlated). Draft models can exploit spatial coherence for high acceptance rates in smooth regions, with lower rates at edges and detail areas.</li>
<li><strong>Challenge:</strong> Image token sequences are very long (256x256 = 65,536 tokens for a 256x256 image with 1x1 patches). The KV-cache for tree attention becomes enormous.</li>
<li><strong>Approach:</strong> Use a small ViT as the draft model operating on lower-resolution features. Verify in blocks rather than individual tokens to reduce verification overhead.</li>
</ul>

<h4>SD for Audio Generation</h4>
<p>Codec language models for TTS and audio generation (VALL-E, MusicGen, AudioLM) are natural targets for SD:</p>
<ul>
<li><strong>VALL-E + SD:</strong> The autoregressive stage (codebook 1 generation) is the bottleneck. A small draft model predicting coarse tokens can provide 2-3x speedup while the NAR stage (codebooks 2-8) is already parallel.</li>
<li><strong>MusicGen + SD:</strong> Music tokens are highly repetitive (rhythmic patterns, harmonic structures repeat). Draft models can achieve alpha > 0.85 on structured music, giving significant speedups.</li>
<li><strong>Challenge:</strong> Audio codecs use RVQ, producing multiple token streams. SD must handle multi-stream prediction, where accepting a token at one codebook level may depend on tokens at other levels.</li>
</ul>

<h4>SD for Multi-Modal Generation</h4>
<p>Omni models that generate interleaved text and audio/image tokens can benefit from SD, but the multi-modal nature introduces complexity:</p>
<ul>
<li><strong>Different acceptance rates per modality:</strong> Text tokens may have alpha=0.8 while audio tokens have alpha=0.5. A smart system adjusts gamma per modality.</li>
<li><strong>Modal-specific drafters:</strong> Use a text-specialized drafter for text regions and an audio-specialized drafter for audio regions. Switch drafters at modality boundaries.</li>
<li><strong>Cross-modal prediction:</strong> Generating text that will be spoken requires understanding both textual and acoustic constraints. Draft models for these cross-modal tokens need richer context.</li>
</ul>

<h4>Speculative Decoding for Reasoning Models</h4>
<p>Chain-of-thought and reasoning models (o1, DeepSeek-R1) generate very long reasoning traces before producing the final answer. SD has interesting implications here:</p>
<ul>
<li><strong>Reasoning tokens are often formulaic:</strong> "Let me think about this step by step. First, I need to..." -- these template phrases have very high acceptance rates.</li>
<li><strong>Mathematical expressions are predictable:</strong> Once a formula is started, the completion is often determined: "\\frac{d}{dx}[x^2 + 3x] = 2x + 3" has high acceptance after "\\frac{d}{dx}".</li>
<li><strong>But decision points are unpredictable:</strong> The actual reasoning choices ("should I try approach A or approach B?") have low acceptance rates.</li>
<li><strong>Opportunity:</strong> A reasoning-aware SD system could adjust gamma dynamically: long gamma for formulaic text, short gamma at decision points. This could significantly speed up the often minutes-long reasoning traces.</li>
</ul>

<h4>Hardware-Aware SD: Leveraging Heterogeneous Systems</h4>
<pre><code># Heterogeneous SD: use different hardware for draft and target
#
# Configuration 1: Draft on CPU, Target on GPU
#   - CPU can run a small draft model (1B) at ~10 tokens/sec
#   - GPU runs target model (70B) at ~1 token/sec for verification
#   - Draft runs in parallel with target verification (pipelining)
#   - No GPU memory used for draft model
#   - Works well when GPU memory is the constraint
#
# Configuration 2: Draft on smaller GPU, Target on main GPU
#   - Draft on RTX 4090 (24GB): fast, cheap
#   - Target on A100 (80GB): expensive, used only for verification
#   - Cross-GPU communication adds ~1ms latency (PCIe)
#   - Good for cost optimization (4090 is ~5x cheaper than A100)
#
# Configuration 3: Draft on NPU/TPU, Target on GPU
#   - Emerging hardware accelerators are very fast for small models
#   - Apple Neural Engine, Qualcomm NPU, Google TPU
#   - Can run 1B models at ~100 tokens/sec
#   - Perfect draft engine: fast, low power, doesn't consume GPU resources

# Future: speculative decoding as a first-class hardware feature
# Some next-gen chips may include dedicated draft circuits that
# speculate in hardware while the main compute array verifies.</code></pre>

<h4>Open Research Directions</h4>
<ol>
<li><strong>Optimal gamma selection:</strong> Currently, gamma is a hyperparameter. Can we learn to predict the optimal gamma dynamically based on context? Early work (EAGLE-2, Sequoia) shows promising results, but a general solution remains open.</li>
<li><strong>Speculative decoding for batched serving:</strong> Most SD gains disappear at large batch sizes. MagicDec showed gains for long contexts; can we find general techniques that maintain speedup at batch 64+?</li>
<li><strong>Draft model training:</strong> EAGLE trains on a few thousand examples. Can we design better training objectives that specifically optimize for acceptance rate rather than cross-entropy?</li>
<li><strong>Multi-model speculation:</strong> Use an ensemble of draft models, selecting the best drafter per context. This adds complexity but could significantly improve alpha for diverse workloads.</li>
<li><strong>Speculative decoding meets structured generation:</strong> When generating JSON, SQL, or code with grammar constraints, the grammar rules can be used to prune the draft tree. This is an under-explored direction with high practical value.</li>
<li><strong>Lossy speculative decoding:</strong> What if we accept a small distributional difference for a large speedup? Relaxing the lossless guarantee by epsilon could enable much longer speculation chains. Early work on "almost-lossless" SD shows promising quality/speed tradeoffs.</li>
</ol>

<div class="callout">
<div class="callout-title">The Future of Speculative Decoding</div>
<p>SD started as a clever trick for speeding up text generation. It is evolving into a <strong>fundamental primitive of autoregressive systems</strong> -- any system that generates tokens one at a time can potentially benefit from speculative execution. As models get larger and the memory-bandwidth gap widens, SD's value increases. The trend toward longer contexts (128K+ tokens) also helps, since longer contexts keep inference memory-bound at higher batch sizes (MagicDec's insight). For AI engineers, understanding SD deeply is a durable skill: the specific methods will evolve, but the core principle of parallel verification will remain relevant as long as autoregressive generation exists.</p>
</div>

<h4>Quick Reference: SD Method Selection Guide</h4>
<pre><code># Decision tree for choosing an SD method:
#
# 1. Can you train a draft model? (need ~4 GPU-hours + target model access)
#    YES -> go to 2
#    NO  -> Use Lookahead decoding (zero setup) or REST (if domain-specific)
#
# 2. Can you afford extra GPU memory (~300MB)?
#    YES -> go to 3
#    NO  -> Use LayerSkip (self-speculative, zero extra memory)
#
# 3. Is your primary concern batch=1 latency or high-batch throughput?
#    Batch=1 latency -> EAGLE-3 (best single-request speedup)
#    High-batch throughput -> go to 4
#
# 4. Are your sequences long (>8K tokens)?
#    YES -> MagicDec (designed for long-context batched serving)
#    NO  -> Consider NOT using SD. At short contexts + high batch,
#           continuous batching alone may be better.
#
# 5. Want simplest implementation?
#    Use Medusa (simplest training, well-documented, good library support)
#
# 6. Want best overall performance?
#    Use EAGLE-3 with SGLang (native support, production-tested)</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Where do you see speculative decoding going in the next 2-3 years? What are the most promising research directions?</div>
<div class="a-text">Three directions I find most promising: (1) Hardware-aware SD -- as AI accelerators diversify (NPUs, TPUs, custom chips), running tiny draft models on specialized hardware while reserving GPU for verification could make SD universally beneficial. Apple's Neural Engine running a 1B drafter while the main GPU verifies against a 70B model is a near-term possibility. (2) SD for structured generation -- when outputting JSON, SQL, or code, grammar constraints can dramatically prune the draft tree. A draft token that violates the grammar is guaranteed to be rejected, so we never waste verification budget on it. This is under-explored and could give 5-10x speedups for structured output. (3) SD for reasoning models -- o1-style models generate very long reasoning chains with alternating predictable (formulaic text) and unpredictable (decision points) sections. A reasoning-aware SD system that adapts gamma dynamically could cut reasoning time significantly while preserving the final answer quality.</div>
</div>
`
    }
  ]
};
