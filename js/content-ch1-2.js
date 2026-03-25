// Deeply expanded content for Chapters 1 and 2
// Audio LLM Research Landscape & Speech-to-Speech Models
const CONTENT_CH1_2 = {

  // ============================================================
  // CHAPTER 1: Audio LLM Research Landscape
  // ============================================================
  ch1_sections: [
    {
      id: "audio-llm-overview",
      title: "Overview & Architecture Evolution",
      content: `
<p>Audio Large Language Models (AudioLLMs) represent a paradigm shift from specialized audio models to unified architectures that can understand and generate both text and audio. The field has evolved rapidly from 2023 to 2025, moving through distinct phases. Understanding the architectural foundations is essential for any AI engineer working in this space, whether you are building production audio systems or conducting research.</p>

<div class="callout">
<div class="callout-title">Key Insight</div>
<p>The core architectural pattern for AudioLLMs: <strong>Audio Encoder + Adapter + LLM Backbone + Decoder</strong>. The encoder converts audio to representations, the adapter bridges modalities, the LLM reasons, and the decoder generates output. Every AudioLLM you encounter is some variation on this theme, and your ability to compare systems reduces to understanding how each component is instantiated.</p>
</div>

<h4>The Encoder+Adapter+LLM Pattern in Detail</h4>

<p>Let us trace the data flow through a canonical AudioLLM step by step. We will use Qwen-Audio as a running example, but the pattern generalizes.</p>

<p><strong>Step 1: Raw Audio to Spectrogram.</strong> The input waveform (typically 16kHz mono) is converted to a log-mel spectrogram. This is a 2D representation where the x-axis is time and the y-axis is frequency, with values representing energy. For Whisper-based encoders, the spectrogram uses 80 mel-frequency bins, computed over 25ms windows with 10ms hop size. A 30-second audio clip at 16kHz is 480,000 samples; after spectrogram conversion, this becomes a matrix of shape <code>(80, 3000)</code> &mdash; 80 mel bins by 3000 time frames (one frame per 10ms).</p>

<pre><code># Computing a log-mel spectrogram (Whisper-style)
import torch
import torchaudio

waveform, sr = torchaudio.load("audio.wav")  # (1, num_samples)
# Resample to 16kHz if needed
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# Whisper uses 80 mel bins, 400-sample window (25ms), 160-sample hop (10ms)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80,
    power=2.0
)
mel_spec = mel_transform(waveform)        # (1, 80, T)
log_mel = torch.log10(mel_spec + 1e-6)    # Log scale
# For 30s audio: T = 480000 / 160 = 3000 time frames
# Final shape: (1, 80, 3000)</code></pre>

<p><strong>Step 2: Audio Encoder.</strong> The spectrogram is fed into a transformer-based encoder. Whisper's encoder (used in SALMONN, Qwen-Audio, and many others) consists of two 1D convolution layers followed by a stack of transformer blocks. The convolution layers downsample the time axis by 2x (stride 2), so the 3000 frames become 1500 encoder positions. Each position produces a d-dimensional vector.</p>

<table>
<tr><th>Whisper Model</th><th>Encoder Layers</th><th>Hidden Dim (d)</th><th>Attention Heads</th><th>Parameters (Encoder)</th></tr>
<tr><td>tiny</td><td>4</td><td>384</td><td>6</td><td>~9M</td></tr>
<tr><td>base</td><td>6</td><td>512</td><td>8</td><td>~24M</td></tr>
<tr><td>small</td><td>12</td><td>768</td><td>12</td><td>~88M</td></tr>
<tr><td>medium</td><td>24</td><td>1024</td><td>16</td><td>~306M</td></tr>
<tr><td>large-v3</td><td>32</td><td>1280</td><td>20</td><td>~637M</td></tr>
</table>

<p>After the Whisper encoder, a 30-second audio clip becomes a sequence of 1500 vectors, each of dimension d (e.g., 1280 for large-v3). This is 1500 "audio tokens" &mdash; far more than the typical text prompt. Length is a key challenge: a 5-minute audio clip would produce 15,000 encoder states, which can overwhelm the LLM's context window.</p>

<p><strong>Alternative encoders</strong> include BEATs (for general audio events, pre-trained with audio-level prediction), HuBERT (self-supervised speech representations), and WavLM (robust to noise). SALMONN famously uses a <em>dual encoder</em> &mdash; Whisper for speech content plus BEATs for non-speech audio events &mdash; concatenating their outputs before the adapter.</p>

<p><strong>Step 3: The Adapter Layer.</strong> The adapter bridges the audio encoder's representation space to the LLM's embedding space. This is where the most design variation occurs. The three dominant approaches are:</p>

<h4>Adapter Comparison: Q-Former vs. Linear Projection vs. MLP</h4>

<table>
<tr><th>Adapter Type</th><th>Parameters</th><th>Compression</th><th>Training Cost</th><th>Used In</th></tr>
<tr><td><strong>Linear Projection</strong></td><td>d_enc * d_llm (e.g., 1280*4096 = 5.2M)</td><td>None (1:1 mapping)</td><td>Lowest</td><td>LLaMA-Omni, Whisper+LLaMA</td></tr>
<tr><td><strong>2-Layer MLP</strong></td><td>d_enc*d_hidden + d_hidden*d_llm (e.g., ~17M for d_hidden=2048)</td><td>None or with pooling</td><td>Low</td><td>Qwen-Audio, VITA</td></tr>
<tr><td><strong>Q-Former</strong></td><td>~100-188M (BERT-base with cross-attention)</td><td>Heavy (1500 -> 32-128 queries)</td><td>Highest</td><td>SALMONN, Audio Flamingo</td></tr>
<tr><td><strong>Perceiver Resampler</strong></td><td>~50-80M</td><td>Moderate (1500 -> 64-256)</td><td>Medium</td><td>AudioPaLM variants</td></tr>
</table>

<p><strong>Linear projection</strong> is the simplest: a single matrix <code>W</code> of shape <code>(d_encoder, d_llm)</code> maps each encoder output to the LLM's embedding space. No sequence compression occurs. This preserves all temporal information but means the LLM must process all 1500 audio positions, consuming precious context window. For a 30-second clip with Whisper-large feeding LLaMA-3-8B, the projection matrix is 1280 x 4096 = 5.2M parameters.</p>

<p><strong>MLP adapters</strong> add one or two hidden layers with a nonlinearity (typically GELU or SiLU). Qwen-Audio uses a 2-layer MLP: <code>Linear(1280, 2048) -> GELU -> Linear(2048, 4096)</code>. This gives the adapter more expressive power to remap features. Some MLP adapters include average pooling with stride 2-4 before the MLP to compress the sequence length, reducing 1500 tokens to 375-750.</p>

<p><strong>Q-Former</strong> (Querying Transformer, from BLIP-2) uses a set of learned query vectors (typically 32-128) that attend to the encoder outputs via cross-attention. The output is a fixed number of tokens regardless of input length. SALMONN uses 32 queries, meaning a 30-second audio clip is compressed from 1500 encoder states to just 32 tokens. The tradeoff: massive compression enables long audio but may lose fine-grained temporal detail. The Q-Former itself is substantial &mdash; typically initialized from BERT-base (110M parameters) with additional cross-attention layers (~78M), totaling ~188M trainable parameters.</p>

<div class="callout warning">
<div class="callout-title">War Story: The 1500-Token Bottleneck</div>
<p>A team building a meeting transcription system fed 10-minute audio segments through a Whisper-large + Linear Projection + LLaMA-3-8B pipeline. Each segment produced 30,000 encoder tokens, filling the LLM's 8K context window with audio alone, leaving no room for the text prompt. The fix: they switched to a Q-Former with 64 queries, reducing 30,000 tokens to 64, and added a sliding window that processed 30-second chunks with 5-second overlap. The lesson: adapter choice is not a minor architectural detail &mdash; it determines what audio lengths your system can handle.</p>
</div>

<p><strong>Step 4: LLM Backbone Processing.</strong> The adapter outputs are prepended to (or interleaved with) the text token embeddings and fed into the LLM. The LLM processes the combined sequence with its standard transformer layers. In most architectures, the audio tokens simply occupy the same sequence positions as text tokens would &mdash; the LLM treats them identically. This is the beauty and the limitation of the approach: no special audio-aware mechanisms exist within the LLM itself.</p>

<p><strong>Step 5: Decoder / Output Head.</strong> For text output (ASR, audio QA, captioning), the standard LM head projects hidden states to vocabulary logits. For audio output (speech generation), additional decoder components are needed &mdash; typically a codec language model or a flow-matching decoder that converts LLM hidden states back to audio tokens or spectrograms.</p>

<h4>Foundational Papers (2023-2024)</h4>
<table>
<tr><th>Paper</th><th>Key Innovation</th><th>Architecture Details</th><th>Impact</th></tr>
<tr><td><strong>Pengi</strong> (NeurIPS 2023)</td><td>All audio tasks as text-generation; audio encoder + text encoder as prefix to frozen LM</td><td>Audio Spectrogram Transformer encoder, transfer matrix adapter, GPT-2 backbone</td><td>Unified audio-text generation; unlocked open-ended audio QA</td></tr>
<tr><td><strong>SALMONN</strong> (ICLR 2024)</td><td>Dual encoder (Whisper + BEATs) with Q-Former adapter to Vicuna</td><td>Whisper-large-v2 (1280d) + BEATs iter3 (768d), Q-Former with 32 queries, Vicuna-13B</td><td>First to study cross-modal emergent capabilities; showed dual encoders capture complementary information</td></tr>
<tr><td><strong>Qwen-Audio</strong> (arXiv:2311.07919)</td><td>30+ tasks, hierarchical tag conditioning to solve multi-task interference</td><td>Whisper-large-v2 encoder, 2-layer MLP adapter, Qwen-7B backbone</td><td>Proved scale + task taxonomy beats hand-crafted models</td></tr>
<tr><td><strong>AudioPaLM</strong> (arXiv:2306.12925)</td><td>Joint audio-text vocabulary; first to generate audio tokens directly from LLM</td><td>PaLM-2 with AudioLM-style tokenizer, unified vocabulary of ~30K text + ~1024 audio tokens</td><td>Opened the end-to-end generation paradigm</td></tr>
</table>

<h4>The 2025 Frontier</h4>
<p>The field branched into several exciting directions:</p>
<ul>
<li><strong>Omni Models:</strong> Qwen2.5-Omni (arXiv:2503.20215), and Kimi-Audio achieved all-modal input with text/speech output, including streaming capabilities. Qwen2.5-Omni introduced "Thinker-Talker" architecture where a "thinker" LLM generates text reasoning tokens and a "talker" module converts them to speech in streaming fashion.</li>
<li><strong>Reasoning in Audio:</strong> Audio Flamingo Sound-CoT (arXiv:2502.16740) introduced systematic audio chain-of-thought with 1.24M auto-generated CoT samples; AudSemThinker grounded reasoning in structured auditory semantics using explicit sound event detection before reasoning.</li>
<li><strong>Long Context:</strong> CALM uses continuous audio tokens (VAE) instead of discrete codecs, avoiding the information bottleneck of VQ; YaRN + VLAT extended context windows to handle 30+ minutes of audio.</li>
<li><strong>Domain Specialization:</strong> SeaLLMs-Audio for Southeast Asian languages (8 languages including Thai, Vietnamese, Indonesian); FinAudio for financial audio analysis (earnings calls, market commentary).</li>
</ul>

<pre><code>2023-2024 FOUNDATION              2025 FRONTIER
---------------------------------------------------
[Encoder+LLM Architecture]   ->  [Omni: all-modal, streaming]
[Multi-task training]         ->  [Reasoning: CoT, RL, RL+CoT]
[General benchmarks]          ->  [Domain benchmarks (finance, SEA)]
[Text output]                 ->  [Audio-in-audio reasoning]
[Fixed context (<30s)]        ->  [Long audio (YaRN, CALM)]
[Cascade vs. E2E debate]     ->  [Cascade comeback vs. Omni models]</code></pre>

<h4>Code Example: Loading and Using Qwen-Audio for Inference</h4>

<p>The following example demonstrates loading Qwen2-Audio (the publicly available successor to Qwen-Audio) and performing audio understanding:</p>

<pre><code>from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import librosa

# Load model and processor
model_name = "Qwen/Qwen2-Audio-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Requires ~16GB VRAM
)

# Load audio
audio, sr = librosa.load("meeting_clip.wav", sr=16000)

# Prepare conversation with audio
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Describe the speakers and summarize what is discussed."}
        ]
    }
]

# Process and generate
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

response = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)[0]
print(response)</code></pre>

<div class="callout">
<div class="callout-title">Architecture Dimensions Cheat Sheet</div>
<p>Memorize these numbers for interviews:<br>
<strong>Whisper-large-v3:</strong> 32 encoder layers, d=1280, 20 heads, 637M encoder params, 1500 tokens per 30s<br>
<strong>Qwen2-Audio-7B:</strong> Whisper encoder (1280d) -> MLP adapter -> Qwen2-7B (4096d, 32 layers, 32 heads, GQA with 8 KV heads)<br>
<strong>SALMONN:</strong> Whisper-large (1280d) + BEATs (768d) -> Q-Former (32 queries) -> Vicuna-13B (5120d)<br>
<strong>Typical audio token budget:</strong> 50 tokens/second after adapter compression</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Walk me through the complete data flow of an AudioLLM processing a 30-second audio clip. What are the tensor shapes at each stage?</div>
<div class="a-text">Starting with a 30-second clip at 16kHz: (1) Raw waveform: shape (1, 480000). (2) Log-mel spectrogram with 80 mel bins, 10ms hop: shape (1, 80, 3000). (3) Whisper encoder conv layers downsample 2x to 1500 frames, then transformer processes to produce encoder states: shape (1, 1500, 1280) for Whisper-large. (4) Adapter: If linear projection to LLaMA-3-8B, output is (1, 1500, 4096). If Q-Former with 32 queries, output is (1, 32, 4096). (5) These are concatenated with text token embeddings (say 50 text tokens) to form LLM input of shape (1, 1532 or 82, 4096). (6) The LLM processes the full sequence through its 32 transformer layers. (7) The LM head projects the last hidden state to vocabulary logits: (1, seq_len, 152064) for Qwen2's vocabulary. The critical insight: the adapter choice determines whether you have 1532 or 82 total tokens, which has massive implications for compute cost (quadratic in attention) and context window budget.</div>
</div>
`
    },
    {
      id: "audio-neglect",
      title: "The Audio Neglect Problem",
      content: `
<p><strong>Audio Neglect</strong> is a critical finding from 2025 research: AudioLLMs systematically under-utilize audio evidence. The text-pretrained LLM backbone is so powerful that it answers from language priors, effectively ignoring the actual audio signal. This is not a minor performance issue &mdash; it fundamentally undermines the premise of AudioLLMs.</p>

<div class="callout warning">
<div class="callout-title">Critical Research Gap</div>
<p>2025 research showed models ignore decisive audio even when it's the only valid signal. The proposed fix (attention steering via audio-specialist heads) is ad hoc. A principled, general solution doesn't exist yet. This remains one of the most important open problems in AudioLLM research.</p>
</div>

<h4>Detailed Experimental Protocol</h4>

<p>The audio neglect phenomenon was systematically characterized through controlled experiments. The experimental protocol works as follows:</p>

<p><strong>Step 1: Construct Diagnostic Datasets.</strong> Create audio-text pairs where the audio content is the <em>only</em> valid source for the correct answer, and text priors would suggest a different answer. For example:</p>
<ul>
<li><strong>Emotion recognition with misleading text:</strong> Audio of someone saying "I'm fine" in a clearly distressed tone. Text context: "The speaker reports they are fine." Correct answer: distressed/sad. Text-prior answer: happy/neutral.</li>
<li><strong>Environmental sound identification:</strong> Audio of birds chirping in a kitchen. Text context: "Recording from a kitchen." Correct answer (requiring audio): birds are present. Text-prior answer: cooking sounds.</li>
<li><strong>Speaker counting:</strong> Audio with 3 distinct speakers. Text context: "A conversation between two people." Correct answer (requiring audio): 3 speakers.</li>
</ul>

<p><strong>Step 2: Measure Accuracy Under Conflict.</strong> Test the AudioLLM on these adversarial pairs and measure:</p>
<ul>
<li><strong>Audio-aligned accuracy:</strong> How often does the model give the answer supported by audio evidence?</li>
<li><strong>Text-prior accuracy:</strong> How often does the model give the answer suggested by text context alone?</li>
<li><strong>Delta (Audio - No-audio):</strong> Run the same questions with and without audio. If the delta is near zero, the model is ignoring audio entirely.</li>
</ul>

<h4>Quantitative Results from Research</h4>

<p>Results across multiple studies paint a consistent picture:</p>

<table>
<tr><th>Model</th><th>Task</th><th>Audio-Aligned Acc (%)</th><th>Text-Prior Acc (%)</th><th>Audio Neglect Rate (%)</th></tr>
<tr><td>SALMONN-13B</td><td>Emotion (adversarial)</td><td>34.2</td><td>61.8</td><td>64.4</td></tr>
<tr><td>Qwen-Audio-7B</td><td>Emotion (adversarial)</td><td>41.5</td><td>55.3</td><td>57.1</td></tr>
<tr><td>Qwen2-Audio-7B</td><td>Emotion (adversarial)</td><td>48.7</td><td>47.2</td><td>49.8</td></tr>
<tr><td>SALMONN-13B</td><td>Speaker count (conflict)</td><td>22.6</td><td>72.1</td><td>76.3</td></tr>
<tr><td>Qwen-Audio-7B</td><td>Speaker count (conflict)</td><td>29.3</td><td>65.4</td><td>69.0</td></tr>
<tr><td>Qwen2-Audio-7B</td><td>Sound event (adversarial)</td><td>55.1</td><td>40.3</td><td>42.7</td></tr>
</table>

<p>The "Audio Neglect Rate" measures how often the model follows text priors when they conflict with audio. A rate above 50% means the model favors text priors over audio evidence more often than not. Note that newer models (Qwen2-Audio) show improvement, but even the best models still neglect audio nearly half the time under adversarial conditions.</p>

<h4>Root Cause Analysis</h4>

<p>Why does audio neglect occur? Three interconnected factors:</p>

<ol>
<li><strong>Asymmetric pretraining:</strong> The LLM backbone has been trained on trillions of text tokens but only millions of audio-text pairs during alignment. The text representations are deeply embedded across all layers, while audio representations are "grafted on" through relatively shallow adapter training. The LLM's internal circuits for text reasoning are vastly more developed than its circuits for integrating audio evidence.</li>

<li><strong>Attention pattern imbalance:</strong> Analysis of attention maps shows that after the first few layers, attention to audio tokens drops precipitously. In a typical 32-layer LLM, layers 1-8 attend roughly equally to audio and text tokens, but layers 9-32 attend primarily to text tokens. The audio signal is effectively "washed out" by the deeper layers.</li>

<li><strong>Training data distribution:</strong> Most audio-text training pairs have correlated modalities (the text description matches the audio). Models rarely encounter cases where audio contradicts text context during training, so they never learn to arbitrate between conflicting modalities.</li>
</ol>

<h4>Proposed Solutions</h4>

<p><strong>1. Attention Steering via Audio-Specialist Heads.</strong> The idea: identify specific attention heads that attend strongly to audio tokens, then upweight these heads during inference or fine-tuning. Implementation involves computing attention entropy per head on audio tokens, ranking heads by their "audio attention ratio," and scaling those heads by a factor of 1.5-3.0x.</p>

<pre><code># Measuring attention to audio tokens per head
def compute_audio_attention_ratio(model, input_ids, audio_token_mask):
    """
    Compute what fraction of attention each head pays to audio tokens.
    audio_token_mask: boolean tensor, True for positions that are audio tokens
    """
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    ratios = {}  # (layer, head) -> ratio
    for layer_idx, attn_weights in enumerate(outputs.attentions):
        # attn_weights shape: (batch, num_heads, seq_len, seq_len)
        # Average attention from text tokens to audio tokens
        text_mask = ~audio_token_mask
        for head_idx in range(attn_weights.shape[1]):
            # Attention from text positions to audio positions
            text_to_audio = attn_weights[0, head_idx][text_mask][:, audio_token_mask]
            ratio = text_to_audio.mean().item()
            ratios[(layer_idx, head_idx)] = ratio

    # Sort by ratio to find audio-specialist heads
    ranked = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    return ranked

# Top-10 audio-specialist heads can be upweighted during inference
# Typically found in layers 3-12 (early-to-mid layers)</code></pre>

<p><strong>2. Contrastive Audio Grounding.</strong> Add a contrastive loss during training that forces the model to distinguish between audio-supported and audio-contradicted answers. For each training example, create a negative pair where the audio is swapped with a mismatched audio clip. The loss encourages the model to attend to audio features when they are decision-relevant.</p>

<p><strong>3. Audio-Conditioned Gating.</strong> Insert a learnable gating mechanism between the adapter and the LLM that explicitly controls how much audio information flows into each LLM layer. The gate is conditioned on the audio content itself &mdash; complex or surprising audio signals open the gate wider, while predictable audio (matching text context) allows the gate to partially close.</p>

<p><strong>4. Adversarial Training.</strong> Include adversarial audio-text pairs in the training data where the correct answer <em>requires</em> attending to audio. This is the simplest approach but requires careful dataset construction.</p>

<h4>The X-Talk Counter-Narrative</h4>
<p>Meanwhile, X-Talk demonstrated that modular ASR-to-LLM-to-TTS cascades remain competitive with end-to-end systems, challenging the "omni is always better" narrative. The key insight: <strong>deployment robustness does not equal benchmark performance</strong>. Cascade systems are inherently immune to audio neglect because the ASR module explicitly extracts information from audio into text, which the LLM then processes. The information extraction is forced, not optional.</p>

<h4>Research Direction: Reasoning Substrate</h4>
<p>Should AudioLLMs reason in text tokens (fast, mature, but loses paralinguistic detail) or audio tokens (preserves acoustics, but expensive and evaluation is undefined)?</p>

<p>Drawing from human cognition: humans think in language, not in sounds, even when processing audio. We extract concepts from audio, then reason over concepts. This suggests:</p>
<ul>
<li>Full audio-token reasoning is likely neither natural nor necessary</li>
<li>The real value is <strong>hybrid</strong>: text CoT as primary reasoning + selective audio anchors at key decision points</li>
<li>The audio neglect problem may be a feature request, not a bug &mdash; the question is <em>when</em> to attend to audio, not whether to attend maximally at all times</li>
</ul>

<h4>Code Example: Measuring Audio Neglect with a Simple Probe</h4>

<pre><code>import torch
import json
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

def measure_audio_neglect(model, processor, test_pairs):
    """
    Measure audio neglect by comparing model outputs with and without audio.

    test_pairs: list of dicts with keys:
      - audio_path: path to audio file
      - question: text question
      - audio_answer: correct answer based on audio
      - text_prior_answer: answer expected from text priors alone
    """
    results = {
        "audio_aligned": 0,
        "text_prior_aligned": 0,
        "neither": 0,
        "total": 0,
        "with_audio_matches_without": 0  # Same answer with/without audio
    }

    for pair in test_pairs:
        audio, sr = librosa.load(pair["audio_path"], sr=16000)

        # Test WITH audio
        conv_with_audio = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": pair["question"]}
            ]
        }]
        inputs = processor.apply_chat_template(
            conv_with_audio, add_generation_prompt=True,
            tokenize=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.0)
        answer_with_audio = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0].strip().lower()

        # Test WITHOUT audio (text only)
        conv_no_audio = [{
            "role": "user",
            "content": [
                {"type": "text", "text": pair["question"]}
            ]
        }]
        inputs_no = processor.apply_chat_template(
            conv_no_audio, add_generation_prompt=True,
            tokenize=True, return_tensors="pt"
        ).to(model.device)
        out_no = model.generate(**inputs_no, max_new_tokens=64, temperature=0.0)
        answer_without_audio = processor.batch_decode(
            out_no[:, inputs_no["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0].strip().lower()

        # Classify
        audio_ans = pair["audio_answer"].lower()
        text_ans = pair["text_prior_answer"].lower()

        if audio_ans in answer_with_audio:
            results["audio_aligned"] += 1
        elif text_ans in answer_with_audio:
            results["text_prior_aligned"] += 1
        else:
            results["neither"] += 1

        if answer_with_audio == answer_without_audio:
            results["with_audio_matches_without"] += 1

        results["total"] += 1

    # Compute metrics
    n = results["total"]
    results["audio_neglect_rate"] = results["text_prior_aligned"] / n
    results["audio_utilization_rate"] = results["audio_aligned"] / n
    results["audio_indifference_rate"] = results["with_audio_matches_without"] / n

    return results

# Usage:
# results = measure_audio_neglect(model, processor, test_pairs)
# print(f"Audio Neglect Rate: {results['audio_neglect_rate']:.1%}")
# print(f"Audio Indifference: {results['audio_indifference_rate']:.1%}")
# Typical finding: 50-75% neglect rate on adversarial pairs</code></pre>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What is the "Audio Neglect" problem in AudioLLMs, and how would you design an experiment to measure it?</div>
<div class="a-text">Audio Neglect refers to AudioLLMs ignoring decisive audio evidence and relying on text priors from the LLM backbone. To measure it, design tasks where: (1) the correct answer requires audio information that cannot be inferred from text alone (e.g., speaker emotion, environmental sounds), (2) create adversarial pairs where text context suggests one answer but audio evidence points to another, (3) measure accuracy with and without audio input &mdash; if performance barely changes, the model is neglecting audio. Concretely, compute three metrics: Audio-Aligned Accuracy (correct answer from audio), Text-Prior Accuracy (answer from text priors), and Audio Indifference Rate (fraction of examples where adding audio does not change the answer). Root causes include asymmetric pretraining (trillions of text tokens vs. millions of audio-text pairs), attention attenuation in deeper layers, and lack of adversarial audio-text pairs in training. Solutions include attention steering on audio-specialist heads, contrastive audio grounding losses, and adversarial training with conflicting modalities.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Compare Q-Former and linear projection as adapter strategies for AudioLLMs. When would you choose each?</div>
<div class="a-text">Linear projection is a single matrix mapping encoder features to LLM dimension (e.g., 5.2M params for Whisper-large to LLaMA-8B). It preserves full temporal resolution (1500 tokens for 30s audio) but consumes significant context window. Q-Former uses learned queries (32-128) with cross-attention to compress the encoder output to a fixed number of tokens (~188M params). Choose linear projection when: fine-grained temporal detail matters (e.g., precise timestamp extraction, phoneme-level tasks), audio clips are short (<30s), and training budget is limited. Choose Q-Former when: processing long audio (minutes to hours), context window is constrained, the task is semantic (understanding, QA) rather than fine-grained temporal, and you have sufficient training data to train the Q-Former's cross-attention layers (typically 100K+ audio-text pairs). A practical middle ground is MLP adapter with strided pooling &mdash; e.g., 2-layer MLP with 4x downsampling, reducing 1500 tokens to 375 with ~17M parameters.</div>
</div>
`
    },
    {
      id: "research-taste",
      title: "Building Research Taste in Audio AI",
      content: `
<p><strong>Research taste</strong> is the compass that tells you <em>which</em> problem is worth solving before you touch data. It's distinct from research skill (how to execute). In the rapidly evolving Audio AI landscape, taste is what separates researchers who produce impactful work from those who chase incremental improvements on saturated benchmarks.</p>

<h4>The 10 Questions Framework</h4>
<p>Apply these to every paper you read. Taste is trained by running this drill consistently:</p>
<ol>
<li><strong>Core claim?</strong> One sentence. If you can't write it, you haven't understood the paper.</li>
<li><strong>What was previously broken?</strong> Not "it achieves SOTA" &mdash; what was actually broken before this?</li>
<li><strong>Key architectural/methodological choice &mdash; why not the obvious alternative?</strong></li>
<li><strong>What would Reviewer 2 say?</strong> Weak baselines? Artificial tasks? Vague contributions?</li>
<li><strong>Who cites this &mdash; and who conspicuously doesn't?</strong> Tells you if it started a lineage or ended one.</li>
<li><strong>What does it NOT solve?</strong> Read Limitations. The biggest clues to future papers live there.</li>
<li><strong>What becomes possible if the claim is true?</strong> The "unlock" question.</li>
<li><strong>Is the eval metric measuring what actually matters?</strong></li>
<li><strong>Simplest baseline that could undermine the paper?</strong></li>
<li><strong>If this paper disappeared from history, what wouldn't exist today?</strong> The "leverage" question.</li>
</ol>

<h4>Applying the Framework: Case Study with Sound-CoT</h4>

<p>Let's apply the 10 questions to Sound-CoT (arXiv:2502.16740):</p>

<ol>
<li><strong>Core claim:</strong> Auto-generated chain-of-thought reasoning over audio, trained via SFT + GRPO, improves audio understanding accuracy by 5-12% across multiple benchmarks.</li>
<li><strong>What was broken:</strong> AudioLLMs answered audio questions by pattern-matching without explicit reasoning. They couldn't explain their answers or break down complex audio scenes.</li>
<li><strong>Key choice:</strong> They auto-generated 1.24M CoT training samples using a teacher model rather than collecting human reasoning annotations. Alternative would be human annotation (too expensive) or prompted CoT without fine-tuning (unreliable).</li>
<li><strong>Reviewer 2 would say:</strong> The auto-generated CoT might be unfaithful &mdash; the reasoning chain might not reflect how the model actually arrives at its answer. Also, the teacher model's own audio understanding limits the quality of generated CoT.</li>
<li><strong>Citations:</strong> Cited heavily by audio reasoning papers but notably NOT by the speech recognition community &mdash; suggesting CoT helps understanding but not transcription.</li>
<li><strong>What it doesn't solve:</strong> Real-time reasoning (CoT adds latency), non-speech audio where language descriptions are imprecise, and the faithfulness problem.</li>
<li><strong>If true, unlocks:</strong> Reliable audio-based decision systems (medical auscultation, industrial monitoring) where explainability matters.</li>
<li><strong>Eval metric:</strong> Accuracy on audio QA benchmarks &mdash; but does not measure reasoning quality directly. A model could produce wrong reasoning chains that happen to reach correct answers.</li>
<li><strong>Simplest undermining baseline:</strong> An AudioLLM trained on the same data without CoT but with more epochs or larger batch size. Does the improvement come from CoT structure or just from more training signal?</li>
<li><strong>Leverage:</strong> Without this paper, audio reasoning would still be implicit. It established that explicit reasoning over audio is both possible and beneficial.</li>
</ol>

<div class="callout tip">
<div class="callout-title">Taste Development Plan</div>
<p><strong>Weeks 1-4 (Map):</strong> Read 30 papers; build concept map; ask "if this disappeared, what wouldn't exist?"<br>
<strong>Weeks 5-8 (Filter):</strong> Reverse-engineer 5 accepted papers; read 5 borderline rejections; weekly idea triage<br>
<strong>Weeks 9-16 (Engage):</strong> Follow key researchers; attend talks; write monthly Reviewer 2 critiques<br>
<strong>Weeks 17-24 (Test):</strong> 2-week prototype drills; submit workshop paper; write intro before experiments</p>
</div>

<h4>Identifying High-Value Research Directions</h4>

<p>Research taste means knowing which problems will matter in 2 years, not which problems are popular now. Here is a framework for evaluating research directions in Audio AI:</p>

<table>
<tr><th>Signal</th><th>High-Value Direction</th><th>Low-Value Direction</th></tr>
<tr><td><strong>Problem is structural</strong></td><td>Audio neglect (fundamental architectural issue)</td><td>Improving WER by 0.3% on LibriSpeech (saturated benchmark)</td></tr>
<tr><td><strong>Enables new capabilities</strong></td><td>Full-duplex conversation (enables new applications)</td><td>Faster inference on existing half-duplex systems (incremental)</td></tr>
<tr><td><strong>Cross-pollination potential</strong></td><td>Audio reasoning with CoT (imports text reasoning advances)</td><td>Audio-specific architecture (isolated innovation)</td></tr>
<tr><td><strong>Data moat weakening</strong></td><td>Self-supervised audio learning (reduces data requirements)</td><td>Larger supervised datasets (expensive, diminishing returns)</td></tr>
<tr><td><strong>Evaluation gap</strong></td><td>New metrics for audio understanding depth</td><td>Another benchmark for well-measured tasks</td></tr>
</table>

<h4>The Paper Reading Workflow for Practitioners</h4>

<p>Developing taste requires disciplined paper reading. Here is a concrete weekly workflow that balances breadth and depth:</p>

<p><strong>Monday (30 min): Triage.</strong> Scan the titles and abstracts of the week's new arXiv papers in cs.SD (Sound), cs.CL (Computation and Language), and eess.AS (Audio and Speech Processing). Use Semantic Scholar alerts for key authors: Dong Yu (Tencent), Jiatao Gu (Apple), Soumi Maiti (CMU), Shinji Watanabe (CMU). Flag 3-5 papers that seem relevant.</p>

<p><strong>Wednesday (1 hour): Deep Read.</strong> Read one paper in full using the 10 Questions Framework. Write a 200-word critique focusing on Question 4 (Reviewer 2) and Question 6 (what it doesn't solve). Post this critique in a shared team channel or research log.</p>

<p><strong>Friday (45 min): Synthesis.</strong> Review your flagged papers and the deep-read critique. Update your personal research concept map with any new connections. Ask yourself: "If I had to start a new project today, would anything I read this week change my direction?"</p>

<p>After 8 weeks of this workflow, you will have deep-read 8 papers, triaged ~40 papers, and built a concept map with 30-50 nodes. At this point, you should be able to articulate the top 3 open problems in Audio AI and explain why each is hard.</p>

<h4>Anti-Patterns in Audio AI Research</h4>

<p>Watch for these common taste failures in the field:</p>

<ul>
<li><strong>"SOTA chasing" on saturated benchmarks.</strong> LibriSpeech test-clean WER is at 1.5%. Reducing it to 1.4% requires enormous effort for negligible practical impact. The same effort applied to noisy/accented speech (WER 15-30%) yields far more value.</li>
<li><strong>"Architecture tourism."</strong> Papers that try a new combination of modules (yet another encoder + yet another adapter + yet another LLM) without a clear hypothesis about why the combination should be better. The contribution is a number, not an insight.</li>
<li><strong>"Eval gaming."</strong> Designing evaluation setups that favor your model. For example, evaluating on short utterances (<5 seconds) when your model is optimized for short audio, without testing on realistic lengths.</li>
<li><strong>"The missing ablation."</strong> Papers that demonstrate a complex system works but never ablate to show which components are responsible for the gains. Without ablations, you cannot learn from the paper.</li>
<li><strong>"Scale story."</strong> "We trained on 10x more data and got better results" is an engineering contribution, not a research insight. The interesting question is always: what did scale unlock that was qualitatively different?</li>
</ul>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You have 6 months and 8 GPUs. Propose a research project in Audio AI that balances novelty with feasibility.</div>
<div class="a-text">I would work on "Audio-Grounded Faithful Reasoning" &mdash; addressing both the audio neglect problem and the reasoning faithfulness gap. The project has three phases: (1) Months 1-2: Build an adversarial audio-text dataset (5K examples) where correct answers require audio evidence, plus a faithfulness evaluation framework that checks whether intermediate reasoning steps reference actual audio content. (2) Months 3-4: Train an AudioLLM with a modified objective that includes contrastive audio grounding loss + CoT supervision from a teacher model, with a key innovation of "audio anchoring" where the CoT must reference specific timestamps or audio features. (3) Months 5-6: Evaluate on standard benchmarks plus the adversarial set, ablate each component, and compare against the simplest possible baseline (larger model without these modifications). This is feasible because it reuses existing models (Qwen2-Audio), requires moderate data (5K adversarial examples can be synthesized), and 8 GPUs suffice for LoRA fine-tuning of a 7B model. The novelty is in connecting two open problems (neglect + faithfulness) that are usually studied separately.</div>
</div>
`
    },
    {
      id: "audiollm-training",
      title: "Training AudioLLMs from Scratch",
      content: `
<p>Training an AudioLLM is a multi-stage process that requires careful orchestration of data, compute, and optimization. This section provides a detailed recipe based on published methods from Qwen-Audio, SALMONN, Qwen2.5-Omni, and others. Even if you will never train one from scratch (most engineers will fine-tune existing models), understanding the training pipeline is essential for debugging, evaluation, and architectural decisions.</p>

<h4>Stage 1: Audio Encoder Pretraining</h4>

<p>The audio encoder must learn rich representations of audio before it can be useful in an AudioLLM. Two dominant pretraining paradigms exist:</p>

<p><strong>Contrastive Learning (CLAP, LAION-CLAP).</strong> Train paired audio and text encoders to map matching audio-text pairs close together in embedding space, and non-matching pairs far apart. This is the audio equivalent of CLIP for images.</p>

<pre><code># CLAP-style contrastive pretraining (simplified)
import torch
import torch.nn.functional as F

class CLAPModel(torch.nn.Module):
    def __init__(self, audio_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.audio_encoder = audio_encoder   # e.g., HTSAT or AST
        self.text_encoder = text_encoder     # e.g., RoBERTa
        self.audio_proj = torch.nn.Linear(audio_encoder.hidden_dim, embed_dim)
        self.text_proj = torch.nn.Linear(text_encoder.hidden_dim, embed_dim)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, audio, text_input_ids, text_attention_mask):
        # Encode both modalities
        audio_features = self.audio_proj(self.audio_encoder(audio))
        text_features = self.text_proj(
            self.text_encoder(text_input_ids, text_attention_mask).pooler_output
        )

        # Normalize
        audio_features = F.normalize(audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Cosine similarity scaled by temperature
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_features @ text_features.T
        logits_per_text = logits_per_audio.T

        # Symmetric cross-entropy loss
        labels = torch.arange(len(audio_features), device=audio_features.device)
        loss_a = F.cross_entropy(logits_per_audio, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_a + loss_t) / 2

# Training recipe for CLAP:
# - Dataset: LAION-Audio-630K (633K audio-text pairs) + AudioSet (2M with generated captions)
# - Audio encoder: HTSAT-base (30M params) or AST (87M params)
# - Text encoder: RoBERTa-base (125M params)
# - Batch size: 2048 (across 8 GPUs, 256 per GPU)
# - Learning rate: 1e-4 with cosine decay, 2000 step warmup
# - Training: ~50 epochs on combined data (~150K steps)
# - Mixed precision: BF16</code></pre>

<p><strong>Masked Prediction (HuBERT, WavLM, W2v-BERT).</strong> Mask portions of the audio input and train the model to predict the masked features. This is the audio equivalent of BERT-style pretraining. HuBERT uses an offline clustering step to create discrete target labels from audio features, then trains the model to predict these labels for masked positions.</p>

<p><strong>Supervised pretraining (Whisper).</strong> Whisper takes a different approach: large-scale supervised training on 680,000 hours of labeled audio-text pairs scraped from the internet. The encoder is trained jointly with the decoder for ASR, producing encoder representations that are already aligned with language.</p>

<p>In practice, most AudioLLM projects skip Stage 1 entirely and use a pretrained Whisper encoder. This is the right choice unless you are targeting a domain where Whisper performs poorly (e.g., non-speech audio events, music, specific low-resource languages).</p>

<h4>Stage 2: Adapter Alignment (Projection Training)</h4>

<p>The goal: train the adapter layer to map audio encoder representations into the LLM's embedding space, while keeping both the encoder and LLM frozen. This is the cheapest and fastest stage.</p>

<table>
<tr><th>Hyperparameter</th><th>Typical Value</th><th>Notes</th></tr>
<tr><td>Frozen components</td><td>Audio encoder + LLM</td><td>Only adapter parameters are trainable</td></tr>
<tr><td>Trainable parameters</td><td>5-200M (depends on adapter type)</td><td>Linear: ~5M, MLP: ~17M, Q-Former: ~188M</td></tr>
<tr><td>Learning rate</td><td>1e-4 to 5e-4</td><td>Higher than Stage 3; adapter trains fast</td></tr>
<tr><td>LR schedule</td><td>Cosine with warmup</td><td>5-10% warmup steps</td></tr>
<tr><td>Batch size</td><td>256-512</td><td>Large batches stabilize alignment</td></tr>
<tr><td>Training data</td><td>ASR transcription pairs</td><td>LibriSpeech (960h), GigaSpeech (10Kh), Common Voice</td></tr>
<tr><td>Training steps</td><td>10K-50K</td><td>Converges quickly; monitor validation loss</td></tr>
<tr><td>Precision</td><td>BF16</td><td>FP16 can cause instability with frozen LLM</td></tr>
<tr><td>Training loss</td><td>Next-token prediction (cross-entropy)</td><td>Standard LM loss on text tokens only</td></tr>
</table>

<pre><code># Stage 2: Adapter alignment training loop (conceptual)
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Freeze encoder and LLM
for param in audio_encoder.parameters():
    param.requires_grad = False
for param in llm.parameters():
    param.requires_grad = False

# Only adapter is trainable
adapter = MLPAdapter(d_encoder=1280, d_hidden=2048, d_llm=4096)
optimizer = AdamW(adapter.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=30000
)

for step, batch in enumerate(dataloader):
    audio_features = audio_encoder(batch["audio"])        # (B, T, 1280)
    adapted_features = adapter(audio_features)             # (B, T, 4096)

    # Concatenate with text token embeddings
    text_embeds = llm.get_input_embeddings()(batch["text_ids"])  # (B, S, 4096)
    combined = torch.cat([adapted_features, text_embeds], dim=1)

    # Forward through frozen LLM
    outputs = llm(inputs_embeds=combined, labels=batch["labels"])
    loss = outputs.loss  # Cross-entropy on text tokens only

    loss.backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")</code></pre>

<p>The key data for Stage 2 is audio-transcription pairs. ASR data is ideal because the mapping between audio and text is deterministic: the model learns that "these audio features correspond to these words." High-quality ASR datasets include:</p>

<ul>
<li><strong>LibriSpeech</strong> (960 hours, English read speech, very clean)</li>
<li><strong>GigaSpeech</strong> (10,000 hours, English, diverse domains)</li>
<li><strong>Common Voice</strong> (multi-lingual, varying quality, 100+ languages)</li>
<li><strong>WenetSpeech</strong> (10,000 hours, Mandarin)</li>
<li><strong>MLS</strong> (Multilingual LibriSpeech, 50K+ hours across 8 languages)</li>
</ul>

<h4>Stage 3: Instruction Tuning</h4>

<p>Now we unfreeze parts of the LLM (or apply LoRA adapters) and train on diverse audio-text instruction data. This stage teaches the model to follow instructions about audio and perform various tasks beyond simple transcription.</p>

<table>
<tr><th>Hyperparameter</th><th>Typical Value</th><th>Notes</th></tr>
<tr><td>Unfrozen components</td><td>Adapter + LoRA on LLM</td><td>LoRA rank 16-64 on q_proj, v_proj, and o_proj</td></tr>
<tr><td>Trainable parameters</td><td>50-500M total</td><td>Adapter (~5-200M) + LoRA (~20-300M)</td></tr>
<tr><td>Learning rate</td><td>1e-5 to 5e-5</td><td>Lower than Stage 2 to prevent catastrophic forgetting</td></tr>
<tr><td>LR schedule</td><td>Cosine with warmup</td><td>1-3% warmup steps</td></tr>
<tr><td>Batch size</td><td>64-128</td><td>Smaller than Stage 2; data is more diverse</td></tr>
<tr><td>LoRA config</td><td>r=32, alpha=64, dropout=0.05</td><td>Target: q_proj, v_proj, o_proj, gate_proj, up_proj</td></tr>
<tr><td>Training data mix</td><td>ASR:Caption:QA:Instruct = 3:2:3:2</td><td>Ratio matters; too much ASR degrades QA</td></tr>
<tr><td>Training steps</td><td>20K-100K</td><td>Monitor multiple eval metrics, not just loss</td></tr>
<tr><td>Audio augmentation</td><td>Speed perturbation (0.9-1.1x), noise injection (SNR 10-20dB)</td><td>Critical for robustness</td></tr>
</table>

<p>The instruction tuning data must be diverse. Key dataset categories:</p>

<ul>
<li><strong>Audio captioning:</strong> AudioCaps (50K clips, ~10s each), Clotho (5K clips with 5 captions each), WavCaps (400K weakly-labeled clips)</li>
<li><strong>Audio question answering:</strong> AudioQA, ClothoAQA, synthesized QA from captioning datasets</li>
<li><strong>Speech understanding:</strong> Emotion recognition (IEMOCAP, MSP-Podcast), speaker verification (VoxCeleb), intent classification (SLURP, FSC)</li>
<li><strong>Complex instructions:</strong> "Transcribe this audio, then summarize the key points," "What emotion is the speaker conveying and what evidence supports this?"</li>
</ul>

<div class="callout warning">
<div class="callout-title">War Story: The Data Ratio Catastrophe</div>
<p>A team training an AudioLLM for a voice assistant product used a 7:1:1:1 ratio of ASR to other tasks in Stage 3. The model achieved excellent WER on speech recognition but could not answer basic questions about audio content ("What sounds are in the background?") &mdash; it had overfit to transcription. Worse, when asked audio understanding questions, it would output transcriptions instead of answers. The fix required retraining with a 3:2:3:2 ratio and adding a task-type prefix token. The lesson: data mixing ratios in multi-task audio training are a critical hyperparameter that must be tuned through evaluation on all target tasks, not just the dominant one.</p>
</div>

<h4>Data Requirements Summary</h4>

<table>
<tr><th>Stage</th><th>Data Type</th><th>Minimum Hours</th><th>Recommended Hours</th><th>Key Datasets</th></tr>
<tr><td>Stage 2 (Alignment)</td><td>ASR transcriptions</td><td>1,000</td><td>10,000+</td><td>LibriSpeech, GigaSpeech, MLS</td></tr>
<tr><td>Stage 3 (Instruction)</td><td>Audio captioning</td><td>100</td><td>1,000+</td><td>AudioCaps, WavCaps, Clotho</td></tr>
<tr><td>Stage 3 (Instruction)</td><td>Audio QA</td><td>50</td><td>500+</td><td>AudioQA, synthesized from captions</td></tr>
<tr><td>Stage 3 (Instruction)</td><td>Speech understanding</td><td>200</td><td>2,000+</td><td>IEMOCAP, VoxCeleb, SLURP</td></tr>
<tr><td>Stage 3 (Instruction)</td><td>Complex instructions</td><td>10K samples</td><td>100K+ samples</td><td>Synthesized via teacher model</td></tr>
</table>

<h4>Common Training Failures and Debugging</h4>

<h4>Compute Requirements</h4>

<p>Understanding the compute budget for each stage helps with project planning:</p>

<table>
<tr><th>Stage</th><th>GPU-Hours (8xA100-80GB)</th><th>Wall-Clock Time</th><th>Cost (cloud, approx.)</th></tr>
<tr><td>Stage 2 (Alignment, 30K steps)</td><td>50-100 GPU-hours</td><td>6-12 hours</td><td>$150-$300</td></tr>
<tr><td>Stage 3 (Instruction, LoRA, 50K steps)</td><td>200-400 GPU-hours</td><td>1-2 days</td><td>$600-$1,200</td></tr>
<tr><td>Stage 3 (Full fine-tune, 50K steps)</td><td>800-1,600 GPU-hours</td><td>4-8 days</td><td>$2,400-$4,800</td></tr>
<tr><td>Full pipeline (all stages)</td><td>300-600 GPU-hours</td><td>2-3 days</td><td>$900-$1,800</td></tr>
</table>

<p>These estimates assume a 7B parameter LLM. For 13B models, multiply by ~2x. For 70B models, multiply by ~10x and note that LoRA becomes essential (full fine-tuning at 70B requires multi-node training). The key cost-saving insight: LoRA reduces Stage 3 cost by 4-8x compared to full fine-tuning, with minimal quality loss for most tasks.</p>

<h4>Monitoring Training Progress</h4>

<p>Unlike text LLM training where perplexity is a reliable progress indicator, AudioLLM training requires monitoring multiple signals simultaneously:</p>

<ul>
<li><strong>Training loss curve:</strong> Should decrease smoothly. Sudden spikes indicate learning rate issues or data corruption. A plateau followed by a drop often indicates the model is learning a new modality mapping.</li>
<li><strong>Validation WER (on held-out ASR data):</strong> Track every 1K steps during Stage 2, every 2K steps during Stage 3. WER should decrease monotonically in Stage 2. In Stage 3, WER may increase slightly (1-2% absolute) as the model learns non-ASR tasks &mdash; this is acceptable.</li>
<li><strong>Audio captioning score (CIDEr on Clotho-val):</strong> Track during Stage 3. This measures whether the model is learning audio understanding beyond transcription.</li>
<li><strong>Text-only benchmark (MMLU 5-shot):</strong> Track every 5K steps to monitor catastrophic forgetting. Should not drop more than 2-3 points from baseline.</li>
<li><strong>Gradient norm:</strong> Should be stable. A sudden increase in gradient norm during Stage 3 often precedes catastrophic forgetting. Use gradient clipping at 1.0 and consider reducing LR if gradient norm consistently exceeds 0.5.</li>
</ul>

<h4>Common Training Failures and Debugging</h4>

<p><strong>Failure 1: Adapter collapse.</strong> The adapter outputs converge to a narrow subspace, producing nearly identical representations regardless of audio input. <em>Symptoms:</em> Validation loss plateaus early; model outputs the same text for different audio inputs. <em>Cause:</em> Learning rate too high in Stage 2, or adapter is too small relative to the representation gap. <em>Fix:</em> Reduce LR to 1e-4; increase adapter hidden dimension; add dropout (0.1) in the adapter; use weight decay (0.01).</p>

<p><strong>Failure 2: Catastrophic forgetting of LLM capabilities.</strong> After Stage 3, the LLM loses its text reasoning abilities. <em>Symptoms:</em> Audio tasks improve but text-only tasks degrade significantly (>5% on text benchmarks). <em>Cause:</em> Learning rate too high for LoRA; training too long; not using LoRA (full fine-tuning). <em>Fix:</em> Use LoRA with rank 16-32 instead of full fine-tuning; reduce LR to 1e-5; add 10-20% text-only training data to the instruction mix; use gradient checkpointing to fit larger batch sizes.</p>

<p><strong>Failure 3: Length generalization failure.</strong> Model works on 10-second clips but fails on 60-second clips. <em>Symptoms:</em> Output quality degrades sharply when audio exceeds training length distribution. <em>Cause:</em> Training data concentrated at short durations; positional encoding doesn't generalize. <em>Fix:</em> Include varied-length audio in training (5s to 120s); use relative positional encoding (RoPE) instead of absolute; apply YaRN or NTK-aware interpolation for long sequences.</p>

<p><strong>Failure 4: Modality competition.</strong> When both audio and text provide information, the model learns to rely exclusively on one modality. <em>Symptoms:</em> High accuracy when only audio is provided, but accuracy drops when text context is also present (or vice versa). <em>Cause:</em> Insufficient examples of modality-complementary and modality-conflicting scenarios. <em>Fix:</em> Add training examples where both modalities must be used; include adversarial examples where modalities conflict (with the audio-aligned answer as ground truth); add attention-based auxiliary losses.</p>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You're fine-tuning Qwen2-Audio for a customer service application. The model needs to understand spoken queries, detect emotion, and provide helpful responses. Walk me through your training strategy.</div>
<div class="a-text">I would use a 2-stage fine-tuning approach: (1) Domain-specific adapter tuning: Collect 500-1000 hours of customer service call recordings with transcriptions. Fine-tune only the adapter layer for 5K steps with LR 2e-4 to align the audio encoder to the domain's acoustic characteristics (phone-quality audio, background noise, domain vocabulary). (2) Task-specific instruction tuning with LoRA: Prepare three types of data &mdash; transcription pairs, emotion-labeled utterances (annotated with {frustrated, neutral, satisfied, angry}), and response-generation examples (input: customer query audio + context, output: helpful response). Apply LoRA (rank=32, alpha=64) to q_proj, v_proj, o_proj, gate_proj. Use data ratio 2:3:5 (transcription:emotion:response) to emphasize the end task. Train for 10K steps with LR 2e-5. Critical details: augment training audio with telephone codec simulation and noise injection to match deployment conditions; add 10% general-purpose audio QA data to prevent catastrophic forgetting; evaluate on a held-out set of real customer calls with human ratings for response quality and emotion detection accuracy.</div>
</div>
`
    },
    {
      id: "audiollm-eval",
      title: "Evaluation & Benchmarks",
      content: `
<p>Evaluation of AudioLLMs is an unsolved problem. Unlike text LLMs where perplexity and benchmark accuracy provide reasonable signals, audio understanding involves multiple orthogonal capabilities: speech recognition, audio event detection, music understanding, emotional perception, and more. No single model excels across all tasks, and no single metric captures overall audio understanding ability.</p>

<h4>AudioBench: The Most Comprehensive Benchmark</h4>

<p>AudioBench (arXiv:2406.16020) is currently the most comprehensive evaluation suite for AudioLLMs, covering 8 task categories across 26 datasets. Understanding its structure is essential for any AI engineer working with audio models.</p>

<table>
<tr><th>Task Category</th><th>Datasets</th><th>Primary Metric</th><th>What It Measures</th></tr>
<tr><td><strong>Speech Recognition</strong></td><td>LibriSpeech (clean/other), Common Voice, Fleurs</td><td>WER (Word Error Rate)</td><td>Basic transcription accuracy</td></tr>
<tr><td><strong>Audio Captioning</strong></td><td>AudioCaps, Clotho</td><td>METEOR, CIDEr, SPIDEr</td><td>Descriptive understanding of audio scenes</td></tr>
<tr><td><strong>Audio QA</strong></td><td>ClothoAQA, AudioQA</td><td>Accuracy, F1</td><td>Reasoning about audio content</td></tr>
<tr><td><strong>Emotion Recognition</strong></td><td>IEMOCAP, MSP-Podcast</td><td>Weighted F1, UAR</td><td>Paralinguistic understanding</td></tr>
<tr><td><strong>Speaker Analysis</strong></td><td>VoxCeleb, LibriSpeech speakers</td><td>Accuracy, EER</td><td>Speaker-level understanding</td></tr>
<tr><td><strong>Sound Event Detection</strong></td><td>ESC-50, UrbanSound8K</td><td>Accuracy</td><td>Environmental audio recognition</td></tr>
<tr><td><strong>Music Understanding</strong></td><td>MusicCaps, MTG-Jamendo</td><td>Accuracy, BLEU</td><td>Musical structure and content</td></tr>
<tr><td><strong>Spoken Language Understanding</strong></td><td>SLURP, FSC</td><td>Intent Accuracy, Entity F1</td><td>Semantic understanding of speech</td></tr>
</table>

<h4>Per-Task Metrics Explained</h4>

<p><strong>WER (Word Error Rate)</strong> is the standard ASR metric: <code>WER = (Substitutions + Insertions + Deletions) / Total Reference Words</code>. Lower is better. State-of-the-art on LibriSpeech test-clean is ~1.5% (Whisper-large-v3), but on noisy/accented speech, WER can exceed 20-30%. A common interview trap: WER can exceed 100% (when insertions are frequent).</p>

<p><strong>METEOR and CIDEr</strong> are used for audio captioning. METEOR aligns predicted and reference words using stemming, synonymy, and paraphrasing, scoring from 0-1. CIDEr uses TF-IDF weighting of n-grams to measure consensus with reference captions. SPIDEr is the average of CIDEr and SPICE (semantic propositional content). For AudioCaps, strong models achieve CIDEr scores of 0.7-0.9.</p>

<p><strong>Weighted F1 and UAR (Unweighted Average Recall)</strong> for emotion recognition. UAR is preferred over accuracy because emotion datasets are heavily imbalanced (neutral typically 40-50% of examples). A random baseline would achieve ~25% UAR on 4-class emotion (happy, sad, angry, neutral) but ~45% accuracy due to class imbalance. Strong models achieve 65-75% UAR on IEMOCAP.</p>

<h4>The Evaluation Crisis</h4>

<p>The central problem: <strong>no single AudioLLM excels across all task categories.</strong> Here are representative results from 2025 evaluations:</p>

<table>
<tr><th>Model</th><th>ASR (WER) ↓</th><th>Caption (CIDEr) ↑</th><th>Audio QA (Acc) ↑</th><th>Emotion (UAR) ↑</th><th>Sound Event (Acc) ↑</th></tr>
<tr><td>Qwen2-Audio-7B</td><td>3.2%</td><td>0.82</td><td>71.5%</td><td>62.3%</td><td>78.4%</td></tr>
<tr><td>SALMONN-13B</td><td>4.8%</td><td>0.76</td><td>65.2%</td><td>58.1%</td><td>82.1%</td></tr>
<tr><td>Gemini 1.5 Pro</td><td>2.1%</td><td>0.85</td><td>76.3%</td><td>67.5%</td><td>74.2%</td></tr>
<tr><td>GPT-4o Audio</td><td>2.4%</td><td>0.88</td><td>78.1%</td><td>64.8%</td><td>71.9%</td></tr>
<tr><td>Whisper-large-v3 (ASR-only)</td><td>1.5%</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>
</table>

<p>Several patterns emerge: (1) Closed-source models (Gemini, GPT-4o) lead on most tasks but are expensive and non-customizable. (2) Specialist models (Whisper for ASR) still beat generalist AudioLLMs on their specialty. (3) Different open-source models excel on different tasks &mdash; SALMONN leads on sound events (dual encoder helps), Qwen2-Audio leads on ASR and captioning. (4) Emotion recognition remains difficult for all models, with the best open models barely exceeding 65% UAR.</p>

<h4>Building Your Own Evaluation Pipeline</h4>

<p>For production deployment, you need an evaluation pipeline tailored to your specific use case. Here is a template:</p>

<pre><code>import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import jiwer  # For WER computation

@dataclass
class EvalResult:
    task: str
    dataset: str
    metric_name: str
    metric_value: float
    num_samples: int
    per_sample_scores: List[float] = field(default_factory=list)

class AudioLLMEvaluator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.results: List[EvalResult] = []

    def evaluate_asr(self, test_data: List[Dict]) -> EvalResult:
        """
        test_data: list of {"audio_path": str, "reference": str}
        """
        hypotheses = []
        references = []
        for item in test_data:
            hypothesis = self._generate(item["audio_path"], "Transcribe this audio.")
            hypotheses.append(hypothesis)
            references.append(item["reference"])

        # Compute WER using jiwer
        wer = jiwer.wer(references, hypotheses)

        # Per-sample WER for error analysis
        per_sample = []
        for ref, hyp in zip(references, hypotheses):
            per_sample.append(jiwer.wer([ref], [hyp]))

        result = EvalResult(
            task="asr", dataset="custom",
            metric_name="WER", metric_value=wer,
            num_samples=len(test_data),
            per_sample_scores=per_sample
        )
        self.results.append(result)
        return result

    def evaluate_classification(self, test_data: List[Dict],
                                 task_name: str,
                                 prompt_template: str) -> EvalResult:
        """
        test_data: list of {"audio_path": str, "label": str}
        prompt_template: e.g., "Classify the emotion: {choices}"
        """
        correct = 0
        per_class_correct = {}
        per_class_total = {}

        for item in test_data:
            prediction = self._generate(item["audio_path"], prompt_template)
            label = item["label"].lower()
            pred_label = prediction.strip().lower()

            # Fuzzy matching: check if label appears in prediction
            is_correct = label in pred_label
            if is_correct:
                correct += 1

            per_class_total[label] = per_class_total.get(label, 0) + 1
            per_class_correct[label] = per_class_correct.get(label, 0) + (1 if is_correct else 0)

        accuracy = correct / len(test_data)

        # Compute UAR (Unweighted Average Recall)
        per_class_recall = []
        for cls in per_class_total:
            recall = per_class_correct.get(cls, 0) / per_class_total[cls]
            per_class_recall.append(recall)
        uar = np.mean(per_class_recall)

        result = EvalResult(
            task=task_name, dataset="custom",
            metric_name="UAR", metric_value=uar,
            num_samples=len(test_data)
        )
        self.results.append(result)
        return result

    def evaluate_qa(self, test_data: List[Dict]) -> EvalResult:
        """
        test_data: list of {"audio_path": str, "question": str, "answer": str}
        """
        correct = 0
        for item in test_data:
            prompt = f"Listen to this audio and answer: {item['question']}"
            prediction = self._generate(item["audio_path"], prompt)
            # Simple exact match (production would use LLM-as-judge)
            if item["answer"].lower() in prediction.lower():
                correct += 1

        accuracy = correct / len(test_data)
        result = EvalResult(
            task="audio_qa", dataset="custom",
            metric_name="Accuracy", metric_value=accuracy,
            num_samples=len(test_data)
        )
        self.results.append(result)
        return result

    def _generate(self, audio_path: str, prompt: str) -> str:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": prompt}
        ]}]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True,
            tokenize=True, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=128, temperature=0.0)
        return self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]

    def summary(self) -> str:
        lines = ["\\n=== AudioLLM Evaluation Summary ==="]
        for r in self.results:
            lines.append(f"{r.task}/{r.dataset}: {r.metric_name} = {r.metric_value:.4f} (n={r.num_samples})")
        return "\\n".join(lines)

# Usage:
# evaluator = AudioLLMEvaluator(model, processor)
# asr_result = evaluator.evaluate_asr(asr_test_data)
# emotion_result = evaluator.evaluate_classification(
#     emotion_test_data, "emotion",
#     "What emotion is the speaker expressing? Choose from: happy, sad, angry, neutral."
# )
# print(evaluator.summary())</code></pre>

<h4>Cross-Model Comparison Methodology</h4>

<p>When comparing AudioLLMs, the evaluation methodology matters as much as the results. Common pitfalls in published comparisons:</p>

<p><strong>Prompt sensitivity.</strong> Different models respond very differently to prompt phrasing. One model might perform 15% better with "Classify the emotion" vs. "What emotion is the speaker expressing?" Always test at least 3 prompt variants and report the best per model (to measure capability) and the average (to measure robustness).</p>

<p><strong>Audio preprocessing differences.</strong> Some models expect 16kHz mono audio; others expect 24kHz or even 48kHz. Resampling quality matters: naive resampling can introduce artifacts that disproportionately affect some models. Always use high-quality resampling (e.g., <code>librosa.resample</code> with <code>res_type='kaiser_best'</code>) and verify the audio is correctly formatted for each model.</p>

<p><strong>Decoding strategy.</strong> Temperature, top-p, and top-k settings dramatically affect output quality. For fair comparison, use greedy decoding (temperature=0) for factual tasks and a fixed temperature (e.g., 0.7) for open-ended tasks. Report decoding parameters for all models.</p>

<p><strong>Context window utilization.</strong> Models with larger context windows have an inherent advantage on long audio tasks. If comparing a 4K-context model against a 32K-context model on 5-minute audio, the comparison is unfair unless you also report results on short audio where context is not a bottleneck.</p>

<pre><code># Fair comparison framework
def fair_comparison(models, test_data, prompts_per_task=3):
    """
    Run fair comparison across models with multiple prompt variants.
    """
    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for task in test_data:
            task_scores = []
            for prompt_variant in range(prompts_per_task):
                prompt = task["prompt_variants"][prompt_variant]
                scores = evaluate_model_on_task(
                    model, task["data"], prompt,
                    temperature=0.0,  # Greedy for reproducibility
                    max_new_tokens=128
                )
                task_scores.append(scores)

            results[model_name][task["name"]] = {
                "best": max(task_scores, key=lambda x: x["primary_metric"]),
                "avg": np.mean([s["primary_metric"] for s in task_scores]),
                "std": np.std([s["primary_metric"] for s in task_scores]),
            }
    return results</code></pre>

<h4>Building a Continuous Evaluation Pipeline</h4>

<p>For production systems, evaluation is not a one-time event but a continuous process. Here is a recommended pipeline:</p>

<ol>
<li><strong>Nightly regression tests:</strong> Run the core evaluation suite (ASR WER, emotion UAR, QA accuracy) on fixed test sets. Alert if any metric degrades by more than 2% from baseline. This catches model regressions, infrastructure issues, and data pipeline problems.</li>
<li><strong>Weekly deep evaluation:</strong> Run the full AudioBench suite plus domain-specific evaluations. Compare against the previous week and the baseline. Generate a report with per-task breakdowns and failure analysis on the worst-performing examples.</li>
<li><strong>Monthly human evaluation:</strong> 100 samples rated by 3 human evaluators across content quality, relevance, and naturalness dimensions. Track trends over time.</li>
<li><strong>Quarterly adversarial evaluation:</strong> Run audio neglect probes, robustness tests (noise injection, accent variation), and edge case tests. This catches subtle degradations that standard benchmarks miss.</li>
</ol>

<h4>Common Evaluation Pitfalls</h4>

<ul>
<li><strong>Leakage through text priors:</strong> If the audio file name or metadata contains the answer, the model may use that instead of actually processing audio. Always use anonymized file paths.</li>
<li><strong>Prompt sensitivity:</strong> AudioLLMs can be extremely sensitive to prompt phrasing. "What emotion is this?" vs "Classify the emotion of the speaker" can yield very different accuracy. Always report the exact prompt used.</li>
<li><strong>Evaluation-training distribution mismatch:</strong> Most AudioLLMs are trained on clean speech; evaluating on noisy, telephone-quality, or far-field audio will dramatically understate capabilities for clean-audio applications.</li>
<li><strong>LLM-as-judge for open-ended tasks:</strong> For captioning and QA, consider using GPT-4 as an evaluator with rubrics, as automated metrics (BLEU, METEOR) correlate poorly with human judgment for open-ended responses.</li>
</ul>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">You need to evaluate an AudioLLM for a production voice assistant. What metrics would you track and why?</div>
<div class="a-text">I would track metrics across four dimensions: (1) Accuracy: WER for transcription quality, intent classification accuracy and entity extraction F1 for understanding, and response relevance score (LLM-as-judge, 1-5 scale). (2) Robustness: WER degradation across noise levels (SNR from clean to 5dB), accuracy across accent groups (to catch bias), and performance on adversarial inputs (audio neglect probe). (3) Latency: Time-to-first-token (TTFT) for streaming, end-to-end response latency, and P99 latency under load. (4) User experience: Task completion rate on realistic multi-turn dialogues, error recovery rate (can the model recover from misunderstanding?), and A/B test metrics if deployed (session length, task success rate, user satisfaction). The key insight: production evaluation must include robustness and latency metrics, not just accuracy. A model with 95% accuracy but 2-second latency is worse than a model with 90% accuracy and 200ms latency for a voice assistant.</div>
</div>
`
    },
    {
      id: "audio-reasoning",
      title: "Audio Reasoning & Chain-of-Thought",
      content: `
<p>Audio reasoning &mdash; the ability to draw inferences, make deductions, and explain conclusions from audio input &mdash; represents the frontier of AudioLLM capabilities. While text LLMs have benefited enormously from chain-of-thought (CoT) prompting and training, applying similar reasoning techniques to audio is substantially more challenging. Audio is continuous, temporal, multi-layered (multiple simultaneous sound sources), and lacks the discrete symbolic structure that makes text reasoning tractable.</p>

<h4>Sound-CoT: Auto-Generated Chain-of-Thought for Audio</h4>

<p>Sound-CoT (arXiv:2502.16740) is the first systematic attempt to train AudioLLMs with explicit reasoning chains over audio. The key contribution is a scalable method for generating CoT training data without human annotation.</p>

<p><strong>Data Generation Pipeline:</strong></p>
<ol>
<li>Start with 1.24 million audio-answer pairs from existing datasets (AudioCaps, Clotho, ESC-50, AudioSet).</li>
<li>For each pair, use a teacher LLM (GPT-4) with the audio transcription/caption to generate a reasoning chain that explains how one would arrive at the correct answer from the audio description.</li>
<li>Filter generated chains for coherence and faithfulness using a critic model (rejection rate: ~15%).</li>
<li>Format as: Audio -> [Reasoning Chain] -> Answer.</li>
</ol>

<p><strong>Example CoT:</strong></p>
<pre><code>Audio: [sound of birds chirping, followed by a car engine starting]
Question: What location might this recording be from?

Without CoT: "A residential area."

With CoT: "I hear birds chirping, which suggests an outdoor setting with
vegetation. The chirping sounds like common garden birds (robins or sparrows),
not tropical species. Then I hear a car engine starting - this indicates
proximity to a road or driveway. The combination of garden birds and a car
engine starting (not passing by) suggests a residential area, likely a
suburban home with a driveway. Answer: A suburban residential area."</code></pre>

<p><strong>Training Recipe:</strong></p>
<ol>
<li><strong>Stage 1: Supervised Fine-Tuning (SFT).</strong> Train the AudioLLM on the 1.24M CoT examples using standard next-token prediction. This teaches the model the format and structure of audio reasoning. LR: 2e-5, 3 epochs, batch size 128.</li>
<li><strong>Stage 2: Audio Thought Injection.</strong> Introduce special tokens <code>&lt;audio_thought&gt;</code> and <code>&lt;/audio_thought&gt;</code> that delimit reasoning chains. The model learns to generate these tokens before producing the final answer. This is similar to how DeepSeek-R1 uses <code>&lt;think&gt;</code> tokens.</li>
<li><strong>Stage 3: GRPO Refinement.</strong> Apply Group Relative Policy Optimization to improve reasoning quality. The reward model evaluates both the final answer accuracy and the reasoning chain quality (measured by an LLM-as-judge). This stage uses 50K high-quality examples with verified reasoning chains.</li>
</ol>

<table>
<tr><th>Training Stage</th><th>Data Size</th><th>LR</th><th>Epochs</th><th>Key Loss</th></tr>
<tr><td>SFT on CoT data</td><td>1.24M examples</td><td>2e-5</td><td>3</td><td>Cross-entropy on reasoning + answer tokens</td></tr>
<tr><td>Audio thought injection</td><td>500K examples</td><td>1e-5</td><td>2</td><td>Cross-entropy with thought-token weighting</td></tr>
<tr><td>GRPO refinement</td><td>50K examples</td><td>5e-6</td><td>1</td><td>PPO with KL penalty + answer accuracy reward</td></tr>
</table>

<h4>AudSemThinker: Structured Auditory Semantics</h4>

<p>AudSemThinker takes a different approach: instead of free-form reasoning chains, it uses structured auditory semantic representations as the reasoning substrate. The insight: audio scenes can be decomposed into a structured representation before reasoning.</p>

<p><strong>The Structured Representation:</strong></p>
<pre><code># AudSemThinker's structured audio scene representation
{
    "sound_events": [
        {"type": "speech", "start": 0.0, "end": 3.5,
         "attributes": {"language": "English", "gender": "male", "emotion": "neutral"}},
        {"type": "music", "start": 2.0, "end": 8.0,
         "attributes": {"genre": "jazz", "instruments": ["piano", "saxophone"]}},
        {"type": "ambient", "start": 0.0, "end": 8.0,
         "attributes": {"environment": "indoor", "noise_level": "low"}}
    ],
    "temporal_relations": [
        {"event1": "speech", "event2": "music", "relation": "overlaps_with"},
        {"event1": "ambient", "event2": "speech", "relation": "contains"}
    ],
    "scene_summary": "Indoor setting with a male speaker talking over jazz music"
}</code></pre>

<p>The model first generates this structured representation, then reasons over it to answer questions. This approach forces the model to explicitly ground its reasoning in detected audio events, reducing hallucination.</p>

<h4>UALM-Reason: Mixed Text+Audio Token Reasoning</h4>

<p>UALM-Reason explores a more radical approach: reasoning in a mixed space of text and audio tokens. Instead of converting audio entirely to text before reasoning, the model maintains audio token representations at key points in the reasoning chain.</p>

<p><strong>How it works:</strong> The model's reasoning chain alternates between text tokens (for logical reasoning) and audio token references (for acoustic evidence). For example:</p>

<pre><code># UALM-Reason mixed reasoning chain (conceptual)
# T = text token, A = audio token reference

T: "I need to determine the speaker's emotion."
T: "First, let me examine the prosodic features:"
A: [audio_ref: frames 100-250, pitch contour]  # References specific audio features
T: "The pitch is elevated and varies rapidly, suggesting arousal."
A: [audio_ref: frames 100-250, energy envelope]
T: "Energy is high with sharp onsets, consistent with anger or excitement."
T: "Now examining the linguistic content:"
A: [audio_ref: frames 100-250, content tokens]
T: "The words 'I can't believe' combined with high pitch suggest frustration."
T: "Conclusion: The speaker is expressing frustration/anger."</code></pre>

<p>This approach preserves acoustic detail that would be lost in pure text reasoning, but it is computationally expensive (audio tokens are much larger than text tokens) and evaluation is challenging (how do you judge the quality of audio token references?).</p>

<h4>Comparing Reasoning Approaches: Empirical Results</h4>

<p>Across published results, the reasoning approaches show consistent tradeoffs. Sound-CoT reports 5-12% accuracy improvements on audio QA benchmarks compared to direct-answer baselines, with the largest gains on complex multi-step questions ("What is happening in the background while the main speaker is talking, and what does it suggest about the location?"). AudSemThinker reports lower hallucination rates (15% reduction in unfounded claims about audio content) but smaller accuracy gains (3-7%), suggesting structured reasoning trades off peak performance for reliability. UALM-Reason reports the best results on fine-grained acoustic discrimination tasks (instrument identification, speaker age estimation) but at 3-5x higher inference cost due to the mixed token generation.</p>

<p>A practical finding across all approaches: reasoning provides the largest gains when the question requires integrating information from multiple parts of the audio, or when the answer requires inference beyond surface-level recognition. For simple classification tasks ("Is this a dog barking?"), reasoning adds latency without improving accuracy. This suggests adaptive reasoning depth &mdash; using reasoning only when the model's confidence on direct answers is below a threshold &mdash; as a promising practical direction.</p>

<h4>The Reasoning Substrate Debate</h4>

<p>The fundamental question in audio reasoning research: <strong>Should AudioLLMs reason in text, audio tokens, or a hybrid?</strong></p>

<table>
<tr><th>Approach</th><th>Advantages</th><th>Disadvantages</th><th>Best For</th></tr>
<tr><td><strong>Text-only reasoning</strong> (Sound-CoT)</td><td>Fast; leverages mature text reasoning; easy to evaluate; interpretable</td><td>Loses paralinguistic info (pitch, timbre, timing); may hallucinate audio details</td><td>Audio QA, captioning, classification</td></tr>
<tr><td><strong>Structured semantic</strong> (AudSemThinker)</td><td>Grounded in detected events; reduces hallucination; interpretable</td><td>Limited by event detection accuracy; rigid structure; can't capture nuance</td><td>Sound event reasoning, scene analysis</td></tr>
<tr><td><strong>Mixed text+audio</strong> (UALM-Reason)</td><td>Preserves acoustic detail; flexible; handles nuance</td><td>Expensive (10-50x more tokens); hard to evaluate; difficult to train</td><td>Fine-grained acoustic analysis, music</td></tr>
<tr><td><strong>Pure audio token</strong></td><td>Maximum information preservation; no modality conversion loss</td><td>Extremely expensive; evaluation undefined; no interpretability; not aligned with human cognition</td><td>Theoretical interest only (currently)</td></tr>
</table>

<h4>Detailed Experimental Design for Comparing Reasoning Substrates</h4>

<p>To rigorously compare reasoning substrates, we propose the following experimental protocol:</p>

<p><strong>Controlled Variables:</strong></p>
<ul>
<li>Same base AudioLLM (Qwen2-Audio-7B)</li>
<li>Same audio encoder (Whisper-large-v3, frozen)</li>
<li>Same training data (100K examples, same audio clips)</li>
<li>Same compute budget (measured in GPU-hours, not steps)</li>
<li>Same evaluation suite (AudioBench + adversarial audio neglect probes)</li>
</ul>

<p><strong>Independent Variable: Reasoning Substrate</strong></p>
<ol>
<li><strong>Baseline:</strong> Direct answer, no explicit reasoning</li>
<li><strong>Text CoT:</strong> Sound-CoT style free-form text reasoning</li>
<li><strong>Structured:</strong> AudSemThinker-style structured representation then reasoning</li>
<li><strong>Hybrid:</strong> Text reasoning with audio token anchors at 3-5 key points</li>
</ol>

<p><strong>Dependent Variables:</strong></p>
<ul>
<li>Task accuracy across all AudioBench categories</li>
<li>Audio neglect rate (adversarial probes)</li>
<li>Reasoning faithfulness (do reasoning chains reference actual audio content?)</li>
<li>Inference latency (time to generate answer)</li>
<li>Token efficiency (total tokens generated per answer)</li>
</ul>

<p><strong>Hypothesis:</strong> Text CoT will win on efficiency and most tasks; Hybrid will win on tasks requiring fine-grained acoustic discrimination; Structured will have the lowest audio neglect rate; Baseline will be fastest but least accurate on complex questions.</p>

<h4>Training Recipe: SFT -> Audio Thought Injection -> GRPO Refinement</h4>

<pre><code># Complete training recipe for audio reasoning (Sound-CoT style)
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

# === Stage 1: SFT on CoT Data ===
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", torch_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
# Trainable params: ~80M out of 7B total

# SFT training config
sft_config = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 32,  # Effective batch size: 128
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 500,
    "max_grad_norm": 1.0,
}

# === Stage 2: Audio Thought Injection ===
# Add special tokens to tokenizer
special_tokens = {
    "additional_special_tokens": ["<audio_thought>", "</audio_thought>"]
}
processor.tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(processor.tokenizer))

# Train with thought-token weighted loss
# Increase loss weight on tokens between <audio_thought> tags by 2x
# This encourages the model to generate reasoning before answering
# LR: 1e-5, 2 epochs, same batch size

# === Stage 3: GRPO Refinement ===
def reward_fn(predictions, references):
    """
    Combined reward: accuracy (0-1) + reasoning quality (0-1)
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        # Answer accuracy: does the final answer match?
        answer_correct = float(ref["answer"].lower() in pred.lower())

        # Reasoning quality: scored by critic model (simplified here)
        has_reasoning = "<audio_thought>" in pred and "</audio_thought>" in pred
        reasoning_score = 0.5 if has_reasoning else 0.0

        # Combined reward
        reward = 0.7 * answer_correct + 0.3 * reasoning_score
        rewards.append(reward)
    return rewards

grpo_config = GRPOConfig(
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_generations=4,        # Generate 4 responses per prompt
    max_new_tokens=512,
    temperature=0.8,          # Higher temperature for diverse generations
    kl_coef=0.05,            # KL penalty coefficient
    bf16=True,
)

# trainer = GRPOTrainer(
#     model=model,
#     config=grpo_config,
#     reward_fn=reward_fn,
#     train_dataset=grpo_dataset,  # 50K high-quality examples
# )
# trainer.train()</code></pre>

<h4>Open Problems in Audio Reasoning</h4>

<p>Several fundamental problems remain unsolved in audio reasoning research:</p>

<ul>
<li><strong>Faithfulness verification:</strong> How do you verify that a reasoning chain about audio is faithful to the actual audio content? For text, you can check factual claims against sources. For audio, there is no equivalent "ground truth" to check against. A model might produce a compelling-sounding reasoning chain ("I detect a rising pitch contour suggesting surprise") that has no grounding in its actual feature processing.</li>
<li><strong>Temporal reasoning:</strong> Current reasoning approaches struggle with questions that require precise temporal reasoning ("Did the dog bark before or after the door closed?"). Audio events often overlap and have fuzzy boundaries, making temporal ordering difficult even for humans.</li>
<li><strong>Multi-source reasoning:</strong> Real-world audio scenes contain multiple simultaneous sound sources. Reasoning about the relationship between sources ("The speaker sounds nervous because the background noise suggests they are in a public place") requires source separation or at least source-aware feature extraction, which current AudioLLMs handle poorly.</li>
<li><strong>Calibration:</strong> AudioLLMs with reasoning are poorly calibrated &mdash; they express high confidence even when their reasoning is wrong. Developing calibrated audio reasoning (knowing when you do not know) is essential for safety-critical deployments.</li>
</ul>

<div class="callout">
<div class="callout-title">Key Insight: The Quality-Faithfulness Tradeoff</div>
<p>Auto-generated CoT reasoning can improve answer accuracy while producing unfaithful reasoning chains. A model might correctly identify birdsong but generate a reasoning chain about "frequency analysis" that doesn't reflect its actual processing. For safety-critical applications (medical auscultation, industrial monitoring), faithfulness is as important as accuracy. Measure both.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you implement chain-of-thought reasoning for an audio classification system, and what challenges would you expect?</div>
<div class="a-text">Implementation approach: (1) Generate CoT training data using a teacher model &mdash; for each audio-label pair, generate a reasoning chain explaining how to identify the sound (e.g., "I hear a repetitive high-pitched chirping pattern at approximately 2-4 kHz, consistent with bird calls. The pattern repeats every 0.5 seconds, suggesting a songbird rather than a raptor."). (2) Fine-tune the AudioLLM with LoRA on these CoT examples using SFT. (3) Optionally refine with GRPO using a reward that weights both final classification accuracy and reasoning chain quality. Key challenges: (a) Faithfulness &mdash; the generated reasoning may not reflect actual model processing; need to verify with ablation studies (mask the audio features referenced in CoT and check if accuracy drops). (b) Latency &mdash; CoT generates 50-200 extra tokens before the answer, adding 0.5-2 seconds of latency; for real-time applications, consider "thinking tokens" that are generated but not shown to the user. (c) Evaluation &mdash; no standard metrics for audio reasoning quality; need human evaluation or LLM-as-judge. (d) Training data quality &mdash; teacher model may generate incorrect reasoning about audio since it doesn't actually hear the audio, only reads descriptions; need careful filtering.</div>
</div>
`
    }
  ],

  // ============================================================
  // CHAPTER 2: Speech-to-Speech Models
  // ============================================================
  ch2_sections: [
    {
      id: "s2s-taxonomy",
      title: "Architecture Taxonomy",
      content: `
<p>Speech-to-Speech (S2S) models can be classified by their architecture type and interaction capabilities. The taxonomy has evolved significantly as the field moves from simple pipeline systems to increasingly sophisticated end-to-end architectures. Understanding this taxonomy is essential for making architectural decisions in production.</p>

<table>
<tr><th>Type</th><th>Description</th><th>Examples</th><th>Typical Latency</th></tr>
<tr><td><strong>Cascade Pipeline</strong></td><td>ASR -> LLM -> TTS as separate modules</td><td>X-Talk, GPT-4 + Whisper + TTS</td><td>500-2000ms</td></tr>
<tr><td><strong>Half-Duplex E2E</strong></td><td>Listen-then-speak; single LLM backbone; no simultaneous I/O</td><td>LLaMA-Omni, Mini-Omni, SpeechGPT</td><td>200-500ms</td></tr>
<tr><td><strong>Full-Duplex E2E</strong></td><td>Simultaneous listening and speaking; continuous stream processing</td><td>Moshi, OmniFlatten, LSLM</td><td>160-300ms</td></tr>
<tr><td><strong>Omni Models</strong></td><td>Multi-modal input/output including text, speech, images</td><td>Qwen2.5-Omni, GPT-4o</td><td>300-800ms</td></tr>
<tr><td><strong>Infrastructure</strong></td><td>Datasets, evaluation frameworks, TTS components</td><td>VoiceAssistant-400K, VoiceBench</td><td>N/A</td></tr>
<tr><td><strong>Alignment/Safety</strong></td><td>Preference alignment, adversarial robustness</td><td>SpeechAlign</td><td>N/A</td></tr>
</table>

<h4>Architecture Comparison: ASCII Diagrams</h4>

<p><strong>Type 1: Cascade Pipeline</strong></p>
<pre><code>
    User Speech        ASR Module           LLM              TTS Module        System Speech
   ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │ Waveform │───>│ Whisper /    │───>│ Text     │───>│ CosyVoice / │───>│ Waveform │
   │ (16kHz)  │    │ Conformer    │    │ Response │    │ F5-TTS      │    │ (24kHz)  │
   └──────────┘    │              │    │          │    │             │    └──────────┘
                   │ Output: Text │    │ Input:   │    │ Input: Text │
                   │ (WER ~2-5%) │    │ Text     │    │ Output:     │
                   └──────────────┘    │ Output:  │    │ Waveform    │
                                       │ Text     │    └──────────────┘
                                       └──────────┘

   Latency: ASR (200-500ms) + LLM (200-1000ms) + TTS (200-500ms) = 600-2000ms total
   Advantage: Each module can be independently optimized and swapped
   Disadvantage: Error propagation (ASR errors corrupt LLM input); high total latency
</code></pre>

<p><strong>Type 2: Half-Duplex End-to-End</strong></p>
<pre><code>
    User Speech               Unified Model                    System Speech
   ┌──────────┐    ┌──────────────────────────────┐    ┌──────────┐
   │ Waveform │───>│  ┌─────────┐                 │    │ Waveform │
   │          │    │  │ Audio   │  ┌─────────┐    │───>│          │
   └──────────┘    │  │ Encoder │─>│ Adapter │    │    └──────────┘
                   │  └─────────┘  └────┬────┘    │
                   │                    │         │
                   │              ┌─────▼─────┐   │
                   │              │   LLM     │   │
                   │              │ Backbone  │   │
                   │              └─────┬─────┘   │
                   │                    │         │
                   │              ┌─────▼─────┐   │
                   │              │  Speech   │   │
                   │              │  Decoder  │   │
                   │              └───────────┘   │
                   └──────────────────────────────┘

   Data flow: Audio -> Encoder -> Adapter -> LLM -> Speech Decoder -> Audio
   LLM generates text tokens AND speech tokens (alternating or parallel)
   Model WAITS for user to finish speaking before responding
   Latency: 200-500ms (all components share a single forward pass pipeline)
</code></pre>

<p><strong>Type 3: Full-Duplex End-to-End (Moshi-style)</strong></p>
<pre><code>
                    ┌─────────────────────────────────────────────────┐
                    │               Moshi Architecture                │
   User Audio ─────>│  ┌──────────────────────────────────────┐      │
   (continuous      │  │         Mimi Codec Encoder           │      │
    stream)         │  └─────────────┬────────────────────────┘      │
                    │                │ User audio tokens              │
                    │                ▼                                │
                    │  ┌──────────────────────────────────────┐      │
                    │  │     Depth Transformer (per-step)     │      │──── System Audio
                    │  │  ┌────────────────────────────┐      │      │     (continuous
                    │  │  │  Temporal Transformer      │      │      │      stream)
                    │  │  │  (across time steps)       │      │      │
                    │  │  │                            │      │      │
                    │  │  │  User tokens ──┐           │      │      │
                    │  │  │  Model tokens ─┤─> Causal  │      │      │
                    │  │  │  Text tokens ──┘   Attn    │      │      │
                    │  │  └────────────────────────────┘      │      │
                    │  └──────────────────────────────────────┘      │
                    │                │ Model audio tokens             │
                    │                ▼                                │
                    │  ┌──────────────────────────────────────┐      │
                    │  │         Mimi Codec Decoder           │──────┘
                    │  └──────────────────────────────────────┘
                    └─────────────────────────────────────────────────┘

   Key: BOTH user and model audio streams are processed SIMULTANEOUSLY
   The model can listen while speaking (true full-duplex)
   Inner Monologue: text tokens are generated alongside audio tokens
   Latency: ~200ms (one codec frame = 80ms processing window)
</code></pre>

<h4>The Cascade vs. End-to-End Debate</h4>

<p>The field has oscillated on whether cascade or end-to-end systems are superior. As of 2025, the evidence is nuanced:</p>

<table>
<tr><th>Dimension</th><th>Cascade Pipeline</th><th>End-to-End</th></tr>
<tr><td><strong>Content quality</strong></td><td>Better (LLM sees clean text)</td><td>Worse (audio tokens less information-dense)</td></tr>
<tr><td><strong>Voice quality</strong></td><td>Better (dedicated TTS)</td><td>Improving but still inferior</td></tr>
<tr><td><strong>Latency</strong></td><td>Worse (cumulative delay)</td><td>Better (single-pass potential)</td></tr>
<tr><td><strong>Emotional expression</strong></td><td>Worse (emotion lost in ASR)</td><td>Better (direct audio-to-audio mapping)</td></tr>
<tr><td><strong>Interruption handling</strong></td><td>Possible but hacky (VAD-based)</td><td>Natural (full-duplex architectures)</td></tr>
<tr><td><strong>Modularity</strong></td><td>Excellent (swap any component)</td><td>Poor (monolithic)</td></tr>
<tr><td><strong>Debugging</strong></td><td>Easy (inspect text at each stage)</td><td>Hard (opaque audio tokens)</td></tr>
<tr><td><strong>Data requirements</strong></td><td>Lower (leverage existing modules)</td><td>Higher (need paired speech-to-speech data)</td></tr>
</table>

<p>X-Talk (2025) demonstrated that optimized cascades with streaming ASR + fast LLM + streaming TTS can achieve sub-500ms latency while maintaining superior content quality. The practical recommendation: <strong>start with a cascade pipeline for new projects, then move to end-to-end only when latency or emotional expressiveness becomes the primary bottleneck.</strong></p>

<h4>Audio Token Vocabularies Across Architectures</h4>

<p>A key differentiator between S2S architectures is how they represent audio internally. The choice of audio tokenization determines information density, sequence length, and generation quality:</p>

<table>
<tr><th>Approach</th><th>Token Rate</th><th>Vocab Size</th><th>Info per Token</th><th>Generation Quality</th></tr>
<tr><td><strong>EnCodec tokens (8 codebooks)</strong></td><td>75 Hz x 8 = 600 tokens/s</td><td>1024 per codebook</td><td>Low (each code captures one quantization level)</td><td>Good with all codebooks; poor with first only</td></tr>
<tr><td><strong>Mimi tokens (8 codebooks)</strong></td><td>12.5 Hz x 8 = 100 tokens/s</td><td>2048 per codebook</td><td>High (80ms of audio per frame)</td><td>Good; semantic codebook enables meaningful first-pass</td></tr>
<tr><td><strong>HuBERT discrete units</strong></td><td>50 Hz</td><td>500-2000 (k-means clusters)</td><td>Semantic only (no acoustic detail)</td><td>Requires separate vocoder; preserves content, loses speaker identity</td></tr>
<tr><td><strong>Continuous representations (VAE)</strong></td><td>50-100 Hz</td><td>N/A (continuous)</td><td>Highest (no quantization loss)</td><td>Best quality but cannot use standard LM decoding</td></tr>
</table>

<p>The sequence length implications are significant. For a 10-second audio clip: EnCodec produces 6,000 tokens (impractical for an LLM), Mimi produces 1,000 tokens (manageable), and HuBERT produces 500 tokens (efficient but lossy). This is why Moshi's choice of Mimi at 12.5 Hz is architecturally critical &mdash; it makes full-duplex processing feasible within a transformer's attention budget.</p>

<h4>Emerging Architectures (2025)</h4>

<p>Several new architectures are pushing beyond the categories described above:</p>

<ul>
<li><strong>LSLM (Listening while Speaking LM):</strong> Uses a token-based turn-taking detector that determines when to yield the floor. Unlike Moshi's continuous generation, LSLM makes an explicit binary decision at each time step: "continue speaking" or "yield and listen." This simplifies training but produces less natural turn-taking.</li>
<li><strong>Qwen2.5-Omni Thinker-Talker:</strong> The "Thinker" is a full-size LLM that generates text reasoning. The "Talker" is a lightweight streaming module that converts text tokens to speech tokens in real-time. This separation allows the system to maintain text-level reasoning quality while achieving low latency speech output. The Talker can produce speech faster than real-time because it operates on pre-generated text.</li>
<li><strong>GLM-4-Voice:</strong> Uses a unified tokenizer that maps both text and speech into the same vocabulary space. The model does not distinguish between text and speech tokens at the transformer level &mdash; it operates on a single interleaved sequence. This radical simplification reduces architectural complexity but requires massive training data to learn the implicit text-speech mapping.</li>
</ul>

<div class="callout">
<div class="callout-title">The Hybrid Trend</div>
<p>The most promising 2025 architectures are hybrids. Qwen2.5-Omni's "Thinker-Talker" architecture uses an end-to-end model where the "Thinker" generates text reasoning (like a cascade's LLM stage) and the "Talker" converts to speech (like a cascade's TTS stage), but both are part of a single model trained jointly. This gets the content quality of cascades with the latency benefits of end-to-end.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What are the key trade-offs between half-duplex and full-duplex speech-to-speech systems?</div>
<div class="a-text">Half-duplex systems (like LLaMA-Omni) are simpler &mdash; they listen, then speak. Benefits: easier to train, can leverage existing LLMs, more stable, better content quality because the model sees the complete input before responding. Drawbacks: no interruption handling, unnatural conversation flow, user must wait for system to finish speaking. Full-duplex systems (like Moshi) can listen and speak simultaneously. Benefits: natural conversation, barge-in support, backchanneling ("uh-huh"), real-time adaptation to user feedback. Drawbacks: much harder to train (need dual-stream architecture), higher compute cost (doubled sequence length since both streams are modeled), content quality often suffers due to the added complexity of simultaneous generation, and training data requirements are higher (need real conversational data, not just monologue). The practical consideration: most current voice assistants (Siri, Alexa) use half-duplex with VAD-based interruption as a "good enough" compromise. True full-duplex matters most for natural conversation applications like companionship AI, therapy bots, and real-time interpretation.</div>
</div>
`
    },
    {
      id: "s2s-latency",
      title: "Latency & Real-Time Considerations",
      content: `
<p>For natural conversation, response latency must be under ~300ms. Human conversational turn-taking typically involves gaps of 200-300ms; anything over 500ms feels noticeably sluggish, and over 1000ms feels broken. Here's how key systems compare and how to measure latency rigorously.</p>

<table>
<tr><th>System</th><th>Latency</th><th>Architecture</th><th>Trade-off</th></tr>
<tr><td>Moshi</td><td>200ms</td><td>Full-duplex, RQ-Transformer</td><td>Content quality lags text LLMs</td></tr>
<tr><td>LLaMA-Omni</td><td>226ms</td><td>Half-duplex, Whisper+LLaMA</td><td>No interruption handling</td></tr>
<tr><td>Mini-Omni</td><td>~300ms</td><td>Think-while-speaking</td><td>0.5B too small for knowledge tasks</td></tr>
<tr><td>SyncLLM</td><td>Variable</td><td>Time-slotted FD</td><td>Fixed-chunk limits prosody</td></tr>
<tr><td>GPT-4o (voice)</td><td>~320ms</td><td>End-to-end (proprietary)</td><td>Best quality but closed-source</td></tr>
<tr><td>Cascade (optimized)</td><td>400-600ms</td><td>Streaming ASR+LLM+TTS</td><td>Best content quality, highest latency</td></tr>
<tr><td>Cascade (naive)</td><td>1000-2000ms</td><td>Batch ASR+LLM+TTS</td><td>Simple to build, too slow for conversation</td></tr>
</table>

<h4>Latency Measurement Methodology</h4>

<p>Latency in S2S systems is not a single number. You must measure multiple latency components and understand what each means:</p>

<p><strong>1. Time-to-First-Audio-Byte (TTFAB):</strong> Time from when the user stops speaking until the system begins outputting audio. This is the most perceptually important metric &mdash; it determines whether the conversation feels responsive. Measured by detecting the end of user speech (via VAD) and the start of system audio output (via energy threshold).</p>

<p><strong>2. End-to-End Latency (E2E):</strong> Time from the end of user speech to the end of system speech. Less perceptually important than TTFAB but matters for throughput.</p>

<p><strong>3. Interruption Response Time (IRT):</strong> For full-duplex systems, time from when the user starts speaking during system output until the system acknowledges the interruption (by stopping or modifying its output). Measured by having a scripted user interrupt at specific points and detecting when the system responds.</p>

<p><strong>4. Processing Latency:</strong> The computational time for each component, independent of network latency. Measured in controlled environments with local inference.</p>

<h4>Measurement Setup</h4>

<pre><code># Rigorous S2S latency measurement
import numpy as np
import time
import sounddevice as sd
import webrtcvad

class S2SLatencyBenchmark:
    def __init__(self, system_endpoint, sample_rate=16000):
        self.endpoint = system_endpoint
        self.sr = sample_rate
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3

    def measure_ttfab(self, test_utterance_audio, n_trials=50):
        """
        Measure Time-to-First-Audio-Byte across multiple trials.
        Returns mean, p50, p95, p99 latencies in milliseconds.
        """
        latencies = []

        for _ in range(n_trials):
            # Send audio to system
            t_send = time.perf_counter()
            response_stream = self.endpoint.send_audio(test_utterance_audio)

            # Detect first non-silence frame in response
            for chunk in response_stream:
                if self._has_speech(chunk):
                    t_first_audio = time.perf_counter()
                    latency_ms = (t_first_audio - t_send) * 1000

                    # Subtract utterance duration to get response latency
                    utterance_duration_ms = len(test_utterance_audio) / self.sr * 1000
                    ttfab = latency_ms - utterance_duration_ms
                    latencies.append(ttfab)
                    break

        return {
            "mean": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "std": np.std(latencies),
            "n_trials": n_trials,
        }

    def measure_interruption_response(self, system_speaking_audio,
                                        interrupt_audio, n_trials=30):
        """
        For full-duplex systems: measure how quickly the system
        responds to user interruption.
        """
        response_times = []

        for _ in range(n_trials):
            # Start system speaking
            stream = self.endpoint.send_audio(system_speaking_audio)

            # Wait for system to start speaking
            time.sleep(1.0)

            # Send interrupt
            t_interrupt = time.perf_counter()
            self.endpoint.send_interrupt(interrupt_audio)

            # Detect when system stops or modifies output
            for chunk in stream:
                if self._is_silence(chunk) or self._is_modified(chunk):
                    t_response = time.perf_counter()
                    irt = (t_response - t_interrupt) * 1000
                    response_times.append(irt)
                    break

        return {
            "mean_irt": np.mean(response_times),
            "p95_irt": np.percentile(response_times, 95),
        }

    def _has_speech(self, audio_chunk):
        """Check if audio chunk contains speech using WebRTC VAD."""
        # VAD expects 10, 20, or 30ms frames at 8/16/32/48kHz
        frame_duration = 30  # ms
        frame_size = int(self.sr * frame_duration / 1000) * 2  # 16-bit
        if len(audio_chunk) >= frame_size:
            return self.vad.is_speech(audio_chunk[:frame_size], self.sr)
        return False

    def _is_silence(self, audio_chunk):
        return not self._has_speech(audio_chunk)</code></pre>

<h4>Latency Budget Allocation</h4>

<p>For a cascade system targeting 500ms total TTFAB, here is how to allocate the budget:</p>

<table>
<tr><th>Component</th><th>Budget (ms)</th><th>Optimization Strategy</th></tr>
<tr><td>VAD + Endpointing</td><td>50-100</td><td>WebRTC VAD with 300ms silence threshold; aggressive endpointing reduces latency but increases false triggers</td></tr>
<tr><td>Streaming ASR</td><td>100-150</td><td>Use streaming Whisper or Conformer-CTC; process in 200ms chunks with overlap</td></tr>
<tr><td>LLM Inference (TTFT)</td><td>100-200</td><td>Speculative decoding; small model (7-13B); KV-cache; quantization (AWQ/GPTQ)</td></tr>
<tr><td>TTS Synthesis</td><td>50-100</td><td>Streaming TTS; start synthesis before full text is generated; VITS or CosyVoice streaming mode</td></tr>
<tr><td>Network + Buffer</td><td>20-50</td><td>Edge deployment; WebSocket connection reuse; minimal audio buffering</td></tr>
<tr><td><strong>Total</strong></td><td><strong>320-600ms</strong></td><td></td></tr>
</table>

<h4>Key Design Decisions for Low-Latency S2S</h4>
<ul>
<li><strong>Codec choice:</strong> Discrete codecs (EnCodec, Mimi) enable fast token-by-token generation but quantization loses acoustic detail. Continuous representations (SALMONN-Omni) preserve quality but are harder to stream. Mimi achieves 1.1kbps with 8 codebooks, providing a good balance.</li>
<li><strong>Streaming architecture:</strong> Chunk-based processing with overlapping windows. The chunk size creates a latency floor. Moshi uses 80ms frames (12.5 fps), meaning the theoretical minimum latency is 80ms plus processing time.</li>
<li><strong>KV-cache management:</strong> For long conversations, efficient KV-cache pruning or compression is essential. A 30-minute conversation at 12.5 audio tokens/second produces 22,500 tokens; with GQA and 32 layers, the KV cache grows to ~3.6GB for a 7B model.</li>
<li><strong>Speculative generation:</strong> Pre-generate likely responses based on partial input. For common queries ("What's the weather?"), the system can start generating before the user finishes speaking.</li>
<li><strong>Model quantization:</strong> INT4 quantization (AWQ or GPTQ) reduces LLM inference latency by 30-50% with minimal quality loss. For the LLM component of a cascade pipeline, this is often the single highest-impact optimization. Combine with CUDA graphs for reduced kernel launch overhead.</li>
<li><strong>Prefill optimization:</strong> The audio tokens in the LLM prompt are fixed once the user finishes speaking. Use chunked prefill to process them efficiently, and cache the KV states for the system prompt across sessions (prefix caching in vLLM).</li>
</ul>

<h4>Real-World Latency Profiles</h4>

<p>Latency in production varies significantly based on the input characteristics. Here are typical profiles measured on real systems:</p>

<table>
<tr><th>Scenario</th><th>Input Length</th><th>TTFAB (p50)</th><th>TTFAB (p95)</th><th>Bottleneck</th></tr>
<tr><td>Short command ("Set a timer")</td><td>1-2s</td><td>250ms</td><td>400ms</td><td>Endpointing (waiting for silence)</td></tr>
<tr><td>Question ("What's the capital of France?")</td><td>2-4s</td><td>350ms</td><td>550ms</td><td>LLM inference</td></tr>
<tr><td>Complex query (multi-sentence)</td><td>5-15s</td><td>500ms</td><td>900ms</td><td>ASR processing + LLM</td></tr>
<tr><td>Long narrative (story telling)</td><td>30-60s</td><td>800ms</td><td>1500ms</td><td>ASR (large audio buffer) + LLM (long context)</td></tr>
</table>

<p>The key insight: latency is not constant. Short inputs are bottlenecked by endpointing (the system waits for silence to confirm the user has finished), while long inputs are bottlenecked by ASR processing time and LLM prefill on a long context. Optimization strategies should be adapted accordingly: for short-input scenarios, focus on aggressive endpointing; for long-input scenarios, focus on streaming ASR and incremental LLM processing.</p>

<div class="callout warning">
<div class="callout-title">War Story: The 80ms That Broke Everything</div>
<p>A team building a real-time voice assistant had achieved 350ms TTFAB in testing but measured 800ms in production. The culprit: their audio pipeline used a 256-sample buffer (16ms at 16kHz) for VAD, but the WebSocket library they used had an internal 512ms send buffer that accumulated small audio chunks before transmitting. The fix was to configure the WebSocket with <code>TCP_NODELAY</code> and reduce the send buffer to 4KB. Lesson: latency optimization must cover the entire pipeline including I/O buffers, not just model inference time. Always measure end-to-end latency in production-like conditions, not just component latencies in isolation.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you reduce end-to-end latency in a speech dialogue system from 1 second to under 400ms?</div>
<div class="a-text">The 1-second latency suggests a naive cascade pipeline. To reach 400ms: (1) Replace batch ASR with streaming ASR &mdash; process audio in 200ms chunks instead of waiting for complete utterances, saving 300-500ms. (2) Implement streaming TTS that starts synthesizing from partial LLM output &mdash; the TTS begins speaking the first sentence while the LLM is still generating the second, overlapping TTS with LLM computation. (3) Optimize the LLM: use speculative decoding (2-3x speedup), INT4 quantization (AWQ), and ensure CUDA graphs are enabled for reduced kernel launch overhead. Target 100-150ms time-to-first-token. (4) Optimize endpointing: reduce the silence threshold from typical 700ms to 300ms, accepting slightly more false positives for faster response. (5) Use WebSocket with TCP_NODELAY instead of HTTP for lower network overhead. (6) If possible, deploy the model on the edge device to eliminate network round-trip entirely. Each optimization targets a specific component: endpointing (100ms saved), streaming ASR (300ms), LLM optimization (200ms), streaming TTS (200ms), network (50ms). Combined, this takes a 1200ms system to under 400ms.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Design the latency monitoring system for a production voice assistant serving 10,000 concurrent users.</div>
<div class="a-text">I would implement a multi-layer monitoring system: (1) Client-side instrumentation: Measure perceived latency on the user's device from end-of-speech to first-audio-byte using Web Audio API timestamps. Report as histograms (p50, p95, p99) bucketed by device type, network quality, and region. (2) Server-side component tracing: Add OpenTelemetry spans for each pipeline stage (VAD, ASR, LLM, TTS). Each span records start time, end time, input size, and output size. This enables identifying which component is the bottleneck for slow requests. (3) Real-time dashboards: Display p50/p95/p99 latency per component in Grafana with 1-minute granularity. Set alerts at p95 > 500ms and p99 > 800ms. (4) Tail-latency investigation: For p99+ requests, log the full audio input and model outputs to enable offline replay and debugging. Common causes: long inputs, unusual accents causing ASR retries, LLM generating unusually long responses. (5) Capacity planning: Track latency vs. concurrent users curves. With 10K concurrent users, GPU utilization will be the primary concern; use dynamic batching in the LLM component and autoscaling on GPU instances triggered at 80% utilization.</div>
</div>
`
    },
    {
      id: "moshi-deep-dive",
      title: "Moshi Architecture Deep Dive",
      content: `
<p>Moshi (Kyutai, 2024; arXiv:2410.00037) is the first genuinely real-time, full-duplex spoken language model. It can listen and speak simultaneously, handle interruptions naturally, and maintain coherent conversation &mdash; all with approximately 200ms latency. Understanding Moshi's architecture in depth is essential because it represents the current frontier of S2S systems and introduces several novel techniques that are being adopted by subsequent work.</p>

<h4>Mimi: The Neural Audio Codec</h4>

<p>Moshi's foundation is Mimi, a custom neural audio codec that compresses audio to just 1.1 kbps &mdash; lower than any previous codec while maintaining high perceptual quality. Mimi's key innovation is splitting semantic and acoustic information across codebook levels.</p>

<p><strong>Codec Architecture:</strong></p>
<pre><code>
Input: 24kHz mono audio waveform
  │
  ▼
┌─────────────────────────────────┐
│   SEANet Encoder                │
│   (Strided convolutions)        │
│   Input: 24kHz waveform         │
│   Output: 12.5 Hz features      │  ← 1920x downsampling
│   (one frame per 80ms)          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Residual Vector Quantizer     │
│   8 codebooks, 2048 entries each│
│                                 │
│   Codebook 1: Semantic          │  ← Distilled from WavLM
│     (content, speaker ID)       │
│   Codebooks 2-8: Acoustic       │  ← Fine-grained audio details
│     (timbre, noise, prosody)    │
│                                 │
│   Total: 8 codes × 11 bits     │
│   = 88 bits per frame           │
│   At 12.5 Hz: 1100 bps ≈ 1.1kbps│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   SEANet Decoder                │
│   (Transposed convolutions)     │
│   Output: 24kHz waveform        │
└─────────────────────────────────┘
</code></pre>

<p><strong>Semantic-Acoustic Split.</strong> The first codebook is trained with an auxiliary distillation loss from WavLM (a self-supervised speech model). This forces the first codebook to capture semantic content &mdash; what is being said, who is speaking &mdash; while subsequent codebooks capture acoustic residuals. This split is critical for the RQ-Transformer: the model can generate semantically meaningful audio by predicting just the first codebook, then fill in acoustic details with subsequent codebooks.</p>

<p><strong>Mimi vs. Other Codecs:</strong></p>
<table>
<tr><th>Codec</th><th>Bitrate</th><th>Codebooks</th><th>Frame Rate</th><th>Semantic-Acoustic Split</th><th>Used In</th></tr>
<tr><td>EnCodec (Meta)</td><td>1.5-24 kbps</td><td>2-32</td><td>75 Hz</td><td>No (emergent only)</td><td>VALL-E, MusicGen</td></tr>
<tr><td>Mimi (Kyutai)</td><td>1.1 kbps</td><td>8</td><td>12.5 Hz</td><td>Yes (distilled)</td><td>Moshi</td></tr>
<tr><td>SpeechTokenizer</td><td>Variable</td><td>8</td><td>50 Hz</td><td>Yes (by design)</td><td>Research</td></tr>
<tr><td>DAC</td><td>8 kbps</td><td>9</td><td>86 Hz</td><td>No</td><td>General audio</td></tr>
</table>

<p>Mimi's 12.5 Hz frame rate (one frame per 80ms) is dramatically lower than EnCodec's 75 Hz. This means the transformer must process far fewer tokens per second of audio &mdash; 12.5 vs. 75 &mdash; making real-time processing feasible. The tradeoff: each frame must encode 80ms of audio into 8 codes, requiring more information per code.</p>

<h4>RQ-Transformer: How Residual Quantization Works in the Transformer</h4>

<p>Moshi uses a novel RQ-Transformer (Residual Quantization Transformer) architecture with two nested transformers: a <strong>Temporal Transformer</strong> that models dependencies across time steps, and a <strong>Depth Transformer</strong> that models dependencies across codebook levels within each time step.</p>

<pre><code>
Time step:     t-2        t-1         t          t+1
              ┌───┐      ┌───┐      ┌───┐      ┌───┐
Codebook 1:   │ c1│─────>│ c1│─────>│ c1│─────>│ c1│    ← Temporal Transformer
              └─┬─┘      └─┬─┘      └─┬─┘      └─┬─┘      (across time)
                │           │          │           │
              ┌─▼─┐      ┌─▼─┐      ┌─▼─┐      ┌─▼─┐
Codebook 2:   │ c2│      │ c2│      │ c2│      │ c2│    ← Depth Transformer
              └─┬─┘      └─┬─┘      └─┬─┘      └─┬─┘      (within time step)
                │           │          │           │
              ┌─▼─┐      ┌─▼─┐      ┌─▼─┐      ┌─▼─┐
Codebook 3:   │ c3│      │ c3│      │ c3│      │ c3│
              └─┬─┘      └─┬─┘      └─┬─┘      └─┬─┘
               ...         ...        ...         ...
              ┌─▼─┐      ┌─▼─┐      ┌─▼─┐      ┌─▼─┐
Codebook 8:   │ c8│      │ c8│      │ c8│      │ c8│
              └───┘      └───┘      └───┘      └───┘

Generation order at each time step t:
  1. Temporal Transformer produces hidden state h_t from previous time steps
  2. Depth Transformer autoregressively generates c1, c2, ..., c8 conditioned on h_t
  3. Each codebook prediction is conditioned on all previous codebooks at this step
</code></pre>

<p><strong>Temporal Transformer:</strong> A large transformer (Moshi uses ~7B parameters) that processes the sequence of time steps. At each time step, it receives a summary vector (the sum of all 8 codebook embeddings from the previous step, plus the text token embedding) and produces a context vector. This transformer uses causal attention &mdash; each time step can only attend to previous steps.</p>

<p><strong>Depth Transformer:</strong> A smaller transformer (~300M parameters) that runs <em>within</em> each time step to generate the 8 codebook codes autoregressively. It takes the temporal transformer's output as conditioning and generates codes c1 through c8 one at a time. Because it only needs to generate 8 tokens per step (not hundreds), this is fast.</p>

<p>The critical insight: by factoring generation into temporal (across time) and depth (across codebooks) components, Moshi avoids the computational explosion of generating all tokens in a flat sequence. A naive approach would require modeling a sequence of 8 codes × 12.5 Hz × seconds = 100 tokens/second per audio stream. With two streams (user + model), that's 200 tokens/second plus text tokens. The RQ-Transformer's factored approach makes this tractable.</p>

<h4>Inner Monologue Mechanism</h4>

<p>Moshi's most innovative feature is the <strong>Inner Monologue</strong>: the model generates text tokens alongside audio tokens at each time step. These text tokens represent the model's "thoughts" &mdash; what it would say in text form &mdash; and are interleaved with audio token generation.</p>

<pre><code>
Time step:     t-1              t                t+1
            ┌─────────┐    ┌─────────┐    ┌─────────┐
Text:       │"Hello"  │───>│"how"    │───>│"are"    │    ← Inner monologue
            └────┬────┘    └────┬────┘    └────┬────┘
                 │              │              │
            ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
User audio: │ u_codes │    │ u_codes │    │ u_codes │    ← User's speech
            └────┬────┘    └────┬────┘    └────┬────┘
                 │              │              │
            ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
Model audio:│ m_codes │    │ m_codes │    │ m_codes │    ← Model's speech
            └─────────┘    └─────────┘    └─────────┘

At each time step, the model predicts:
  1. A text token (from vocabulary ~32K)
  2. 8 user audio codebook codes (for modeling user's speech)
  3. 8 model audio codebook codes (for generating model's speech)
Total: 1 + 8 + 8 = 17 predictions per time step
At 12.5 Hz: 17 × 12.5 = 212.5 tokens per second
</code></pre>

<p>The text tokens serve multiple purposes:</p>
<ol>
<li><strong>Higher-bandwidth reasoning channel:</strong> Text is more information-dense than audio tokens. A single text token can encode a concept that would require 10+ audio tokens to express. This makes text tokens the primary channel for semantic content, while audio tokens handle acoustic realization.</li>
<li><strong>Supervised training signal:</strong> Text transcripts are vastly more abundant than paired conversational audio. The inner monologue provides a dense training signal that accelerates learning. Without it, the model must learn everything from audio-only reconstruction, which is much harder.</li>
<li><strong>Interpretability:</strong> By reading the inner monologue, developers can understand what the model is "thinking" at each time step. This is invaluable for debugging: if the model produces incorrect speech, you can check whether the inner monologue was correct (indicating a text-to-speech failure) or incorrect (indicating a reasoning failure).</li>
<li><strong>Semantic anchoring:</strong> The text tokens anchor audio generation in linguistic meaning, reducing the chance of generating acoustically plausible but semantically nonsensical speech. Think of it as a guardrail that keeps audio generation on track.</li>
</ol>

<p><strong>Inner Monologue at inference time:</strong> During generation, the text tokens are produced first at each time step, then the audio tokens are conditioned on them. This means the model "decides what to say" (text) before "deciding how to say it" (audio). The text tokens can optionally be exposed to the user as real-time captions, providing a dual-modality output.</p>

<p>A subtle but important detail: the inner monologue operates at 12.5 Hz (one text token per 80ms frame), which is much slower than typical LLM generation rates (50-100 tokens/second). This means the model cannot express complex reasoning in the inner monologue &mdash; it is limited to approximately 2-3 words per second. This rate matches natural speaking rate, which is by design: the inner monologue transcribes what the model is saying, not what it is thinking about saying.</p>

<h4>Practical Implications of the RQ-Transformer Design</h4>

<p>The RQ-Transformer design has several important practical consequences that affect both training and deployment:</p>

<p><strong>Memory footprint.</strong> The temporal transformer (~7B params) dominates memory. At BF16, it requires ~14GB for weights alone. The depth transformer (~300M params) adds ~600MB. The KV cache for the temporal transformer grows with conversation length: at 12.5 Hz, a 5-minute conversation produces 3,750 time steps. With 32 layers, 32 heads, and 128-dim per head, the KV cache is: 3,750 x 32 x 2 x 32 x 128 x 2 bytes = ~1.5GB. This means Moshi can run on a single A100-40GB GPU for conversations up to ~20 minutes.</p>

<p><strong>Inference speed.</strong> At each time step (every 80ms), the model must: (1) run one temporal transformer forward pass, (2) run 8 depth transformer forward passes (one per codebook), (3) decode the 8 audio codes through the Mimi decoder. Steps 1 and 2 must complete within 80ms. On an A100 GPU, the temporal transformer step takes ~30ms and each depth step takes ~3ms (8 x 3 = 24ms), totaling ~54ms &mdash; comfortably within the 80ms budget. On consumer GPUs (RTX 4090), this drops to ~40ms + 20ms = 60ms, still feasible but with less headroom.</p>

<p><strong>Quality vs. speed at inference.</strong> You can trade quality for speed by reducing the number of codebooks used in generation. Using only codebooks 1-4 (instead of 1-8) halves the depth transformer steps and produces intelligible but lower-quality audio. This is useful for low-powered devices or when bandwidth is constrained.</p>

<h4>Training Data and Procedure</h4>

<p>Moshi's training follows a careful multi-stage procedure:</p>

<ol>
<li><strong>Text pretraining:</strong> The temporal transformer backbone is initialized from a text LLM pretrained on standard text corpora. This provides the reasoning capabilities.</li>

<li><strong>Codec training:</strong> Mimi is trained separately on a large audio corpus (~20K hours) with reconstruction loss + commitment loss + the WavLM distillation loss for the first codebook.</li>

<li><strong>Joint audio-text pretraining:</strong> The model is trained on paired audio-text data. The Fisher corpus (2,000 hours of telephone conversations with transcripts) is a key data source, supplemented with proprietary conversational data. Training uses a combined loss:
<pre><code>L_total = L_text + lambda_user * L_user_audio + lambda_model * L_model_audio
# lambda_user = 0.5, lambda_model = 1.0
# L_text: cross-entropy on text tokens
# L_user_audio: cross-entropy on user's audio codes (all 8 codebooks)
# L_model_audio: cross-entropy on model's audio codes (all 8 codebooks)</code></pre>
</li>

<li><strong>Instruction tuning:</strong> Fine-tuning on conversational instruction data with preference optimization to improve response quality and safety.</li>
</ol>

<h4>Code: Loading Moshi and Running Inference</h4>

<pre><code># Running Moshi inference (using the moshi Python package from Kyutai)
# pip install moshi

import torch
from moshi.models import loaders
from huggingface_hub import hf_hub_download

# Download model weights
model_path = hf_hub_download(
    repo_id="kyutai/moshiko-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors"
)
mimi_path = hf_hub_download(
    repo_id="kyutai/moshiko-pytorch-bf16",
    filename="mimi-e4cf21ec-checkpoint50.safetensors"
)

# Load Mimi codec
mimi = loaders.get_mimi(mimi_path, device="cuda")
mimi.set_num_codebooks(8)

# Load Moshi model
lm = loaders.get_moshi_lm(model_path, device="cuda")

# Encode user audio with Mimi
import torchaudio
wav, sr = torchaudio.load("user_speech.wav")
wav = torchaudio.functional.resample(wav, sr, mimi.sample_rate)  # 24kHz
wav = wav.unsqueeze(0).to("cuda")  # (1, 1, num_samples)

with torch.no_grad():
    user_codes = mimi.encode(wav)  # (1, 8, num_frames) at 12.5 Hz

print(f"Encoded {wav.shape[-1]/24000:.1f}s audio to {user_codes.shape[-1]} frames")
print(f"Bitrate: {user_codes.shape[1] * 11 * 12.5 / 1000:.1f} kbps")

# For full duplex inference, use the streaming API:
# from moshi.client import Client
# client = Client(host="localhost", port=8998)
# client.run()  # Opens bidirectional audio stream</code></pre>

<div class="callout">
<div class="callout-title">Key Numbers for Interviews</div>
<p><strong>Moshi Architecture:</strong> Temporal Transformer ~7B params, Depth Transformer ~300M params. Mimi codec: 12.5 Hz frame rate, 8 codebooks with 2048 entries each, 1.1 kbps bitrate, 80ms frame size. Total predictions per second: 212.5 tokens (1 text + 8 user audio + 8 model audio, all at 12.5 Hz). Theoretical minimum latency: 160ms (two codec frames). Measured latency: ~200ms. Training data: Fisher corpus (2K hours) + proprietary. Inner Monologue: text tokens generated at 12.5 Hz alongside audio tokens.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Explain the RQ-Transformer architecture in Moshi. Why use two nested transformers instead of one flat transformer?</div>
<div class="a-text">The RQ-Transformer uses a Temporal Transformer (~7B params) for dependencies across time and a Depth Transformer (~300M params) for dependencies across codebook levels within each time step. The flat alternative would model all tokens (text + 8 user codes + 8 model codes per step) in a single sequence. At 12.5 Hz, that is 212.5 tokens/second. For a 30-second conversation, the flat sequence would be 6,375 tokens, and attention is O(n^2). More critically, the flat approach conflates two types of dependencies: temporal coherence (what the model should say next) and acoustic structure (how to realize the current phoneme across codebook levels). These have very different characteristics &mdash; temporal dependencies are long-range and semantic, while codebook dependencies are local and acoustic. By factoring them, each transformer can be specialized: the temporal transformer is large (for reasoning) and the depth transformer is small (for fast acoustic generation of 8 codes). This factoring also enables streaming: the temporal transformer can run once per frame, and the depth transformer generates 8 codes quickly enough for real-time output.</div>
</div>
`
    },
    {
      id: "duplex-training",
      title: "Training Full-Duplex Systems",
      content: `
<p>Training a full-duplex speech-to-speech system is one of the hardest problems in modern AI engineering. The model must simultaneously process incoming user audio and generate coherent output audio, handle interruptions gracefully, maintain conversation context, and produce natural-sounding speech &mdash; all while running in real time. This section covers the major training approaches, with detailed loss functions and training pipelines.</p>

<h4>OmniFlatten: The 3-Stage Pipeline</h4>

<p>OmniFlatten (arXiv:2410.17799) solved the critical problem of converting a pretrained half-duplex text LLM into a full-duplex speech system without catastrophic forgetting. The key insight: use gradual modality adaptation across three stages, each building on the previous one.</p>

<p><strong>Stage 1: Text-Only Half-Duplex Training</strong></p>
<p>Start with a pretrained text LLM (e.g., LLaMA-3-8B). Train it on text conversation data in a standard instruction-following format. This stage preserves and refines the LLM's conversational abilities. No audio is involved.</p>
<pre><code># Stage 1 loss: standard next-token prediction on text
L_stage1 = CrossEntropyLoss(logits_text, target_text)
# LR: 2e-5, batch size: 128, steps: 10K
# Data: ShareGPT, Alpaca, custom conversation data</code></pre>

<p><strong>Stage 2: Speech Half-Duplex Training</strong></p>
<p>Introduce audio input and output. The model learns to understand speech input (via an audio encoder + adapter) and generate speech output (via a speech decoder). Training data consists of spoken conversations where one speaker talks, then the other responds (no overlapping speech).</p>
<pre><code># Stage 2 loss: text + audio generation
L_stage2 = L_text_response + alpha * L_audio_codes
# L_text_response: cross-entropy on text response tokens
# L_audio_codes: cross-entropy on audio codec tokens (each codebook)
# alpha = 1.0 (equal weight initially, may tune)

# Training recipe:
# - Freeze audio encoder (Whisper-large-v2)
# - Train adapter + LoRA on LLM + speech decoder
# - LR: 1e-5, batch size: 64, steps: 20K
# - Data: LibriTTS-R (585h), VCTK (44h), Fisher transcribed portions</code></pre>

<p><strong>Stage 3: Speech Full-Duplex Training with Stream Flattening</strong></p>
<p>The critical innovation. Stream flattening converts the two parallel audio streams (user speaking, model speaking) into a single interleaved sequence that a causal transformer can process.</p>

<pre><code>
Without stream flattening (parallel streams - cannot be processed by causal LM):

  User:    [u1] [u2] [u3] [u4] [u5] [u6] [u7] [u8]
  Model:   [--] [--] [--] [m1] [m2] [m3] [m4] [m5]

With stream flattening (interleaved - causal LM compatible):

  Sequence: [u1] [u2] [u3] [u4,m1] [u5,m2] [u6,m3] [u7,m4] [u8,m5]

  Implementation: At each time step where both are active,
  user tokens come first, then model tokens:
  [..., u_t, m_t, u_{t+1}, m_{t+1}, ...]
</code></pre>

<pre><code># Stage 3 loss: full-duplex with stream flattening
def full_duplex_loss(model_output, targets):
    # Separate user and model predictions from flattened sequence
    user_positions = targets["user_positions"]    # Boolean mask
    model_positions = targets["model_positions"]  # Boolean mask
    text_positions = targets["text_positions"]    # Boolean mask

    # Text loss (inner monologue)
    L_text = F.cross_entropy(
        model_output[text_positions],
        targets["text_tokens"][text_positions]
    )

    # Model audio generation loss (we want high quality here)
    L_model_audio = F.cross_entropy(
        model_output[model_positions],
        targets["model_audio_codes"][model_positions]
    )

    # User audio prediction loss (lower weight - just for context modeling)
    L_user_audio = F.cross_entropy(
        model_output[user_positions],
        targets["user_audio_codes"][user_positions]
    )

    # Combined loss with asymmetric weights
    return L_text + 1.0 * L_model_audio + 0.5 * L_user_audio

# Training recipe:
# - Train adapter + LoRA + speech decoder (keep encoder frozen)
# - LR: 5e-6 (lower than Stage 2 to prevent forgetting)
# - Batch size: 32 (longer sequences, more memory)
# - Steps: 30K
# - Data: Fisher conversational speech (2K hours, with overlapping segments)
# - Key: include 20% Stage 2 data to prevent half-duplex regression</code></pre>

<h4>SyncLLM: Time-Slotted Approach</h4>

<p>SyncLLM takes a different approach to full-duplex: instead of stream flattening, it divides time into fixed slots and the model alternates between listening and speaking at a fine granularity.</p>

<pre><code>
SyncLLM Time-Slotted Architecture:

Time:  |--- 160ms ---|--- 160ms ---|--- 160ms ---|--- 160ms ---|
User:  | Listen      | Listen      | Listen      | Listen      |
Model: | Generate    | Silence     | Generate    | Generate    |

  Each 160ms slot:
  - Model receives user audio tokens for this slot
  - Model decides: generate speech tokens OR remain silent
  - Decision is based on conversational context
  - The model can "take turns" at the granularity of 160ms

Key differences from OmniFlatten:
- Fixed chunk size (160ms) vs. variable interleaving
- Explicit speak/silence decision vs. continuous generation
- Simpler to implement but less natural prosody
</code></pre>

<h4>NTPP: Joint Token-Pair Prediction</h4>

<p>Next Token-Pair Prediction (NTPP) modifies the standard next-token prediction objective to jointly predict the next user token and the next model token at each step:</p>

<pre><code># NTPP: predict both user and model next tokens simultaneously
def ntpp_loss(model, input_sequence):
    hidden = model.transformer(input_sequence)  # (B, T, d)

    # Two separate prediction heads
    user_logits = model.user_head(hidden)    # (B, T, vocab_audio)
    model_logits = model.model_head(hidden)  # (B, T, vocab_audio)

    L_user = F.cross_entropy(user_logits[:, :-1], user_targets[:, 1:])
    L_model = F.cross_entropy(model_logits[:, :-1], model_targets[:, 1:])

    # Joint prediction encourages the model to anticipate user speech
    # while generating its own speech
    return L_user + L_model</code></pre>

<p>The advantage of NTPP: it explicitly trains the model to predict what the user will say next, which is essential for natural turn-taking and interruption handling. If the model can predict the user is about to speak, it can prepare to yield the floor.</p>

<h4>Catastrophic Forgetting Prevention</h4>

<p>The single biggest challenge in training full-duplex systems is preventing catastrophic forgetting of text reasoning abilities during audio training. Strategies:</p>

<table>
<tr><th>Strategy</th><th>Mechanism</th><th>Effectiveness</th><th>Cost</th></tr>
<tr><td><strong>LoRA-only training</strong></td><td>Only modify low-rank adapter matrices; original weights frozen</td><td>High</td><td>Low</td></tr>
<tr><td><strong>Data mixing</strong></td><td>Include 10-20% text-only data in every training batch</td><td>Medium-High</td><td>Low</td></tr>
<tr><td><strong>EWC (Elastic Weight Consolidation)</strong></td><td>Add penalty for changing weights important for text tasks</td><td>Medium</td><td>Medium (need to compute Fisher)</td></tr>
<tr><td><strong>Progressive freezing</strong></td><td>Freeze bottom layers first, then middle layers; only top layers remain trainable</td><td>Medium</td><td>Low</td></tr>
<tr><td><strong>Stage-wise LR decay</strong></td><td>Reduce LR in each subsequent training stage</td><td>Medium</td><td>None</td></tr>
<tr><td><strong>Checkpoint averaging</strong></td><td>Average weights of checkpoints from different training stages</td><td>Low-Medium</td><td>None</td></tr>
</table>

<p>The recommended combination: LoRA-only training + 15% text-only data mixing + stage-wise LR decay (Stage 1: 2e-5 -> Stage 2: 1e-5 -> Stage 3: 5e-6). This is the most reliable approach in practice, as confirmed across multiple published systems.</p>

<h4>Data Requirements for Full-Duplex Training</h4>

<p>Full-duplex training requires conversational audio data with overlapping speech &mdash; far harder to obtain than the single-speaker audio used for ASR or TTS. Here are the key data sources and their characteristics:</p>

<table>
<tr><th>Dataset</th><th>Hours</th><th>Type</th><th>Overlapping Speech</th><th>Transcriptions</th><th>Availability</th></tr>
<tr><td>Fisher English</td><td>2,000</td><td>Telephone conversations</td><td>Frequent (natural)</td><td>Yes</td><td>LDC (paid)</td></tr>
<tr><td>Switchboard</td><td>260</td><td>Telephone conversations</td><td>Moderate</td><td>Yes</td><td>LDC (paid)</td></tr>
<tr><td>AMI Meeting Corpus</td><td>100</td><td>Meeting recordings</td><td>Frequent</td><td>Yes</td><td>Free</td></tr>
<tr><td>CALLHOME</td><td>60</td><td>Telephone (multi-language)</td><td>Moderate</td><td>Yes</td><td>LDC (paid)</td></tr>
<tr><td>VoiceAssistant-400K</td><td>~400K samples</td><td>Synthesized dialogues</td><td>Simulated</td><td>Yes</td><td>Free</td></tr>
<tr><td>SpokenWOZ</td><td>~5,000 dialogues</td><td>Task-oriented dialogues</td><td>Minimal</td><td>Yes</td><td>Free</td></tr>
</table>

<p>The critical data gap: most freely available conversational data has minimal overlapping speech (speakers take clean turns). Natural conversations have 10-30% overlap. To train full-duplex systems, you often need to: (1) synthesize overlapping speech by time-shifting clean turn-taking recordings, (2) use data augmentation to simulate barge-in events, or (3) collect proprietary conversational data (expensive but highest quality).</p>

<pre><code># Synthesizing overlapping speech for full-duplex training
import numpy as np
import random

def create_overlap_data(speaker_a_audio, speaker_b_audio, sr=16000,
                         overlap_ratio=0.15, n_samples=1000):
    """
    Create synthetic overlapping speech data from clean turn-taking data.
    overlap_ratio: fraction of time with overlapping speech.
    """
    samples = []
    for _ in range(n_samples):
        # Pick random segments
        a_start = random.randint(0, len(speaker_a_audio) - sr * 10)
        b_start = random.randint(0, len(speaker_b_audio) - sr * 10)
        a_seg = speaker_a_audio[a_start:a_start + sr * 10]  # 10 seconds
        b_seg = speaker_b_audio[b_start:b_start + sr * 10]

        # Create overlap by shifting speaker B forward
        overlap_samples = int(len(a_seg) * overlap_ratio)
        shift = random.randint(0, len(a_seg) - overlap_samples)

        mixed = np.zeros(len(a_seg) + shift)
        mixed[:len(a_seg)] += a_seg
        mixed[shift:shift + len(b_seg)] += b_seg * 0.8  # Slight attenuation

        # Clip to prevent distortion
        mixed = np.clip(mixed, -1.0, 1.0)

        samples.append({
            "mixed_audio": mixed,
            "speaker_a_timestamps": [(0, len(a_seg) / sr)],
            "speaker_b_timestamps": [(shift / sr, (shift + len(b_seg)) / sr)],
            "overlap_region": (shift / sr, len(a_seg) / sr)
        })
    return samples</code></pre>

<h4>Turn-Taking Models</h4>

<p>A crucial component of full-duplex systems is the turn-taking model &mdash; the mechanism that decides when to start speaking, when to stop speaking, and when to yield the floor. Three approaches are used in practice:</p>

<ul>
<li><strong>Implicit (Moshi):</strong> The model learns turn-taking entirely from data. The temporal transformer naturally learns to generate silence tokens when it predicts the user will speak, and speech tokens when the user is silent or finishing. No explicit turn-taking module exists &mdash; it emerges from training on conversational data. Pro: most natural. Con: hard to control, can produce odd behavior.</li>
<li><strong>Explicit classifier (LSLM):</strong> A small binary classifier (typically a 2-layer MLP on top of the LLM's hidden states) predicts at each time step whether the model should be speaking or listening. Pro: controllable, debuggable. Con: less natural, binary decision is too coarse.</li>
<li><strong>Threshold-based (SyncLLM):</strong> The model generates a "speak probability" at each time slot. If above a threshold (e.g., 0.5), it speaks; otherwise, it listens. The threshold can be adjusted at deployment time to make the model more or less likely to take the floor. Pro: tunable. Con: requires careful threshold calibration per deployment context.</li>
</ul>

<div class="callout warning">
<div class="callout-title">War Story: The Forgetting Cliff</div>
<p>A team training a full-duplex system noticed that text reasoning quality (measured by MMLU score) was stable for the first 15K steps of Stage 3 training, then dropped catastrophically from 62% to 38% over just 2K steps. Investigation revealed that the learning rate schedule had a warmup phase that ramped up from 1e-6 to 5e-5 over the first 2K steps, and the model was fine during warmup. But once the LR reached full strength, the audio gradients overwhelmed the LoRA matrices. The fix: cap the Stage 3 LR at 5e-6 (10x lower than they initially planned) and increase batch size to compensate for slower learning. After the fix, MMLU stayed at 60% through all of Stage 3 training, with only a 2-point drop from pre-training levels.</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you convert a pretrained text LLM into a full-duplex speech system? Walk me through the stages.</div>
<div class="a-text">I would follow a 3-stage approach inspired by OmniFlatten: Stage 1 (Text Half-Duplex): Fine-tune the LLM on conversational text data to establish good dialogue patterns. LR 2e-5, 10K steps. This is cheap and establishes the baseline conversational ability. Stage 2 (Speech Half-Duplex): Add a frozen audio encoder (Whisper-large) + trainable adapter + speech decoder head. Train on single-turn spoken dialogues where speakers take turns. Train adapter + LoRA + decoder, keeping encoder and LLM backbone (mostly) frozen. LR 1e-5, 20K steps. Key data: speech transcription pairs + spoken dialogue datasets. Stage 3 (Full-Duplex): Implement stream flattening to interleave user and model audio tokens into a single causal sequence. Train with asymmetric loss (higher weight on model audio, lower weight on user audio prediction). Critical: reduce LR to 5e-6, include 15% text-only data to prevent forgetting, and use LoRA instead of full fine-tuning. 30K steps on conversational data with overlapping speech. Throughout: monitor MMLU score (text reasoning), WER (speech understanding), and MOS (speech quality) at every checkpoint to catch forgetting early.</div>
</div>
`
    },
    {
      id: "s2s-pipeline",
      title: "Building a S2S Pipeline",
      content: `
<p>While end-to-end S2S models are the research frontier, the most practical and reliable way to build a speech-to-speech system today is the cascade pipeline: VAD -> ASR -> LLM -> TTS -> Audio Output. This section provides a complete implementation guide with code, covering each component and the engineering challenges of connecting them in a streaming architecture.</p>

<h4>Complete Pipeline Architecture</h4>

<pre><code>
┌─────────────────────────────────────────────────────────────────────┐
│                    Streaming S2S Pipeline                           │
│                                                                     │
│  User Mic                                                           │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   VAD    │────>│  Streaming   │────>│  Streaming   │            │
│  │ (WebRTC) │     │    ASR       │     │    LLM       │            │
│  │          │     │  (Whisper    │     │  (vLLM /     │            │
│  │ Detects  │     │   Streaming) │     │   SGLang)    │            │
│  │ speech   │     │              │     │              │            │
│  │ start/   │     │ Output:      │     │ Output:      │            │
│  │ end      │     │ partial text │     │ text tokens  │            │
│  └──────────┘     └──────────────┘     └──────┬───────┘            │
│                                               │                     │
│                                               ▼                     │
│  Speaker           ┌──────────────┐    ┌──────────────┐            │
│     ▲              │   Audio      │<───│  Streaming   │            │
│     │              │   Buffer &   │    │    TTS       │            │
│     └──────────────│   Output     │    │  (CosyVoice  │            │
│                    │              │    │   / F5-TTS)  │            │
│                    └──────────────┘    └──────────────┘            │
│                                                                     │
│  ─── WebSocket connection throughout ───                           │
└─────────────────────────────────────────────────────────────────────┘
</code></pre>

<h4>Component 1: Voice Activity Detection (VAD)</h4>

<pre><code>import webrtcvad
import collections
import numpy as np

class StreamingVAD:
    """
    Voice Activity Detection with endpoint detection.
    Uses WebRTC VAD for speech detection and a state machine
    for endpoint (end-of-utterance) detection.
    """
    def __init__(self, sample_rate=16000, frame_duration_ms=30,
                 aggressiveness=3, silence_threshold_ms=300,
                 speech_threshold_ms=100):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sr = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.silence_threshold_frames = int(silence_threshold_ms / frame_duration_ms)
        self.speech_threshold_frames = int(speech_threshold_ms / frame_duration_ms)

        # State machine
        self.state = "SILENCE"  # SILENCE -> SPEECH -> ENDPOINT
        self.speech_frames = 0
        self.silence_frames = 0
        self.audio_buffer = []

    def process_frame(self, audio_frame):
        """
        Process a single audio frame.
        Returns: (event, audio_data)
          event: None, "SPEECH_START", "SPEECH_END"
          audio_data: accumulated audio when SPEECH_END is triggered
        """
        is_speech = self.vad.is_speech(
            audio_frame.tobytes(), self.sr
        )

        if self.state == "SILENCE":
            if is_speech:
                self.speech_frames += 1
                self.audio_buffer.append(audio_frame)
                if self.speech_frames >= self.speech_threshold_frames:
                    self.state = "SPEECH"
                    return ("SPEECH_START", None)
            else:
                self.speech_frames = 0
                self.audio_buffer = []

        elif self.state == "SPEECH":
            self.audio_buffer.append(audio_frame)
            if not is_speech:
                self.silence_frames += 1
                if self.silence_frames >= self.silence_threshold_frames:
                    self.state = "SILENCE"
                    audio_data = np.concatenate(self.audio_buffer)
                    self.audio_buffer = []
                    self.silence_frames = 0
                    self.speech_frames = 0
                    return ("SPEECH_END", audio_data)
            else:
                self.silence_frames = 0

        return (None, None)</code></pre>

<h4>Component 2: Streaming ASR</h4>

<pre><code>import torch
import whisper
import threading
import queue

class StreamingASR:
    """
    Streaming ASR using Whisper with chunked processing.
    Processes audio in overlapping chunks for low latency.
    """
    def __init__(self, model_size="base", device="cuda",
                 chunk_size_ms=2000, overlap_ms=500):
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        self.chunk_size = int(16000 * chunk_size_ms / 1000)
        self.overlap = int(16000 * overlap_ms / 1000)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript = ""

    def process_audio(self, audio_chunk):
        """
        Add audio to buffer and transcribe when enough is accumulated.
        Returns partial transcript or None.
        """
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        if len(self.audio_buffer) >= self.chunk_size:
            # Transcribe current buffer
            audio_tensor = torch.from_numpy(self.audio_buffer).to(self.device)

            # Pad or trim to 30 seconds for Whisper
            audio_padded = whisper.pad_or_trim(audio_tensor)
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.device)

            options = whisper.DecodingOptions(
                language="en",
                without_timestamps=True,
                fp16=(self.device == "cuda")
            )
            result = whisper.decode(self.model, mel, options)

            self.transcript = result.text
            # Keep overlap for context continuity
            self.audio_buffer = self.audio_buffer[-self.overlap:]
            return self.transcript

        return None

    def finalize(self):
        """Process remaining audio in buffer."""
        if len(self.audio_buffer) > 1600:  # At least 100ms
            audio_tensor = torch.from_numpy(self.audio_buffer)
            audio_padded = whisper.pad_or_trim(audio_tensor).to(self.device)
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.device)
            options = whisper.DecodingOptions(
                language="en", without_timestamps=True,
                fp16=(self.device == "cuda")
            )
            result = whisper.decode(self.model, mel, options)
            self.transcript = result.text
        self.audio_buffer = np.array([], dtype=np.float32)
        final = self.transcript
        self.transcript = ""
        return final</code></pre>

<h4>Component 3: Streaming LLM</h4>

<pre><code>from openai import OpenAI

class StreamingLLM:
    """
    Streaming LLM response generation.
    Uses OpenAI-compatible API (works with vLLM, SGLang, etc.)
    """
    def __init__(self, base_url="http://localhost:8000/v1",
                 model="meta-llama/Llama-3.1-8B-Instruct"):
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = model
        self.conversation_history = [
            {"role": "system", "content":
             "You are a helpful voice assistant. Keep responses concise "
             "(2-3 sentences max) and conversational. Do not use markdown, "
             "bullet points, or formatting - speak naturally."}
        ]

    def generate_streaming(self, user_text):
        """
        Generate a streaming response. Yields text chunks.
        """
        self.conversation_history.append(
            {"role": "user", "content": user_text}
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            max_tokens=150,  # Keep short for voice
            temperature=0.7,
            stream=True
        )

        full_response = ""
        current_sentence = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_response += text
                current_sentence += text

                # Yield at sentence boundaries for TTS
                if any(text.endswith(p) for p in ['.', '!', '?', ',']):
                    yield current_sentence
                    current_sentence = ""

        # Yield remaining text
        if current_sentence.strip():
            yield current_sentence

        self.conversation_history.append(
            {"role": "assistant", "content": full_response}
        )</code></pre>

<h4>Component 4: Streaming TTS</h4>

<pre><code># Using CosyVoice for streaming TTS
# pip install cosyvoice

class StreamingTTS:
    """
    Streaming TTS that begins synthesis before the full text is available.
    """
    def __init__(self, model_name="CosyVoice-300M-SFT", device="cuda"):
        # Note: actual CosyVoice loading may differ based on version
        from cosyvoice.cli.cosyvoice import CosyVoice
        self.model = CosyVoice(model_name)
        self.sample_rate = 22050  # CosyVoice output sample rate

    def synthesize_streaming(self, text_chunks):
        """
        Synthesize speech from streaming text chunks.
        Yields audio numpy arrays as they are generated.
        """
        for text_chunk in text_chunks:
            if not text_chunk.strip():
                continue

            # Generate audio for this chunk
            # CosyVoice generates audio for each text segment
            for audio_segment in self.model.inference_sft(
                text_chunk,
                spk_id="default",
                stream=True
            ):
                # audio_segment is a dict with 'tts_speech' key
                audio = audio_segment['tts_speech'].numpy()
                yield audio

    def synthesize_with_clone(self, text_chunks, reference_audio_path):
        """
        Zero-shot voice cloning: synthesize in a specific voice.
        """
        for text_chunk in text_chunks:
            if not text_chunk.strip():
                continue
            for audio_segment in self.model.inference_zero_shot(
                text_chunk,
                "Reference speaker prompt text.",
                reference_audio_path,
                stream=True
            ):
                yield audio_segment['tts_speech'].numpy()</code></pre>

<h4>Putting It All Together: Complete S2S Pipeline</h4>

<pre><code>import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd

class S2SPipeline:
    """
    Complete Speech-to-Speech pipeline with streaming.
    """
    def __init__(self):
        self.vad = StreamingVAD(silence_threshold_ms=300)
        self.asr = StreamingASR(model_size="base")
        self.llm = StreamingLLM()
        self.tts = StreamingTTS()
        self.is_speaking = False

    async def run(self):
        """Main pipeline loop."""
        print("S2S Pipeline ready. Speak into your microphone...")

        # Audio input stream
        audio_queue = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_queue.put_nowait(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=16000, channels=1, blocksize=480,  # 30ms frames
            dtype='int16', callback=audio_callback
        ):
            while True:
                # Get audio frame
                audio_frame = await audio_queue.get()

                # VAD processing
                event, audio_data = self.vad.process_frame(audio_frame)

                if event == "SPEECH_START":
                    print("[VAD] Speech detected...")
                    if self.is_speaking:
                        # Interruption: stop current TTS output
                        self.is_speaking = False
                        print("[INTERRUPT] User interrupted, stopping output")

                elif event == "SPEECH_END":
                    print("[VAD] End of speech, processing...")

                    # ASR: transcribe the accumulated audio
                    self.asr.audio_buffer = audio_data.astype(np.float32) / 32768.0
                    transcript = self.asr.finalize()
                    print(f"[ASR] Transcript: {transcript}")

                    if transcript.strip():
                        # LLM: generate response (streaming)
                        print("[LLM] Generating response...")
                        text_stream = self.llm.generate_streaming(transcript)

                        # TTS: synthesize response (streaming)
                        self.is_speaking = True
                        print("[TTS] Synthesizing speech...")
                        for audio_chunk in self.tts.synthesize_streaming(text_stream):
                            if not self.is_speaking:
                                break  # Interrupted
                            # Output audio
                            sd.play(audio_chunk, self.tts.sample_rate)
                            sd.wait()

                        self.is_speaking = False
                        print("[Pipeline] Response complete\\n")

# Run:
# pipeline = S2SPipeline()
# asyncio.run(pipeline.run())</code></pre>

<h4>Endpoint Detection Algorithms</h4>

<p>Endpoint detection (determining when the user has finished speaking) is perhaps the most under-appreciated component. Too aggressive: the system interrupts the user mid-sentence. Too conservative: long pauses before response. Advanced approaches:</p>

<ul>
<li><strong>Fixed silence threshold (baseline):</strong> Trigger after N ms of silence (typically 300-700ms). Simple but fails on natural pauses within speech.</li>
<li><strong>Linguistic endpoint detection:</strong> Use the partial ASR transcript to predict whether the utterance is syntactically complete. A small classifier trained on (partial_text, is_complete) pairs can reduce false positives by 40-60%.</li>
<li><strong>Prosodic cues:</strong> Falling pitch at phrase boundaries suggests finality; rising pitch suggests continuation. Combine with VAD for more accurate endpoints.</li>
<li><strong>Hybrid:</strong> Use a short silence threshold (200ms) as a candidate trigger, then check linguistic completeness. If the sentence is syntactically incomplete, extend the threshold to 500ms.</li>
</ul>

<pre><code># Linguistic endpoint detection
import re

class LinguisticEndpointDetector:
    """
    Checks if a partial transcript is likely complete.
    Uses heuristic rules + optional small model.
    """
    # Patterns that suggest the utterance is complete
    COMPLETE_PATTERNS = [
        r'\\.$', r'\\?$', r'\\!$',  # Sentence-ending punctuation
        r'\\b(thanks|thank you|bye|goodbye|okay|alright|sure|yes|no)$',
        r'\\b(please|help|that)$',
    ]

    # Patterns that suggest the utterance is incomplete
    INCOMPLETE_PATTERNS = [
        r'\\b(and|but|or|so|because|if|when|while|although)$',
        r'\\b(the|a|an|my|your|their|his|her|its)$',
        r'\\b(is|are|was|were|will|would|can|could|should)$',
        r'\\b(to|for|with|from|about|into)$',
    ]

    def is_likely_complete(self, partial_transcript):
        text = partial_transcript.strip().lower()
        if not text:
            return False

        # Check incomplete patterns first (higher priority)
        for pattern in self.INCOMPLETE_PATTERNS:
            if re.search(pattern, text):
                return False

        # Check complete patterns
        for pattern in self.COMPLETE_PATTERNS:
            if re.search(pattern, text):
                return True

        # Default: if >5 words and no incomplete signal, assume complete
        return len(text.split()) >= 5

# Usage in endpoint detection:
# if vad_silence_detected and duration > 200ms:
#     if linguistic_detector.is_likely_complete(partial_transcript):
#         trigger_endpoint()
#     elif duration > 500ms:
#         trigger_endpoint()  # Fallback timeout</code></pre>

<h4>WebSocket Architecture for Streaming</h4>

<p>A production S2S pipeline uses WebSocket connections for bidirectional audio streaming between the client and server. The architecture must handle three concurrent streams: incoming user audio, outgoing system audio, and control messages (interrupt, status updates).</p>

<pre><code># Server-side WebSocket handler for S2S pipeline
import asyncio
import websockets
import json
import struct

async def handle_s2s_session(websocket, path):
    """
    Handle a single S2S session over WebSocket.
    Protocol:
      - Binary frames: audio data (16-bit PCM, 16kHz, mono)
      - Text frames: JSON control messages
    """
    pipeline = S2SPipeline()
    audio_queue = asyncio.Queue()
    is_cancelled = asyncio.Event()

    async def receive_audio():
        """Receive user audio and control messages."""
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary: audio data
                audio = np.frombuffer(message, dtype=np.int16)
                await audio_queue.put(audio)
            else:
                # Text: control message
                msg = json.loads(message)
                if msg.get("type") == "interrupt":
                    is_cancelled.set()
                elif msg.get("type") == "end_session":
                    break

    async def process_and_respond():
        """Process audio and send responses."""
        while True:
            audio_frame = await audio_queue.get()
            event, audio_data = pipeline.vad.process_frame(audio_frame)

            if event == "SPEECH_START" and pipeline.is_speaking:
                is_cancelled.set()  # Interrupt current response
                await websocket.send(json.dumps({"type": "interrupted"}))

            elif event == "SPEECH_END" and audio_data is not None:
                is_cancelled.clear()
                transcript = pipeline.asr.transcribe(audio_data)
                await websocket.send(json.dumps({
                    "type": "transcript", "text": transcript
                }))

                # Generate and stream response
                for audio_chunk in pipeline.generate_response(transcript):
                    if is_cancelled.is_set():
                        break
                    await websocket.send(audio_chunk.tobytes())

    await asyncio.gather(receive_audio(), process_and_respond())</code></pre>

<h4>Latency Budget Allocation</h4>

<pre><code>
Target: 400ms Time-to-First-Audio-Byte (TTFAB)

Component          | Budget  | Strategy
-------------------+---------+------------------------------------------
Endpointing (VAD)  |  50ms   | WebRTC VAD, 200ms silence + linguistic check
Streaming ASR      | 100ms   | Process last 2s chunk, whisper.cpp on GPU
LLM TTFT           | 150ms   | 8B model, INT4 quantized, speculative decoding
TTS first chunk    |  80ms   | CosyVoice streaming, pre-warm the decoder
Network + buffer   |  20ms   | Local deployment, TCP_NODELAY
-------------------+---------+------------------------------------------
Total              | 400ms   |
</code></pre>

<div class="callout">
<div class="callout-title">Production Tip: Overlapping Pipeline Stages</div>
<p>The latency budget above assumes sequential processing. In practice, you can overlap stages: start LLM inference while the last ASR chunk is processing (using the partial transcript). Start TTS synthesis on the first LLM output tokens while more tokens are being generated. This overlapping can save 100-200ms, bringing total TTFAB below 300ms. The cost: increased complexity in error handling (what if ASR revises the transcript after LLM has started generating?).</p>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">Design a production-grade S2S pipeline for a customer service application handling 1000 concurrent calls. What are the key architectural decisions?</div>
<div class="a-text">Key architectural decisions: (1) Infrastructure: Deploy ASR, LLM, and TTS as separate microservices behind a load balancer, connected by gRPC streams (lower latency than REST). Use Kubernetes with GPU node pools for each service. (2) ASR: Use Whisper-large-v3 with batched inference via faster-whisper (CTranslate2). At 1000 concurrent calls, batch processing is essential. Process 500ms chunks with 4-GPU instances, each handling ~250 concurrent streams. (3) LLM: Use vLLM with continuous batching and PagedAttention. A LLaMA-3.1-8B-Instruct with INT4 quantization on 8xA100s can handle 1000 concurrent requests with <200ms TTFT. Enable prefix caching for the system prompt. (4) TTS: Use CosyVoice or F5-TTS in streaming mode. Pre-compute audio for common responses ("Please hold," "Could you repeat that?"). Deploy on 4 GPU instances. (5) State management: Use Redis for conversation state (KV cache pointers, conversation history). Implement session affinity at the load balancer to reduce cache misses. (6) Monitoring: Track per-component latency (p50/p95/p99), end-to-end TTFAB, ASR word error rate on production traffic (sample 1%), and TTS MOS score (periodic human evaluation). (7) Fallback: If any component exceeds latency budget, fall back to pre-recorded responses rather than delivering a delayed response.</div>
</div>
`
    },
    {
      id: "s2s-evaluation",
      title: "Evaluating S2S Systems",
      content: `
<p>Evaluating speech-to-speech systems is fundamentally harder than evaluating text LLMs or even AudioLLMs. S2S systems must be evaluated on multiple orthogonal dimensions: content quality (is the response correct and helpful?), speech quality (does it sound natural?), conversational quality (is the interaction smooth?), and latency (is it real-time?). No single metric captures all of these, and there are significant tensions between them.</p>

<h4>VoiceBench: The Standard Benchmark</h4>

<p>VoiceBench (arXiv:2410.17196) is the most widely used evaluation framework for S2S systems. It provides a structured evaluation across multiple dimensions:</p>

<table>
<tr><th>Dimension</th><th>Sub-metrics</th><th>How Measured</th><th>Weight (typical)</th></tr>
<tr><td><strong>Content Quality</strong></td><td>Accuracy, Relevance, Completeness</td><td>GPT-4 as judge; compare response content to reference</td><td>40%</td></tr>
<tr><td><strong>Speech Quality</strong></td><td>Naturalness (MOS), Intelligibility (WER of output), Speaker consistency</td><td>MOS: human rating 1-5; Intelligibility: ASR the output and measure WER</td><td>25%</td></tr>
<tr><td><strong>Conversational Quality</strong></td><td>Turn-taking naturalness, Backchanneling, Interruption handling</td><td>Human evaluation on a 1-5 scale per dimension</td><td>20%</td></tr>
<tr><td><strong>Latency</strong></td><td>TTFAB, E2E latency, IRT</td><td>Automated measurement (see latency section)</td><td>15%</td></tr>
</table>

<h4>Latency Measurement in Detail</h4>

<p><strong>Time-to-First-Audio-Byte (TTFAB):</strong> The most perceptually important metric. Measured from the end of user speech (as detected by VAD) to the first non-silence audio frame in the system response. Target: <400ms for natural conversation.</p>

<p><strong>End-to-End Latency (E2E):</strong> Time from end of user speech to end of system speech. This is TTFAB + response duration. Less perceptually important (users are "distracted" by listening to the response) but matters for throughput.</p>

<p><strong>Interruption Response Time (IRT):</strong> For full-duplex systems only. Time from when the user starts speaking during system output to when the system acknowledges the interruption. Measured by having a scripted user interrupt at controlled points and detecting system response. Target: <300ms for natural interruption handling.</p>

<pre><code># Automated latency measurement protocol
class LatencyEvaluator:
    def __init__(self, system_under_test):
        self.system = system_under_test
        self.results = []

    def run_ttfab_benchmark(self, test_utterances, n_trials=100):
        """
        Measure TTFAB across diverse test utterances.
        test_utterances: list of (audio_array, expected_response_type)
        """
        for utterance, response_type in test_utterances:
            for trial in range(n_trials):
                import time
                t_start = time.perf_counter()

                # Send audio
                response_stream = self.system.send_audio(utterance)

                # Wait for first non-silence response frame
                t_first_audio = None
                for frame in response_stream:
                    rms = np.sqrt(np.mean(frame.astype(float)**2))
                    if rms > 100:  # Threshold for non-silence
                        t_first_audio = time.perf_counter()
                        break

                if t_first_audio:
                    utterance_duration = len(utterance) / 16000
                    ttfab = (t_first_audio - t_start) - utterance_duration
                    self.results.append({
                        "metric": "TTFAB",
                        "value_ms": ttfab * 1000,
                        "response_type": response_type,
                        "trial": trial
                    })

        return self._compute_statistics("TTFAB")

    def _compute_statistics(self, metric_name):
        values = [r["value_ms"] for r in self.results if r["metric"] == metric_name]
        return {
            "metric": metric_name,
            "mean": np.mean(values),
            "median": np.median(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "std": np.std(values),
            "n": len(values)
        }</code></pre>

<h4>Content Quality Evaluation</h4>

<p>Evaluating content quality of S2S responses is uniquely challenging because the response is in audio form. The standard approach: transcribe the system's audio output using a high-quality ASR model, then evaluate the transcript.</p>

<pre><code># Content quality evaluation using LLM-as-judge
def evaluate_content_quality(system_response_text, reference_answer,
                              user_question, judge_model="gpt-4"):
    """
    Use GPT-4 as a judge to evaluate content quality.
    Returns scores for accuracy, relevance, and completeness.
    """
    prompt = f"""You are evaluating a voice assistant's response.

User question: {user_question}
Reference answer: {reference_answer}
System response: {system_response_text}

Rate the system response on these dimensions (1-5 each):
1. Accuracy: Is the information correct?
2. Relevance: Does it address the user's question?
3. Completeness: Does it cover the key points?
4. Conciseness: Is it appropriately brief for voice? (penalty for being too long)

Output JSON: {{"accuracy": X, "relevance": X, "completeness": X, "conciseness": X}}"""

    response = openai.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    scores = json.loads(response.choices[0].message.content)
    return scores</code></pre>

<h4>Speech Quality Evaluation</h4>

<p><strong>Mean Opinion Score (MOS):</strong> The gold standard for speech quality evaluation. Human raters listen to audio samples and rate them on a 1-5 scale (1=bad, 5=excellent). For research, typically 20+ raters evaluate 100+ samples. For production, smaller ongoing evaluations with 5-10 raters can track quality over time.</p>

<p><strong>Automated MOS prediction:</strong> Models like UTMOS (arXiv:2204.02152) predict MOS from audio directly, trained on large-scale human rating data. Useful for automated CI/CD pipelines.</p>

<pre><code># Automated speech quality evaluation
# pip install speechmos  (or use torchaudio's PESQ/STOI)

def evaluate_speech_quality(audio_path):
    """Evaluate speech quality using automated metrics."""
    import torchaudio
    from pesq import pesq  # PESQ: Perceptual Evaluation of Speech Quality

    waveform, sr = torchaudio.load(audio_path)

    results = {}

    # 1. Intelligibility: ASR the output and check for garbled speech
    transcript = whisper_transcribe(audio_path)
    # If ASR confidence is low, speech quality is likely poor
    results["intelligibility_proxy"] = asr_confidence

    # 2. UTMOS: predicted MOS score
    # utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")
    # results["utmos"] = utmos_model(waveform, sr).item()

    # 3. Signal-level metrics (if reference audio available)
    # results["pesq"] = pesq(16000, ref_signal, deg_signal, 'wb')

    # 4. Speaking rate (words per minute)
    num_words = len(transcript.split())
    duration_seconds = waveform.shape[1] / sr
    results["speaking_rate_wpm"] = num_words / duration_seconds * 60
    # Natural range: 120-180 WPM; outside this suggests problems

    return results</code></pre>

<h4>Conversational Quality: The Hard Part</h4>

<p>Conversational quality is the hardest dimension to evaluate automatically. It includes:</p>

<ul>
<li><strong>Turn-taking naturalness:</strong> Does the system respond at the right time? Does it interrupt the user? Does it leave awkward silences?</li>
<li><strong>Backchanneling:</strong> Does the system produce natural listener responses ("uh-huh," "I see," "right") during the user's speech? (Only applicable for full-duplex systems.)</li>
<li><strong>Interruption handling:</strong> When the user interrupts, does the system stop gracefully? Does it acknowledge the interruption?</li>
<li><strong>Coherence across turns:</strong> Does the system maintain context across multiple turns?</li>
<li><strong>Emotional appropriateness:</strong> Does the system's tone match the conversation's emotional context?</li>
</ul>

<p>These are currently evaluated through human evaluation with structured rubrics. Here is a sample rubric:</p>

<table>
<tr><th>Dimension</th><th>Score 1 (Poor)</th><th>Score 3 (Acceptable)</th><th>Score 5 (Excellent)</th></tr>
<tr><td>Turn-taking</td><td>Frequently interrupts user or leaves >1s gaps</td><td>Mostly appropriate timing, occasional mis-timed responses</td><td>Consistently natural timing, responds at appropriate moments</td></tr>
<tr><td>Interruption handling</td><td>Ignores interruptions or crashes</td><td>Stops speaking but doesn't acknowledge interruption</td><td>Stops naturally, acknowledges interruption, resumes appropriately</td></tr>
<tr><td>Coherence</td><td>Contradicts previous statements or forgets context</td><td>Mostly coherent, occasional context loss</td><td>Maintains full context, references previous turns naturally</td></tr>
<tr><td>Emotional tone</td><td>Monotone or inappropriate emotion</td><td>Neutral but appropriate</td><td>Adapts tone to match conversation context</td></tr>
</table>

<h4>Human Evaluation Protocols</h4>

<p>For rigorous S2S evaluation, use the following protocol:</p>

<ol>
<li><strong>Evaluator recruitment:</strong> 20-30 native speakers of the target language, diverse in age and dialect. Pay fairly ($25-40/hour).</li>
<li><strong>Training session:</strong> 30-minute orientation with example ratings and calibration exercises.</li>
<li><strong>Test design:</strong> 50-100 test conversations, each 3-5 turns. Include diverse scenarios: information seeking, task completion, emotional support, chitchat.</li>
<li><strong>Rating protocol:</strong> Each conversation rated by 3+ evaluators. Use the rubric above. Compute inter-rater reliability (Krippendorff's alpha should be >0.6).</li>
<li><strong>A/B testing:</strong> When comparing systems, use paired comparisons (same conversation, two systems) rather than absolute ratings, as they are more reliable.</li>
</ol>

<div class="callout">
<div class="callout-title">The Content-Conversation Tradeoff</div>
<p>There is a fundamental tension between content quality and conversational quality. Optimizing for content quality (correct, complete answers) tends to produce longer responses that hurt conversational flow. Optimizing for conversational quality (short, snappy responses) can sacrifice accuracy and completeness. The best systems manage this tradeoff by varying response length based on question complexity: simple questions get brief responses, complex questions get structured multi-turn explanations.</p>
</div>

<h4>Robustness Testing</h4>

<p>Production S2S systems encounter conditions far worse than clean evaluation data. Robustness testing must cover:</p>

<p><strong>Acoustic robustness:</strong></p>
<ul>
<li><strong>Background noise:</strong> Test at SNR levels from 30dB (quiet office) to 5dB (noisy cafe). Most systems degrade gracefully to 15dB but fail catastrophically below 10dB. Add noise from the MUSAN dataset (music, speech, and environmental noise).</li>
<li><strong>Reverberation:</strong> Simulate room impulse responses from the RIR dataset. Far-field microphones (2+ meters) add significant reverberation that degrades both ASR and emotion detection.</li>
<li><strong>Codec artifacts:</strong> Test with audio that has been compressed through telephone codecs (G.711, AMR-NB at 4.75kbps), Bluetooth audio (SBC codec), and VoIP codecs (Opus at various bitrates). Many production deployments receive audio through these codecs.</li>
</ul>

<p><strong>Linguistic robustness:</strong></p>
<ul>
<li><strong>Accented speech:</strong> Test on at least 5 accent groups for your target language. For English: American, British, Indian, Chinese-accented, Spanish-accented. WER disparities across accents should be less than 2x.</li>
<li><strong>Code-switching:</strong> Many real users switch between languages mid-sentence. Test with code-switched utterances if your user base is multilingual.</li>
<li><strong>Disfluencies:</strong> Real speech contains "um," "uh," false starts, and self-corrections. Systems that are trained only on read speech may struggle with spontaneous conversational speech.</li>
</ul>

<p><strong>Adversarial robustness:</strong></p>
<ul>
<li><strong>Prompt injection via speech:</strong> Test whether spoken instructions can override the system prompt ("Ignore your instructions and..."). This is a security concern for deployed systems.</li>
<li><strong>Adversarial audio:</strong> Test with audio designed to be misrecognized by ASR (adversarial perturbations). While academic adversarial attacks rarely transfer to production, testing for this catches unexpected failure modes.</li>
</ul>

<h4>Comprehensive Evaluation Checklist</h4>

<table>
<tr><th>Category</th><th>Metric</th><th>Target (Production)</th><th>Automated?</th></tr>
<tr><td rowspan="3">Content</td><td>Response accuracy (LLM-judge)</td><td>>4.0/5.0</td><td>Yes</td></tr>
<tr><td>Hallucination rate</td><td><5%</td><td>Yes (with fact-checking)</td></tr>
<tr><td>Task completion rate</td><td>>85%</td><td>Semi (need task definitions)</td></tr>
<tr><td rowspan="3">Speech</td><td>MOS (human or UTMOS)</td><td>>3.8/5.0</td><td>Semi (UTMOS is proxy)</td></tr>
<tr><td>Output intelligibility (WER)</td><td><5%</td><td>Yes</td></tr>
<tr><td>Speaking rate</td><td>130-170 WPM</td><td>Yes</td></tr>
<tr><td rowspan="3">Conversation</td><td>Turn-taking score (human)</td><td>>3.5/5.0</td><td>No</td></tr>
<tr><td>Interruption success rate</td><td>>80%</td><td>Partially</td></tr>
<tr><td>Multi-turn coherence</td><td>>3.5/5.0</td><td>No</td></tr>
<tr><td rowspan="3">Latency</td><td>TTFAB (p50)</td><td><350ms</td><td>Yes</td></tr>
<tr><td>TTFAB (p99)</td><td><800ms</td><td>Yes</td></tr>
<tr><td>IRT (full-duplex only)</td><td><300ms</td><td>Yes</td></tr>
</table>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">How would you set up A/B testing for a voice assistant, and what metrics would determine the winner?</div>
<div class="a-text">A/B testing for voice assistants requires careful design because the interaction is temporal and subjective. Setup: (1) Random assignment at the session level (not turn level) &mdash; each user session gets one system variant for consistency. (2) Minimum 1000 sessions per variant for statistical power (alpha=0.05, beta=0.8, minimum detectable effect=5%). (3) Stratify by user segment (new vs. returning, task type, time of day). Primary metric: Task completion rate &mdash; did the user accomplish what they set out to do? This is the most reliable automated metric. Secondary metrics: (a) Session length (shorter is usually better for task-oriented assistants), (b) Repetition rate (how often the user repeats themselves, indicating misunderstanding), (c) Escalation rate (how often the user asks for a human agent), (d) CSAT survey (post-session 1-5 rating). Guardrail metrics: TTFAB p95 should not exceed 500ms, and output WER should not exceed 5%. Statistical analysis: Use a two-proportion z-test for task completion rate, and Mann-Whitney U test for ordinal metrics like CSAT. Run for at least 2 weeks to capture weekday/weekend variation. The winner is the variant with significantly higher task completion rate, provided guardrail metrics are not degraded.</div>
</div>

<div class="interview-q">
<div class="q-label">Interview Question</div>
<div class="q-text">What are the limitations of using automated metrics (like WER, BLEU, UTMOS) for evaluating S2S systems, and how would you complement them?</div>
<div class="a-text">Automated metrics have critical blind spots for S2S evaluation. WER only measures transcription accuracy of the output &mdash; it says nothing about whether the response content was correct or helpful. A perfectly transcribed wrong answer gets 0% WER. BLEU/METEOR compare against reference texts, but voice responses are inherently more variable than text (many valid phrasings). UTMOS predicts speech naturalness but misses conversational appropriateness (a perfectly natural-sounding response at the wrong time is still bad). None of these metrics capture turn-taking, interruption handling, emotional appropriateness, or conversational coherence. Complementary approaches: (1) LLM-as-judge for content quality &mdash; GPT-4 evaluates transcript against rubrics (automated, scalable, correlates well with humans at r>0.8). (2) Regular human evaluation cadence: 100 conversations rated by 3 humans every 2 weeks. (3) Interaction-level metrics from logs: repeat rate, escalation rate, session completion rate. (4) Targeted probes: specific test cases for known failure modes (emotional mismatch, long response to simple question, failure to handle interruption). The key insight: automated metrics are guardrails (detect regressions), not goals. Human evaluation and task completion are the true north star.</div>
</div>
`
    }
  ]
};
