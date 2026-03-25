// Deeply Expanded Content for Chapter 15
// Chapter 15: Data Structures & Algorithms for AI Engineers (8 sections, ~12,000 words)

const CONTENT_CH15 = {

  // ============================================================
  // CHAPTER 15: Data Structures & Algorithms for AI Engineers
  // ============================================================
  ch15_sections: [
    // ----------------------------------------------------------
    // 15.1 Tries and BPE Tokenizers
    // ----------------------------------------------------------
    {
      id: "dsa-tokenizer",
      title: "Tries and BPE Tokenizers",
      content: `
<p>Tokenization is the very first operation in every modern language model pipeline, yet most engineers treat it as a black box. Understanding the data structures that power tokenizers&mdash;particularly the <strong>trie</strong> and the <strong>priority queue</strong>&mdash;gives you the ability to debug encoding anomalies, build custom tokenizers for domain-specific vocabularies, and reason about why certain prompts consume more tokens than you expect.</p>

<div class="callout">
<div class="callout-title">Why This Matters for AI Engineers</div>
<p>A tokenizer determines the "resolution" at which your model sees text. Inefficient tokenization inflates sequence length, increases latency, and burns through context windows. Languages like Thai, Tibetan, and many African languages are particularly penalized by tokenizers trained predominantly on English web text. Understanding the underlying algorithms empowers you to diagnose these problems and build better solutions.</p>
</div>

<h4>1. The Trie Data Structure</h4>
<p>A <strong>trie</strong> (from "re<em>trie</em>val") is a tree-like data structure where each node represents a character (or byte) and paths from root to marked nodes represent complete strings. Tries excel at prefix matching&mdash;exactly the operation tokenizers need when finding the longest matching token in a vocabulary.</p>

<p>Key properties of a trie:</p>
<ul>
  <li><strong>Lookup time:</strong> O(m) where m is the length of the query string, independent of vocabulary size</li>
  <li><strong>Prefix matching:</strong> Finding all strings with a given prefix is naturally efficient</li>
  <li><strong>Space:</strong> Shared prefixes share nodes, making it memory-efficient for vocabularies with common prefixes</li>
  <li><strong>Ordered iteration:</strong> Lexicographic traversal comes naturally via DFS</li>
</ul>

<pre><code>class TrieNode:
    """A single node in a trie structure."""
    __slots__ = ['children', 'token_id', 'is_end']

    def __init__(self):
        self.children = {}      # char -> TrieNode
        self.token_id = None    # If this node marks end of a token
        self.is_end = False

class TokenizerTrie:
    """Trie-based vocabulary lookup for tokenization.

    This is the core data structure used by many tokenizer implementations
    to perform longest-prefix matching against a vocabulary.
    """

    def __init__(self):
        self.root = TrieNode()
        self.vocab_size = 0

    def insert(self, token_bytes: bytes, token_id: int):
        """Insert a token into the trie.

        Args:
            token_bytes: The byte representation of the token
            token_id: The integer ID assigned to this token
        """
        node = self.root
        for byte in token_bytes:
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node = node.children[byte]
        node.is_end = True
        node.token_id = token_id
        self.vocab_size += 1

    def longest_prefix_match(self, data: bytes, start: int) -> tuple:
        """Find the longest token in the vocabulary that matches
        starting at position 'start' in the data.

        Returns:
            (token_id, length) of the longest match, or (None, 0) if no match.

        This is the critical operation during tokenization. By greedily
        matching the longest prefix at each position, we implement the
        "greedy left-to-right" encoding strategy.
        """
        node = self.root
        best_match = (None, 0)  # (token_id, length)

        for i in range(start, len(data)):
            byte = data[i]
            if byte not in node.children:
                break
            node = node.children[byte]
            if node.is_end:
                best_match = (node.token_id, i - start + 1)

        return best_match

    def tokenize_greedy(self, text: str) -> list:
        """Tokenize text using greedy longest-match.

        This is a simplified version of how many tokenizers work.
        Real implementations use BPE merge rules instead of pure
        greedy matching.
        """
        data = text.encode('utf-8')
        tokens = []
        pos = 0

        while pos < len(data):
            token_id, length = self.longest_prefix_match(data, pos)
            if token_id is not None:
                tokens.append(token_id)
                pos += length
            else:
                # Fallback: encode unknown byte as individual byte token
                tokens.append(data[pos] + 256)  # Byte fallback range
                pos += 1

        return tokens</code></pre>

<h4>2. Byte Pair Encoding (BPE) Algorithm</h4>
<p>BPE is the dominant tokenization algorithm used by GPT, LLaMA, Mistral, and most modern language models. Originally a data compression technique (Gage, 1994), it was adapted for NLP by Sennrich et al. (2016). The key insight: start with individual characters (or bytes) and iteratively merge the most frequent pair into a new token.</p>

<p>The BPE training algorithm:</p>
<ol>
  <li>Initialize vocabulary with all individual bytes (256 tokens for byte-level BPE)</li>
  <li>Count frequency of all adjacent token pairs in the training corpus</li>
  <li>Find the most frequent pair &rarr; this requires a <strong>priority queue</strong> (max-heap)</li>
  <li>Merge all occurrences of that pair into a new token; add it to the vocabulary</li>
  <li>Update pair frequencies (only pairs involving the merged tokens change)</li>
  <li>Repeat steps 3-5 until desired vocabulary size is reached</li>
</ol>

<pre><code>import heapq
from collections import defaultdict

class BPETrainer:
    """Train a Byte Pair Encoding tokenizer from scratch.

    This implementation demonstrates the core algorithm with
    priority-queue-based pair selection for efficiency.
    """

    def __init__(self, target_vocab_size: int = 32000):
        self.target_vocab_size = target_vocab_size
        self.merges = []          # List of (token_a, token_b) merge rules
        self.vocab = {}           # token_id -> bytes

    def _get_pair_counts(self, token_sequences: list) -> dict:
        """Count frequency of all adjacent pairs across the corpus.

        Args:
            token_sequences: List of (token_list, frequency) tuples.
                Each token_list represents a word as a sequence of token IDs.

        Returns:
            Dictionary mapping (token_a, token_b) -> count
        """
        pair_counts = defaultdict(int)
        for tokens, freq in token_sequences:
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += freq
        return pair_counts

    def _merge_pair(self, token_sequences: list, pair: tuple,
                    new_token_id: int) -> list:
        """Merge all occurrences of a pair in all sequences.

        This is the O(n) scan where n is total tokens in corpus.
        In production implementations, this is optimized with
        linked lists and index structures.
        """
        merged = []
        a, b = pair
        for tokens, freq in token_sequences:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == a and tokens[i + 1] == b):
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            merged.append((new_tokens, freq))
        return merged

    def train(self, text_corpus: dict):
        """Train BPE on a word-frequency dictionary.

        Args:
            text_corpus: Dictionary mapping word -> frequency.
                Example: {"hello": 1000, "world": 800, ...}

        The algorithm uses a max-heap (priority queue) to efficiently
        find the most frequent pair at each step.
        """
        # Step 1: Initialize vocabulary with individual bytes
        for i in range(256):
            self.vocab[i] = bytes([i])

        next_token_id = 256

        # Step 2: Convert words to byte sequences
        token_sequences = []
        for word, freq in text_corpus.items():
            byte_tokens = list(word.encode('utf-8'))
            token_sequences.append((byte_tokens, freq))

        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.target_vocab_size - 256

        for merge_idx in range(num_merges):
            # Count all pairs
            pair_counts = self._get_pair_counts(token_sequences)

            if not pair_counts:
                print(f"No more pairs to merge at step {merge_idx}")
                break

            # Use a max-heap to find the most frequent pair.
            # Python's heapq is a min-heap, so we negate frequencies.
            # In production, you'd maintain the heap incrementally
            # rather than rebuilding it each iteration.
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break  # No pair occurs more than once

            # Create new token from merge
            a, b = best_pair
            new_bytes = self.vocab[a] + self.vocab[b]
            self.vocab[next_token_id] = new_bytes
            self.merges.append(best_pair)

            # Merge all occurrences
            token_sequences = self._merge_pair(
                token_sequences, best_pair, next_token_id
            )

            if merge_idx % 1000 == 0:
                print(f"Merge {merge_idx}: {self.vocab[a]} + "
                      f"{self.vocab[b]} -> {new_bytes} "
                      f"(count: {best_count})")

            next_token_id += 1

        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
        return self.vocab, self.merges</code></pre>

<h4>3. BPE Encoding: Applying Merge Rules</h4>
<p>After training, encoding new text requires applying the learned merge rules in priority order. This is where the trie data structure becomes essential for efficient lookup:</p>

<pre><code>class BPEEncoder:
    """Encode text using trained BPE merge rules.

    The encoding process applies merge rules in the exact order
    they were learned during training. This is critical: applying
    merges in a different order can produce different tokenizations.
    """

    def __init__(self, vocab: dict, merges: list):
        self.vocab = vocab
        self.merges = merges
        # Build a merge priority lookup: (a, b) -> priority (lower = merge first)
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}
        # Build inverse vocabulary for decoding
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        # Build trie for fast token lookup
        self.trie = TokenizerTrie()
        for token_id, token_bytes in vocab.items():
            self.trie.insert(token_bytes, token_id)

    def encode(self, text: str) -> list:
        """Encode text into a list of token IDs.

        Algorithm:
        1. Start with individual bytes
        2. Repeatedly find the highest-priority merge pair
           and merge it, until no more merges apply.

        Uses a priority queue to efficiently find the next merge.
        """
        byte_tokens = list(text.encode('utf-8'))

        # Convert to token IDs (initially, each byte is its own token)
        tokens = byte_tokens[:]

        while len(tokens) >= 2:
            # Find the pair with the highest priority (lowest index)
            # that exists in our merge rules
            best_pair = None
            best_priority = float('inf')
            best_position = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_position = i

            if best_pair is None:
                break  # No more applicable merges

            # Merge the best pair everywhere it occurs
            new_token_id = 256 + best_priority  # Token ID from merge index
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == best_pair[0] and
                    tokens[i + 1] == best_pair[1]):
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, token_ids: list) -> str:
        """Decode token IDs back to text."""
        byte_arrays = [self.vocab[tid] for tid in token_ids]
        return b''.join(byte_arrays).decode('utf-8', errors='replace')


# Example usage: train and use a BPE tokenizer
def demo_bpe_tokenizer():
    """Demonstrate the complete BPE training and encoding pipeline."""

    # Simulated word frequencies from a corpus
    corpus = {
        "low": 5, "lower": 2, "newest": 6,
        "widest": 3, "new": 4, "the": 10,
        "there": 5, "their": 4
    }

    # Train
    trainer = BPETrainer(target_vocab_size=300)
    vocab, merges = trainer.train(corpus)

    # Encode
    encoder = BPEEncoder(vocab, merges)
    text = "the newest lower"
    tokens = encoder.encode(text)
    decoded = encoder.decode(tokens)

    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Num tokens: {len(tokens)}")</code></pre>

<h4>4. Byte-Level BPE and the GPT Tokenizer</h4>
<p>Modern tokenizers like those used by GPT-4 and LLaMA operate at the <strong>byte level</strong> rather than the character level. This provides a crucial property: <em>any</em> text in <em>any</em> language can be tokenized without "unknown token" fallbacks, because every possible byte (0-255) is in the base vocabulary.</p>

<p>The <code>tiktoken</code> library (used by OpenAI) implements this with a particularly clever optimization: it uses a <strong>regex-based pre-tokenization</strong> pattern to split text into chunks before applying BPE. This prevents merges from crossing word boundaries, which would create unwanted tokens like "the dog" as a single unit.</p>

<pre><code>import regex  # Note: regex, not re (supports Unicode categories)

# The GPT-4 pre-tokenization pattern (simplified)
GPT4_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|"""           # Contractions
    r"""[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|"""  # Words (with optional leading punct)
    r"""\\p{N}{1,3}|"""                       # Numbers (up to 3 digits)
    r""" ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|"""  # Punctuation
    r"""\\s*[\\r\\n]|"""                      # Newlines
    r"""\\s+(?!\\S)|"""                        # Trailing whitespace
    r"""\\s+"""                                # Other whitespace
)

def pretokenize(text: str) -> list:
    """Split text into chunks before BPE, preventing cross-word merges.

    This is a critical preprocessing step. Without it, BPE would learn
    merges that span word boundaries, creating tokens like 'is a' or
    'of the' which would be inefficient and harm generalization.
    """
    chunks = GPT4_PATTERN.findall(text)
    # Convert each chunk to bytes for byte-level BPE
    return [chunk.encode('utf-8') for chunk in chunks]</code></pre>

<h4>5. Complexity Analysis</h4>
<table>
<tr><th>Operation</th><th>Time Complexity</th><th>Space Complexity</th><th>Notes</th></tr>
<tr><td>Trie insertion</td><td>O(m)</td><td>O(m)</td><td>m = token length in bytes</td></tr>
<tr><td>Trie longest-prefix match</td><td>O(m)</td><td>O(1)</td><td>Independent of vocab size</td></tr>
<tr><td>BPE training (naive)</td><td>O(V &times; N)</td><td>O(N)</td><td>V = merges, N = corpus tokens</td></tr>
<tr><td>BPE training (optimized)</td><td>O(N log N)</td><td>O(N)</td><td>With incremental heap updates</td></tr>
<tr><td>BPE encoding</td><td>O(n &times; m)</td><td>O(n)</td><td>n = input length, m = max merges applicable</td></tr>
<tr><td>Trie total space</td><td>&mdash;</td><td>O(V &times; L)</td><td>V = vocab size, L = avg token length</td></tr>
</table>

<div class="callout">
<div class="callout-title">Interview Question: Tokenizer Debugging</div>
<p><strong>Q:</strong> A user reports that prompts in Japanese consume 3x more tokens than equivalent English prompts of the same "meaning." You suspect a tokenizer issue. How would you diagnose and fix this?</p>
<p><strong>A:</strong> Japanese characters are encoded as 3 bytes each in UTF-8. If the BPE tokenizer's training corpus was predominantly English, it will have learned merges primarily for common English byte patterns. Japanese byte sequences will be under-represented in the merge rules, causing them to be split into more sub-tokens. Diagnosis: (1) check the training corpus language distribution, (2) compare token-to-character ratios across languages, (3) examine the merge list for Japanese-specific patterns. Fix: retrain the tokenizer with a more balanced multilingual corpus, or use a SentencePiece Unigram model which handles this more gracefully. Alternatively, add a language-specific pre-tokenization step that respects character boundaries.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Trie vs Hash Table</div>
<p><strong>Q:</strong> Why do tokenizers use a trie instead of a hash table for vocabulary lookup? After all, a hash table has O(1) average-case lookup.</p>
<p><strong>A:</strong> The critical operation in tokenization is <em>longest prefix matching</em>, not exact matching. With a hash table, you would need to check every possible prefix length: hash("h"), hash("he"), hash("hel"), hash("hell"), hash("hello")&mdash;that is O(m) hash lookups, each taking O(m) time to hash, giving O(m&sup2;) total. A trie traverses the string once, checking at each character whether it is at a valid token boundary, giving O(m) total. Additionally, tries support ordered iteration and prefix enumeration, which are useful for vocabulary analysis and debugging.</p>
</div>
`
    },

    // ----------------------------------------------------------
    // 15.2 Priority Queues and Beam Search
    // ----------------------------------------------------------
    {
      id: "dsa-beam-search",
      title: "Priority Queues and Beam Search",
      content: `
<p>Beam search is the workhorse decoding algorithm for sequence-to-sequence models, machine translation, speech recognition, and increasingly for structured reasoning with LLMs. At its core, beam search is a <strong>best-first search</strong> algorithm powered by a <strong>priority queue</strong> (heap). Understanding the data structure gives you the ability to implement efficient custom decoding strategies, tune beam parameters intelligently, and diagnose pathological decoding behaviors.</p>

<div class="callout">
<div class="callout-title">Key Insight</div>
<p>Beam search is a compromise between greedy search (beam=1, fast but suboptimal) and exhaustive search (beam=|V|, optimal but exponential). The priority queue is what makes it tractable: it lets us efficiently maintain and prune the k best candidates at each step.</p>
</div>

<h4>1. Heap-Based Priority Queue</h4>
<p>A <strong>binary heap</strong> supports insert and extract-min (or extract-max) in O(log n) time. For beam search, we use a <strong>min-heap of size k</strong> (the beam width) to maintain the top-k candidates:</p>

<pre><code>import heapq
from dataclasses import dataclass, field
from typing import List, Optional
import math

@dataclass(order=True)
class BeamCandidate:
    """A single candidate in beam search.

    The @dataclass(order=True) enables comparison based on fields
    in declaration order. We put score first so candidates are
    ordered by score in the heap.
    """
    score: float                                    # Log-probability (negative, lower = better for min-heap)
    sequence: List[int] = field(compare=False)      # Token IDs generated so far
    is_finished: bool = field(compare=False, default=False)
    hidden_state: Optional[object] = field(compare=False, default=None)  # Model state for continuation

class BeamSearchPriorityQueue:
    """Fixed-size priority queue for beam search.

    Maintains the top-k candidates using a min-heap.
    When a new candidate is better than the worst in the beam,
    we replace the worst. This gives us O(log k) insertion
    with automatic pruning.
    """

    def __init__(self, beam_width: int):
        self.beam_width = beam_width
        self.heap = []  # Min-heap: worst candidate is at top

    def push(self, candidate: BeamCandidate):
        """Add a candidate, evicting the worst if at capacity."""
        if len(self.heap) < self.beam_width:
            heapq.heappush(self.heap, candidate)
        elif candidate.score > self.heap[0].score:
            # New candidate is better than the worst in beam
            heapq.heapreplace(self.heap, candidate)
            # heapreplace is O(log k), more efficient than pop + push

    def get_all_sorted(self) -> List[BeamCandidate]:
        """Return all candidates sorted by score (best first)."""
        return sorted(self.heap, key=lambda c: c.score, reverse=True)

    def __len__(self):
        return len(self.heap)</code></pre>

<h4>2. Beam Search with Length Penalty</h4>
<p>A critical issue with naive beam search is <strong>length bias</strong>: shorter sequences have higher log-probabilities simply because they multiply fewer (less-than-one) probabilities. The length penalty addresses this:</p>

<p>The <strong>length penalty</strong> (Wu et al., 2016 - Google's NMT) normalizes scores:</p>
<p><code>score = log_prob / lp(length)</code> where <code>lp(y) = ((5 + |y|) / (5 + 1))^alpha</code></p>
<p>When alpha = 0: no penalty. alpha = 1: full length normalization. Typical values: 0.6-1.0.</p>

<pre><code>import numpy as np

def length_penalty(length: int, alpha: float = 0.6) -> float:
    """Compute length penalty factor (Wu et al., 2016).

    Args:
        length: Current sequence length
        alpha: Penalty exponent. 0 = no penalty, 1 = linear.
               Values 0.6-1.0 are typical.

    Returns:
        Penalty factor (>= 1.0). Divide log-prob by this to get
        the length-normalized score.
    """
    return ((5 + length) / (5 + 1)) ** alpha


def beam_search(model, encoder_output, beam_width: int = 5,
                max_length: int = 200, alpha: float = 0.6,
                eos_token_id: int = 2) -> List[BeamCandidate]:
    """Full beam search implementation with length penalty.

    Args:
        model: A seq2seq model with a .decode_step(token, state) method
               that returns (logits, new_state)
        encoder_output: Pre-computed encoder representations
        beam_width: Number of candidates to maintain
        max_length: Maximum output sequence length
        alpha: Length penalty exponent
        eos_token_id: End-of-sequence token ID

    Returns:
        List of completed hypotheses sorted by normalized score.

    Complexity:
        Time: O(T * k * V * log k) where T=max_length, k=beam_width, V=vocab_size
        Space: O(k * T) for storing beam candidates
    """

    # Initialize beam with start token
    bos_token_id = 1  # Beginning of sequence
    initial_state = model.init_decoder_state(encoder_output)

    active_beams = [
        BeamCandidate(
            score=0.0,
            sequence=[bos_token_id],
            hidden_state=initial_state
        )
    ]

    finished_beams = BeamSearchPriorityQueue(beam_width)

    for step in range(max_length):
        if not active_beams:
            break

        all_candidates = BeamSearchPriorityQueue(beam_width)

        for beam in active_beams:
            # Get model predictions for this beam
            logits, new_state = model.decode_step(
                beam.sequence[-1], beam.hidden_state
            )

            # Convert logits to log probabilities
            log_probs = log_softmax(logits)

            # Only consider top-k tokens (pruning for efficiency)
            # This reduces the inner loop from O(V) to O(k)
            top_k_log_probs, top_k_indices = top_k(log_probs, beam_width)

            for log_prob, token_id in zip(top_k_log_probs, top_k_indices):
                new_score = beam.score + log_prob
                new_sequence = beam.sequence + [token_id]

                # Apply length penalty for ranking
                lp = length_penalty(len(new_sequence), alpha)
                normalized_score = new_score / lp

                candidate = BeamCandidate(
                    score=normalized_score,
                    sequence=new_sequence,
                    is_finished=(token_id == eos_token_id),
                    hidden_state=new_state
                )

                if candidate.is_finished:
                    finished_beams.push(candidate)
                else:
                    all_candidates.push(candidate)

        active_beams = [c for c in all_candidates.get_all_sorted()]

        # Early stopping: if best finished beam is better than
        # all active beams, we can stop
        if len(finished_beams) >= beam_width:
            best_finished = finished_beams.get_all_sorted()[0].score
            best_active = max(b.score for b in active_beams) if active_beams else float('-inf')
            if best_finished >= best_active:
                break

    # If no finished beams, return the active beams
    if len(finished_beams) == 0:
        return sorted(active_beams, key=lambda c: c.score, reverse=True)

    return finished_beams.get_all_sorted()


def log_softmax(logits):
    """Numerically stable log-softmax."""
    max_val = max(logits)
    shifted = [x - max_val for x in logits]
    log_sum = math.log(sum(math.exp(x) for x in shifted))
    return [x - log_sum for x in shifted]

def top_k(values, k):
    """Return top-k values and their indices using a min-heap.

    Time complexity: O(n log k) where n = len(values)
    This is more efficient than sorting when k << n.
    """
    # Use a min-heap of size k
    heap = []
    for i, v in enumerate(values):
        if len(heap) < k:
            heapq.heappush(heap, (v, i))
        elif v > heap[0][0]:
            heapq.heapreplace(heap, (v, i))

    # Sort by value descending
    result = sorted(heap, reverse=True)
    return [v for v, i in result], [i for v, i in result]</code></pre>

<h4>3. Diverse Beam Search</h4>
<p>Standard beam search often produces near-identical outputs because high-probability beams tend to converge. <strong>Diverse Beam Search</strong> (Vijayakumar et al., 2018) splits the beam into groups and penalizes similarity between groups:</p>

<pre><code>def diverse_beam_search(model, encoder_output, beam_width: int = 10,
                        num_groups: int = 5, diversity_penalty: float = 0.5,
                        max_length: int = 200, alpha: float = 0.6,
                        eos_token_id: int = 2):
    """Diverse beam search that encourages variety in outputs.

    The key idea: divide beams into G groups. When scoring candidates
    for group g, penalize tokens that were already selected by
    groups 0..g-1. This is implemented by subtracting a
    diversity_penalty for each previous group that selected the
    same token at this timestep.

    Args:
        num_groups: Number of diversity groups (beam_width must be divisible)
        diversity_penalty: Lambda - higher values = more diversity

    Complexity: Same as standard beam search times num_groups factor.
    """
    assert beam_width % num_groups == 0, \
        "beam_width must be divisible by num_groups"
    group_size = beam_width // num_groups

    # Initialize each group with the start token
    bos_token_id = 1
    initial_state = model.init_decoder_state(encoder_output)

    groups = []
    for g in range(num_groups):
        groups.append([
            BeamCandidate(
                score=0.0,
                sequence=[bos_token_id],
                hidden_state=initial_state
            )
        ])

    for step in range(max_length):
        # Track which tokens each group selects at this step
        selected_tokens_by_group = []

        for g in range(num_groups):
            candidates = BeamSearchPriorityQueue(group_size)

            for beam in groups[g]:
                logits, new_state = model.decode_step(
                    beam.sequence[-1], beam.hidden_state
                )
                log_probs = log_softmax(logits)

                # Apply diversity penalty: subtract penalty for tokens
                # already chosen by previous groups
                penalized_log_probs = list(log_probs)
                for prev_group_tokens in selected_tokens_by_group:
                    for token_id in prev_group_tokens:
                        penalized_log_probs[token_id] -= diversity_penalty

                top_k_probs, top_k_ids = top_k(
                    penalized_log_probs, group_size * 2
                )

                for lp_val, token_id in zip(top_k_probs, top_k_ids):
                    # Use ORIGINAL (unpenalized) log-prob for the actual score
                    original_lp = log_probs[token_id]
                    new_score = beam.score + original_lp
                    new_sequence = beam.sequence + [token_id]

                    lp = length_penalty(len(new_sequence), alpha)
                    normalized_score = new_score / lp

                    candidates.push(BeamCandidate(
                        score=normalized_score,
                        sequence=new_sequence,
                        hidden_state=new_state
                    ))

            # Record which tokens this group selected
            group_beams = candidates.get_all_sorted()
            selected_tokens = set()
            for b in group_beams:
                if b.sequence:
                    selected_tokens.add(b.sequence[-1])
            selected_tokens_by_group.append(selected_tokens)

            groups[g] = group_beams

    # Flatten all groups and return sorted
    all_beams = []
    for g in groups:
        all_beams.extend(g)
    return sorted(all_beams, key=lambda c: c.score, reverse=True)</code></pre>

<h4>4. Complexity Comparison of Decoding Strategies</h4>
<table>
<tr><th>Strategy</th><th>Time per Step</th><th>Space</th><th>Quality</th><th>Use Case</th></tr>
<tr><td>Greedy (beam=1)</td><td>O(V)</td><td>O(1)</td><td>Lowest</td><td>Real-time chat, streaming</td></tr>
<tr><td>Beam search (beam=k)</td><td>O(k &times; V &times; log k)</td><td>O(k &times; T)</td><td>High</td><td>Translation, ASR</td></tr>
<tr><td>Diverse beam (G groups)</td><td>O(G &times; k/G &times; V &times; log k)</td><td>O(k &times; T)</td><td>High + diverse</td><td>Caption generation, multi-answer</td></tr>
<tr><td>Sampling (top-p)</td><td>O(V log V)</td><td>O(1)</td><td>Variable</td><td>Creative text generation</td></tr>
</table>

<div class="callout">
<div class="callout-title">Interview Question: Beam Search Pathology</div>
<p><strong>Q:</strong> You are using beam search for machine translation and notice that it consistently produces outputs that are 20-30% shorter than reference translations. What is happening and how do you fix it?</p>
<p><strong>A:</strong> This is the classic <strong>length bias</strong> problem. Log-probabilities are always negative, so longer sequences accumulate more negative score. Beam search favors shorter sequences because they have higher (less negative) total log-probability. Fixes: (1) Apply length penalty (alpha=0.6-1.0) to normalize scores by length. (2) Use minimum length constraints that prevent EOS from being selected before a threshold. (3) Use length-reward: add a small positive bonus for each generated token. The root cause is that the model's probability of EOS is too high at intermediate positions, which itself may indicate insufficient training data of longer sequences.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Priority Queue Choice</div>
<p><strong>Q:</strong> Why does beam search use a min-heap (priority queue) instead of just sorting all candidates at each step?</p>
<p><strong>A:</strong> At each step, we generate k &times; V candidates (each beam expanded by the full vocabulary). Sorting all of them is O(kV log(kV)). With a min-heap of size k, we can find the top-k candidates in O(kV log k), saving a factor of log(V/k). For V=50,000 and k=5, this is roughly a 10x speedup. Furthermore, the heap-based approach allows early termination: if a candidate's score is lower than the current minimum in the heap, we can skip it immediately without insertion. The heapreplace operation is particularly efficient as it combines pop and push in a single O(log k) operation.</p>
</div>
`
    },

    // ----------------------------------------------------------
    // 15.3 Hash Tables for KV-Cache
    // ----------------------------------------------------------
    {
      id: "dsa-kv-cache",
      title: "Hash Tables for KV-Cache",
      content: `
<p>The KV-cache is one of the most critical performance components in LLM inference. It stores previously computed key and value tensors from the attention mechanism, avoiding redundant computation during autoregressive generation. Managing this cache efficiently&mdash;especially in multi-tenant serving environments&mdash;requires sophisticated use of <strong>hash tables</strong>, <strong>page tables</strong>, and <strong>eviction policies</strong>. These are fundamental data structure problems solved with elegant algorithms.</p>

<div class="callout">
<div class="callout-title">Why KV-Cache Management is a DS&amp;A Problem</div>
<p>A single LLaMA-70B request with a 4K context window requires approximately 2.5 GB of KV-cache memory in FP16. When serving hundreds of concurrent requests with varying context lengths, you face a classic memory management problem: fragmentation, allocation, eviction, and sharing&mdash;the same problems that operating system kernels solve with page tables and LRU caches.</p>
</div>

<h4>1. Hash Table Fundamentals Applied to KV-Cache</h4>
<p>A hash table provides O(1) average-case lookup, insertion, and deletion. For KV-cache, we need to:</p>
<ul>
  <li>Map <strong>(request_id, layer, head, position)</strong> to a block of cached K/V tensors</li>
  <li>Quickly determine whether a prefix has already been computed (prefix caching)</li>
  <li>Efficiently reclaim memory when requests complete</li>
</ul>

<pre><code>from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import hashlib
import numpy as np

@dataclass
class KVBlock:
    """A fixed-size block of KV-cache memory.

    Analogous to a memory page in an OS. Each block stores
    KV tensors for a fixed number of token positions.
    """
    block_id: int
    key_data: np.ndarray       # Shape: (block_size, num_heads, head_dim)
    value_data: np.ndarray     # Shape: (block_size, num_heads, head_dim)
    num_filled: int = 0        # How many positions are actually filled
    ref_count: int = 0         # Number of sequences referencing this block
    block_size: int = 16       # Tokens per block

    @property
    def is_full(self) -> bool:
        return self.num_filled >= self.block_size


class KVBlockAllocator:
    """Page-table-style block allocator for KV-cache.

    This is the core memory management system used by systems like
    vLLM's PagedAttention. Instead of pre-allocating contiguous memory
    per request (which causes massive fragmentation), we allocate
    fixed-size blocks on demand.

    The key insight: just as virtual memory decouples logical addresses
    from physical memory, PagedAttention decouples logical token positions
    from physical GPU memory locations.
    """

    def __init__(self, num_blocks: int, block_size: int,
                 num_heads: int, head_dim: int):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-allocate all blocks (like a physical memory pool)
        self.blocks = {}
        self.free_blocks = list(range(num_blocks))

        for i in range(num_blocks):
            self.blocks[i] = KVBlock(
                block_id=i,
                key_data=np.zeros((block_size, num_heads, head_dim)),
                value_data=np.zeros((block_size, num_heads, head_dim)),
                block_size=block_size
            )

        # Page table: maps (request_id, logical_block_idx) -> physical_block_id
        self.page_table: Dict[Tuple[int, int], int] = {}

    def allocate_block(self, request_id: int, logical_idx: int) -> Optional[int]:
        """Allocate a physical block for a logical position.

        Returns:
            Physical block ID, or None if OOM.
        """
        if not self.free_blocks:
            return None  # Out of memory - trigger eviction

        physical_id = self.free_blocks.pop()
        self.page_table[(request_id, logical_idx)] = physical_id
        self.blocks[physical_id].ref_count = 1
        return physical_id

    def free_request(self, request_id: int):
        """Free all blocks belonging to a request.

        Decrements reference count and returns block to free list
        when ref_count reaches 0 (copy-on-write support).
        """
        keys_to_remove = [
            key for key in self.page_table
            if key[0] == request_id
        ]
        for key in keys_to_remove:
            physical_id = self.page_table.pop(key)
            block = self.blocks[physical_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                block.num_filled = 0
                self.free_blocks.append(physical_id)

    def get_block(self, request_id: int, logical_idx: int) -> Optional[KVBlock]:
        """Look up a block via the page table. O(1) hash table lookup."""
        physical_id = self.page_table.get((request_id, logical_idx))
        if physical_id is not None:
            return self.blocks[physical_id]
        return None

    def memory_usage(self) -> dict:
        """Report memory utilization statistics."""
        total = len(self.blocks)
        used = total - len(self.free_blocks)
        return {
            "total_blocks": total,
            "used_blocks": used,
            "free_blocks": len(self.free_blocks),
            "utilization": used / total if total > 0 else 0,
            "fragmentation": 0.0  # Paged allocation eliminates fragmentation!
        }</code></pre>

<h4>2. LRU Eviction with OrderedDict</h4>
<p>When GPU memory is exhausted, we need an eviction policy. <strong>Least Recently Used (LRU)</strong> is the standard choice, and Python's <code>OrderedDict</code> provides an elegant O(1) implementation:</p>

<pre><code>class LRUKVCache:
    """LRU eviction policy for KV-cache blocks.

    Uses OrderedDict which maintains insertion order and supports
    O(1) move-to-end and pop-from-front operations. This is
    implemented internally as a hash table + doubly linked list.

    The invariant: the least recently used item is always at the
    front of the OrderedDict, and the most recently used is at the end.
    """

    def __init__(self, max_blocks: int, block_size: int = 16,
                 num_heads: int = 32, head_dim: int = 128):
        self.max_blocks = max_blocks
        self.allocator = KVBlockAllocator(
            max_blocks, block_size, num_heads, head_dim
        )
        # OrderedDict tracks access order for LRU eviction
        # Key: (request_id, logical_idx), Value: physical_block_id
        self.access_order = OrderedDict()

    def access(self, request_id: int, logical_idx: int) -> Optional[KVBlock]:
        """Access a KV-cache block, updating LRU order.

        Move the accessed entry to the end (most recent).
        Time complexity: O(1) amortized.
        """
        key = (request_id, logical_idx)
        if key in self.access_order:
            # Move to end (most recently used)
            self.access_order.move_to_end(key)
            return self.allocator.get_block(request_id, logical_idx)
        return None

    def allocate(self, request_id: int, logical_idx: int) -> Optional[KVBlock]:
        """Allocate a new block, evicting LRU if necessary."""
        key = (request_id, logical_idx)

        # Try to allocate directly
        physical_id = self.allocator.allocate_block(request_id, logical_idx)

        if physical_id is None:
            # Out of memory: evict least recently used
            if not self.access_order:
                return None  # Nothing to evict

            evict_key, evict_physical = self.access_order.popitem(last=False)
            evict_req, evict_idx = evict_key

            # Free the evicted block
            self.allocator.free_request(evict_req)

            # Retry allocation
            physical_id = self.allocator.allocate_block(request_id, logical_idx)
            if physical_id is None:
                return None

        self.access_order[key] = physical_id
        return self.allocator.blocks[physical_id]

    def evict_request(self, request_id: int):
        """Evict all blocks for a completed request."""
        keys_to_remove = [
            k for k in self.access_order if k[0] == request_id
        ]
        for key in keys_to_remove:
            del self.access_order[key]
        self.allocator.free_request(request_id)</code></pre>

<h4>3. Prefix Caching with Hash-Based Matching</h4>
<p>Many LLM requests share common prefixes (system prompts, few-shot examples, shared context). <strong>Prefix caching</strong> avoids recomputing KV-cache for shared prefixes by using content-based hashing to detect matches:</p>

<pre><code>class PrefixCache:
    """Content-addressed prefix cache for KV-cache sharing.

    Uses cryptographic hashing to identify matching prefixes across
    requests. When two requests share the same system prompt, only
    one copy of the KV-cache for that prefix is stored.

    This is the technique behind "automatic prefix caching" in
    vLLM and similar systems. It saves both memory and computation.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        # Hash table: content_hash -> physical_block_id
        self.prefix_table: Dict[str, int] = {}
        # Reference counting for safe deallocation
        self.ref_counts: Dict[str, int] = {}

    def _hash_token_block(self, tokens: tuple, layer: int,
                          parent_hash: str = "") -> str:
        """Compute content hash for a block of tokens.

        The hash includes:
        - The tokens themselves
        - The layer index
        - The parent block's hash (for chain integrity)

        This creates a hash chain (like a Merkle tree) that ensures
        two blocks match only if ALL preceding tokens also match.
        """
        content = f"{parent_hash}|layer={layer}|tokens={tokens}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def lookup_or_allocate(self, token_ids: list, layer: int,
                           allocator: KVBlockAllocator,
                           request_id: int) -> list:
        """Look up prefix in cache, allocating only for new suffixes.

        Returns:
            List of (physical_block_id, was_cached) tuples.

        This is the key function: it walks through the token sequence
        block by block, checking the hash table for each block.
        As soon as a cache miss occurs, all subsequent blocks are
        freshly allocated.
        """
        results = []
        parent_hash = ""
        num_blocks = (len(token_ids) + self.block_size - 1) // self.block_size

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(token_ids))
            block_tokens = tuple(token_ids[start:end])

            content_hash = self._hash_token_block(
                block_tokens, layer, parent_hash
            )

            if content_hash in self.prefix_table:
                # Cache HIT: reuse existing block
                physical_id = self.prefix_table[content_hash]
                self.ref_counts[content_hash] += 1
                results.append((physical_id, True))
            else:
                # Cache MISS: allocate new block
                physical_id = allocator.allocate_block(
                    request_id, block_idx
                )
                if physical_id is not None:
                    self.prefix_table[content_hash] = physical_id
                    self.ref_counts[content_hash] = 1
                    results.append((physical_id, False))
                else:
                    results.append((None, False))  # OOM

            parent_hash = content_hash

        return results

    def report_stats(self) -> dict:
        """Report cache hit statistics."""
        total_refs = sum(self.ref_counts.values())
        unique_blocks = len(self.prefix_table)
        return {
            "unique_blocks_cached": unique_blocks,
            "total_references": total_refs,
            "sharing_ratio": total_refs / unique_blocks if unique_blocks > 0 else 0,
            "memory_saved_factor": total_refs / unique_blocks if unique_blocks > 0 else 1,
        }</code></pre>

<h4>4. Copy-on-Write for Forked Sequences</h4>
<p>When generating multiple completions from the same prompt (e.g., n=4 in the API), the prompt's KV-cache can be shared using <strong>copy-on-write</strong> (COW)&mdash;another operating systems concept applied to AI serving:</p>

<pre><code>class CopyOnWriteKVManager:
    """Copy-on-Write KV-cache for forked sequences.

    When multiple sequences share a prefix, they share the same
    physical blocks. Only when a sequence modifies a shared block
    (by appending new tokens) does it get its own copy.

    This is identical to how Unix fork() works: parent and child
    share memory pages until one of them writes.
    """

    def __init__(self, allocator: KVBlockAllocator):
        self.allocator = allocator
        # Maps (request_id, logical_idx) -> physical_block_id
        self.mappings: Dict[Tuple[int, int], int] = {}

    def fork(self, parent_id: int, child_id: int, num_blocks: int):
        """Fork a sequence: child shares parent's blocks.

        This is O(num_blocks) pointer copies, NOT data copies.
        The actual KV tensors are not duplicated.
        """
        for logical_idx in range(num_blocks):
            parent_key = (parent_id, logical_idx)
            if parent_key in self.mappings:
                physical_id = self.mappings[parent_key]
                # Increment reference count (shared block)
                self.allocator.blocks[physical_id].ref_count += 1
                # Child points to same physical block
                self.mappings[(child_id, logical_idx)] = physical_id

    def write(self, request_id: int, logical_idx: int,
              key_data, value_data):
        """Write to a block, copying if shared (COW).

        If ref_count > 1, this block is shared. We must:
        1. Allocate a new block
        2. Copy the old data
        3. Write the new data
        4. Decrement old block's ref_count
        """
        key = (request_id, logical_idx)
        physical_id = self.mappings.get(key)

        if physical_id is None:
            # New block needed
            physical_id = self.allocator.allocate_block(
                request_id, logical_idx
            )
            self.mappings[key] = physical_id
            block = self.allocator.blocks[physical_id]
        else:
            block = self.allocator.blocks[physical_id]
            if block.ref_count > 1:
                # COW: copy before writing
                new_physical = self.allocator.free_blocks.pop()
                new_block = self.allocator.blocks[new_physical]
                # Copy existing data
                np.copyto(new_block.key_data, block.key_data)
                np.copyto(new_block.value_data, block.value_data)
                new_block.num_filled = block.num_filled
                new_block.ref_count = 1
                # Update reference counts
                block.ref_count -= 1
                self.mappings[key] = new_physical
                block = new_block

        # Perform the actual write
        pos = block.num_filled
        block.key_data[pos] = key_data
        block.value_data[pos] = value_data
        block.num_filled += 1</code></pre>

<div class="callout">
<div class="callout-title">Interview Question: KV-Cache Memory</div>
<p><strong>Q:</strong> You are serving a LLaMA-13B model and notice that GPU memory utilization is only 40% but requests are being rejected with OOM errors. What could explain this?</p>
<p><strong>A:</strong> This is the <strong>memory fragmentation</strong> problem. Without paged allocation, each request pre-allocates a contiguous block of GPU memory for its maximum possible sequence length. Even though total free memory is 60%, it may be scattered in non-contiguous chunks too small for any single request. The solution is PagedAttention (vLLM): allocate KV-cache in fixed-size blocks (like OS pages) mapped through a page table. This eliminates fragmentation because any free block can serve any request. Typical improvement: from 40% to 95%+ memory utilization, effectively 2-3x throughput increase.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Prefix Cache Design</div>
<p><strong>Q:</strong> Design a prefix cache that works across multiple GPUs in a distributed inference setup. What data structure challenges arise?</p>
<p><strong>A:</strong> The core challenge is maintaining a <strong>distributed hash table</strong> for prefix matching across GPUs. Key considerations: (1) The prefix hash table itself fits in CPU memory since it stores only hashes to block IDs, not the actual tensors. (2) Use consistent hashing to assign prefix blocks to GPUs, so the same system prompt always routes to the same GPU. (3) Reference counting must be atomic across GPUs&mdash;use a centralized coordinator or distributed reference counting with lease-based expiration. (4) Hash collisions in the prefix table are catastrophic (incorrect KV data), so use cryptographic hashes (SHA-256) not fast hashes (xxHash). The system resembles a distributed memory cache like memcached, but with GPU memory as the backing store.</p>
</div>
`
    },

    // ----------------------------------------------------------
    // 15.4 Matrix Operations for Attention
    // ----------------------------------------------------------
    {
      id: "dsa-attention",
      title: "Matrix Operations for Attention",
      content: `
<p>The attention mechanism is the computational heart of the Transformer. Understanding it as a <strong>matrix multiplication problem</strong> reveals why it is both powerful and expensive, and understanding <strong>tiled computation</strong> reveals the key insight behind FlashAttention&mdash;the single most impactful optimization in modern LLM inference.</p>

<div class="callout">
<div class="callout-title">The Memory Wall</div>
<p>Modern GPUs can perform far more arithmetic operations per second than they can move data to/from memory. An A100 GPU does 312 TFLOPS of FP16 math but has only 2 TB/s of memory bandwidth. Standard attention is <strong>memory-bound</strong>, not compute-bound: it reads and writes the N&times;N attention matrix to HBM, which dominates runtime. FlashAttention eliminates this bottleneck by never materializing the full attention matrix.</p>
</div>

<h4>1. Attention as Matrix Multiplication</h4>
<p>Standard scaled dot-product attention:</p>
<p><code>Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V</code></p>

<p>Where Q, K, V have shape (N, d) for sequence length N and head dimension d:</p>
<ul>
  <li><strong>QK^T:</strong> (N, d) &times; (d, N) = (N, N) &mdash; O(N&sup2;d) FLOPs, O(N&sup2;) memory</li>
  <li><strong>softmax:</strong> Applied row-wise to the N &times; N matrix &mdash; O(N&sup2;) operations</li>
  <li><strong>Score &times; V:</strong> (N, N) &times; (N, d) = (N, d) &mdash; O(N&sup2;d) FLOPs</li>
</ul>

<p>Total: <strong>O(N&sup2;d) FLOPs</strong> and <strong>O(N&sup2;) memory</strong>. For N=128K, the attention matrix alone is 128K &times; 128K &times; 2 bytes = 32 GB in FP16&mdash;larger than most GPU memories.</p>

<pre><code>import numpy as np

def standard_attention(Q, K, V):
    """Standard attention - simple but memory-hungry.

    Args:
        Q: Query matrix, shape (N, d)
        K: Key matrix, shape (N, d)
        V: Value matrix, shape (N, d)

    Returns:
        Output matrix, shape (N, d)

    Memory: O(N^2) for the attention score matrix
    FLOPs: O(N^2 * d)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores - creates N x N matrix!
    scores = Q @ K.T / np.sqrt(d_k)     # (N, N) - this is the memory bottleneck

    # Step 2: Softmax (row-wise)
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)  # Numerical stability
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Step 3: Weighted sum of values
    output = attention_weights @ V       # (N, d)

    return output</code></pre>

<h4>2. Tiled Matrix Multiplication: The Foundation</h4>
<p>Before understanding FlashAttention, you must understand <strong>tiling</strong> (also called blocking). Tiled matrix multiplication processes matrices in small blocks that fit in fast cache (SRAM), reducing slow memory (HBM) accesses:</p>

<pre><code>def tiled_matmul(A, B, block_size=64):
    """Tiled (blocked) matrix multiplication.

    Instead of accessing memory in a pattern that causes cache misses,
    we process the matrices in small blocks that fit in L1/L2 cache.

    Standard matmul: O(N^3) FLOPs, O(N^3) memory accesses
    Tiled matmul: O(N^3) FLOPs, O(N^3 / sqrt(M)) memory accesses
    where M is the cache size in elements.

    The FLOP count is identical, but memory accesses are dramatically
    reduced. This is why tiling speeds up computation even though
    it doesn't reduce the amount of arithmetic.
    """
    N = A.shape[0]
    M = B.shape[1]
    K_dim = A.shape[1]
    C = np.zeros((N, M))

    for i in range(0, N, block_size):
        for j in range(0, M, block_size):
            # Accumulator for this output block
            c_block = np.zeros((
                min(block_size, N - i),
                min(block_size, M - j)
            ))

            for k in range(0, K_dim, block_size):
                # Load blocks into "cache" (SRAM in GPU terms)
                a_block = A[i:i+block_size, k:k+block_size]
                b_block = B[k:k+block_size, j:j+block_size]

                # Multiply blocks - this happens in fast memory
                c_block += a_block @ b_block

            # Write result block back
            C[i:i+block_size, j:j+block_size] = c_block

    return C</code></pre>

<h4>3. Online Softmax: The Key to Tiled Attention</h4>
<p>The challenge with tiling attention is the softmax. Standard softmax requires two passes over the data: one to find the maximum (for numerical stability), and one to compute the exponentials and normalize. The <strong>online softmax</strong> algorithm (Milakov & Gimelshein, 2018) computes softmax in a single pass, enabling tiled attention:</p>

<pre><code>def online_softmax(x):
    """Online (streaming) softmax - single pass over data.

    Standard softmax requires two passes:
      Pass 1: max_val = max(x)
      Pass 2: exp(x - max_val) / sum(exp(x - max_val))

    Online softmax maintains running max and running sum,
    correcting previous computations when a new max is found.
    This enables processing data in tiles without ever seeing
    the full vector at once.
    """
    running_max = float('-inf')
    running_sum = 0.0
    output = np.zeros_like(x)

    for i in range(len(x)):
        new_max = max(running_max, x[i])

        if running_max != float('-inf'):
            # Correction factor: rescale previous sum to new max
            correction = np.exp(running_max - new_max)
            running_sum = running_sum * correction

        running_sum += np.exp(x[i] - new_max)
        running_max = new_max

    # Final pass to compute normalized values
    for i in range(len(x)):
        output[i] = np.exp(x[i] - running_max) / running_sum

    return output


def online_softmax_single_pass(x):
    """True single-pass online softmax with incremental output.

    This version computes the softmax incrementally, which is
    exactly what FlashAttention needs: process one tile of K
    at a time, updating the running output.
    """
    n = len(x)
    m = float('-inf')  # Running max
    l = 0.0            # Running sum of exp(x_i - m)

    for i in range(n):
        m_new = max(m, x[i])
        l = l * np.exp(m - m_new) + np.exp(x[i] - m_new)
        m = m_new

    # Normalized output
    return np.array([np.exp(x[i] - m) / l for i in range(n)])</code></pre>

<h4>4. Tiled Attention (FlashAttention Core Idea)</h4>
<p>FlashAttention (Dao et al., 2022) combines tiling with online softmax to compute exact attention without ever materializing the N&times;N attention matrix in HBM:</p>

<pre><code>def flash_attention_simplified(Q, K, V, block_size=64):
    """Simplified FlashAttention algorithm.

    The key insight: we can compute attention output incrementally,
    processing blocks of K and V at a time, while keeping a running
    softmax normalization. The output for each query is updated as
    we process each key-value block.

    Memory: O(N) instead of O(N^2) - the N x N matrix is NEVER stored
    FLOPs: Same O(N^2 * d) - we don't save computation, we save memory

    The speedup comes from reduced HBM I/O:
    - Standard: O(N^2 * d) HBM accesses (read/write the NxN matrix)
    - Flash: O(N^2 * d^2 / M) HBM accesses (M = SRAM size)
    For typical d=128 and M=100KB, this is 10-20x fewer memory accesses.
    """
    N, d = Q.shape
    output = np.zeros((N, d))

    # Process queries in blocks
    for q_start in range(0, N, block_size):
        q_end = min(q_start + block_size, N)
        Q_block = Q[q_start:q_end]         # Load Q tile to SRAM
        block_len = q_end - q_start

        # For each query block, maintain running softmax state
        # m_i: running row-wise maximum of attention scores
        # l_i: running row-wise sum of exp(scores - m_i)
        # O_i: running output accumulator (unnormalized)
        m_i = np.full((block_len, 1), float('-inf'))
        l_i = np.zeros((block_len, 1))
        O_i = np.zeros((block_len, d))

        # Iterate over K, V blocks
        for kv_start in range(0, N, block_size):
            kv_end = min(kv_start + block_size, N)
            K_block = K[kv_start:kv_end]   # Load K tile to SRAM
            V_block = V[kv_start:kv_end]   # Load V tile to SRAM

            # Compute attention scores for this tile
            # (block_size, d) @ (d, block_size) = (block_size, block_size)
            S_block = Q_block @ K_block.T / np.sqrt(d)

            # Online softmax update
            m_new = np.maximum(m_i, S_block.max(axis=-1, keepdims=True))

            # Rescale previous accumulations to new max
            exp_correction = np.exp(m_i - m_new)
            exp_scores = np.exp(S_block - m_new)

            l_new = exp_correction * l_i + exp_scores.sum(axis=-1, keepdims=True)

            # Update output: rescale old output and add new contribution
            O_i = (exp_correction * l_i * O_i / l_new +
                   exp_scores @ V_block / l_new)

            # Note: the above is mathematically equivalent to:
            # O_i = exp_correction * l_i / l_new * O_i + exp_scores / l_new @ V_block
            # which separates the rescaling of old output from new contribution

            m_i = m_new
            l_i = l_new

        output[q_start:q_end] = O_i

    return output


# Verification: FlashAttention produces identical results
def verify_flash_attention():
    """Verify that flash attention matches standard attention."""
    np.random.seed(42)
    N, d = 256, 64
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    standard_out = standard_attention(Q, K, V)
    flash_out = flash_attention_simplified(Q, K, V, block_size=32)

    max_diff = np.abs(standard_out - flash_out).max()
    print(f"Max difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Results differ by {max_diff}"
    print("FlashAttention output matches standard attention!")</code></pre>

<h4>5. IO Complexity Analysis</h4>
<table>
<tr><th>Algorithm</th><th>FLOPs</th><th>HBM Reads/Writes</th><th>Extra Memory</th><th>Exact?</th></tr>
<tr><td>Standard Attention</td><td>O(N&sup2;d)</td><td>O(N&sup2; + Nd)</td><td>O(N&sup2;)</td><td>Yes</td></tr>
<tr><td>FlashAttention</td><td>O(N&sup2;d)</td><td>O(N&sup2;d&sup2;/M)</td><td>O(N)</td><td>Yes</td></tr>
<tr><td>FlashAttention-2</td><td>O(N&sup2;d)</td><td>O(N&sup2;d&sup2;/M)</td><td>O(N)</td><td>Yes</td></tr>
<tr><td>Multi-Query Attention</td><td>O(N&sup2;d/H)</td><td>O(N&sup2;/H + Nd)</td><td>O(N&sup2;/H)</td><td>Yes</td></tr>
<tr><td>Linear Attention</td><td>O(Nd&sup2;)</td><td>O(Nd)</td><td>O(d&sup2;)</td><td>Approximate</td></tr>
</table>
<p>M = SRAM size (typically ~100KB per SM on A100), H = number of attention heads, N = sequence length, d = head dimension.</p>

<div class="callout">
<div class="callout-title">Interview Question: FlashAttention</div>
<p><strong>Q:</strong> FlashAttention doesn't reduce the number of FLOPs compared to standard attention. So why is it 2-4x faster?</p>
<p><strong>A:</strong> Because standard attention is <strong>memory-bound</strong>, not compute-bound. The GPU spends most of its time waiting for data to move between HBM (slow, high-capacity) and SRAM (fast, small). Standard attention reads Q, K from HBM, writes the N&times;N score matrix to HBM, reads it back for softmax, writes softmax output to HBM, reads it back for V multiplication, and writes the final output. FlashAttention keeps all intermediate results (the tile-sized score matrices) in SRAM and only reads inputs and writes outputs to HBM. The total HBM traffic drops from O(N&sup2;) to O(N&sup2;d/M), which for typical sizes is 10-20x less. This moves attention from memory-bound to compute-bound, allowing the GPU to actually utilize its arithmetic throughput.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Causal Masking</div>
<p><strong>Q:</strong> How does FlashAttention handle causal masking efficiently?</p>
<p><strong>A:</strong> In causal (autoregressive) attention, positions can only attend to previous positions. In tiled FlashAttention, when processing a Q-block [i:i+B] against a K-block [j:j+B], there are three cases: (1) If j+B &le; i, the entire tile is unmasked&mdash;process normally. (2) If j &gt; i+B, the entire tile is masked&mdash;skip it entirely (free speedup!). (3) If the tile straddles the diagonal, apply the mask within the tile. Case (2) means FlashAttention with causal masking processes roughly half the tiles, giving approximately 2x speedup over non-causal FlashAttention for autoregressive models. This is implemented in FlashAttention-2 and is why causal LLM inference is faster than bidirectional models of the same size.</p>
</div>
`
    },

    // ----------------------------------------------------------
    // 15.5 Graphs for Computation
    // ----------------------------------------------------------
    {
      id: "dsa-graphs",
      title: "Graphs for Computation",
      content: `
<p>Deep learning is, at its mathematical core, a problem of composing differentiable functions into a <strong>computation graph</strong> and then computing gradients through that graph via <strong>reverse-mode automatic differentiation</strong> (backpropagation). Understanding computation graphs, topological sorting, and graph partitioning gives you the foundation to understand autograd systems, pipeline parallelism, and model compilation.</p>

<div class="callout">
<div class="callout-title">The Big Picture</div>
<p>Every time you call <code>loss.backward()</code> in PyTorch, you are running a topological sort on a directed acyclic graph (DAG), then traversing it in reverse order to compute gradients. Every time you use pipeline parallelism to train a large model across GPUs, you are solving a graph partitioning problem. These are classical graph algorithms applied to AI.</p>
</div>

<h4>1. Computation Graphs and Autograd</h4>
<p>A computation graph is a DAG where:</p>
<ul>
  <li><strong>Nodes</strong> represent operations (add, multiply, relu, matmul) or variables (weights, inputs)</li>
  <li><strong>Edges</strong> represent data flow (tensors flowing between operations)</li>
  <li>The graph is <strong>acyclic</strong> because each operation produces a new tensor (no circular dependencies)</li>
</ul>

<p>Here is a complete implementation of a simple autograd engine, demonstrating the key graph algorithms:</p>

<pre><code>import math

class Value:
    """A scalar value with automatic differentiation support.

    This is a simplified version of what PyTorch's autograd does.
    Each Value is a node in the computation graph. Operations on
    Values create new nodes with edges back to their inputs
    (the _prev set), building the graph dynamically.

    Based on Andrej Karpathy's micrograd, extended with additional
    operations and detailed commentary on the graph algorithms.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0

        # Graph structure
        self._prev = set(_children)   # Parent nodes (inputs to this op)
        self._op = _op                 # Operation that created this node
        self.label = label

        # Backward function: computes gradients for parent nodes
        # This is the "local gradient" in the chain rule
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # Gradients accumulate (+=) because a value may be used
            # multiple times in the graph (fan-out)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d(exp(x))/dx = exp(x)
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # d(log(x))/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Compute gradients via reverse-mode autodiff (backpropagation).

        This is the heart of the algorithm:
        1. Topologically sort the computation graph
        2. Set the gradient of the output to 1.0
        3. Traverse in reverse topological order, calling each
           node's _backward() function

        The topological sort ensures that when we process a node,
        all nodes that USE its output have already propagated their
        gradients back. This is what makes the chain rule work.
        """
        # Step 1: Topological sort using DFS
        topo_order = []
        visited = set()

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    dfs(parent)
                topo_order.append(node)

        dfs(self)

        # Step 2: Reverse traverse and accumulate gradients
        self.grad = 1.0  # d(output)/d(output) = 1

        for node in reversed(topo_order):
            node._backward()


# Demonstration: training a tiny neural network
def demo_autograd():
    """Build and train a small network to learn XOR."""

    # Create weights (these are the learnable parameters)
    # 2 inputs -> 2 hidden neurons -> 1 output
    w1 = [[Value(0.5, label='w1_00'), Value(-0.3, label='w1_01')],
           [Value(0.2, label='w1_10'), Value(0.8, label='w1_11')]]
    b1 = [Value(0.0, label='b1_0'), Value(0.0, label='b1_1')]
    w2 = [Value(0.6, label='w2_0'), Value(-0.4, label='w2_1')]
    b2 = Value(0.0, label='b2')

    params = [w1[0][0], w1[0][1], w1[1][0], w1[1][1],
              b1[0], b1[1], w2[0], w2[1], b2]

    # XOR training data
    data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]

    learning_rate = 0.1

    for epoch in range(100):
        total_loss = Value(0.0)

        for inputs, target in data:
            x0 = Value(inputs[0])
            x1 = Value(inputs[1])

            # Forward pass: 2-layer network
            h0 = (x0 * w1[0][0] + x1 * w1[1][0] + b1[0]).tanh()
            h1 = (x0 * w1[0][1] + x1 * w1[1][1] + b1[1]).tanh()
            out = (h0 * w2[0] + h1 * w2[1] + b2).tanh()

            # MSE loss for this sample
            loss = (out - Value(target)) ** 2
            total_loss = total_loss + loss

        # Backpropagation
        for p in params:
            p.grad = 0.0  # Zero gradients

        total_loss.backward()

        # Gradient descent update
        for p in params:
            p.data -= learning_rate * p.grad

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.data:.6f}")</code></pre>

<h4>2. Topological Sort for Backpropagation</h4>
<p>The topological sort in <code>backward()</code> above uses DFS, but there is an alternative: <strong>Kahn's algorithm</strong> using a queue, which can be more efficient for large graphs and enables parallel execution:</p>

<pre><code>from collections import deque

def kahns_topological_sort(output_node):
    """Kahn's algorithm for topological sorting.

    Advantages over DFS-based sort:
    1. Naturally identifies parallelizable groups (nodes at the same "level")
    2. Detects cycles (if not all nodes are visited)
    3. More cache-friendly memory access pattern

    In deep learning frameworks, this is used for:
    - Scheduling operations on GPU streams
    - Identifying which operations can run in parallel
    - Memory allocation planning (when to free intermediate tensors)
    """
    # Step 1: Compute in-degree for each node
    all_nodes = set()
    in_degree = {}

    def collect_nodes(node):
        if node in all_nodes:
            return
        all_nodes.add(node)
        in_degree[node] = 0
        for parent in node._prev:
            collect_nodes(parent)

    collect_nodes(output_node)

    # Count outgoing edges (how many nodes depend on each node)
    out_edges = {node: set() for node in all_nodes}
    for node in all_nodes:
        for parent in node._prev:
            out_edges[parent].add(node)
            in_degree[node] = in_degree.get(node, 0) + 1

    # Step 2: Initialize queue with source nodes (in-degree 0)
    queue = deque()
    for node in all_nodes:
        if in_degree.get(node, 0) == 0:
            queue.append(node)

    # Step 3: Process nodes level by level
    topo_order = []
    levels = []  # Groups of parallelizable nodes

    while queue:
        # All nodes in current queue can execute in parallel
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            topo_order.append(node)
            level.append(node)

            for dependent in out_edges[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        levels.append(level)

    assert len(topo_order) == len(all_nodes), "Cycle detected in graph!"

    return topo_order, levels</code></pre>

<h4>3. DAGs for Pipeline Parallelism</h4>
<p>Pipeline parallelism splits a model across GPUs by partitioning the computation graph. The scheduling problem&mdash;deciding which GPU processes which micro-batch at which time&mdash;is a DAG scheduling problem:</p>

<pre><code>def pipeline_schedule_1f1b(num_stages: int, num_microbatches: int):
    """Generate 1F1B (one forward, one backward) pipeline schedule.

    This is the standard pipeline parallelism schedule used by
    Megatron-LM and DeepSpeed. It minimizes the "pipeline bubble"
    (idle time) compared to naive fill-drain scheduling.

    The schedule has three phases:
    1. Warmup: fill the pipeline with forward passes
    2. Steady state: alternate one forward and one backward
    3. Cooldown: drain the pipeline with remaining backward passes

    Returns:
        List of (stage, timestep, operation) tuples
    """
    schedule = []

    for stage in range(num_stages):
        timestep = 0
        fwd_mb = 0  # Next micro-batch for forward
        bwd_mb = 0  # Next micro-batch for backward

        # Phase 1: Warmup (num_stages - stage - 1 forward passes)
        warmup_count = num_stages - stage - 1
        warmup_count = min(warmup_count, num_microbatches)

        for i in range(warmup_count):
            schedule.append((stage, timestep, f'F{fwd_mb}'))
            fwd_mb += 1
            timestep += 1

        # Phase 2: Steady state (1F1B)
        steady_count = num_microbatches - warmup_count
        for i in range(steady_count):
            if fwd_mb < num_microbatches:
                schedule.append((stage, timestep, f'F{fwd_mb}'))
                fwd_mb += 1
                timestep += 1
            schedule.append((stage, timestep, f'B{bwd_mb}'))
            bwd_mb += 1
            timestep += 1

        # Phase 3: Cooldown (remaining backward passes)
        while bwd_mb < num_microbatches:
            schedule.append((stage, timestep, f'B{bwd_mb}'))
            bwd_mb += 1
            timestep += 1

    return schedule


def visualize_pipeline_schedule(num_stages=4, num_microbatches=8):
    """Print a visual pipeline schedule."""
    schedule = pipeline_schedule_1f1b(num_stages, num_microbatches)

    # Organize by stage and timestep
    grid = {}
    max_time = 0
    for stage, time, op in schedule:
        grid[(stage, time)] = op
        max_time = max(max_time, time)

    print(f"Pipeline Schedule: {num_stages} stages, "
          f"{num_microbatches} micro-batches")
    print("-" * (max_time * 5 + 15))

    for stage in range(num_stages):
        row = f"GPU {stage}: "
        for t in range(max_time + 1):
            op = grid.get((stage, t), '  ')
            row += f"{op:>4} "
        print(row)

    # Calculate bubble ratio
    total_slots = num_stages * (max_time + 1)
    used_slots = len(schedule)
    bubble = 1.0 - used_slots / total_slots
    print(f"\\nPipeline bubble: {bubble:.1%}")</code></pre>

<h4>4. Model Parallelism as Graph Partitioning</h4>
<p>Distributing a model across GPUs requires partitioning the computation graph to minimize cross-GPU communication. This is a variant of the classical <strong>graph partitioning</strong> problem:</p>

<pre><code>def greedy_graph_partition(nodes: list, edges: list,
                           num_partitions: int,
                           node_weights: dict,
                           edge_weights: dict) -> dict:
    """Greedy graph partitioning for model parallelism.

    Assigns nodes (layers/operations) to partitions (GPUs) to:
    1. Balance computation load across partitions
    2. Minimize communication (cut edges) between partitions

    This is NP-hard in general, so we use a greedy heuristic.
    Production systems use more sophisticated approaches like
    METIS or learned partitioning.

    Args:
        nodes: List of node IDs
        edges: List of (src, dst, weight) tuples
        num_partitions: Number of GPUs/partitions
        node_weights: Dict of node_id -> compute cost
        edge_weights: Dict of (src, dst) -> communication cost

    Returns:
        Dict mapping node_id -> partition_id
    """
    assignment = {}
    partition_loads = [0.0] * num_partitions

    # Sort nodes by compute cost (descending) - process heavy nodes first
    sorted_nodes = sorted(nodes, key=lambda n: node_weights.get(n, 1.0),
                         reverse=True)

    # Build adjacency list
    adj = {n: [] for n in nodes}
    for src, dst, w in edges:
        adj[src].append((dst, w))
        adj[dst].append((src, w))

    for node in sorted_nodes:
        # For each partition, compute the cost of assigning this node
        best_partition = 0
        best_cost = float('inf')

        for p in range(num_partitions):
            # Load imbalance cost
            hypothetical_load = partition_loads[p] + node_weights.get(node, 1.0)
            max_load = max(partition_loads[q] if q != p else hypothetical_load
                         for q in range(num_partitions))
            imbalance_cost = max_load

            # Communication cost: edges to nodes in OTHER partitions
            comm_cost = 0
            for neighbor, weight in adj[node]:
                if neighbor in assignment and assignment[neighbor] != p:
                    comm_cost += weight

            total_cost = imbalance_cost + comm_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_partition = p

        assignment[node] = best_partition
        partition_loads[best_partition] += node_weights.get(node, 1.0)

    return assignment</code></pre>

<div class="callout">
<div class="callout-title">Interview Question: Autograd Complexity</div>
<p><strong>Q:</strong> What is the time and space complexity of backpropagation relative to the forward pass?</p>
<p><strong>A:</strong> Time: backpropagation takes approximately <strong>2x the time</strong> of the forward pass. For each operation in the graph, backward computes local gradients and multiplies by the incoming gradient&mdash;roughly the same cost as the forward operation, plus the additional multiply. The topological sort itself is O(V+E) which is negligible. Space: backpropagation requires storing all intermediate activations from the forward pass (for computing local gradients), so space is O(N) where N is the number of intermediate tensors. This is why gradient checkpointing trades compute for memory: recompute activations during backward instead of storing them, reducing space from O(N) to O(sqrt(N)) at the cost of one additional forward pass.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Pipeline Bubble</div>
<p><strong>Q:</strong> In pipeline parallelism with P stages and M micro-batches, what is the pipeline bubble ratio, and how do you minimize it?</p>
<p><strong>A:</strong> The bubble ratio is approximately <strong>(P-1)/(P-1+M)</strong>. The pipeline needs P-1 steps to fill (warmup) and P-1 steps to drain (cooldown), during which some GPUs are idle. Total work is M forward + M backward = 2M operations per stage. The ideal (no bubble) schedule would take 2M timesteps. The actual schedule takes 2M + (P-1) timesteps due to the bubble. Minimization strategies: (1) Increase M relative to P (more micro-batches), (2) Use interleaved schedules (Megatron v3) which assign multiple non-consecutive stages to each GPU, reducing the effective P, (3) Use async pipeline (PipeDream) which overlaps forward and backward of different micro-batches, (4) Zero-bubble pipeline parallelism (Qi et al., 2023) which eliminates the bubble entirely by interleaving warmup backward passes.</p>
</div>
`
    },

    // ----------------------------------------------------------
    // 15.6 Dynamic Programming for Sequences
    // ----------------------------------------------------------
    {
      id: "dsa-dp",
      title: "Dynamic Programming for Sequences",
      content: `
<p>Dynamic programming (DP) is the art of solving problems by breaking them into overlapping subproblems and reusing solutions. In AI engineering, DP appears in some of the most critical algorithms: edit distance for evaluating speech recognition (WER), CTC decoding for end-to-end ASR, the Viterbi algorithm for HMMs, and sequence alignment for bioinformatics. These algorithms share a common structure that, once understood, makes them all approachable.</p>

<div class="callout">
<div class="callout-title">The DP Pattern in AI</div>
<p>Almost every DP algorithm in AI follows this template: define a recurrence over a sequence (or pair of sequences), fill a table bottom-up (or top-down with memoization), and trace back through the table to reconstruct the optimal solution. The challenge is defining the right subproblem and recurrence.</p>
</div>

<h4>1. Edit Distance for WER Calculation</h4>
<p>The <strong>Word Error Rate (WER)</strong> is the standard metric for speech recognition. It is computed as the edit distance between the hypothesis and reference word sequences, normalized by reference length. The edit distance itself is a classic DP algorithm:</p>

<pre><code>def edit_distance(hypothesis: list, reference: list) -> dict:
    """Compute edit distance between two word sequences.

    This is the Wagner-Fischer algorithm (1974), which is the standard
    method for computing WER in speech recognition.

    Operations and their costs:
    - Substitution (S): replace a word (cost 1)
    - Insertion (I): hypothesis has an extra word (cost 1)
    - Deletion (D): hypothesis is missing a word (cost 1)
    - Correct (C): words match (cost 0)

    Time complexity: O(n * m) where n = len(reference), m = len(hypothesis)
    Space complexity: O(n * m) for the full table (O(n) if only distance needed)

    Returns:
        Dictionary with WER, edit distance, and operation counts.
    """
    n = len(reference)
    m = len(hypothesis)

    # DP table: dp[i][j] = min edit distance between
    # reference[:i] and hypothesis[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Backtrace table for reconstructing alignment
    backtrace = [[None] * (m + 1) for _ in range(n + 1)]

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i                    # Delete all reference words
        backtrace[i][0] = 'D'
    for j in range(m + 1):
        dp[0][j] = j                    # Insert all hypothesis words
        backtrace[0][j] = 'I'
    backtrace[0][0] = None

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i-1] == hypothesis[j-1]:
                # Words match: no cost
                dp[i][j] = dp[i-1][j-1]
                backtrace[i][j] = 'C'  # Correct
            else:
                # Try all three operations, pick minimum cost
                sub_cost = dp[i-1][j-1] + 1  # Substitution
                del_cost = dp[i-1][j] + 1      # Deletion
                ins_cost = dp[i][j-1] + 1      # Insertion

                dp[i][j] = min(sub_cost, del_cost, ins_cost)

                if dp[i][j] == sub_cost:
                    backtrace[i][j] = 'S'  # Substitution
                elif dp[i][j] == del_cost:
                    backtrace[i][j] = 'D'  # Deletion
                else:
                    backtrace[i][j] = 'I'  # Insertion

    # Trace back to count operations
    i, j = n, m
    counts = {'S': 0, 'I': 0, 'D': 0, 'C': 0}
    alignment = []

    while i > 0 or j > 0:
        op = backtrace[i][j]
        counts[op] += 1

        if op == 'C' or op == 'S':
            alignment.append((reference[i-1], hypothesis[j-1], op))
            i -= 1
            j -= 1
        elif op == 'D':
            alignment.append((reference[i-1], '***', op))
            i -= 1
        elif op == 'I':
            alignment.append(('***', hypothesis[j-1], op))
            j -= 1

    alignment.reverse()

    edit_dist = dp[n][m]
    wer = edit_dist / n if n > 0 else 0.0

    return {
        'wer': wer,
        'edit_distance': edit_dist,
        'counts': counts,
        'alignment': alignment,
        'reference_length': n,
    }


# Example usage
ref = "the cat sat on the mat".split()
hyp = "the cat sit on a mat".split()
result = edit_distance(hyp, ref)
# WER = (1 sub + 1 sub) / 6 = 33.3%
print(f"WER: {result['wer']:.1%}")
print(f"Operations: {result['counts']}")</code></pre>

<h4>2. CTC Decoding Algorithm</h4>
<p><strong>Connectionist Temporal Classification (CTC)</strong> is the loss function and decoding algorithm that enabled end-to-end speech recognition (Graves et al., 2006). The CTC forward algorithm is a DP that sums over all possible alignments between an input sequence and an output label sequence:</p>

<pre><code>import numpy as np

def ctc_greedy_decode(log_probs: np.ndarray, idx_to_char: dict,
                      blank_idx: int = 0) -> str:
    """CTC greedy (best-path) decoding.

    At each timestep, select the most likely character. Then collapse
    repeated characters and remove blanks.

    Args:
        log_probs: Shape (T, C) - log probabilities at each timestep
                   T = number of frames, C = number of classes (including blank)
        idx_to_char: Mapping from class index to character
        blank_idx: Index of the blank/CTC token

    Returns:
        Decoded string

    Time complexity: O(T)
    This is fast but suboptimal - beam search gives better results.
    """
    T = log_probs.shape[0]

    # Step 1: Argmax at each timestep
    best_path = np.argmax(log_probs, axis=1)  # Shape: (T,)

    # Step 2: Collapse repeated tokens and remove blanks
    decoded = []
    prev = None
    for t in range(T):
        token = best_path[t]
        if token != prev:  # Collapse repeats
            if token != blank_idx:  # Remove blanks
                decoded.append(idx_to_char.get(token, '?'))
        prev = token

    return ''.join(decoded)


def ctc_beam_search_decode(log_probs: np.ndarray, idx_to_char: dict,
                           beam_width: int = 10,
                           blank_idx: int = 0) -> list:
    """CTC beam search decoding.

    Unlike greedy decoding, beam search considers multiple hypotheses
    and properly handles the many-to-one CTC mapping (where different
    frame-level alignments produce the same output string).

    The key insight: we maintain two scores for each prefix:
    - p_b: probability of the prefix ending in blank at time t
    - p_nb: probability of the prefix NOT ending in blank at time t

    This distinction is necessary because "aa" (with blank between)
    and "a" (single character) map to different outputs.

    Args:
        log_probs: Shape (T, C) - log probabilities
        beam_width: Number of hypotheses to maintain
        blank_idx: Index of blank token

    Returns:
        List of (decoded_string, score) tuples, sorted by score.

    Time complexity: O(T * beam_width * C)
    Space complexity: O(beam_width)
    """
    T, C = log_probs.shape

    # Initialize: empty prefix with probability 1 (log-prob 0)
    # Each beam entry: prefix -> (p_blank, p_non_blank)
    beams = {
        '': (0.0, float('-inf'))  # (log_p_blank, log_p_non_blank)
    }

    for t in range(T):
        new_beams = {}

        for prefix, (p_b, p_nb) in beams.items():
            # Total probability of this prefix
            p_total = np.logaddexp(p_b, p_nb)

            for c in range(C):
                log_p = log_probs[t, c]

                if c == blank_idx:
                    # Blank extends any prefix without changing it
                    key = prefix
                    new_p_b = p_total + log_p

                    if key in new_beams:
                        old_b, old_nb = new_beams[key]
                        new_beams[key] = (
                            np.logaddexp(old_b, new_p_b),
                            old_nb
                        )
                    else:
                        new_beams[key] = (new_p_b, float('-inf'))

                else:
                    char = idx_to_char.get(c, '?')

                    # Case 1: Character is same as last character of prefix
                    if prefix and char == prefix[-1]:
                        # Can only extend if previous was blank
                        # (otherwise it would be a repeat, not a new char)
                        new_prefix = prefix + char
                        new_p_nb = p_b + log_p  # Must come from blank

                        if new_prefix in new_beams:
                            old_b, old_nb = new_beams[new_prefix]
                            new_beams[new_prefix] = (
                                old_b,
                                np.logaddexp(old_nb, new_p_nb)
                            )
                        else:
                            new_beams[new_prefix] = (float('-inf'), new_p_nb)

                        # Also: the repeat could be a continuation (not new char)
                        key = prefix
                        cont_p_nb = p_nb + log_p  # Continue existing char

                        if key in new_beams:
                            old_b, old_nb = new_beams[key]
                            new_beams[key] = (
                                old_b,
                                np.logaddexp(old_nb, cont_p_nb)
                            )
                        else:
                            new_beams[key] = (float('-inf'), cont_p_nb)
                    else:
                        # Different character: extend the prefix
                        new_prefix = prefix + char
                        new_p_nb = p_total + log_p

                        if new_prefix in new_beams:
                            old_b, old_nb = new_beams[new_prefix]
                            new_beams[new_prefix] = (
                                old_b,
                                np.logaddexp(old_nb, new_p_nb)
                            )
                        else:
                            new_beams[new_prefix] = (float('-inf'), new_p_nb)

        # Prune to top-k beams
        scored = [
            (prefix, np.logaddexp(p_b, p_nb), p_b, p_nb)
            for prefix, (p_b, p_nb) in new_beams.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:beam_width]

        beams = {s[0]: (s[2], s[3]) for s in scored}

    # Return final results
    results = []
    for prefix, (p_b, p_nb) in beams.items():
        score = np.logaddexp(p_b, p_nb)
        results.append((prefix, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results</code></pre>

<h4>3. Viterbi Algorithm for HMM-Based Models</h4>
<p>The Viterbi algorithm finds the most likely sequence of hidden states given observations. While HMMs are less common in modern NLP, Viterbi remains important for CRF layers in NER models and for understanding the DP structure shared by many sequence algorithms:</p>

<pre><code>def viterbi(observations: list, states: list,
            start_prob: dict, trans_prob: dict,
            emit_prob: dict) -> tuple:
    """Viterbi algorithm for finding the most likely state sequence.

    This is a DP over sequences where:
    - Subproblem: dp[t][s] = probability of the best path ending
      in state s at time t
    - Recurrence: dp[t][s] = max over all previous states s' of
      dp[t-1][s'] * trans_prob[s'][s] * emit_prob[s][obs[t]]

    Used in: CRF layers, POS tagging, named entity recognition,
    speech recognition (HMM-GMM era), bioinformatics.

    Args:
        observations: List of observed symbols
        states: List of possible hidden states
        start_prob: Dict of state -> initial probability
        trans_prob: Dict of (state, state) -> transition probability
        emit_prob: Dict of (state, observation) -> emission probability

    Returns:
        (best_path, best_probability)

    Time complexity: O(T * S^2) where T = sequence length, S = num states
    Space complexity: O(T * S) for the DP table and backtrace
    """
    T = len(observations)

    # DP table: dp[t][s] = log probability of best path ending at state s, time t
    dp = [{} for _ in range(T)]
    backpointer = [{} for _ in range(T)]

    # Initialization (t = 0)
    for s in states:
        dp[0][s] = (math.log(start_prob.get(s, 1e-10)) +
                    math.log(emit_prob.get((s, observations[0]), 1e-10)))
        backpointer[0][s] = None

    # Recursion
    for t in range(1, T):
        for s in states:
            best_prev_state = None
            best_score = float('-inf')

            for s_prev in states:
                score = (dp[t-1][s_prev] +
                        math.log(trans_prob.get((s_prev, s), 1e-10)) +
                        math.log(emit_prob.get((s, observations[t]), 1e-10)))

                if score > best_score:
                    best_score = score
                    best_prev_state = s_prev

            dp[t][s] = best_score
            backpointer[t][s] = best_prev_state

    # Termination: find the best final state
    best_final_state = max(states, key=lambda s: dp[T-1][s])
    best_prob = dp[T-1][best_final_state]

    # Backtrace
    path = [best_final_state]
    for t in range(T-1, 0, -1):
        path.append(backpointer[t][path[-1]])
    path.reverse()

    return path, math.exp(best_prob)


# Example: Simple NER with Viterbi
def demo_viterbi_ner():
    """Demonstrate Viterbi for named entity recognition."""
    states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']

    # Simplified probabilities (normally learned from data)
    start_prob = {'O': 0.8, 'B-PER': 0.1, 'I-PER': 0.0,
                  'B-ORG': 0.1, 'I-ORG': 0.0}

    # Transition probabilities encode valid BIO sequences
    trans_prob = {
        ('O', 'O'): 0.7, ('O', 'B-PER'): 0.15, ('O', 'B-ORG'): 0.15,
        ('B-PER', 'I-PER'): 0.7, ('B-PER', 'O'): 0.2, ('B-PER', 'B-ORG'): 0.1,
        ('I-PER', 'I-PER'): 0.5, ('I-PER', 'O'): 0.4, ('I-PER', 'B-ORG'): 0.1,
        ('B-ORG', 'I-ORG'): 0.7, ('B-ORG', 'O'): 0.2, ('B-ORG', 'B-PER'): 0.1,
        ('I-ORG', 'I-ORG'): 0.5, ('I-ORG', 'O'): 0.4, ('I-ORG', 'B-PER'): 0.1,
    }

    observations = ['John', 'works', 'at', 'Google']

    # Emission probabilities (simplified)
    emit_prob = {
        ('O', 'John'): 0.01, ('B-PER', 'John'): 0.8, ('I-PER', 'John'): 0.3,
        ('O', 'works'): 0.8, ('B-PER', 'works'): 0.01, ('B-ORG', 'works'): 0.01,
        ('O', 'at'): 0.9, ('B-PER', 'at'): 0.001, ('B-ORG', 'at'): 0.001,
        ('O', 'Google'): 0.01, ('B-ORG', 'Google'): 0.8, ('I-ORG', 'Google'): 0.1,
    }

    path, prob = viterbi(observations, states, start_prob, trans_prob, emit_prob)

    for word, tag in zip(observations, path):
        print(f"{word:>10} -> {tag}")
    # Expected: John -> B-PER, works -> O, at -> O, Google -> B-ORG</code></pre>

<h4>4. Complexity Comparison of Sequence DP Algorithms</h4>
<table>
<tr><th>Algorithm</th><th>Time</th><th>Space</th><th>Use in AI</th></tr>
<tr><td>Edit Distance</td><td>O(nm)</td><td>O(nm)</td><td>WER, text similarity</td></tr>
<tr><td>CTC Forward</td><td>O(T &times; L)</td><td>O(T &times; L)</td><td>End-to-end ASR, OCR</td></tr>
<tr><td>CTC Beam Search</td><td>O(T &times; k &times; C)</td><td>O(k)</td><td>ASR decoding</td></tr>
<tr><td>Viterbi</td><td>O(T &times; S&sup2;)</td><td>O(T &times; S)</td><td>NER (CRF), POS tagging</td></tr>
<tr><td>Forward-Backward (HMM)</td><td>O(T &times; S&sup2;)</td><td>O(T &times; S)</td><td>HMM training, CRF marginals</td></tr>
<tr><td>Needleman-Wunsch</td><td>O(nm)</td><td>O(nm)</td><td>Protein alignment</td></tr>
</table>

<div class="callout">
<div class="callout-title">Interview Question: CTC vs Attention</div>
<p><strong>Q:</strong> Compare CTC decoding with attention-based decoding for speech recognition. What are the trade-offs?</p>
<p><strong>A:</strong> CTC assumes <strong>conditional independence</strong> between output tokens given the input&mdash;it cannot model output dependencies (e.g., language model probabilities). This makes it fast (parallelizable in training, no autoregressive decoding) but limits quality. Attention-based decoding models P(y_t | y_1..y_{t-1}, x) directly, capturing output dependencies, but is autoregressive (slow) and can suffer from attention failures (skipping, repeating). Modern hybrid systems (e.g., Whisper uses attention, but some systems use CTC + attention jointly) combine both: CTC provides monotonic alignment constraint while attention provides output dependencies. CTC decoding is O(T) greedy or O(T*k*C) beam, while attention decoding is O(T_out * T_in) per step due to cross-attention.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Space Optimization</div>
<p><strong>Q:</strong> Edit distance uses O(nm) space for the full DP table. How can you reduce this to O(min(n,m)) if you only need the distance (not the alignment)?</p>
<p><strong>A:</strong> Since each row of the DP table depends only on the previous row, you can use <strong>two rolling arrays</strong> instead of the full table: one for the current row and one for the previous row. After computing the current row, swap them. This reduces space from O(nm) to O(min(n,m)) by iterating over the longer sequence and maintaining arrays of the shorter sequence's length. If you DO need the alignment (backtrace), you can use Hirschberg's algorithm which computes the alignment in O(nm) time but only O(min(n,m)) space using a divide-and-conquer approach that runs the DP forward and backward, finding the midpoint of the optimal path, then recursing on each half.</p>
</div>
`
    },

    // ----------------------------------------------------------
    // 15.7 Sampling Algorithms for LLMs
    // ----------------------------------------------------------
    {
      id: "dsa-sampling",
      title: "Sampling Algorithms for LLMs",
      content: `
<p>Sampling is how LLMs generate text. While beam search finds the most likely sequence, sampling introduces controlled randomness to produce diverse, creative, and human-like text. The choice and implementation of sampling algorithm profoundly affects output quality. This section covers every major sampling method, from the simple to the cutting-edge, with complete implementations and complexity analysis.</p>

<div class="callout">
<div class="callout-title">Why Not Just Use Greedy/Beam Search?</div>
<p>Maximization-based decoding (greedy, beam search) produces text that is high-probability but often <strong>repetitive, generic, and boring</strong>. Human language is not the most likely sequence of words&mdash;it contains surprises, variety, and creativity. Holtzman et al. (2020) showed that human text has higher perplexity than beam-search output, and that the probability of human-chosen words follows a much broader distribution than argmax would suggest. Sampling bridges this gap.</p>
</div>

<h4>1. Temperature Scaling</h4>
<p>Temperature is the simplest and most fundamental sampling control. It scales the logits before softmax, sharpening or flattening the probability distribution:</p>

<pre><code>import numpy as np
from typing import List, Tuple

def temperature_scale(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to logits.

    temperature < 1.0: Sharper distribution (more deterministic)
    temperature = 1.0: Original distribution (no change)
    temperature > 1.0: Flatter distribution (more random)
    temperature -> 0:   Approaches argmax (greedy)
    temperature -> inf:  Approaches uniform distribution

    Mathematical basis:
    softmax(x/T)_i = exp(x_i/T) / sum(exp(x_j/T))

    As T -> 0, softmax approaches a one-hot vector at the argmax.
    As T -> inf, softmax approaches 1/|V| for all tokens.

    Args:
        logits: Raw model output logits, shape (vocab_size,)
        temperature: Temperature parameter

    Returns:
        Scaled logits (apply softmax after this to get probabilities)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive. Use 1e-8 for near-greedy.")
    if temperature == 1.0:
        return logits
    return logits / temperature


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def sample_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample a token using temperature scaling.

    Args:
        logits: Raw logits from the model
        temperature: Sampling temperature

    Returns:
        Sampled token index
    """
    scaled_logits = temperature_scale(logits, temperature)
    probs = softmax(scaled_logits)
    return np.random.choice(len(probs), p=probs)</code></pre>

<h4>2. Top-k Sampling with Partial Sort</h4>
<p>Top-k sampling (Fan et al., 2018) restricts sampling to the k most likely tokens. The key algorithmic insight is that we do NOT need a full sort&mdash;a <strong>partial sort</strong> (selection algorithm) suffices:</p>

<pre><code>def top_k_sampling(logits: np.ndarray, k: int = 50,
                   temperature: float = 1.0) -> int:
    """Top-k sampling: sample from only the k most likely tokens.

    Algorithm:
    1. Find the k-th largest logit (using partial sort / quickselect)
    2. Mask all logits below this threshold to -infinity
    3. Apply temperature and softmax to the remaining k tokens
    4. Sample from this truncated distribution

    Why partial sort matters:
    - Full sort: O(V log V) where V = vocab size (e.g., 128K)
    - Partial sort (np.argpartition): O(V) average case
    - This is a ~17x speedup for V=128K, k=50

    The risk with top-k: for peaked distributions, k tokens may include
    very unlikely ones. For flat distributions, k may exclude important ones.
    Top-p addresses this by adapting the cutoff dynamically.
    """
    # Apply temperature first
    scaled_logits = temperature_scale(logits, temperature)

    # Partial sort to find top-k indices: O(V) average case
    # np.argpartition does NOT fully sort - it only guarantees
    # that the k-th element is in its final sorted position
    top_k_indices = np.argpartition(scaled_logits, -k)[-k:]

    # Get the logits for top-k tokens
    top_k_logits = scaled_logits[top_k_indices]

    # Softmax over only the top-k tokens
    top_k_probs = softmax(top_k_logits)

    # Sample from the truncated distribution
    chosen_idx = np.random.choice(len(top_k_probs), p=top_k_probs)

    return top_k_indices[chosen_idx]


def top_k_with_quickselect(logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Manual implementation of partial sort using quickselect.

    This is what np.argpartition does internally. Understanding it
    helps you reason about the O(V) complexity.

    Quickselect (Hoare, 1961):
    - Like quicksort, but only recurse into the partition containing
      the k-th element
    - Average case: O(n), worst case: O(n^2) (mitigated by random pivot)
    """
    n = len(logits)
    indices = np.arange(n)

    def partition(arr, idx, left, right, pivot_idx):
        pivot_val = arr[pivot_idx]
        # Move pivot to end
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        idx[pivot_idx], idx[right] = idx[right], idx[pivot_idx]

        store = left
        for i in range(left, right):
            if arr[i] > pivot_val:  # We want top-k, so partition by >
                arr[i], arr[store] = arr[store], arr[i]
                idx[i], idx[store] = idx[store], idx[i]
                store += 1

        arr[right], arr[store] = arr[store], arr[right]
        idx[right], idx[store] = idx[store], idx[right]
        return store

    def quickselect(arr, idx, left, right, k):
        if left == right:
            return

        pivot_idx = np.random.randint(left, right + 1)
        pivot_pos = partition(arr, idx, left, right, pivot_idx)

        if k == pivot_pos:
            return
        elif k < pivot_pos:
            quickselect(arr, idx, left, pivot_pos - 1, k)
        else:
            quickselect(arr, idx, pivot_pos + 1, right, k)

    arr = logits.copy()
    idx = indices.copy()
    quickselect(arr, idx, 0, n - 1, k)

    return arr[:k], idx[:k]</code></pre>

<h4>3. Top-p (Nucleus) Sampling</h4>
<p>Top-p sampling (Holtzman et al., 2020) dynamically adjusts the number of tokens by selecting the smallest set whose cumulative probability exceeds p. This adapts to the shape of the distribution:</p>

<pre><code>def top_p_sampling(logits: np.ndarray, p: float = 0.9,
                   temperature: float = 1.0,
                   min_tokens_to_keep: int = 1) -> int:
    """Top-p (nucleus) sampling: sample from the smallest set of tokens
    whose cumulative probability exceeds p.

    Algorithm:
    1. Apply temperature
    2. Compute probabilities via softmax
    3. Sort probabilities in descending order
    4. Find the cutoff index where cumulative probability exceeds p
    5. Mask everything below the cutoff
    6. Re-normalize and sample

    The key advantage over top-k: the number of tokens considered
    adapts to the distribution shape.
    - Peaked distribution: few tokens (maybe 2-5) suffice
    - Flat distribution: many tokens (maybe 100+) are included

    Time complexity: O(V log V) due to the sort
    (Can be optimized to O(V) average with quickselect + running sum)

    Args:
        p: Cumulative probability threshold (0.9 = keep tokens summing to 90%)
        min_tokens_to_keep: Always keep at least this many tokens
    """
    # Apply temperature
    scaled_logits = temperature_scale(logits, temperature)

    # Compute probabilities
    probs = softmax(scaled_logits)

    # Sort in descending order
    sorted_indices = np.argsort(probs)[::-1]  # O(V log V)
    sorted_probs = probs[sorted_indices]

    # Compute cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)

    # Find cutoff: first index where cumsum exceeds p
    cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
    cutoff_idx = max(cutoff_idx, min_tokens_to_keep)
    cutoff_idx = min(cutoff_idx, len(probs))

    # Keep only tokens within the nucleus
    nucleus_indices = sorted_indices[:cutoff_idx]
    nucleus_probs = sorted_probs[:cutoff_idx]

    # Re-normalize
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    # Sample
    chosen_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
    return nucleus_indices[chosen_idx]


def combined_top_k_top_p(logits: np.ndarray, k: int = 50,
                          p: float = 0.9,
                          temperature: float = 1.0) -> int:
    """Combined top-k and top-p: apply both filters.

    This is what most production LLM APIs actually use:
    1. First filter to top-k candidates
    2. Then apply top-p within those candidates
    3. This gives the safety of top-k (bounded max tokens)
       with the adaptivity of top-p

    This is the default in HuggingFace transformers when both
    top_k and top_p are specified.
    """
    scaled_logits = temperature_scale(logits, temperature)

    # Step 1: Top-k filter
    if k > 0 and k < len(logits):
        top_k_indices = np.argpartition(scaled_logits, -k)[-k:]
        mask = np.full_like(scaled_logits, -np.inf)
        mask[top_k_indices] = scaled_logits[top_k_indices]
        scaled_logits = mask

    # Step 2: Compute probs and apply top-p
    probs = softmax(scaled_logits)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)

    # Remove tokens after cumsum exceeds p
    remove_mask = cumsum > p
    # Shift right to keep the first token that pushes past p
    remove_mask[1:] = remove_mask[:-1].copy()
    remove_mask[0] = False

    sorted_probs[remove_mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()

    chosen_idx = np.random.choice(len(sorted_probs), p=sorted_probs)
    return sorted_indices[chosen_idx]</code></pre>

<h4>4. Gumbel-Softmax Trick</h4>
<p>The Gumbel-Softmax trick enables <strong>differentiable sampling</strong> from a categorical distribution, which is crucial for training models with discrete sampling in the forward pass (e.g., VQ-VAE, reinforcement learning from human feedback):</p>

<pre><code>def gumbel_softmax(logits: np.ndarray, temperature: float = 1.0,
                   hard: bool = False) -> np.ndarray:
    """Gumbel-Softmax: differentiable approximation to categorical sampling.

    The Gumbel-Max trick: adding Gumbel noise to logits and taking
    argmax is equivalent to sampling from the categorical distribution.

    argmax(logits + Gumbel(0,1)) ~ Categorical(softmax(logits))

    The Gumbel-Softmax relaxation replaces the non-differentiable
    argmax with a softmax at low temperature, making it differentiable
    while being close to a one-hot sample.

    This is used in:
    - VQ-VAE: differentiable codebook selection
    - RLHF: differentiable token sampling for policy gradients
    - Discrete latent variable models

    Args:
        logits: Unnormalized log probabilities
        temperature: Controls sharpness (lower = closer to one-hot)
        hard: If True, use straight-through estimator for exact one-hot

    Returns:
        Soft (or hard) sample, same shape as logits
    """
    # Sample Gumbel(0, 1) noise
    # Gumbel(0,1) = -log(-log(Uniform(0,1)))
    u = np.random.uniform(0, 1, size=logits.shape)
    u = np.clip(u, 1e-10, 1.0)  # Avoid log(0)
    gumbel_noise = -np.log(-np.log(u))

    # Add noise and apply temperature-scaled softmax
    noisy_logits = (logits + gumbel_noise) / temperature
    y_soft = softmax(noisy_logits)

    if hard:
        # Straight-through estimator: forward pass uses one-hot,
        # backward pass uses soft gradients
        y_hard = np.zeros_like(y_soft)
        y_hard[np.argmax(y_soft)] = 1.0
        # In autograd: y_hard - y_soft.detach() + y_soft
        # This makes the forward pass use y_hard but gradient flows through y_soft
        return y_hard  # In practice: y_hard - stop_gradient(y_soft) + y_soft
    else:
        return y_soft


# Verify Gumbel-Softmax matches categorical sampling
def verify_gumbel_softmax():
    """Verify that Gumbel-Softmax sampling matches categorical sampling."""
    logits = np.array([2.0, 1.0, 0.5, -1.0])
    true_probs = softmax(logits)

    # Sample many times and count
    n_samples = 100000
    counts = np.zeros(len(logits))
    for _ in range(n_samples):
        sample = gumbel_softmax(logits, temperature=0.01, hard=True)
        counts += sample

    empirical_probs = counts / n_samples
    print(f"True probs:      {true_probs}")
    print(f"Empirical probs: {empirical_probs}")
    print(f"Max difference:  {np.abs(true_probs - empirical_probs).max():.4f}")</code></pre>

<h4>5. Speculative Sampling (Speculative Decoding)</h4>
<p>Speculative sampling (Leviathan et al., 2023; Chen et al., 2023) is a breakthrough algorithm that uses a small "draft" model to propose tokens that are then verified by the large "target" model, enabling faster inference without changing the output distribution:</p>

<pre><code>def speculative_sampling(target_model, draft_model, prompt_tokens: list,
                         gamma: int = 4, max_tokens: int = 100,
                         temperature: float = 1.0) -> list:
    """Speculative sampling for faster LLM inference.

    Key insight: a small model (draft) proposes gamma tokens quickly,
    then the large model (target) verifies them all in ONE forward pass.
    Accepted tokens are provably sampled from the target distribution.

    Why it works:
    - The draft model is 10-100x faster than the target
    - The draft proposes gamma tokens in gamma fast forward passes
    - The target verifies all gamma tokens in ONE parallel forward pass
    - If the draft is good, most tokens are accepted (70-90% typical)
    - The expected tokens per target model call: > 1 (speedup!)

    The rejection algorithm ensures the output distribution is EXACTLY
    the target model's distribution, not an approximation.

    Args:
        target_model: Large model (slow but accurate)
        draft_model: Small model (fast, approximate)
        gamma: Number of draft tokens to propose per iteration
        max_tokens: Maximum total tokens to generate

    Returns:
        List of generated token IDs (sampled from target distribution)
    """
    generated = list(prompt_tokens)

    while len(generated) - len(prompt_tokens) < max_tokens:
        # Step 1: Draft model proposes gamma tokens autoregressively
        draft_tokens = []
        draft_probs = []
        draft_input = list(generated)

        for _ in range(gamma):
            # Draft model forward pass (fast)
            q_logits = draft_model.forward(draft_input)
            q_probs = softmax(temperature_scale(q_logits, temperature))

            # Sample from draft distribution
            token = np.random.choice(len(q_probs), p=q_probs)
            draft_tokens.append(token)
            draft_probs.append(q_probs)
            draft_input.append(token)

        # Step 2: Target model scores all positions in ONE forward pass
        # This is the key efficiency: one call instead of gamma calls
        target_logits_batch = target_model.forward_batch(
            generated, draft_tokens
        )
        # target_logits_batch[i] are the logits at position len(generated) + i

        # Step 3: Accept/reject each draft token
        num_accepted = 0
        for i in range(gamma):
            p_probs = softmax(temperature_scale(
                target_logits_batch[i], temperature
            ))
            q_probs_i = draft_probs[i]
            token = draft_tokens[i]

            # Acceptance probability: min(1, p(x) / q(x))
            acceptance_ratio = p_probs[token] / max(q_probs_i[token], 1e-10)

            r = np.random.uniform()

            if r < acceptance_ratio:
                # Accept: token is from target distribution
                generated.append(token)
                num_accepted += 1
            else:
                # Reject: sample from the residual distribution
                # p'(x) = max(0, p(x) - q(x)) / sum(max(0, p(x) - q(x)))
                # This correction ensures we still sample from p(x) exactly
                residual = np.maximum(0, p_probs - q_probs_i)
                residual_sum = residual.sum()

                if residual_sum > 0:
                    residual = residual / residual_sum
                    correction_token = np.random.choice(
                        len(residual), p=residual
                    )
                else:
                    # Fallback: sample from target directly
                    correction_token = np.random.choice(
                        len(p_probs), p=p_probs
                    )

                generated.append(correction_token)
                break  # Restart drafting from this point

        # If ALL gamma tokens were accepted, sample one more from target
        if num_accepted == gamma:
            final_probs = softmax(temperature_scale(
                target_logits_batch[gamma], temperature
            ))
            bonus_token = np.random.choice(len(final_probs), p=final_probs)
            generated.append(bonus_token)

    return generated[len(prompt_tokens):]


def speculative_sampling_analysis(acceptance_rate: float = 0.8,
                                   gamma: int = 4,
                                   draft_speed: float = 10.0,
                                   target_speed: float = 1.0) -> dict:
    """Analyze expected speedup from speculative sampling.

    Args:
        acceptance_rate: Fraction of draft tokens accepted (alpha)
        gamma: Number of draft tokens per iteration
        draft_speed: Relative speed of draft model (tokens/unit_time)
        target_speed: Relative speed of target model (tokens/unit_time)

    Returns:
        Analysis of expected speedup

    The expected number of accepted tokens per iteration:
    E[accepted] = (1 - alpha^(gamma+1)) / (1 - alpha)

    where alpha is the acceptance rate.
    """
    alpha = acceptance_rate

    # Expected accepted tokens per iteration
    expected_accepted = (1 - alpha**(gamma + 1)) / (1 - alpha)

    # Time per iteration: gamma draft passes + 1 target pass
    time_per_iteration = gamma / draft_speed + 1 / target_speed

    # Tokens per unit time with speculative decoding
    spec_throughput = expected_accepted / time_per_iteration

    # Baseline: regular autoregressive with target model
    baseline_throughput = target_speed

    speedup = spec_throughput / baseline_throughput

    return {
        'expected_tokens_per_iter': expected_accepted,
        'time_per_iter': time_per_iteration,
        'speculative_throughput': spec_throughput,
        'baseline_throughput': baseline_throughput,
        'speedup': speedup,
        'gamma': gamma,
        'acceptance_rate': alpha,
    }</code></pre>

<h4>6. Comparison of Sampling Methods</h4>
<table>
<tr><th>Method</th><th>Time Complexity</th><th>Adaptive?</th><th>Deterministic?</th><th>Best For</th></tr>
<tr><td>Greedy (temp=0)</td><td>O(V)</td><td>No</td><td>Yes</td><td>Factual Q&amp;A, code</td></tr>
<tr><td>Temperature</td><td>O(V)</td><td>No</td><td>No</td><td>General creativity control</td></tr>
<tr><td>Top-k</td><td>O(V)</td><td>No</td><td>No</td><td>Simple truncation</td></tr>
<tr><td>Top-p</td><td>O(V log V)</td><td>Yes</td><td>No</td><td>Open-ended generation</td></tr>
<tr><td>Top-k + Top-p</td><td>O(V log V)</td><td>Yes</td><td>No</td><td>Production APIs</td></tr>
<tr><td>Gumbel-Softmax</td><td>O(V)</td><td>No</td><td>No</td><td>Differentiable training</td></tr>
<tr><td>Speculative</td><td>O(gamma * V_draft + V_target)</td><td>Yes</td><td>No</td><td>Inference speedup</td></tr>
</table>

<div class="callout">
<div class="callout-title">Interview Question: Speculative Sampling Correctness</div>
<p><strong>Q:</strong> Prove that speculative sampling produces tokens from exactly the target model's distribution, not an approximation.</p>
<p><strong>A:</strong> Consider a single token position. Let p(x) be the target probability and q(x) be the draft probability for token x. With probability q(x), the draft proposes token x. It is accepted with probability min(1, p(x)/q(x)). So the probability of accepting x is: q(x) * min(1, p(x)/q(x)) = min(q(x), p(x)). If rejected (probability 1 - sum_x min(q(x), p(x))), we sample from the residual distribution max(0, p(x) - q(x)) / Z where Z = sum_x max(0, p(x) - q(x)) = 1 - sum_x min(q(x), p(x)). The total probability of outputting x is: min(q(x), p(x)) + (1 - sum_y min(q(y), p(y))) * max(0, p(x) - q(x)) / Z. If p(x) <= q(x): this equals p(x) + 0 = p(x). If p(x) > q(x): this equals q(x) + Z * (p(x) - q(x)) / Z = q(x) + p(x) - q(x) = p(x). In both cases, the output probability equals p(x). QED.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Top-k vs Top-p</div>
<p><strong>Q:</strong> When would top-k sampling fail but top-p would succeed, and vice versa?</p>
<p><strong>A:</strong> Top-k fails when the distribution is very peaked (e.g., the model is 99% confident about the next word). With k=50, you include 49 tokens that the model considers extremely unlikely, adding noise. Top-p=0.95 would only include 1-2 tokens, correctly reflecting the model's confidence. Conversely, top-k provides a <em>safety bound</em> that top-p lacks: with a very flat distribution (model is uncertain), top-p=0.95 might include thousands of tokens including truly nonsensical ones. Top-k=50 bounds the search space. This is why production systems use both: top-k provides the safety ceiling, top-p provides the adaptive floor. The best practice: top_k=50 and top_p=0.9 together.</p>
</div>
`
    },

    // ----------------------------------------------------------
    // 15.8 Systems DS&A
    // ----------------------------------------------------------
    {
      id: "dsa-systems",
      title: "Systems DS&A",
      content: `
<p>AI systems at scale require not just ML-specific algorithms but also classical systems data structures. This section covers the structures that power large-scale AI infrastructure: bloom filters for data deduplication, consistent hashing for model sharding, ring buffers for streaming, memory-mapped files for large datasets, and producer-consumer queues for inference batching. These are the building blocks of production AI systems.</p>

<div class="callout">
<div class="callout-title">The Systems Layer</div>
<p>Most AI engineering interviews focus on model architecture and training. But at companies running models at scale, the <strong>systems layer</strong>&mdash;serving, data processing, storage, communication&mdash;is where the real engineering challenges live. A brilliant model is useless if you cannot serve it at low latency to millions of users. This section bridges the gap between CS fundamentals and AI systems.</p>
</div>

<h4>1. Bloom Filters for Data Deduplication</h4>
<p>Training data deduplication is critical for LLM quality. Duplicated documents cause models to memorize and regurgitate training data. A <strong>Bloom filter</strong> is a probabilistic data structure that tests set membership in O(1) time and O(m) space, where m is far smaller than storing the actual set:</p>

<pre><code>import hashlib
import math
from typing import List

class BloomFilter:
    """Bloom filter for probabilistic set membership testing.

    Properties:
    - No false negatives: if it says "not in set", the item is definitely absent
    - Small false positive rate: if it says "in set", it's probably there
    - O(k) insert and query, where k = number of hash functions
    - Memory: m bits (much less than storing actual items)

    Used in AI engineering for:
    - Training data deduplication (dedup at scale)
    - Detecting benchmark contamination
    - Deduplicating web crawl data (Common Crawl processing)
    - Checking if a URL has been scraped
    """

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """Initialize Bloom filter with desired parameters.

        Args:
            expected_items: Expected number of items to insert (n)
            false_positive_rate: Desired false positive probability (p)

        The optimal parameters are:
            m = -n * ln(p) / (ln(2))^2  (bits)
            k = (m / n) * ln(2)          (hash functions)
        """
        self.n = expected_items
        self.p = false_positive_rate

        # Optimal bit array size
        self.m = int(-expected_items * math.log(false_positive_rate) /
                     (math.log(2) ** 2))

        # Optimal number of hash functions
        self.k = int((self.m / expected_items) * math.log(2))
        self.k = max(1, self.k)

        # Bit array (using bytearray for memory efficiency)
        self.bit_array = bytearray(self.m // 8 + 1)
        self.items_added = 0

    def _get_hash_values(self, item: str) -> List[int]:
        """Generate k hash values for an item.

        Uses double hashing: h(i) = h1 + i * h2
        This gives k independent hash values from 2 hash computations.
        (Kirsch & Mitzenmacher, 2006)
        """
        # Two independent hash functions
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha256(item.encode()).hexdigest(), 16)

        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def _set_bit(self, pos: int):
        byte_idx = pos // 8
        bit_idx = pos % 8
        self.bit_array[byte_idx] |= (1 << bit_idx)

    def _get_bit(self, pos: int) -> bool:
        byte_idx = pos // 8
        bit_idx = pos % 8
        return bool(self.bit_array[byte_idx] & (1 << bit_idx))

    def add(self, item: str):
        """Add an item to the Bloom filter. O(k) time."""
        for pos in self._get_hash_values(item):
            self._set_bit(pos)
        self.items_added += 1

    def might_contain(self, item: str) -> bool:
        """Test if an item MIGHT be in the set.

        Returns:
            True: item is PROBABLY in the set (with false_positive_rate chance of error)
            False: item is DEFINITELY NOT in the set (no false negatives)
        """
        return all(self._get_bit(pos) for pos in self._get_hash_values(item))

    def stats(self) -> dict:
        """Report filter statistics."""
        bits_set = sum(bin(byte).count('1') for byte in self.bit_array)
        actual_fpr = (bits_set / self.m) ** self.k if self.m > 0 else 0

        return {
            "bits": self.m,
            "bytes": len(self.bit_array),
            "hash_functions": self.k,
            "items_added": self.items_added,
            "bits_per_item": self.m / max(self.items_added, 1),
            "estimated_fpr": actual_fpr,
            "target_fpr": self.p,
        }


def deduplicate_training_data(documents: list,
                              expected_count: int = 1_000_000) -> list:
    """Deduplicate a corpus using Bloom filter.

    This is a simplified version of what tools like MinHash LSH do
    at larger scale. For exact dedup, Bloom filters are excellent.
    For near-duplicate detection, you need MinHash or SimHash.

    Memory comparison for 1 billion documents:
    - HashSet: ~40 GB (40 bytes per hash)
    - Bloom filter (1% FPR): ~1.2 GB (9.6 bits per item)
    - That's a 33x memory savings!
    """
    bloom = BloomFilter(expected_count, false_positive_rate=0.001)
    unique_docs = []
    duplicates_found = 0

    for doc in documents:
        # Hash the document content
        doc_hash = hashlib.sha256(doc.encode()).hexdigest()

        if bloom.might_contain(doc_hash):
            duplicates_found += 1
            continue  # Skip duplicate

        bloom.add(doc_hash)
        unique_docs.append(doc)

    print(f"Processed {len(documents)} docs, found {duplicates_found} "
          f"duplicates, kept {len(unique_docs)} unique")
    print(f"Bloom filter memory: {len(bloom.bit_array) / 1024:.1f} KB")

    return unique_docs</code></pre>

<h4>2. Consistent Hashing for Model Sharding</h4>
<p>When serving large models across multiple GPUs or machines, you need to route requests to the right shard. <strong>Consistent hashing</strong> ensures that adding or removing a shard only redistributes a minimal number of keys:</p>

<pre><code>import bisect

class ConsistentHashRing:
    """Consistent hashing for model sharding and request routing.

    Standard hashing (hash(key) % N) redistributes ALL keys when N changes.
    Consistent hashing redistributes only K/N keys (K = total keys, N = nodes).

    Used in AI systems for:
    - Routing requests to model replicas
    - Sharding KV-cache across GPUs
    - Distributing embedding table partitions
    - Load balancing inference requests by prefix (for cache locality)

    Virtual nodes improve balance: each physical node gets multiple
    positions on the ring, smoothing out the distribution.
    """

    def __init__(self, nodes: list = None, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}           # hash_value -> node_id
        self.sorted_keys = []    # Sorted hash values for binary search

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring (0 to 2^32-1)."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def add_node(self, node: str):
        """Add a node to the ring with virtual nodes.

        Each physical node gets self.virtual_nodes positions on the ring.
        This ensures even distribution even with few physical nodes.

        Time: O(V * log(N*V)) where V = virtual_nodes, N = physical nodes
        """
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:v{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node
            bisect.insort(self.sorted_keys, hash_val)

    def remove_node(self, node: str):
        """Remove a node and all its virtual nodes from the ring.

        Only keys that were assigned to this node need to be redistributed.
        """
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:v{i}"
            hash_val = self._hash(virtual_key)
            if hash_val in self.ring:
                del self.ring[hash_val]
                self.sorted_keys.remove(hash_val)

    def get_node(self, key: str) -> str:
        """Find which node should handle a given key.

        Algorithm: hash the key, then walk clockwise on the ring
        to find the next node. Uses binary search for O(log N) lookup.

        Args:
            key: The key to route (e.g., request ID, prefix hash)

        Returns:
            The node ID that should handle this key
        """
        if not self.ring:
            return None

        hash_val = self._hash(key)

        # Binary search for the first ring position >= hash_val
        idx = bisect.bisect_left(self.sorted_keys, hash_val)

        # Wrap around if necessary
        if idx >= len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def get_distribution(self, num_samples: int = 10000) -> dict:
        """Analyze key distribution across nodes."""
        counts = {}
        for i in range(num_samples):
            node = self.get_node(f"sample_key_{i}")
            counts[node] = counts.get(node, 0) + 1

        total = sum(counts.values())
        return {
            node: {
                "count": count,
                "percentage": f"{count/total*100:.1f}%"
            }
            for node, count in sorted(counts.items())
        }


# Example: routing inference requests to model shards
def demo_model_sharding():
    """Demonstrate consistent hashing for model serving."""
    ring = ConsistentHashRing(
        nodes=["gpu-0", "gpu-1", "gpu-2", "gpu-3"],
        virtual_nodes=150
    )

    # Route requests
    requests = ["user_123_chat", "user_456_code", "user_789_translate"]
    for req in requests:
        node = ring.get_node(req)
        print(f"Request '{req}' -> {node}")

    # Check distribution
    dist = ring.get_distribution()
    print(f"\\nDistribution across 10K keys:")
    for node, info in dist.items():
        print(f"  {node}: {info['percentage']}")

    # Remove a node (simulating GPU failure)
    print(f"\\nRemoving gpu-2...")
    ring.remove_node("gpu-2")

    # Check which requests moved
    for req in requests:
        node = ring.get_node(req)
        print(f"Request '{req}' -> {node}")</code></pre>

<h4>3. Ring Buffers for Streaming Audio</h4>
<p>Real-time audio processing (live transcription, voice assistants) requires a <strong>ring buffer</strong> (circular buffer) to handle continuous audio streams with fixed memory:</p>

<pre><code>import numpy as np
import threading

class AudioRingBuffer:
    """Lock-free ring buffer for streaming audio processing.

    A ring buffer uses a fixed-size array with read and write pointers
    that wrap around. This gives O(1) read and write without memory
    allocation, making it ideal for real-time audio.

    Used in AI systems for:
    - Streaming ASR: buffer audio until VAD detects speech
    - Voice activity detection: keep a rolling window of audio
    - Real-time audio feature extraction
    - Echo cancellation and noise reduction pipelines
    """

    def __init__(self, capacity_seconds: float = 30.0,
                 sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.capacity = int(capacity_seconds * sample_rate)

        # Pre-allocated buffer (no dynamic allocation in hot path)
        self.buffer = np.zeros((self.capacity, channels), dtype=np.float32)

        # Read and write positions (modular arithmetic)
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0  # Number of samples available to read

        # Thread safety
        self.lock = threading.Lock()

    def write(self, audio_data: np.ndarray) -> int:
        """Write audio samples to the buffer.

        If the buffer is full, oldest data is overwritten (dropping
        strategy for real-time audio). This is intentional: in real-time
        processing, it's better to drop old audio than to block.

        Args:
            audio_data: Audio samples, shape (num_samples,) or (num_samples, channels)

        Returns:
            Number of samples written
        """
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)

        num_samples = len(audio_data)

        with self.lock:
            for i in range(num_samples):
                self.buffer[self.write_pos] = audio_data[i]
                self.write_pos = (self.write_pos + 1) % self.capacity

                if self.count < self.capacity:
                    self.count += 1
                else:
                    # Buffer full: advance read pointer (overwrite oldest)
                    self.read_pos = (self.read_pos + 1) % self.capacity

        return num_samples

    def read(self, num_samples: int) -> np.ndarray:
        """Read and consume samples from the buffer.

        Returns:
            Audio samples, shape (actual_samples, channels).
            May return fewer samples than requested if buffer is not full.
        """
        with self.lock:
            actual = min(num_samples, self.count)
            result = np.zeros((actual, self.channels), dtype=np.float32)

            for i in range(actual):
                result[i] = self.buffer[self.read_pos]
                self.read_pos = (self.read_pos + 1) % self.capacity

            self.count -= actual

        return result

    def peek(self, num_samples: int) -> np.ndarray:
        """Read samples WITHOUT consuming them (look-ahead).

        Useful for VAD: peek at audio to decide if speech is present,
        without removing it from the buffer.
        """
        with self.lock:
            actual = min(num_samples, self.count)
            result = np.zeros((actual, self.channels), dtype=np.float32)

            pos = self.read_pos
            for i in range(actual):
                result[i] = self.buffer[pos]
                pos = (pos + 1) % self.capacity

        return result

    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        """Get the last N seconds of audio (for context).

        This is a common pattern in streaming ASR: when speech is
        detected, grab the last 0.5-1s of audio as context for the
        model, even though those samples were "before" the speech.
        """
        num_samples = int(seconds * self.sample_rate)
        num_samples = min(num_samples, self.count)

        with self.lock:
            result = np.zeros((num_samples, self.channels), dtype=np.float32)
            start_pos = (self.write_pos - num_samples) % self.capacity

            for i in range(num_samples):
                pos = (start_pos + i) % self.capacity
                result[i] = self.buffer[pos]

        return result

    @property
    def available(self) -> int:
        return self.count

    @property
    def duration_available(self) -> float:
        return self.count / self.sample_rate</code></pre>

<h4>4. Memory-Mapped Files for Large Datasets</h4>
<p>Training datasets for LLMs can be hundreds of gigabytes to terabytes. Loading them entirely into RAM is impractical. <strong>Memory-mapped files</strong> let you access data as if it were in memory while the OS handles paging:</p>

<pre><code>import mmap
import struct
import json
import os

class MemmapTokenDataset:
    """Memory-mapped dataset for LLM training.

    Instead of loading the entire dataset into RAM, we memory-map
    the file. The OS pages data in/out as needed, allowing us to
    work with datasets larger than available RAM.

    This is the approach used by:
    - Megatron-LM (NVIDIA)
    - LLaMA training (Meta)
    - GPT-NeoX (EleutherAI)

    File format:
    - Header: JSON metadata (vocab size, sequence length, etc.)
    - Data: Packed uint16 or uint32 token IDs

    Random access is O(1) because we can compute the byte offset
    for any sequence directly: offset = header_size + (idx * seq_len * dtype_size)
    """

    def __init__(self, filepath: str, seq_length: int = 2048,
                 dtype_bytes: int = 2):
        """Open a memory-mapped token dataset.

        Args:
            filepath: Path to the binary token file
            seq_length: Number of tokens per training example
            dtype_bytes: 2 for uint16 (vocab < 65536), 4 for uint32
        """
        self.filepath = filepath
        self.seq_length = seq_length
        self.dtype_bytes = dtype_bytes
        self.dtype = np.uint16 if dtype_bytes == 2 else np.uint32

        # Memory-map the file
        self.file = open(filepath, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header
        header_size_bytes = struct.unpack('I', self.mm[:4])[0]
        header_json = self.mm[4:4+header_size_bytes].decode('utf-8')
        self.header = json.loads(header_json)
        self.data_offset = 4 + header_size_bytes

        # Calculate dataset size
        data_size = len(self.mm) - self.data_offset
        self.total_tokens = data_size // dtype_bytes
        self.num_sequences = self.total_tokens // seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a single training sequence. O(1) random access.

        This does NOT copy the data into RAM unnecessarily.
        The OS reads only the needed pages from disk.
        """
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.num_sequences})")

        start_byte = self.data_offset + idx * self.seq_length * self.dtype_bytes
        end_byte = start_byte + self.seq_length * self.dtype_bytes

        # Read bytes and interpret as numpy array (zero-copy when possible)
        raw_bytes = self.mm[start_byte:end_byte]
        tokens = np.frombuffer(raw_bytes, dtype=self.dtype)

        return tokens

    def __del__(self):
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close()

    @staticmethod
    def create(output_path: str, token_ids: np.ndarray,
               metadata: dict = None):
        """Create a memory-mapped dataset file from token IDs.

        Args:
            output_path: Where to write the file
            token_ids: Flat array of all token IDs
            metadata: Optional metadata dictionary
        """
        if metadata is None:
            metadata = {}

        metadata['total_tokens'] = len(token_ids)
        metadata['dtype'] = str(token_ids.dtype)

        header_json = json.dumps(metadata).encode('utf-8')
        header_size = len(header_json)

        with open(output_path, 'wb') as f:
            # Write header size
            f.write(struct.pack('I', header_size))
            # Write header
            f.write(header_json)
            # Write token data
            f.write(token_ids.tobytes())

        file_size = os.path.getsize(output_path)
        print(f"Created dataset: {file_size / 1e9:.2f} GB, "
              f"{len(token_ids):,} tokens")</code></pre>

<h4>5. Producer-Consumer Queues for Inference Batching</h4>
<p>High-throughput LLM serving requires <strong>dynamic batching</strong>: collecting incoming requests and batching them together for efficient GPU utilization. This is a classical producer-consumer problem:</p>

<pre><code>import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class InferenceRequest:
    """A single inference request in the batching queue."""
    request_id: str
    input_tokens: list
    max_output_tokens: int = 256
    temperature: float = 1.0
    arrived_at: float = field(default_factory=time.time)
    result_future: Any = None  # Will be set with the result

@dataclass
class InferenceBatch:
    """A batch of requests ready for processing."""
    requests: list
    created_at: float = field(default_factory=time.time)

    @property
    def size(self):
        return len(self.requests)

    @property
    def max_input_length(self):
        return max(len(r.input_tokens) for r in self.requests)


class DynamicBatcher:
    """Dynamic batching system for LLM inference.

    Collects incoming requests and forms batches based on:
    1. Maximum batch size (GPU memory constraint)
    2. Maximum wait time (latency SLA)
    3. Token budget (total tokens in batch <= GPU capacity)

    This is the core of serving systems like vLLM, TGI, and Triton.

    The producer-consumer pattern:
    - Producers: API handlers that submit requests
    - Consumer: GPU worker that processes batches
    - Queue: thread-safe buffer between them

    Uses condition variables for efficient waiting (no busy-polling).
    """

    def __init__(self, max_batch_size: int = 32,
                 max_wait_ms: float = 50.0,
                 max_token_budget: int = 16384,
                 process_fn: Callable = None):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_token_budget = max_token_budget
        self.process_fn = process_fn

        # Thread-safe request queue
        self.request_queue = queue.Queue()

        # Condition variable for signaling
        self.batch_ready = threading.Condition()

        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_tokens': 0,
            'avg_batch_size': 0,
            'avg_wait_ms': 0,
        }

        # Start the consumer thread
        self.running = True
        self.consumer_thread = threading.Thread(
            target=self._consumer_loop, daemon=True
        )
        self.consumer_thread.start()

    def submit(self, request: InferenceRequest) -> threading.Event:
        """Submit a request for batched inference.

        Returns an Event that will be set when the result is ready.
        The result will be attached to request.result_future.

        This is non-blocking: the caller can wait on the Event
        or continue doing other work.
        """
        done_event = threading.Event()
        request.result_future = done_event
        self.request_queue.put(request)
        self.stats['total_requests'] += 1

        # Signal the consumer that a new request is available
        with self.batch_ready:
            self.batch_ready.notify()

        return done_event

    def _form_batch(self) -> InferenceBatch:
        """Form a batch from queued requests.

        Greedily adds requests until one of the constraints is hit:
        - max_batch_size
        - max_token_budget
        - no more requests in queue
        """
        requests = []
        total_tokens = 0

        while (len(requests) < self.max_batch_size and
               not self.request_queue.empty()):
            try:
                req = self.request_queue.get_nowait()

                # Check token budget
                req_tokens = len(req.input_tokens) + req.max_output_tokens
                if total_tokens + req_tokens > self.max_token_budget and requests:
                    # Put it back - batch is full by token budget
                    self.request_queue.put(req)
                    break

                requests.append(req)
                total_tokens += req_tokens

            except queue.Empty:
                break

        if requests:
            return InferenceBatch(requests=requests)
        return None

    def _consumer_loop(self):
        """Main consumer loop: forms and processes batches.

        Waits for either:
        1. max_batch_size requests to arrive, OR
        2. max_wait_ms to elapse since first request in current window

        This balances throughput (larger batches) with latency (shorter waits).
        """
        while self.running:
            # Wait for at least one request
            with self.batch_ready:
                while self.request_queue.empty() and self.running:
                    self.batch_ready.wait(timeout=0.1)

            if not self.running:
                break

            # Wait for batch to fill or timeout
            wait_start = time.time()
            while (self.request_queue.qsize() < self.max_batch_size and
                   (time.time() - wait_start) * 1000 < self.max_wait_ms):
                time.sleep(0.001)  # 1ms granularity

            # Form and process batch
            batch = self._form_batch()
            if batch:
                self._process_batch(batch)

    def _process_batch(self, batch: InferenceBatch):
        """Process a batch of requests on the GPU."""
        batch_start = time.time()

        if self.process_fn:
            # Real model inference
            results = self.process_fn(batch)
        else:
            # Placeholder
            results = [f"result_{r.request_id}" for r in batch.requests]

        # Distribute results back to callers
        for req, result in zip(batch.requests, results):
            req.result = result
            if isinstance(req.result_future, threading.Event):
                req.result_future.set()  # Signal completion

        # Update statistics
        batch_time = time.time() - batch_start
        avg_wait = sum(
            time.time() - r.arrived_at for r in batch.requests
        ) / batch.size

        self.stats['total_batches'] += 1
        self.stats['total_tokens'] += sum(
            len(r.input_tokens) for r in batch.requests
        )
        self.stats['avg_batch_size'] = (
            self.stats['total_tokens'] / self.stats['total_batches']
        )
        self.stats['avg_wait_ms'] = avg_wait * 1000

    def shutdown(self):
        """Gracefully shut down the batcher."""
        self.running = False
        with self.batch_ready:
            self.batch_ready.notify_all()
        self.consumer_thread.join(timeout=5.0)


# Example: simulated inference server
def demo_dynamic_batching():
    """Demonstrate dynamic batching with simulated requests."""

    def mock_inference(batch):
        """Simulate model inference with batch processing."""
        time.sleep(0.05)  # Simulate 50ms GPU computation
        return [f"output_{r.request_id}" for r in batch.requests]

    batcher = DynamicBatcher(
        max_batch_size=8,
        max_wait_ms=20.0,
        max_token_budget=4096,
        process_fn=mock_inference
    )

    # Simulate concurrent requests
    events = []
    for i in range(20):
        req = InferenceRequest(
            request_id=f"req_{i}",
            input_tokens=list(range(50 + i * 10)),
            max_output_tokens=100
        )
        event = batcher.submit(req)
        events.append((req, event))
        time.sleep(0.005)  # 5ms between requests

    # Wait for all results
    for req, event in events:
        event.wait(timeout=5.0)
        print(f"{req.request_id}: completed "
              f"(wait: {(time.time() - req.arrived_at)*1000:.1f}ms)")

    print(f"\\nStats: {batcher.stats}")
    batcher.shutdown()</code></pre>

<h4>6. Complexity Summary for Systems Data Structures</h4>
<table>
<tr><th>Data Structure</th><th>Insert</th><th>Lookup</th><th>Delete</th><th>Space</th><th>AI Use Case</th></tr>
<tr><td>Bloom Filter</td><td>O(k)</td><td>O(k)</td><td>N/A</td><td>O(m) bits</td><td>Deduplication</td></tr>
<tr><td>Consistent Hash Ring</td><td>O(V log N)</td><td>O(log N)</td><td>O(V log N)</td><td>O(NV)</td><td>Model sharding</td></tr>
<tr><td>Ring Buffer</td><td>O(1)</td><td>O(1)</td><td>O(1)</td><td>O(C)</td><td>Streaming audio</td></tr>
<tr><td>Memory-Mapped File</td><td>O(1)*</td><td>O(1)*</td><td>N/A</td><td>O(1) RAM</td><td>Large datasets</td></tr>
<tr><td>Producer-Consumer Queue</td><td>O(1)</td><td>O(1)</td><td>O(1)</td><td>O(B)</td><td>Inference batching</td></tr>
</table>
<p>* Amortized; actual I/O depends on OS page cache behavior. V = virtual nodes, N = physical nodes, C = capacity, B = buffer size.</p>

<div class="callout">
<div class="callout-title">Interview Question: Bloom Filter Trade-offs</div>
<p><strong>Q:</strong> You need to deduplicate 10 billion web documents for LLM training. You have 32 GB of RAM. Can you use a Bloom filter? What false positive rate can you achieve?</p>
<p><strong>A:</strong> With 32 GB = 256 billion bits and 10 billion items, we get about 25.6 bits per item. The false positive rate is (1 - e^(-k*n/m))^k where optimal k = (m/n)*ln(2) = 25.6 * 0.693 = 17.7, round to 18. The FPR is approximately (1/2)^k = (1/2)^18 = 3.8 * 10^-6, or about 0.0004%. This means out of 10 billion unique documents, roughly 38,000 would be falsely flagged as duplicates and incorrectly removed. This is excellent for training data dedup where a tiny fraction of false positives is acceptable. For comparison, storing exact hashes (32 bytes each) would require 320 GB&mdash;10x more than available RAM.</p>
</div>

<div class="callout">
<div class="callout-title">Interview Question: Batching Optimization</div>
<p><strong>Q:</strong> In a dynamic batching system, how do you handle requests with very different sequence lengths efficiently?</p>
<p><strong>A:</strong> This is the <strong>padding waste</strong> problem. When batching sequences of lengths [10, 50, 200, 1000], all sequences must be padded to 1000 tokens, wasting 75%+ of computation. Solutions: (1) <strong>Bucket batching</strong>: group requests by similar length into buckets. (2) <strong>Continuous batching</strong> (Orca): instead of waiting for all sequences in a batch to finish, add new sequences and remove finished ones at each generation step. This keeps the batch full and avoids long sequences holding up short ones. (3) <strong>PagedAttention</strong> (vLLM): decouple batch scheduling from memory allocation, allowing sequences of any length to share GPU memory efficiently. (4) <strong>Sequence packing</strong>: pack multiple short sequences into one "long" sequence with separator tokens, used during training. The state of the art (vLLM, TGI) combines continuous batching with PagedAttention for optimal throughput.</p>
</div>
`
    },
  ],
};
