// === AI Chat Module with Persistent Memory ===
(function() {
    'use strict';

    const MEMORY_KEY = 'lt_chat_memory';
    const HISTORY_KEY = 'lt_chat_history';
    const API_KEY_KEY = 'lt_api_key';

    let chatContext = null; // highlighted text context
    let conversationHistory = [];
    let memory = {};
    let isProcessing = false;

    document.addEventListener('DOMContentLoaded', () => {
        loadMemory();
        loadHistory();
        setupChat();
        renderMessages();
    });

    function setupChat() {
        const sendBtn = document.getElementById('chat-send');
        const input = document.getElementById('chat-input');
        const clearBtn = document.getElementById('chat-clear');
        const clearCtxBtn = document.getElementById('clear-context');

        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        clearBtn.addEventListener('click', clearChat);
        clearCtxBtn.addEventListener('click', clearContext);
    }

    function setContext(text) {
        chatContext = text;
        const banner = document.getElementById('chat-context-banner');
        const ctxText = document.getElementById('chat-context-text');
        ctxText.textContent = 'Context: "' + text.substring(0, 100) + (text.length > 100 ? '...' : '') + '"';
        banner.classList.remove('hidden');
    }

    function clearContext() {
        chatContext = null;
        document.getElementById('chat-context-banner').classList.add('hidden');
    }

    async function sendMessage() {
        const input = document.getElementById('chat-input');
        const text = input.value.trim();
        if (!text || isProcessing) return;

        // Build message with context
        let fullMessage = text;
        if (chatContext) {
            fullMessage = `[Highlighted text: "${chatContext}"]\n\n${text}`;
        }

        // Add user message
        addMessage('user', text);
        input.value = '';
        isProcessing = true;

        // Show typing indicator
        showTyping();

        try {
            const response = await getAIResponse(fullMessage);
            hideTyping();
            addMessage('assistant', response);

            // Check if AI suggested content updates
            checkForContentUpdates(response);

            // Update memory with conversation context
            updateMemory(text, response);
        } catch (err) {
            hideTyping();
            addMessage('assistant', getLocalResponse(fullMessage));
            updateMemory(text, 'Local response generated');
        }

        isProcessing = false;
    }

    async function getAIResponse(userMessage) {
        // Try to use API key if stored
        const apiKey = localStorage.getItem(API_KEY_KEY);

        // Build system prompt with memory
        const systemPrompt = buildSystemPrompt();

        // Build messages array with recent history
        const messages = [
            { role: 'system', content: systemPrompt }
        ];

        // Add recent conversation history (last 10 exchanges)
        const recentHistory = conversationHistory.slice(-20);
        recentHistory.forEach(msg => {
            messages.push({ role: msg.role, content: msg.content });
        });

        messages.push({ role: 'user', content: userMessage });

        if (apiKey) {
            // Try Anthropic API
            try {
                const resp = await fetch('https://api.anthropic.com/v1/messages', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'x-api-key': apiKey,
                        'anthropic-version': '2023-06-01',
                        'anthropic-dangerous-direct-browser-access': 'true'
                    },
                    body: JSON.stringify({
                        model: 'claude-sonnet-4-20250514',
                        max_tokens: 1024,
                        system: systemPrompt,
                        messages: messages.filter(m => m.role !== 'system').map(m => ({
                            role: m.role,
                            content: m.content
                        }))
                    })
                });

                if (resp.ok) {
                    const data = await resp.json();
                    return data.content[0].text;
                }
            } catch (e) {
                console.log('API call failed, using local response');
            }
        }

        // Fallback to local intelligent response
        return getLocalResponse(userMessage);
    }

    function buildSystemPrompt() {
        const memoryContext = Object.entries(memory)
            .map(([k, v]) => `- ${k}: ${v}`)
            .join('\n');

        return `You are an AI tutor embedded in an interactive AI Engineering textbook. Your role is to help the user understand concepts in AI/ML, audio AI, speech processing, LLM inference, and related topics.

PERSISTENT MEMORY (user preferences and past topics):
${memoryContext || '(No memory yet)'}

CAPABILITIES:
- Answer questions about highlighted text
- Explain concepts in depth
- Provide interview preparation help
- Suggest updates to textbook content when user identifies gaps

When suggesting content updates, format them as:
[CONTENT_UPDATE]
section: <section-id>
content: <new or modified content in HTML>
[/CONTENT_UPDATE]

Be concise, technical, and helpful. Use code examples where appropriate.`;
    }

    function getLocalResponse(userMessage) {
        const msg = userMessage.toLowerCase();

        // Extract highlighted context if present
        let context = '';
        const ctxMatch = userMessage.match(/\[Highlighted text: "(.*?)"\]/s);
        if (ctxMatch) {
            context = ctxMatch[1];
        }

        // Knowledge base for local responses
        const topics = {
            'attention': 'The attention mechanism computes weighted sums of values (V) based on the similarity between queries (Q) and keys (K). The formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. The scaling factor sqrt(d_k) prevents gradient vanishing in softmax. Multi-head attention runs h parallel attention operations, allowing the model to attend to different representation subspaces.',
            'transformer': 'The Transformer architecture (Vaswani et al., 2017) uses self-attention instead of recurrence. Key components: multi-head self-attention, feed-forward networks, layer normalization, and residual connections. Modern variants include: GPT (decoder-only, autoregressive), BERT (encoder-only, bidirectional), T5 (encoder-decoder). Key innovations since: RoPE (rotary position embeddings), GQA (grouped-query attention), RMSNorm, SwiGLU activations.',
            'speculative': 'Speculative decoding uses a small draft model to propose multiple tokens, then verifies them in parallel with the target model. The EAGLE series is state-of-the-art: EAGLE-1 uses feature-level prediction, EAGLE-2 adds dynamic tree building based on calibrated confidence, EAGLE-3 uses multi-layer feature fusion for up to 6.5x speedup. Key constraint: speedup diminishes at large batch sizes.',
            'vllm': 'vLLM uses PagedAttention to manage KV-cache memory efficiently, inspired by OS virtual memory. It allocates KV-cache in fixed-size blocks on demand, reducing memory waste from 60-80% to near zero. This enables 2-4x more concurrent requests. vLLM also supports continuous batching, prefix caching, and tensor parallelism.',
            'tts': 'Modern TTS has shifted from pipeline-based (text frontend -> acoustic model -> vocoder) to end-to-end neural codec language models. VALL-E treats TTS as language modeling over audio tokens. CosyVoice uses an LLM backbone with Flow Matching decoder. Key concepts: RVQ (residual vector quantization), zero-shot voice cloning, streaming synthesis.',
            'rlhf': 'RLHF/RLVR training for LLMs: SFT (supervised fine-tuning) -> Reward Model training -> RL optimization. GRPO (Group Relative Policy Optimization) simplifies PPO by sampling a group of responses and computing relative advantages, eliminating the need for a separate value model. verl framework provides weight resharding between inference (vLLM) and training (Megatron) layouts.',
            'whisper': 'Whisper is OpenAI\'s ASR model trained on 680K hours of multilingual data. Architecture: encoder-decoder Transformer. The encoder processes mel-spectrograms, the decoder generates text tokens autoregressively. Key features: multilingual, robust to noise, supports timestamps. For production: use VAD preprocessing, chunk long audio with overlap, set language tokens explicitly.',
            'interview': 'For AI engineer interviews, focus on: (1) ML fundamentals - gradient descent, backprop, regularization, attention, (2) Systems - distributed training, serving infrastructure, latency optimization, (3) Domain knowledge - your specific area (audio AI, NLP, CV), (4) Coding - implement core algorithms from scratch, (5) System design - design ML pipelines end-to-end.',
            'audio': 'AudioLLMs combine audio encoders (Whisper, BEATs, HuBERT) with LLM backbones. The 2025 frontier includes: omni-modal models (Qwen3-Omni), audio reasoning (Sound-CoT), and the discovery of "audio neglect" - models ignoring audio evidence in favor of text priors. Key open problem: should reasoning happen in text tokens or audio tokens?',
            'codec': 'Neural audio codecs (EnCodec, Mimi, DAC) compress audio into discrete tokens using Residual Vector Quantization. First codebook captures coarse features (content, speaker), subsequent codebooks capture finer acoustic details. This enables treating audio generation as a language modeling problem.',
            'flow matching': 'Flow Matching learns a velocity field that transforms noise into data directly. Unlike diffusion (many small denoising steps), FM learns straighter paths requiring fewer steps (10-50 vs 50-1000). Used in CosyVoice and other modern TTS systems. Compatible with Optimal Transport for even more efficient generation paths.',
            'grpo': 'GRPO (Group Relative Policy Optimization) simplifies PPO by removing the critic model. For each prompt, sample G responses, compute rewards, normalize advantages within the group. Advantages: simpler implementation, less memory (no value model), works well with verifiable rewards. Used in DeepSeek-R1 and other reasoning models.',
            'distributed': 'Distributed training strategies: Data Parallelism (replicate model, split data), Tensor Parallelism (split layers across GPUs, needs NVLink), Pipeline Parallelism (split layers across nodes), ZeRO (shard optimizer/gradients/params). For 70B on 64 GPUs: TP=8 intra-node, PP=2 inter-node groups, DP=4 replicas.',
            'kv cache': 'KV-cache stores previously computed key-value pairs during autoregressive generation to avoid recomputation. Without it, generating N tokens requires O(N^2) computation. With KV-cache, it\'s O(N). Memory cost: for a 70B model, ~2.5GB per request for 4K tokens. PagedAttention (vLLM) manages this efficiently.',
            'moshi': 'Moshi is the first real-time full-duplex speech LLM. Uses dual-stream architecture (user + model streams), RQ-Transformer on Mimi codec tokens, and Inner Monologue (text reasoning in parallel with audio generation). Achieves 200ms latency. Open challenges: fixed-rate inner monologue, emergent rather than explicit backchanneling.',
            'eagle': 'EAGLE series for speculative decoding: EAGLE-1 (feature-level autoregression, 2.7-3.5x), EAGLE-2 (dynamic trees from calibrated confidence, 3-4.3x), EAGLE-3 (multi-layer fusion, breaks feature prediction ceiling, up to 6.5x). Integrated into SGLang. Main limitation: speedup degrades at batch>8 (EAGLE-3 works up to batch=64).',
            'default': null
        };

        // Find best matching topic
        let bestResponse = null;
        let bestScore = 0;

        for (const [key, response] of Object.entries(topics)) {
            if (key === 'default') continue;
            const words = key.split(' ');
            let score = 0;
            words.forEach(w => {
                if (msg.includes(w)) score += 1;
            });
            // Bonus for context match
            if (context && context.toLowerCase().includes(key)) score += 2;
            if (score > bestScore) {
                bestScore = score;
                bestResponse = response;
            }
        }

        if (bestResponse && bestScore > 0) {
            if (context) {
                return `Regarding the highlighted text:\n\n${bestResponse}\n\nWould you like me to elaborate on any specific aspect?`;
            }
            return bestResponse;
        }

        // Generic responses
        if (msg.includes('help') || msg.includes('what can')) {
            return 'I can help you with:\n\n- **Explaining concepts** from the textbook\n- **Answering questions** about highlighted text\n- **Interview preparation** - ask me any ML interview question\n- **Deep dives** - ask me to elaborate on any topic\n- **Content suggestions** - tell me what\'s missing and I can suggest additions\n\nTry highlighting text and clicking the AI button, or just ask me anything about AI/ML!';
        }

        if (msg.includes('update') || msg.includes('add') || msg.includes('missing')) {
            return 'I\'d be happy to suggest content updates! Please describe what topic or information you\'d like added, and I\'ll format it for inclusion in the textbook. You can also highlight existing text and ask me to expand or modify it.';
        }

        if (context) {
            return `I see you've highlighted: "${context.substring(0, 100)}${context.length > 100 ? '...' : ''}"\n\nThis is an interesting passage. Could you tell me specifically what you'd like to know about it? For example:\n- Explain it in simpler terms\n- Go deeper into the technical details\n- How does this relate to other concepts?\n- How might this come up in an interview?`;
        }

        return 'That\'s a great question! This topic relates to AI engineering concepts covered in this textbook. Could you be more specific about what aspect you\'d like to explore? You can also:\n\n- Highlight text and click the AI button for context-specific help\n- Ask about specific topics like attention, speculative decoding, TTS, etc.\n- Request interview preparation on any topic';
    }

    function checkForContentUpdates(response) {
        const updateMatch = response.match(/\[CONTENT_UPDATE\]([\s\S]*?)\[\/CONTENT_UPDATE\]/);
        if (updateMatch) {
            const update = updateMatch[1];
            const sectionMatch = update.match(/section:\s*(\S+)/);
            const contentMatch = update.match(/content:\s*([\s\S]*)/);

            if (sectionMatch && contentMatch) {
                showUpdateBanner(sectionMatch[1], contentMatch[1].trim());
            }
        }
    }

    function showUpdateBanner(sectionId, newContent) {
        const messagesDiv = document.getElementById('chat-messages');
        const banner = document.createElement('div');
        banner.className = 'content-update-banner';
        banner.innerHTML = `
            <span>AI suggests a content update for this section</span>
            <button onclick="ChatModule.applyUpdate('${sectionId}', this)">Apply Update</button>
        `;
        banner._newContent = newContent;
        banner._sectionId = sectionId;
        messagesDiv.appendChild(banner);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function applyUpdate(sectionId, btnEl) {
        const banner = btnEl.closest('.content-update-banner');
        const section = document.getElementById(sectionId);
        if (section && banner._newContent) {
            // Append new content to section
            const div = document.createElement('div');
            div.className = 'ai-added-content';
            div.innerHTML = banner._newContent;
            section.appendChild(div);
            banner.innerHTML = '<span>Content updated successfully!</span>';

            // Save to local storage for persistence
            const updates = JSON.parse(localStorage.getItem('lt_content_updates') || '{}');
            updates[sectionId] = updates[sectionId] || [];
            updates[sectionId].push(banner._newContent);
            localStorage.setItem('lt_content_updates', JSON.stringify(updates));
        }
    }

    function addMessage(role, content) {
        conversationHistory.push({ role, content, timestamp: Date.now() });
        saveHistory();
        renderMessage(role, content);
    }

    function renderMessage(role, content) {
        const messagesDiv = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = `chat-message ${role}`;

        // Simple markdown-like rendering
        let html = content
            .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n- /g, '</p><ul><li>')
            .replace(/\n/g, '<br>');

        // Wrap in paragraphs
        if (!html.startsWith('<')) html = '<p>' + html + '</p>';

        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        div.innerHTML = `
            <div class="msg-bubble">${html}</div>
            <div class="msg-time">${time}</div>
        `;

        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function renderMessages() {
        const messagesDiv = document.getElementById('chat-messages');
        messagesDiv.innerHTML = '';

        if (conversationHistory.length === 0) {
            // Welcome message
            renderMessage('assistant', 'Welcome to the AI Learning Toolkit! I\'m your AI study assistant. I can help you:\n\n- **Understand concepts** - highlight text and ask me about it\n- **Prepare for interviews** - ask any ML/AI interview question\n- **Explore topics deeper** - request elaboration on any section\n- **Update content** - tell me what\'s missing and I\'ll suggest additions\n\nI remember our conversations across sessions. How can I help you today?');
        } else {
            // Render last 20 messages
            conversationHistory.slice(-20).forEach(msg => {
                renderMessage(msg.role, msg.content);
            });
        }
    }

    function showTyping() {
        const messagesDiv = document.getElementById('chat-messages');
        const typing = document.createElement('div');
        typing.className = 'chat-message assistant';
        typing.id = 'typing-indicator';
        typing.innerHTML = '<div class="msg-bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>';
        messagesDiv.appendChild(typing);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById('typing-indicator');
        if (el) el.remove();
    }

    function clearChat() {
        conversationHistory = [];
        saveHistory();
        renderMessages();
        clearContext();
    }

    // === Memory Management ===
    function loadMemory() {
        try {
            memory = JSON.parse(localStorage.getItem(MEMORY_KEY) || '{}');
        } catch {
            memory = {};
        }
    }

    function saveMemory() {
        localStorage.setItem(MEMORY_KEY, JSON.stringify(memory));
    }

    function updateMemory(userMsg, aiResponse) {
        // Extract key topics discussed
        const topics = extractTopics(userMsg);
        if (topics.length > 0) {
            memory['recent_topics'] = topics.slice(0, 5).join(', ');
        }

        // Track interaction count
        memory['interaction_count'] = (parseInt(memory['interaction_count'] || '0') + 1).toString();
        memory['last_interaction'] = new Date().toISOString();

        // Detect user preferences
        if (userMsg.toLowerCase().includes('interview')) {
            memory['interested_in'] = (memory['interested_in'] || '') + ', interview prep';
        }
        if (userMsg.toLowerCase().includes('audio') || userMsg.toLowerCase().includes('speech')) {
            memory['focus_area'] = 'audio AI / speech processing';
        }
        if (userMsg.toLowerCase().includes('explain') || userMsg.toLowerCase().includes('simpler')) {
            memory['learning_style'] = 'prefers detailed explanations';
        }

        saveMemory();
    }

    function extractTopics(text) {
        const keywords = ['attention', 'transformer', 'speculative', 'decoding', 'vllm', 'tts',
            'speech', 'audio', 'llm', 'training', 'inference', 'rlhf', 'grpo', 'whisper',
            'codec', 'diffusion', 'flow matching', 'moshi', 'eagle', 'kv cache', 'quantization',
            'distributed', 'pytorch', 'benchmark', 'evaluation'];
        return keywords.filter(k => text.toLowerCase().includes(k));
    }

    // === History Management ===
    function loadHistory() {
        try {
            conversationHistory = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        } catch {
            conversationHistory = [];
        }
    }

    function saveHistory() {
        // Keep last 100 messages
        const toSave = conversationHistory.slice(-100);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(toSave));
    }

    // === API Key Management ===
    function setApiKey(key) {
        localStorage.setItem(API_KEY_KEY, key);
    }

    // Export
    window.ChatModule = {
        setContext,
        clearContext,
        applyUpdate,
        setApiKey
    };
})();
