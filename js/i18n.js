// === Internationalization Module ===
(function() {
    'use strict';

    const LANG_KEY = 'lt_language';
    let currentLang = localStorage.getItem(LANG_KEY) || 'en';

    // Chinese translations for UI elements and key content
    const ZH = {
        // UI
        nav_title: 'AI工程师学习手册',
        contents: '目录',
        search_placeholder: '搜索内容... (Ctrl+K)',
        ai_assistant: 'AI助手',
        clear: '清除',
        add_comment: '添加批注',
        save: '保存',
        cancel: '取消',
        comment_placeholder: '写下你的批注...',
        chat_placeholder: '问关于高亮文本或任何话题的问题...',
        study_plan: '学习计划',
        recent_updates: '最近更新',
        chapters_label: '章',
        // Parts
        'part_1': '音频AI基础',
        'part_2': 'LLM推理与优化',
        'part_3': 'ML训练与基础设施',
        'part_4': '软件工程与职业发展',
        'part_5': 'Transformer与LLM基础',
        'part_6': '高级专题',
        // Chapter titles
        'audio-llm-landscape': 'AudioLLM研究全景',
        'speech-to-speech': '语音对话模型',
        'tts-technology': '语音合成（TTS）技术',
        'speculative-decoding': '投机解码',
        'vllm-serving': 'vLLM推理服务',
        'rl-training': 'LLM强化学习训练（RLHF/RLVR）',
        'ml-engineering': 'ML工程最佳实践',
        'agent-development': 'AI Agent开发',
        'system-design': 'AI应用系统设计',
        'interview-prep': 'AI工程师面试指南',
        // New chapters
        'transformer-basics': 'Transformer架构深度解析',
        'general-llms': '通用大语言模型',
        'quantization': '量化技术深度解析',
        'rag-systems': 'RAG检索增强生成',
        'dsa-for-ai': 'AI工程师数据结构与算法',
        // New section titles
        'transformer-overview': '从RNN到Attention',
        'self-attention': '自注意力机制',
        'positional-encoding': '位置编码（RoPE、ALiBi）',
        'ffn-architecture': '前馈网络与SwiGLU',
        'normalization': '归一化：LayerNorm、RMSNorm',
        'modern-architectures': '现代架构（LLaMA、Mistral、MoE）',
        'efficient-attention': 'FlashAttention与GQA',
        'transformer-training': '训练Transformer',
        'llm-landscape': 'LLM全景',
        'scaling-laws': '缩放定律与计算最优训练',
        'tokenization': '分词器深度解析（BPE）',
        'pretraining': '预训练：数据与流程',
        'finetuning': '微调：LoRA、QLoRA、适配器',
        'context-extension': '上下文窗口扩展',
        'llm-evaluation': 'LLM评估',
        'llm-practical': '实用LLM指南',
        'quant-fundamentals': '量化基础',
        'quant-ptq': 'GPTQ：训练后量化',
        'quant-awq': 'AWQ：激活感知量化',
        'quant-fp8': 'FP8量化',
        'quant-gguf': 'GGUF与llama.cpp',
        'quant-qat': '量化感知训练（QLoRA）',
        'quant-benchmarks': '质量vs速度基准',
        'quant-production': '生产环境量化',
        'rag-fundamentals': 'RAG基础',
        'rag-embeddings': '嵌入模型',
        'rag-vector-db': '向量数据库',
        'rag-chunking': '分块策略',
        'rag-retrieval': '检索方法',
        'rag-pipeline': '构建RAG流水线',
        'rag-evaluation': 'RAG评估',
        'rag-advanced': '高级RAG模式',
        'dsa-tokenizer': '字典树与BPE分词器',
        'dsa-beam-search': '优先队列与Beam Search',
        'dsa-kv-cache': '哈希表与KV-Cache',
        'dsa-attention': '注意力矩阵运算',
        'dsa-graphs': '计算图',
        'dsa-dp': '序列动态规划',
        'dsa-sampling': 'LLM采样算法',
        'dsa-systems': '系统级数据结构',
        // Section titles
        'audio-llm-overview': '概览与架构演进',
        'audio-neglect': 'Audio Neglect问题',
        'research-taste': '培养Audio AI研究品味',
        's2s-taxonomy': '架构分类学',
        's2s-latency': '延迟与实时性',
        'tts-evolution': 'TTS演进：从流水线到神经编解码器',
        'tts-flow-matching': 'Flow Matching与现代生成方法',
        'sd-fundamentals': '基础原理与核心协议',
        'sd-eagle': 'EAGLE系列：当前最优',
        'sd-production': '生产环境中的投机解码',
        'vllm-architecture': 'vLLM架构与PagedAttention',
        'vllm-optimization': 'LLM服务优化技术',
        'rlvr-fundamentals': 'GRPO与可验证奖励',
        'distributed-training': '分布式训练基础',
        'benchmark-hygiene': 'ML基准测试规范',
        'asr-pipeline': 'ASR流水线工程',
        'pytorch-gpu': 'PyTorch GPU服务模式',
        'agent-patterns': 'Agent架构模式',
        'agent-dev-practices': 'Agent辅助开发实践',
        'ml-system-design': 'ML系统设计模式',
        'networking-deploy': '受限网络部署',
        'ml-fundamentals-interview': 'ML基础知识',
        'coding-interview': 'ML面试编程题',
        'behavioral-interview': '行为面试与研究讨论',
        // Reading goals
        'week1': '第1周：音频AI基础',
        'week2': '第2周：LLM推理',
        'week3': '第3周：训练与工程',
        'week4': '第4周：应用与面试',
        'week5': '第5周：Transformer与LLM基础',
        'week6': '第6周：量化、RAG与数据结构',
        // Callout titles
        'key_insight': '核心洞察',
        'production_tip': '生产建议',
        'production_war_story': '生产踩坑实录',
        'critical_research_gap': '关键研究空白',
        'interview_question': '面试题',
        'taste_development_plan': '品味培养计划',
        // Changelog
        'changelog_1': '新增6个生产踩坑故事：SD批量失效、vLLM OOM调试、Whisper幻觉修复、LoRA数据损坏、NCCL超时、Agent死循环',
        'changelog_2': '修正KV-cache计算公式（含GQA）；更新扩散模型步数对比',
        'changelog_3': '首次发布：10章内容覆盖音频AI、LLM推理、ML训练和面试准备',
    };

    function t(key) {
        if (currentLang === 'zh' && ZH[key]) return ZH[key];
        return null; // null = use English original
    }

    function getLang() { return currentLang; }

    function setLang(lang) {
        currentLang = lang;
        localStorage.setItem(LANG_KEY, lang);
        // Rebuild UI
        if (window.App) {
            document.getElementById('textbook-content').innerHTML = '';
            window.App.renderContent();
            window.App.renderTOC();
            window.App.updateUILang();
        }
    }

    function toggleLang() {
        setLang(currentLang === 'en' ? 'zh' : 'en');
    }

    window.I18n = { t, getLang, setLang, toggleLang, ZH };
})();
