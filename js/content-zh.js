// Chinese content translations for key sections
// Uses natural Chinese tech community language (not machine translation)
const TEXTBOOK_ZH = {
  parts: [
    {
      title: "音频AI基础",
      chapters: [
        {
          id: "audio-llm-landscape",
          title: "AudioLLM研究全景",
          sections: [
            {
              id: "audio-llm-overview",
              title: "概览与架构演进",
              content: `
<p>音频大语言模型（AudioLLM）代表了从专用音频模型到统一架构的范式转变——一个模型同时理解和生成文本与音频。这个领域从2023到2025年经历了爆发式发展。</p>

<div class="callout">
<div class="callout-title">核心洞察</div>
<p>AudioLLM的核心架构模式：<strong>音频编码器 + 适配器 + LLM骨干网络 + 解码器</strong>。编码器把音频转成表示，适配器桥接模态，LLM负责推理，解码器生成输出。</p>
</div>

<h4>奠基论文（2023-2024）</h4>
<table>
<tr><th>论文</th><th>关键创新</th><th>影响</th></tr>
<tr><td><strong>Pengi</strong>（NeurIPS 2023）</td><td>所有音频任务统一为文本生成；音频编码器+文本编码器作为冻结LM的前缀</td><td>统一了音频-文本生成范式</td></tr>
<tr><td><strong>SALMONN</strong>（ICLR 2024）</td><td>双编码器（Whisper + BEATs）+ Q-Former适配器接Vicuna</td><td>首次研究跨模态涌现能力</td></tr>
<tr><td><strong>Qwen-Audio</strong></td><td>30+任务，层次化标签条件化解决多任务干扰</td><td>证明了规模+任务分类法优于手工模型</td></tr>
<tr><td><strong>AudioPaLM</strong></td><td>联合音频-文本词表；首次从LLM直接生成音频token</td><td>开启了端到端生成范式</td></tr>
</table>

<h4>2025年前沿</h4>
<p>这个领域分化出几个令人兴奋的方向：</p>
<ul>
<li><strong>全模态模型：</strong>Qwen2.5-Omni、Qwen3-Omni和Kimi-Audio实现了全模态输入+文本/语音输出，支持流式处理</li>
<li><strong>音频推理：</strong>Audio Flamingo Sound-CoT引入系统化的音频思维链；AudSemThinker把推理锚定在结构化听觉语义上</li>
<li><strong>长上下文：</strong>CALM用连续音频token（VAE）替代离散编解码器；YaRN + VLAT大幅扩展了上下文窗口</li>
<li><strong>领域专业化：</strong>SeaLLMs-Audio面向东南亚语言；FinAudio面向金融音频分析</li>
</ul>

<pre><code>2023-2024 基础                    2025 前沿
---------------------------------------------------
[编码器+LLM架构]            ->  [全模态、流式处理]
[多任务训练]                ->  [推理：CoT、RL、RL+CoT]
[通用基准]                  ->  [领域基准（金融、东南亚）]
[文本输出]                  ->  [音频内推理]
[固定上下文（<30s）]         ->  [长音频（YaRN、CALM）]
[级联vs端到端之争]          ->  [级联回归 vs 全模态]</code></pre>
`
            },
            {
              id: "audio-neglect",
              title: "Audio Neglect问题",
              content: `
<p><strong>Audio Neglect（音频忽视）</strong>是2025年的一个关键发现：AudioLLM系统性地低估音频证据。文本预训练的LLM骨干网络太强了，直接用语言先验回答问题，等于把音频信号当空气。</p>

<div class="callout warning">
<div class="callout-title">关键研究空白</div>
<p>2025年研究表明，即使音频是唯一有效信号，模型仍然忽略它。提出的修复方案（通过音频专用注意力头进行注意力引导）是临时性的。原则性的通用解决方案尚不存在。</p>
</div>

<p>这个发现动摇了许多已发表的AudioLLM结果的可信度。如果模型主要使用文本先验而非真正处理音频，那benchmark数字可能在"注水"。</p>

<h4>X-Talk的反叙事</h4>
<p>与此同时，X-Talk证明了模块化的ASR->LLM->TTS级联方案仍然具有竞争力，挑战了"全模态一定更好"的叙事。核心洞察：<strong>部署鲁棒性不等于benchmark表现</strong>。</p>

<h4>推理基底：文本token还是音频token？</h4>
<p>AudioLLM应该用文本token推理（快速、成熟，但丢失副语言细节）还是用音频token推理（保留声学信息，但昂贵且评估方法未定义）？</p>

<p>从人类认知角度看：人类即使在处理音频时也是用语言思考，而非用声音思考。我们从音频中提取概念，然后对概念进行推理。这意味着：</p>
<ul>
<li>完全的音频token推理可能既不自然也不必要</li>
<li>真正的价值在于<strong>混合方案</strong>：文本CoT作为主要推理 + 在关键决策点选择性地锚定音频信息</li>
</ul>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">什么是AudioLLM中的"Audio Neglect"问题？你会如何设计实验来衡量它？</div>
<div class="a-text">Audio Neglect指AudioLLM忽视决定性的音频证据，依赖LLM骨干网络的文本先验。衡量方法：(1) 设计正确答案必须依赖音频信息的任务（如说话人情绪、环境声音），(2) 构造对抗样本对——文本上下文暗示一个答案但音频证据指向另一个，(3) 对比有无音频输入时的准确率——如果差距不大，说明模型在忽略音频。</div>
</div>
`
            },
            {
              id: "research-taste",
              title: "培养Audio AI研究品味",
              content: `
<p><strong>研究品味</strong>是在你碰数据之前，告诉你<em>哪个</em>问题值得解决的指南针。它和研究技能不同（技能=怎么做，品味=做什么）。</p>

<h4>论文十问框架</h4>
<p>读每篇论文时都问这十个问题。品味是靠反复练习训练出来的：</p>
<ol>
<li><strong>核心claim？</strong>一句话。写不出来说明你没看懂。</li>
<li><strong>之前什么东西是坏的？</strong>不是"它达到了SOTA"——之前到底什么东西不work？</li>
<li><strong>关键的架构/方法论选择——为什么不用显而易见的替代方案？</strong></li>
<li><strong>Reviewer 2会怎么说？</strong>弱baseline？人造任务？贡献模糊？</li>
<li><strong>谁引用了这篇——谁明显没引用？</strong>这能告诉你它开创了一个流派还是终结了一个。</li>
<li><strong>它没解决什么？</strong>读Limitations。未来论文最大的线索在那里。</li>
<li><strong>如果claim是真的，什么变得可能了？</strong>"解锁"问题。</li>
<li><strong>评测指标是否在衡量真正重要的东西？</strong></li>
<li><strong>最简单的能推翻这篇论文的baseline？</strong></li>
<li><strong>如果这篇论文从历史上消失，今天什么东西不会存在？</strong>"杠杆"问题。</li>
</ol>

<div class="callout tip">
<div class="callout-title">品味培养计划</div>
<p><strong>第1-4周（建图）：</strong>读30篇论文；画概念图；问"如果它消失了，什么不会存在？"<br>
<strong>第5-8周（过滤）：</strong>逆向工程5篇被接收的论文；读5篇borderline rejection；每周想法筛选<br>
<strong>第9-16周（参与）：</strong>关注关键研究者；听报告；每月写Reviewer 2式批评<br>
<strong>第17-24周（检验）：</strong>两周原型冲刺；投workshop论文；先写introduction再跑实验</p>
</div>
`
            }
          ]
        },
        {
          id: "speech-to-speech",
          title: "语音对话模型",
          sections: [
            { id: "s2s-taxonomy", title: "架构分类学", content: null },
            { id: "s2s-latency", title: "延迟与实时性", content: null }
          ]
        },
        {
          id: "tts-technology",
          title: "语音合成（TTS）技术",
          sections: [
            { id: "tts-evolution", title: "TTS演进：从流水线到神经编解码器", content: null },
            { id: "tts-flow-matching", title: "Flow Matching与现代生成方法", content: null }
          ]
        }
      ]
    },
    {
      title: "LLM推理与优化",
      chapters: [
        {
          id: "speculative-decoding",
          title: "投机解码",
          sections: [
            { id: "sd-fundamentals", title: "基础原理与核心协议", content: null },
            { id: "sd-eagle", title: "EAGLE系列：当前最优", content: null },
            { id: "sd-production", title: "生产环境中的投机解码", content: null }
          ]
        },
        {
          id: "vllm-serving",
          title: "vLLM推理服务",
          sections: [
            { id: "vllm-architecture", title: "vLLM架构与PagedAttention", content: null },
            { id: "vllm-optimization", title: "LLM服务优化技术", content: null }
          ]
        }
      ]
    },
    {
      title: "ML训练与基础设施",
      chapters: [
        {
          id: "rl-training",
          title: "LLM强化学习训练（RLHF/RLVR）",
          sections: [
            { id: "rlvr-fundamentals", title: "GRPO与可验证奖励", content: null },
            { id: "distributed-training", title: "分布式训练基础", content: null }
          ]
        },
        {
          id: "ml-engineering",
          title: "ML工程最佳实践",
          sections: [
            { id: "benchmark-hygiene", title: "ML基准测试规范", content: null },
            { id: "asr-pipeline", title: "ASR流水线工程", content: null },
            { id: "pytorch-gpu", title: "PyTorch GPU服务模式", content: null }
          ]
        }
      ]
    },
    {
      title: "软件工程与职业发展",
      chapters: [
        {
          id: "agent-development",
          title: "AI Agent开发",
          sections: [
            { id: "agent-patterns", title: "Agent架构模式", content: null },
            { id: "agent-dev-practices", title: "Agent辅助开发实践", content: null }
          ]
        },
        {
          id: "system-design",
          title: "AI应用系统设计",
          sections: [
            { id: "ml-system-design", title: "ML系统设计模式", content: null },
            { id: "networking-deploy", title: "受限网络部署", content: null }
          ]
        },
        {
          id: "interview-prep",
          title: "AI工程师面试指南",
          sections: [
            { id: "ml-fundamentals-interview", title: "ML基础知识", content: null },
            { id: "coding-interview", title: "ML面试编程题", content: null },
            { id: "behavioral-interview", title: "行为面试与研究讨论", content: null }
          ]
        }
      ]
    }
  ]
};
