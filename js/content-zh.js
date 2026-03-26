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
            { id: "sd-fundamentals", title: "基础原理与核心协议", content: `
<p>LLM推理的瓶颈不在算力，而在<strong>显存带宽</strong>。自回归解码时，每生成一个token都要把整个模型权重从显存搬到计算单元一次，GPU的算力大量闲置。投机解码（Speculative Decoding）的核心思路就是：用一次昂贵的前向传播同时验证多个token，把闲置算力利用起来。</p>

<h4>核心协议四步走</h4>
<ol>
<li>一个轻量的<strong>草稿模型</strong>（Draft Model）自回归地生成 γ 个候选token</li>
<li><strong>目标模型</strong>（Target Model）对这 γ+1 个位置做一次并行前向传播</li>
<li>通过<strong>拒绝采样</strong>（Rejection Sampling）逐个决定接受还是拒绝每个token，同时保证输出分布与目标模型完全一致</li>
<li>期望接受的token数服从几何级数：<code>(1 - α^(γ+1)) / (1 - α)</code>，其中 α 为接受率。直觉上约等于 γ × α，实际可以带来 2-5x 的加速</li>
</ol>

<div class="callout">
<div class="callout-title">为什么投机解码能work</div>
<p>自回归解码时GPU利用率极低（受限于显存带宽）。投机解码把这些闲置的算力用来并行验证多个草稿token。数学上的关键保证：拒绝采样确保输出分布与目标模型<em>完全相同</em>——投机解码是无损的，不牺牲任何生成质量。</p>
</div>

<h4>拒绝采样的数学直觉</h4>
<p>对于草稿模型预测的每个token x：</p>
<ul>
<li>如果草稿概率 q(x) ≤ 目标概率 p(x)：直接接受</li>
<li>如果 q(x) > p(x)：以概率 p(x)/q(x) 接受</li>
<li>被拒绝的token从修正分布 (p - q) 中重新采样</li>
</ul>
<p>这个机制在数学上保证了最终输出严格服从目标模型分布。草稿模型只影响速度，绝不影响质量。</p>

<h4>2024年以来的技术分支</h4>
<ul>
<li><strong>更好的草稿器：</strong>Medusa、Hydra、EAGLE系列——用各种方式提升草稿质量</li>
<li><strong>更好的验证：</strong>树注意力（Tree Attention）、动态树构建——一次验证更多候选路径</li>
<li><strong>无草稿/自投机：</strong>模型用自身浅层特征做投机，不需要额外模型</li>
<li><strong>系统级集成：</strong>与vLLM、SGLang等推理框架的深度整合</li>
<li><strong>非文本模态：</strong>投机解码在图像、音频、视频生成中的应用</li>
</ul>

<pre><code># 投机解码的核心逻辑（伪代码）
def speculative_decode(target_model, draft_model, prompt, gamma=5):
    tokens = prompt
    while not finished:
        # 1. 草稿模型快速生成 gamma 个候选token
        draft_tokens, draft_probs = draft_model.generate(tokens, n=gamma)

        # 2. 目标模型一次性验证所有候选
        target_probs = target_model.forward(tokens + draft_tokens)

        # 3. 拒绝采样：逐个验证
        for i in range(gamma):
            if random() < min(1, target_probs[i] / draft_probs[i]):
                tokens.append(draft_tokens[i])  # 接受
            else:
                # 从修正分布重新采样，然后break
                tokens.append(resample(target_probs[i] - draft_probs[i]))
                break
    return tokens</code></pre>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">投机解码为什么不改变目标模型的输出分布？</div>
<div class="a-text">投机解码使用拒绝采样来决定是否接受每个草稿token。对于每个位置，目标模型计算真实概率 p(x)。如果草稿概率 q(x) ≤ p(x)，token总是被接受。如果 q(x) > p(x)，以概率 p(x)/q(x) 接受。被拒绝的token从修正分布 (p - q) 归一化后重新采样。这在数学上保证了最终输出严格服从目标分布。草稿模型只影响速度，绝不影响质量。</div>
</div>
` },
            { id: "sd-eagle", title: "EAGLE系列：当前最优", content: `
<p>EAGLE系列是草稿头（Draft Head）路线的代表作，三代演进清晰地展示了投机解码的优化方向。</p>

<h4>EAGLE（ICML 2024）</h4>
<p><strong>核心洞察：</strong>在特征层面做自回归比在token层面容易得多。token序列的不确定性很高（下一个词可能是很多种），但倒数第二层的特征向量变化平滑、可预测性强。EAGLE用当前token的embedding和顶层特征作为输入，通过一个额外的Transformer解码器层来预测下一步的特征。</p>
<p><strong>效果：</strong>LLaMA2-Chat 70B上 2.7-3.5x 延迟加速。比Medusa快 1.5-1.6x。完全无损。</p>

<h4>EAGLE-2（EMNLP 2024）</h4>
<p><strong>核心洞察：</strong>EAGLE的草稿模型是"校准良好"的——置信度分数近似于接受率。EAGLE-2利用这个特性构建<strong>上下文感知的动态草稿树</strong>：高置信度的分支多展开，低置信度的分支早剪枝。不再用固定的树结构，而是根据每次具体的生成情况动态调整。</p>
<p><strong>效果：</strong>3.05-4.26x 加速，比EAGLE-1快 20-40%。</p>

<h4>EAGLE-3（2025）</h4>
<p><strong>核心洞察：</strong>彻底放弃了特征预测的路线（这是EAGLE-1/2的天花板）。转而直接使用<strong>多层特征融合</strong>来预测下一个token。引入"训练时测试"技术，让草稿模型在训练时就暴露于模拟推理的多样化上下文中，弥合训练-推理的分布差距。</p>
<p><strong>效果：</strong>最高 6.5x 加速。第一个在 batch=64 时仍保持有效加速的EAGLE变体。单模型投机解码的当前SOTA。</p>

<pre><code># EAGLE草稿头的概念架构
class EAGLEDraftHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        self.feature_proj = nn.Linear(hidden_size, hidden_size)
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.decoder_layer = TransformerDecoderLayer(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, features):
        # 把token embedding和特征层信息结合
        x = self.token_embed(token_ids) + self.feature_proj(features)
        x = self.decoder_layer(x)
        return self.lm_head(x)</code></pre>

<h4>其他重要的草稿方法</h4>
<table>
<tr><th>方法</th><th>思路</th><th>加速比</th></tr>
<tr><td>Medusa</td><td>k个独立的额外LM头，树注意力验证</td><td>2.2-3.6x</td></tr>
<tr><td>Hydra</td><td>串行草稿头（每个以前一个为条件）</td><td>2.7x</td></tr>
<tr><td>HASS</td><td>表示对齐 + 一致性训练</td><td>提升接受率</td></tr>
<tr><td>Lookahead</td><td>基于Jacobi迭代的N-gram草稿</td><td>~1.5-2x</td></tr>
</table>

<div class="callout">
<div class="callout-title">EAGLE系列的演进逻辑</div>
<p>EAGLE-1发现特征比token更好预测 → EAGLE-2发现草稿模型的置信度可以指导树结构 → EAGLE-3发现多层融合比单层特征预测天花板更高。每一代都在解决上一代的核心瓶颈，这是很好的研究品味示范。</p>
</div>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">EAGLE为什么选择在特征层面而非token层面做自回归？这带来了什么好处？</div>
<div class="a-text">Token层面的自回归不确定性高——下一个token可能是词表中的任何词，概率分布很平坦。但在模型倒数第二层的特征空间中，相邻位置的表示变化平滑，可预测性强得多。EAGLE利用这个特性，用一个轻量的Transformer层在特征空间做预测，大幅提升了草稿质量（更高的接受率）。好处是：(1) 草稿模型可以很小（只有一层），参数少、推理快；(2) 接受率高于直接在token层面预测的方法（如Medusa）；(3) 训练简单，只需要蒸馏目标模型的特征。</div>
</div>
` },
            { id: "sd-production", title: "生产环境中的投机解码", content: `
<h4>大batch下加速坍缩问题</h4>
<p>投机解码在生产部署中最大的坑：<strong>加速比随batch size增大而急剧下降</strong>。batch=1时3-5x加速，到了batch=64以上，收益几乎消失。原因是：</p>
<ul>
<li>大batch下验证阶段从显存带宽瓶颈变成了算力瓶颈——GPU不再闲置，投机的前提不成立了</li>
<li>树注意力的显存开销随batch线性增长</li>
<li>草稿模型和目标模型争抢GPU资源</li>
</ul>

<div class="callout warning">
<div class="callout-title">生产踩坑：投机解码反噬</div>
<p>我们在Qwen-2.5-72B服务上部署了EAGLE-2，期望3x延迟降低。在4xA100 80GB上测量10K请求的结果：batch=1测试时P50 TTFT提升3.2x。但在20并发用户的生产环境中（24小时测量）：只有1.1x，几乎不值得增加的复杂度。草稿模型占用的显存本可以通过普通连续批处理多服务30%的并发请求。<strong>教训：</strong>一定要在你实际的并发模式下benchmark投机解码，而不是batch=1。我们最终只在低流量高优先级API层保留了投机解码，从主服务路径移除后吞吐量反而提升了25%。</p>
</div>

<h4>与推理框架的集成</h4>
<p>主流推理框架（vLLM、SGLang、TensorRT-LLM）都已内置投机解码支持：</p>
<ul>
<li><strong>SGLang：</strong>默认使用EAGLE-2作为投机解码方法，支持动态树构建，配合RadixAttention做前缀共享</li>
<li><strong>vLLM：</strong>支持Medusa、EAGLE和外部草稿模型，与连续批处理深度集成</li>
</ul>

<pre><code># 在SGLang中使用EAGLE
python -m sglang.launch_server \\
    --model meta-llama/Llama-3-70B-Instruct \\
    --speculative-algorithm EAGLE \\
    --speculative-draft-model eagle-llama3-70b \\
    --speculative-num-steps 5 \\
    --speculative-eagle-topk 8

# 在vLLM中使用投机解码
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3-70B-Instruct \\
    --speculative-model eagle-llama3-70b \\
    --num-speculative-tokens 5 \\
    --tensor-parallel-size 4</code></pre>

<div class="callout tip">
<div class="callout-title">生产建议</div>
<p>低并发场景（batch 1-8）：用EAGLE-2/3，延迟优化效果最佳。高并发场景（batch 32+）：连续批处理本身的吞吐量往往已经超过投机解码的收益。务必针对你的实际工作负载做性能画像。经验法则：如果你的GPU利用率已经超过70%，投机解码大概率帮不上忙。</p>
</div>

<h4>何时该用投机解码</h4>
<table>
<tr><th>场景</th><th>是否推荐</th><th>原因</th></tr>
<tr><td>低并发、延迟敏感</td><td>强烈推荐</td><td>GPU利用率低，投机的前提成立</td></tr>
<tr><td>高并发、吞吐优先</td><td>通常不推荐</td><td>GPU已饱和，草稿模型反而争抢资源</td></tr>
<tr><td>长文本生成</td><td>推荐</td><td>更多token意味着更多加速机会</td></tr>
<tr><td>代码生成</td><td>效果特别好</td><td>代码的可预测性高，接受率高</td></tr>
<tr><td>创意写作</td><td>效果一般</td><td>高随机性降低接受率</td></tr>
</table>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">在高并发LLM服务中，你会如何决定是否使用投机解码？</div>
<div class="a-text">关键决策因素：(1) 当前GPU利用率——如果已经>70%，投机解码的前提（利用闲置算力）不成立；(2) 并发模式——batch=1-8时收益最大，batch=32+时收益急剧下降；(3) 任务类型——代码等高可预测性任务接受率高，效果好；创意写作等低可预测性任务接受率低；(4) 延迟vs吞吐的优先级——投机解码优化延迟，不一定优化吞吐；(5) 显存预算——草稿模型需要额外显存，可能挤压并发容量。建议在实际流量模式下做A/B测试，而非只看batch=1的benchmark。</div>
</div>
` }
          ]
        },
        {
          id: "vllm-serving",
          title: "vLLM推理服务",
          sections: [
            { id: "vllm-architecture", title: "vLLM架构与PagedAttention", content: `
<p>vLLM是目前部署最广泛的开源LLM推理引擎。它的核心创新是<strong>分页注意力（PagedAttention）</strong>——用操作系统虚拟内存管理的思路来管理KV缓存的显存。</p>

<h4>KV缓存的显存困境</h4>
<p>自回归生成时，每个token需要对所有之前的token做注意力计算。为了避免重复计算，必须把历史的Key和Value张量缓存在显存里（即KV缓存）。问题有多严重？</p>
<ul>
<li>一个13B参数的MHA模型（40层、40头），服务2048 token的序列，每个请求的KV缓存需要约1.6GB显存</li>
<li>朴素分配方式浪费60-80%的KV缓存显存——这是巨大的浪费</li>
<li><strong>内部碎片：</strong>为每个请求预分配最大序列长度的空间，短序列浪费大量显存</li>
<li><strong>外部碎片：</strong>已分配块之间的间隙无法被利用</li>
</ul>

<h4>分页注意力的解决方案</h4>
<p>分页注意力把KV缓存切分成固定大小的"页"（block）。页按需分配，且物理上可以不连续——就像操作系统的虚拟内存一样：</p>

<pre><code># 分页注意力核心概念
# 不再是：每个序列一个连续的缓存区
# 而是：用页表将逻辑块映射到物理块

class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        # 物理KV缓存块（共享池）
        self.k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        # 每个序列的页表（逻辑块 -> 物理块）
        self.page_tables = {}

    def allocate_block(self, seq_id):
        physical_block = self.free_blocks.pop()
        self.page_tables[seq_id].append(physical_block)
        return physical_block

    def free_sequence(self, seq_id):
        # 序列结束时回收所有物理块
        for block in self.page_tables[seq_id]:
            self.free_blocks.append(block)
        del self.page_tables[seq_id]</code></pre>

<div class="callout">
<div class="callout-title">核心收益</div>
<p>分页注意力将KV缓存的显存浪费从60-80%降到接近零，在相同硬件上可以多服务2-4倍的并发请求。此外还支持跨序列的显存共享——当多个请求共享相同的系统提示词时，它们可以共享同一份KV缓存的物理页。</p>
</div>

<h4>连续批处理（Continuous Batching）</h4>
<p>传统静态批处理要等一整个batch的请求都完成才开始处理下一批。连续批处理则是：任何请求完成一步后，立刻插入新请求。好处是：</p>
<ul>
<li>GPU利用率大幅提升——不再有"等最慢的请求"的问题</li>
<li>首token延迟（TTFT）显著降低——新请求不用排队等整个batch结束</li>
<li>尾部延迟改善——短请求不会被长请求拖累</li>
</ul>

<h4>前缀缓存（Prefix Caching）</h4>
<p>当大量请求共享相同的系统提示词（system prompt）时，前缀缓存复用这个公共前缀的KV缓存。对于有长系统提示词的聊天应用，可以将TTFT降低80%以上。</p>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">解释PagedAttention的原理，以及它为什么对LLM推理服务很重要。</div>
<div class="a-text">PagedAttention借鉴了操作系统虚拟内存的分页机制来管理KV缓存。它不再为每个序列预分配最大长度的连续显存，而是按需分配固定大小的物理块，通过页表将每个序列的逻辑块映射到物理位置。好处：(1) 显存碎片几乎为零，(2) 相同硬件上可服务2-4倍的并发请求，(3) 共享前缀的请求可以共享KV缓存的物理页（比如系统提示词），(4) 支持动态序列长度，不需要过度分配。</div>
</div>
` },
            { id: "vllm-optimization", title: "LLM服务优化技术", content: `
<h4>推理量化</h4>
<p>量化是用更少的比特表示模型权重，以换取更低的显存占用和更快的推理速度：</p>
<table>
<tr><th>方法</th><th>位数</th><th>质量影响</th><th>加速效果</th></tr>
<tr><td>FP16</td><td>16</td><td>基线</td><td>1x</td></tr>
<tr><td>GPTQ</td><td>4</td><td>大多数任务影响很小</td><td>显存降低约2-3x</td></tr>
<tr><td>AWQ</td><td>4</td><td>略优于GPTQ</td><td>约2-3x显存节省，kernel更快</td></tr>
<tr><td>GGUF（llama.cpp）</td><td>2-8</td><td>因方法而异</td><td>CPU友好</td></tr>
<tr><td>FP8（H100）</td><td>8</td><td>几乎无损</td><td>约1.5x算力提升</td></tr>
</table>

<h4>张量并行 vs 流水线并行</h4>
<ul>
<li><strong>张量并行（TP）：</strong>把每层的权重矩阵切分到多个GPU上，所有GPU参与计算每一个token。需要高速互联（NVLink），适合同一节点内的并行。对延迟敏感的推理服务首选。</li>
<li><strong>流水线并行（PP）：</strong>把不同的层分配到不同GPU组上，每个GPU只负责一部分层。互联带宽要求低，适合跨节点场景。吞吐导向。</li>
</ul>

<pre><code># vLLM部署示例
# 4卡TP部署70B模型
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3-70B-Instruct \\
    --tensor-parallel-size 4 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.9 \\
    --enable-prefix-caching

# 关键参数说明：
# --tensor-parallel-size: 张量并行度，通常等于GPU数量
# --max-model-len: 最大序列长度，直接影响KV缓存占用
# --gpu-memory-utilization: 分配给KV缓存的显存比例
# --enable-prefix-caching: 开启前缀缓存，共享系统提示词</code></pre>

<div class="callout warning">
<div class="callout-title">生产踩坑：诡异的OOM</div>
<p>我们的vLLM部署在约200并发时不断OOM，尽管用的是80GB A100。<code>nvidia-smi</code>显示只用了71GB，远低于80GB上限。元凶：<code>gpu-memory-utilization</code>设为0.9（72GB），但vLLM会预先为KV缓存块保留显存。模型权重占28GB后，只剩44GB给KV缓存——以我们4096的max_model_len只够约180个并发请求的平均序列长度。<strong>解法：</strong>把<code>max_model_len</code>从4096降到2048（我们实际P99才1200 token），<code>gpu-memory-utilization</code>调到0.95。并发容量直接跳到350+。<strong>教训：</strong>永远根据你实际的流量分布来设置<code>max_model_len</code>，而不是模型的最大能力。</p>
</div>

<h4>性能调优核心指标</h4>
<ul>
<li><strong>TTFT（Time to First Token）：</strong>用户感知延迟的关键，优化方向是前缀缓存和更快的预填充</li>
<li><strong>TPOT（Time Per Output Token）：</strong>每个输出token的生成时间，决定了流式输出的流畅度</li>
<li><strong>吞吐量（Throughput）：</strong>每秒处理的token总数，连续批处理和量化是主要优化手段</li>
<li><strong>GPU利用率：</strong>目标>80%。如果低于50%，可能是batch太小或者有其他瓶颈</li>
</ul>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">如何优化一个LLM推理服务来支撑1000并发用户？</div>
<div class="a-text">核心策略：(1) 使用vLLM/SGLang的连续批处理最大化吞吐量；(2) 开启前缀缓存（如果请求共享系统提示词）；(3) 量化到FP8或INT4（AWQ）以在显存中容纳更多并发请求；(4) 用张量并行跨GPU部署模型；(5) 部署多副本在负载均衡器后面；(6) 实现带优先级的请求队列；(7) 设置合理的max_tokens限制；(8) 中低并发时考虑投机解码；(9) 监控GPU利用率、队列深度和P95延迟；(10) 使用流式响应降低用户感知延迟。关键是根据实际流量模式持续调优，而不是一次性配置。</div>
</div>
` }
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
            { id: "rlvr-fundamentals", title: "GRPO与可验证奖励", content: `
<p>RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励的强化学习）已经成为训练推理模型的主流范式。核心算法是<strong>GRPO（Group Relative Policy Optimization，组相对策略优化）</strong>。</p>

<h4>GRPO的训练循环</h4>
<ol>
<li><strong>采样（Rollout）：</strong>给定一个prompt，从当前策略中采样 G 个候选回复</li>
<li><strong>打分（Reward）：</strong>用可验证的奖励函数对每个回复打分（例如：代码能否执行通过、数学答案是否正确）</li>
<li><strong>更新（Update）：</strong>在组内计算归一化的优势值（advantage），通过PPO-clip损失 + KL惩罚更新策略</li>
</ol>

<div class="callout">
<div class="callout-title">核心工程挑战</div>
<p>采样阶段是推理过程（受益于vLLM/SGLang的连续批处理和KV缓存）。更新阶段是训练过程（需要反向传播、张量并行/流水线并行、梯度检查点）。两者的显存和算力需求完全不同。当前最佳方案：在推理和训练布局之间做<strong>权重重分片（weight resharding）</strong>。</p>
</div>

<h4>GRPO vs PPO：为什么更简单反而更好</h4>
<p>PPO需要训练一个独立的Critic（价值模型）来估计优势值，这带来了额外的显存开销和训练不稳定性。GRPO的巧妙之处在于：</p>
<ul>
<li>不需要Critic模型——直接在一组采样中计算相对优势</li>
<li>优势值 = (该回复的奖励 - 组内平均奖励) / 组内标准差</li>
<li>这种"组内相对排名"天然适配可验证奖励的场景</li>
<li>代价是需要更多采样（每个prompt采G个回复），但避免了Critic训练的复杂性</li>
</ul>

<h4>verl：生产级RLVR框架</h4>
<p>verl（Volcano Engine RL）提供了3D-HybridEngine，可以在vLLM（推理）和Megatron（训练）之间自动做权重重分片：</p>

<table>
<tr><th>维度</th><th>verl</th><th>自己搭</th></tr>
<tr><td>混合引擎</td><td>vLLM和Megatron之间自动重分片</td><td>自己实现要4-6周</td></tr>
<tr><td>采样吞吐</td><td>vLLM/SGLang后端，快3-5x</td><td>只能用HF generate</td></tr>
<tr><td>RL算法</td><td>GRPO、PPO、DAPO、REINFORCE++开箱即用</td><td>需要手动实现</td></tr>
<tr><td>集成成本</td><td>约1周</td><td>从零开始约3-4周</td></tr>
</table>

<h4>炼丹流水线示例：Audio Agent</h4>
<pre><code># 阶段1：冷启动SFT
# 教会模型think/answer格式
model: Qwen3-8B
audio_encoder: whisper-large-v3（冻结）
projection: 2层MLP（可训练）
llm: LoRA rank=64, alpha=128
batch_size: 32, lr: 2e-4, epochs: 3
hardware: 4x H100

# 阶段2：GRPO强化学习（verl）
# 用可验证奖励优化任务完成度
rollout: vLLM后端 + 自定义音频worker
reward: 任务完成度验证（代码执行、答案匹配）
training: Megatron-LM后端, TP=4

# 阶段3：评估
# AudioAgentBench + 分布外测试集</code></pre>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">GRPO是什么？它和PPO在LLM训练中有什么区别？</div>
<div class="a-text">GRPO（组相对策略优化）通过消除对独立Critic/价值模型的需求来简化PPO。PPO用学习到的价值函数来估计优势值，GRPO则是对每个prompt采样G个回复，用组内奖励的相对排名来计算优势值。好处：实现更简单、显存更省（不需要价值模型）、在可验证奖励场景下效果好。代价：每个prompt需要更多采样（即"组"），但避免了训练Critic的不稳定性和复杂性。</div>
</div>
` },
            { id: "distributed-training", title: "分布式训练基础", content: `
<h4>并行策略全景</h4>

<p><strong>数据并行（DP）：</strong>每张GPU持有完整模型副本，处理不同的数据。梯度通过all-reduce聚合。简单易用，但受限于单卡显存——模型必须能放进一张卡。</p>

<p><strong>张量并行（TP）：</strong>把每层的权重矩阵切分到多张GPU上。需要高速互联（NVLink），最适合节点内并行。每个token的计算涉及所有GPU，通信频率最高。</p>

<p><strong>流水线并行（PP）：</strong>把不同层分配到不同GPU组。用微批次（micro-batch）减少流水线气泡。适合跨节点场景，通信量比TP小。</p>

<p><strong>序列并行（SP）：</strong>把序列维度切分到多张GPU上。对于单卡装不下的长序列是刚需。</p>

<p><strong>ZeRO（零冗余优化器）：</strong></p>
<ul>
<li>ZeRO-1：跨GPU分片优化器状态（Adam的一阶/二阶矩）</li>
<li>ZeRO-2：+ 分片梯度</li>
<li>ZeRO-3：+ 分片模型参数（最省显存，但通信量最大）</li>
</ul>

<h4>显存优化技巧</h4>
<pre><code># 梯度检查点——用算力换显存
# 不存储所有中间激活值用于反向传播，
# 而是在反向传播时重新计算它们
model.gradient_checkpointing_enable()

# 混合精度训练
# 前向/反向用FP16/BF16，优化器状态保持FP32
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    loss = model(input_ids, labels=labels).loss
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# BF16 vs FP16的选择：
# BF16：动态范围大（和FP32一样），不容易溢出，H100/A100原生支持
# FP16：精度更高但动态范围小，需要loss scaling防止下溢
# 实践中BF16几乎总是更好的选择</code></pre>

<h4>常见炼丹事故排查</h4>
<table>
<tr><th>症状</th><th>可能原因</th><th>解法</th></tr>
<tr><td>Loss变NaN/Inf</td><td>学习率太大、坏数据</td><td>降低学习率，检查数据流水线，用BF16</td></tr>
<tr><td>Loss过早进入平台期</td><td>学习率太小、数据重复</td><td>提高学习率，检查数据shuffle</td></tr>
<tr><td>训练几步后OOM</td><td>动态shape、显存泄漏</td><td>检查padding策略，排查梯度累积</td></tr>
<tr><td>all-reduce缓慢</td><td>网络瓶颈</td><td>检查NCCL配置，用梯度压缩</td></tr>
<tr><td>梯度范数尖刺</td><td>坏数据batch、模型不稳定</td><td>梯度裁剪，数据过滤</td></tr>
</table>

<div class="callout warning">
<div class="callout-title">生产踩坑：静默的数据损坏</div>
<p>我们对Qwen2.5-7B做LoRA微调，验证集指标很好看，但生产环境输出全是乱码。根因：数据流水线中多worker数据加载存在竞态条件，约2%的训练样本被截断了半句话。模型学会了生成截断的输出。验证集没发现问题是因为BLEU/ROUGE算在剩余98%的干净数据上。<strong>解法：</strong>加了数据完整性校验（每个batch做hash验证），训练过程中记录采样输出（不只看loss），验证中增加"连贯性评分"衡量输出完整性。<strong>教训：</strong>训练时永远要检查实际的模型输出，而不是只看聚合指标。低loss数字可能掩盖灾难性的故障模式。</p>
</div>

<div class="callout warning">
<div class="callout-title">生产踩坑：8节点训练的NCCL超时</div>
<p>在8个节点64张H100上训练，到第500步左右总是挂起，报NCCL超时。单节点训练没问题。排查发现：一个节点的InfiniBand线缆接触不良，持续all-reduce时丢包0.1%。<code>ibstat</code>显示链路状态"Active"，但<code>perfquery</code>暴露了丢包率。<strong>解法：</strong>换线缆，设置<code>NCCL_IB_TIMEOUT=23</code>和<code>NCCL_IB_RETRY_CNT=7</code>环境变量，在每次训练前跑<code>all_reduce_bench</code>做健康检查。<strong>教训：</strong>分布式训练的网络问题表现为随机挂起，不是报错信息。多天训练任务启动前，必须先跑节点间通信的benchmark。</p>
</div>

<div class="interview-q">
<div class="q-label">面试题</div>
<div class="q-text">你需要训练一个70B的模型，有64张H100分布在8个节点上。你会用什么并行策略？</div>
<div class="a-text">推荐策略：TP=8（节点内，利用NVLink），PP=2（跨2组节点），DP=4（4路数据并行）。节点内8张GPU用张量并行保证最低延迟；2路流水线并行跨节点处理模型大小，流水线气泡可控；4路数据并行保证吞吐量。同时使用：梯度检查点减少激活值显存占用、BF16混合精度训练、序列并行处理长序列、ZeRO-1在DP组内分片优化器状态。</div>
</div>
` }
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
