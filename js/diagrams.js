// SVG Diagrams for key AI/ML architectural concepts
// Each diagram uses CSS custom properties for theme-aware rendering

const DIAGRAMS = {

  "flash-attention-tiling": `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 280" width="600" height="280" style="max-width:100%;height:auto;" role="img" aria-label="FlashAttention block-by-block tiling diagram">
  <style>
    .fa-text { font-family: 'Segoe UI', system-ui, sans-serif; fill: var(--text-primary, #1a1a2e); }
    .fa-label { font-size: 11px; }
    .fa-title { font-size: 13px; font-weight: 600; }
    .fa-small { font-size: 10px; }
    .fa-annot { font-size: 10px; font-style: italic; fill: var(--accent, #4361ee); }
    .fa-block { stroke: var(--accent, #4361ee); stroke-width: 1.5; rx: 3; }
    .fa-block-q { fill: var(--accent, #4361ee); fill-opacity: 0.15; }
    .fa-block-kv { fill: var(--accent, #4361ee); fill-opacity: 0.08; }
    .fa-block-active { fill: var(--accent, #4361ee); fill-opacity: 0.35; }
    .fa-line { stroke: var(--border-color, #e0e0e8); stroke-width: 1; }
    .fa-arrow { stroke: var(--accent, #4361ee); stroke-width: 1.5; fill: none; marker-end: url(#fa-arrowhead); }
    .fa-mem-box { stroke: var(--border-color, #e0e0e8); stroke-width: 1.5; stroke-dasharray: 5,3; fill: none; rx: 6; }
  </style>
  <defs>
    <marker id="fa-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--accent, #4361ee)" />
    </marker>
  </defs>

  <!-- SRAM region -->
  <rect x="150" y="8" width="300" height="264" class="fa-mem-box" />
  <text x="300" y="26" text-anchor="middle" class="fa-text fa-annot">SRAM (on-chip, fast)</text>

  <!-- HBM labels -->
  <text x="60" y="26" text-anchor="middle" class="fa-text fa-annot">HBM</text>
  <text x="540" y="26" text-anchor="middle" class="fa-text fa-annot">HBM</text>

  <!-- Q matrix in HBM (left) -->
  <text x="60" y="52" text-anchor="middle" class="fa-text fa-title">Q</text>
  <rect x="20" y="60" width="80" height="30" class="fa-block fa-block-q" />
  <text x="60" y="80" text-anchor="middle" class="fa-text fa-label">Block 0</text>
  <rect x="20" y="94" width="80" height="30" class="fa-block fa-block-active" />
  <text x="60" y="114" text-anchor="middle" class="fa-text fa-label" style="font-weight:600;">Block i</text>
  <rect x="20" y="128" width="80" height="30" class="fa-block fa-block-q" />
  <text x="60" y="148" text-anchor="middle" class="fa-text fa-label">Block 2</text>
  <rect x="20" y="162" width="80" height="12" class="fa-block fa-block-q" />
  <text x="60" y="172" text-anchor="middle" class="fa-text fa-small">...</text>

  <!-- K/V matrix in HBM (right) -->
  <text x="540" y="52" text-anchor="middle" class="fa-text fa-title">K, V</text>
  <rect x="500" y="60" width="80" height="30" class="fa-block fa-block-kv" />
  <text x="540" y="80" text-anchor="middle" class="fa-text fa-label">Block 0</text>
  <rect x="500" y="94" width="80" height="30" class="fa-block fa-block-kv" />
  <text x="540" y="114" text-anchor="middle" class="fa-text fa-label">Block 1</text>
  <rect x="500" y="128" width="80" height="30" class="fa-block fa-block-active" />
  <text x="540" y="148" text-anchor="middle" class="fa-text fa-label" style="font-weight:600;">Block j</text>
  <rect x="500" y="162" width="80" height="12" class="fa-block fa-block-kv" />
  <text x="540" y="172" text-anchor="middle" class="fa-text fa-small">...</text>

  <!-- SRAM working area -->
  <rect x="190" y="60" width="70" height="50" class="fa-block fa-block-active" />
  <text x="225" y="80" text-anchor="middle" class="fa-text fa-label" style="font-weight:600;">Q_i</text>
  <text x="225" y="96" text-anchor="middle" class="fa-text fa-small">(loaded)</text>

  <rect x="340" y="60" width="70" height="50" class="fa-block fa-block-active" />
  <text x="375" y="80" text-anchor="middle" class="fa-text fa-label" style="font-weight:600;">K_j, V_j</text>
  <text x="375" y="96" text-anchor="middle" class="fa-text fa-small">(loaded)</text>

  <!-- Arrows from HBM to SRAM -->
  <path d="M100,109 L188,85" class="fa-arrow" />
  <path d="M500,143 L412,85" class="fa-arrow" />

  <!-- Compute box -->
  <rect x="220" y="130" width="160" height="40" class="fa-block" style="fill:var(--accent,#4361ee);fill-opacity:0.12;" />
  <text x="300" y="148" text-anchor="middle" class="fa-text fa-label" style="font-weight:600;">S_ij = Q_i * K_j^T</text>
  <text x="300" y="162" text-anchor="middle" class="fa-text fa-small">compute attention tile</text>

  <!-- Online softmax -->
  <rect x="220" y="180" width="160" height="34" class="fa-block" style="fill:var(--accent,#4361ee);fill-opacity:0.08;" />
  <text x="300" y="198" text-anchor="middle" class="fa-text fa-label">Online Softmax</text>
  <text x="300" y="210" text-anchor="middle" class="fa-text fa-small">running max &amp; sum</text>

  <!-- Output accumulate -->
  <rect x="220" y="224" width="160" height="34" class="fa-block" style="fill:var(--accent,#4361ee);fill-opacity:0.08;" />
  <text x="300" y="242" text-anchor="middle" class="fa-text fa-label">O_i += softmax(S_ij) * V_j</text>
  <text x="300" y="254" text-anchor="middle" class="fa-text fa-small">accumulate output</text>

  <!-- Flow arrows inside SRAM -->
  <line x1="300" y1="110" x2="300" y2="128" class="fa-arrow" />
  <line x1="300" y1="170" x2="300" y2="178" class="fa-arrow" />
  <line x1="300" y1="214" x2="300" y2="222" class="fa-arrow" />
</svg>`,

  "tree-attention": `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 260" width="600" height="260" style="max-width:100%;height:auto;" role="img" aria-label="Speculative decoding tree attention diagram">
  <style>
    .ta-text { font-family: 'Segoe UI', system-ui, sans-serif; fill: var(--text-primary, #1a1a2e); }
    .ta-label { font-size: 11px; }
    .ta-title { font-size: 13px; font-weight: 600; }
    .ta-small { font-size: 10px; }
    .ta-node { stroke: var(--accent, #4361ee); stroke-width: 1.5; }
    .ta-accept { fill: var(--accent, #4361ee); fill-opacity: 0.25; }
    .ta-reject { fill: #e74c3c; fill-opacity: 0.15; stroke: #e74c3c; }
    .ta-root { fill: var(--accent, #4361ee); fill-opacity: 0.4; }
    .ta-edge { stroke: var(--border-color, #e0e0e8); stroke-width: 1.5; fill: none; }
    .ta-check { fill: #27ae60; font-size: 13px; font-weight: bold; }
    .ta-cross { fill: #e74c3c; font-size: 13px; font-weight: bold; }
    .ta-verify-box { fill: var(--accent, #4361ee); fill-opacity: 0.08; stroke: var(--accent, #4361ee); stroke-width: 1; stroke-dasharray: 4,2; rx: 6; }
  </style>

  <!-- Target Model Verify label -->
  <rect x="170" y="4" width="260" height="24" class="ta-verify-box" />
  <text x="300" y="20" text-anchor="middle" class="ta-text ta-title" style="fill:var(--accent,#4361ee);">Target Model Verify (1 forward pass)</text>

  <!-- Root node (prompt token) -->
  <circle cx="300" cy="62" r="18" class="ta-node ta-root" />
  <text x="300" y="66" text-anchor="middle" class="ta-text ta-label" style="font-weight:600;">the</text>

  <!-- Level 1 branches -->
  <line x1="300" y1="80" x2="180" y2="120" class="ta-edge" />
  <line x1="300" y1="80" x2="300" y2="120" class="ta-edge" />
  <line x1="300" y1="80" x2="420" y2="120" class="ta-edge" />

  <!-- L1 nodes -->
  <circle cx="180" cy="130" r="16" class="ta-node ta-accept" />
  <text x="180" y="134" text-anchor="middle" class="ta-text ta-label">cat</text>
  <text x="200" y="124" class="ta-check">&#10003;</text>

  <circle cx="300" cy="130" r="16" class="ta-node ta-accept" />
  <text x="300" y="134" text-anchor="middle" class="ta-text ta-label">dog</text>
  <text x="320" y="124" class="ta-check">&#10003;</text>

  <circle cx="420" cy="130" r="16" class="ta-node ta-reject" />
  <text x="420" y="134" text-anchor="middle" class="ta-text ta-label">big</text>
  <text x="440" y="124" class="ta-cross">&#10007;</text>

  <!-- Level 2 from "cat" -->
  <line x1="180" y1="146" x2="120" y2="186" class="ta-edge" />
  <line x1="180" y1="146" x2="210" y2="186" class="ta-edge" />

  <circle cx="120" cy="196" r="16" class="ta-node ta-accept" />
  <text x="120" y="200" text-anchor="middle" class="ta-text ta-label">sat</text>
  <text x="140" y="190" class="ta-check">&#10003;</text>

  <circle cx="210" cy="196" r="16" class="ta-node ta-reject" />
  <text x="210" y="200" text-anchor="middle" class="ta-text ta-label">ran</text>
  <text x="230" y="190" class="ta-cross">&#10007;</text>

  <!-- Level 2 from "dog" -->
  <line x1="300" y1="146" x2="270" y2="186" class="ta-edge" />
  <line x1="300" y1="146" x2="340" y2="186" class="ta-edge" />

  <circle cx="270" cy="196" r="16" class="ta-node ta-reject" />
  <text x="270" y="200" text-anchor="middle" class="ta-text ta-label">ate</text>
  <text x="290" y="190" class="ta-cross">&#10007;</text>

  <circle cx="340" cy="196" r="16" class="ta-node ta-accept" />
  <text x="340" y="200" text-anchor="middle" class="ta-text ta-label">ran</text>
  <text x="360" y="190" class="ta-check">&#10003;</text>

  <!-- Level 3 leaf from "sat" -->
  <line x1="120" y1="212" x2="90" y2="236" class="ta-edge" />
  <circle cx="90" cy="244" r="14" class="ta-node ta-accept" />
  <text x="90" y="248" text-anchor="middle" class="ta-text ta-small">on</text>
  <text x="108" y="238" class="ta-check">&#10003;</text>

  <!-- Legend -->
  <circle cx="478" cy="70" r="8" class="ta-node ta-accept" />
  <text x="492" y="74" class="ta-text ta-small">= accepted (p >= q)</text>
  <circle cx="478" cy="92" r="8" class="ta-node ta-reject" />
  <text x="492" y="96" class="ta-text ta-small">= rejected (resample)</text>

  <!-- Draft model annotation -->
  <text x="490" y="140" text-anchor="middle" class="ta-text ta-small" style="fill:var(--accent,#4361ee);">Draft model</text>
  <text x="490" y="154" text-anchor="middle" class="ta-text ta-small" style="fill:var(--accent,#4361ee);">proposes tree</text>
  <text x="490" y="172" text-anchor="middle" class="ta-text ta-small">Target verifies</text>
  <text x="490" y="186" text-anchor="middle" class="ta-text ta-small">all paths in</text>
  <text x="490" y="200" text-anchor="middle" class="ta-text ta-small">parallel</text>
</svg>`,

  "rope-rotation": `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 260" width="600" height="260" style="max-width:100%;height:auto;" role="img" aria-label="RoPE positional encoding rotation diagram">
  <style>
    .ro-text { font-family: 'Segoe UI', system-ui, sans-serif; fill: var(--text-primary, #1a1a2e); }
    .ro-label { font-size: 11px; }
    .ro-title { font-size: 13px; font-weight: 600; }
    .ro-small { font-size: 10px; }
    .ro-formula { font-size: 11px; font-style: italic; fill: var(--accent, #4361ee); font-family: 'SF Mono', 'Consolas', monospace; }
    .ro-axis { stroke: var(--border-color, #e0e0e8); stroke-width: 1; }
    .ro-vec0 { stroke: var(--accent, #4361ee); stroke-width: 2; fill: none; marker-end: url(#ro-arrow-b); }
    .ro-vec1 { stroke: #e74c3c; stroke-width: 2; fill: none; marker-end: url(#ro-arrow-r); stroke-dasharray: 6,3; }
    .ro-arc { stroke: var(--accent, #4361ee); stroke-width: 1.2; fill: var(--accent, #4361ee); fill-opacity: 0.08; }
    .ro-pair-box { fill: var(--accent, #4361ee); fill-opacity: 0.08; stroke: var(--accent, #4361ee); stroke-width: 1; rx: 4; }
    .ro-dim-box { fill: var(--accent, #4361ee); fill-opacity: 0.04; stroke: var(--border-color, #e0e0e8); stroke-width: 1; rx: 3; }
  </style>
  <defs>
    <marker id="ro-arrow-b" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--accent, #4361ee)" />
    </marker>
    <marker id="ro-arrow-r" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#e74c3c" />
    </marker>
  </defs>

  <!-- Title -->
  <text x="300" y="20" text-anchor="middle" class="ro-text ro-title">Rotary Position Embedding (RoPE)</text>

  <!-- Left: 2D rotation visualization -->
  <!-- Axes -->
  <line x1="40" y1="150" x2="210" y2="150" class="ro-axis" />
  <line x1="120" y1="60" x2="120" y2="240" class="ro-axis" />
  <text x="215" y="154" class="ro-text ro-small">d_2i</text>
  <text x="124" y="56" class="ro-text ro-small">d_2i+1</text>

  <!-- Original vector (pos=0) -->
  <line x1="120" y1="150" x2="194" y2="100" class="ro-vec0" />
  <text x="198" y="96" class="ro-text ro-label" style="fill:var(--accent,#4361ee);">pos=0</text>

  <!-- Rotated vector (pos=m) -->
  <line x1="120" y1="150" x2="156" y2="72" class="ro-vec1" />
  <text x="150" y="66" class="ro-text ro-label" style="fill:#e74c3c;">pos=m</text>

  <!-- Rotation arc -->
  <path d="M170,118 A55,55 0 0,0 148,90" class="ro-arc" fill="none" />
  <text x="176" y="100" class="ro-text ro-small" style="fill:var(--accent,#4361ee);">m*theta_i</text>

  <!-- Right: embedding pair visualization -->
  <text x="400" y="52" text-anchor="middle" class="ro-text ro-label" style="font-weight:600;">Embedding vector pairs rotated by position</text>

  <!-- Dimension pair boxes -->
  <rect x="280" y="62" width="60" height="28" class="ro-pair-box" />
  <text x="310" y="80" text-anchor="middle" class="ro-text ro-small" style="font-weight:600;">(d0, d1)</text>
  <text x="310" y="96" text-anchor="middle" class="ro-formula">theta_0</text>

  <rect x="350" y="62" width="60" height="28" class="ro-pair-box" />
  <text x="380" y="80" text-anchor="middle" class="ro-text ro-small" style="font-weight:600;">(d2, d3)</text>
  <text x="380" y="96" text-anchor="middle" class="ro-formula">theta_1</text>

  <rect x="420" y="62" width="60" height="28" class="ro-pair-box" />
  <text x="450" y="80" text-anchor="middle" class="ro-text ro-small" style="font-weight:600;">(d4, d5)</text>
  <text x="450" y="96" text-anchor="middle" class="ro-formula">theta_2</text>

  <rect x="490" y="62" width="40" height="28" class="ro-dim-box" />
  <text x="510" y="80" text-anchor="middle" class="ro-text ro-small">...</text>

  <!-- Rotation matrix -->
  <text x="400" y="126" text-anchor="middle" class="ro-text ro-label">Each pair rotated by 2D rotation matrix:</text>

  <rect x="280" y="136" width="240" height="48" class="ro-dim-box" />
  <text x="400" y="155" text-anchor="middle" class="ro-formula" style="font-size:12px;">[ cos(m*theta_i)  -sin(m*theta_i) ]</text>
  <text x="400" y="175" text-anchor="middle" class="ro-formula" style="font-size:12px;">[ sin(m*theta_i)   cos(m*theta_i) ]</text>

  <!-- Formula -->
  <rect x="260" y="196" width="280" height="24" class="ro-dim-box" />
  <text x="400" y="213" text-anchor="middle" class="ro-formula" style="font-size:12px;">theta_i = 10000^(-2i/d)    m = position index</text>

  <!-- Key insight -->
  <text x="400" y="248" text-anchor="middle" class="ro-text ro-small">Low dims rotate fast (local patterns) | High dims rotate slow (global patterns)</text>
</svg>`,

  "paged-attention": `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 280" width="600" height="280" style="max-width:100%;height:auto;" role="img" aria-label="PagedAttention block table diagram">
  <style>
    .pa-text { font-family: 'Segoe UI', system-ui, sans-serif; fill: var(--text-primary, #1a1a2e); }
    .pa-label { font-size: 11px; }
    .pa-title { font-size: 13px; font-weight: 600; }
    .pa-small { font-size: 10px; }
    .pa-block { stroke: var(--border-color, #e0e0e8); stroke-width: 1; rx: 3; }
    .pa-seqA { fill: var(--accent, #4361ee); fill-opacity: 0.2; }
    .pa-seqB { fill: #e74c3c; fill-opacity: 0.15; }
    .pa-shared { fill: #27ae60; fill-opacity: 0.2; stroke: #27ae60; }
    .pa-free { fill: var(--text-muted, #8888a0); fill-opacity: 0.08; }
    .pa-arrow { stroke: var(--accent, #4361ee); stroke-width: 1.2; fill: none; marker-end: url(#pa-arrowhead); }
    .pa-arrowB { stroke: #e74c3c; stroke-width: 1.2; fill: none; marker-end: url(#pa-arrowhead-r); }
    .pa-table-border { stroke: var(--border-color, #e0e0e8); stroke-width: 1.5; fill: none; rx: 4; }
  </style>
  <defs>
    <marker id="pa-arrowhead" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <path d="M0,0 L7,2.5 L0,5 Z" fill="var(--accent, #4361ee)" />
    </marker>
    <marker id="pa-arrowhead-r" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <path d="M0,0 L7,2.5 L0,5 Z" fill="#e74c3c" />
    </marker>
  </defs>

  <!-- Title -->
  <text x="300" y="20" text-anchor="middle" class="pa-text pa-title">PagedAttention: Virtual Memory for KV-Cache</text>

  <!-- Sequence A: Logical Blocks -->
  <text x="10" y="52" class="pa-text pa-label" style="font-weight:600;fill:var(--accent,#4361ee);">Sequence A (logical)</text>
  <rect x="10" y="58" width="50" height="28" class="pa-block pa-shared" />
  <text x="35" y="76" text-anchor="middle" class="pa-text pa-small">Sys 0</text>
  <rect x="64" y="58" width="50" height="28" class="pa-block pa-shared" />
  <text x="89" y="76" text-anchor="middle" class="pa-text pa-small">Sys 1</text>
  <rect x="118" y="58" width="50" height="28" class="pa-block pa-seqA" />
  <text x="143" y="76" text-anchor="middle" class="pa-text pa-small">A_0</text>
  <rect x="172" y="58" width="50" height="28" class="pa-block pa-seqA" />
  <text x="197" y="76" text-anchor="middle" class="pa-text pa-small">A_1</text>

  <!-- Sequence B: Logical Blocks -->
  <text x="10" y="118" class="pa-text pa-label" style="font-weight:600;fill:#e74c3c;">Sequence B (logical)</text>
  <rect x="10" y="124" width="50" height="28" class="pa-block pa-shared" />
  <text x="35" y="142" text-anchor="middle" class="pa-text pa-small">Sys 0</text>
  <rect x="64" y="124" width="50" height="28" class="pa-block pa-shared" />
  <text x="89" y="142" text-anchor="middle" class="pa-text pa-small">Sys 1</text>
  <rect x="118" y="124" width="50" height="28" class="pa-block pa-seqB" />
  <text x="143" y="142" text-anchor="middle" class="pa-text pa-small">B_0</text>

  <!-- Page Table -->
  <rect x="250" y="40" width="110" height="128" class="pa-table-border" />
  <text x="305" y="58" text-anchor="middle" class="pa-text pa-label" style="font-weight:600;">Page Table</text>
  <line x1="250" y1="62" x2="360" y2="62" style="stroke:var(--border-color,#e0e0e8);stroke-width:1;" />

  <text x="268" y="78" class="pa-text pa-small">Sys 0</text>
  <text x="330" y="78" text-anchor="middle" class="pa-text pa-small">-> Phys 0</text>
  <line x1="250" y1="84" x2="360" y2="84" style="stroke:var(--border-color,#e0e0e8);stroke-width:0.5;" />

  <text x="268" y="98" class="pa-text pa-small">Sys 1</text>
  <text x="330" y="98" text-anchor="middle" class="pa-text pa-small">-> Phys 1</text>
  <line x1="250" y1="104" x2="360" y2="104" style="stroke:var(--border-color,#e0e0e8);stroke-width:0.5;" />

  <text x="268" y="118" class="pa-text pa-small" style="fill:var(--accent,#4361ee);">A_0</text>
  <text x="330" y="118" text-anchor="middle" class="pa-text pa-small">-> Phys 3</text>
  <line x1="250" y1="124" x2="360" y2="124" style="stroke:var(--border-color,#e0e0e8);stroke-width:0.5;" />

  <text x="268" y="138" class="pa-text pa-small" style="fill:var(--accent,#4361ee);">A_1</text>
  <text x="330" y="138" text-anchor="middle" class="pa-text pa-small">-> Phys 5</text>
  <line x1="250" y1="144" x2="360" y2="144" style="stroke:var(--border-color,#e0e0e8);stroke-width:0.5;" />

  <text x="268" y="158" class="pa-text pa-small" style="fill:#e74c3c;">B_0</text>
  <text x="330" y="158" text-anchor="middle" class="pa-text pa-small">-> Phys 4</text>

  <!-- Arrows from logical to page table -->
  <path d="M222,72 L248,72" class="pa-arrow" />
  <path d="M168,138 L248,138" class="pa-arrowB" />

  <!-- Physical Memory (GPU) -->
  <text x="495" y="42" text-anchor="middle" class="pa-text pa-label" style="font-weight:600;">Physical GPU Memory</text>

  <rect x="410" y="52" width="50" height="28" class="pa-block pa-shared" />
  <text x="435" y="70" text-anchor="middle" class="pa-text pa-small">P0</text>
  <text x="435" y="82" text-anchor="middle" class="pa-text pa-small" style="fill:#27ae60;">shared</text>

  <rect x="464" y="52" width="50" height="28" class="pa-block pa-shared" />
  <text x="489" y="70" text-anchor="middle" class="pa-text pa-small">P1</text>
  <text x="489" y="82" text-anchor="middle" class="pa-text pa-small" style="fill:#27ae60;">shared</text>

  <rect x="518" y="52" width="50" height="28" class="pa-block pa-free" />
  <text x="543" y="70" text-anchor="middle" class="pa-text pa-small">P2</text>
  <text x="543" y="82" text-anchor="middle" class="pa-text pa-small" style="fill:var(--text-muted,#8888a0);">free</text>

  <rect x="410" y="98" width="50" height="28" class="pa-block pa-seqA" />
  <text x="435" y="116" text-anchor="middle" class="pa-text pa-small">P3</text>
  <text x="435" y="128" text-anchor="middle" class="pa-text pa-small" style="fill:var(--accent,#4361ee);">A_0</text>

  <rect x="464" y="98" width="50" height="28" class="pa-block pa-seqB" />
  <text x="489" y="116" text-anchor="middle" class="pa-text pa-small">P4</text>
  <text x="489" y="128" text-anchor="middle" class="pa-text pa-small" style="fill:#e74c3c;">B_0</text>

  <rect x="518" y="98" width="50" height="28" class="pa-block pa-seqA" />
  <text x="543" y="116" text-anchor="middle" class="pa-text pa-small">P5</text>
  <text x="543" y="128" text-anchor="middle" class="pa-text pa-small" style="fill:var(--accent,#4361ee);">A_1</text>

  <!-- Arrow from page table to physical -->
  <path d="M360,90 L408,78" class="pa-arrow" />

  <!-- Shared prefix annotation -->
  <rect x="130" y="190" width="340" height="44" style="fill:var(--accent,#4361ee);fill-opacity:0.06;stroke:var(--accent,#4361ee);stroke-width:1;stroke-dasharray:4,2;rx:6;" />
  <text x="300" y="208" text-anchor="middle" class="pa-text pa-label" style="fill:#27ae60;font-weight:600;">Common prefix (system prompt) shares physical blocks</text>
  <text x="300" y="224" text-anchor="middle" class="pa-text pa-small">Seq A and Seq B both point to P0, P1 -- zero copy overhead</text>

  <!-- Memory savings note -->
  <text x="300" y="256" text-anchor="middle" class="pa-text pa-small">Fragmentation: ~0% (vs 60-80% with contiguous allocation)</text>
  <text x="300" y="270" text-anchor="middle" class="pa-text pa-small">Blocks allocated on demand, freed immediately on completion</text>
</svg>`,

  "encoder-adapter-llm": `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 220" width="600" height="220" style="max-width:100%;height:auto;" role="img" aria-label="AudioLLM encoder-adapter-LLM architecture">
  <style>
    .ea-text { font-family: 'Segoe UI', system-ui, sans-serif; fill: var(--text-primary, #1a1a2e); }
    .ea-label { font-size: 11px; }
    .ea-title { font-size: 13px; font-weight: 600; }
    .ea-small { font-size: 10px; }
    .ea-box { stroke: var(--border-color, #e0e0e8); stroke-width: 1.5; rx: 6; }
    .ea-input { fill: var(--accent, #4361ee); fill-opacity: 0.08; }
    .ea-encoder { fill: var(--accent, #4361ee); fill-opacity: 0.18; }
    .ea-adapter { fill: #e74c3c; fill-opacity: 0.12; stroke: #e74c3c; }
    .ea-llm { fill: var(--accent, #4361ee); fill-opacity: 0.25; }
    .ea-output { fill: #27ae60; fill-opacity: 0.12; stroke: #27ae60; }
    .ea-arrow { stroke: var(--accent, #4361ee); stroke-width: 1.5; fill: none; marker-end: url(#ea-arrowhead); }
    .ea-trainable { font-size: 9px; fill: #e74c3c; font-weight: 600; }
    .ea-frozen { font-size: 9px; fill: var(--text-muted, #8888a0); }
  </style>
  <defs>
    <marker id="ea-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--accent, #4361ee)" />
    </marker>
  </defs>

  <!-- Title -->
  <text x="300" y="20" text-anchor="middle" class="ea-text ea-title">AudioLLM Architecture Pipeline</text>

  <!-- Stage 1: Audio Waveform -->
  <rect x="8" y="50" width="72" height="70" class="ea-box ea-input" />
  <text x="44" y="74" text-anchor="middle" class="ea-text ea-label" style="font-weight:600;">Audio</text>
  <text x="44" y="88" text-anchor="middle" class="ea-text ea-small">Waveform</text>
  <!-- Waveform icon -->
  <path d="M20,104 Q26,96 30,104 Q34,112 38,104 Q42,96 46,104 Q50,112 54,104 Q58,96 62,104" stroke="var(--accent,#4361ee)" stroke-width="1.2" fill="none" />

  <!-- Arrow -->
  <line x1="82" y1="85" x2="96" y2="85" class="ea-arrow" />

  <!-- Stage 2: Mel Spectrogram -->
  <rect x="98" y="50" width="72" height="70" class="ea-box ea-input" />
  <text x="134" y="74" text-anchor="middle" class="ea-text ea-label" style="font-weight:600;">Mel</text>
  <text x="134" y="88" text-anchor="middle" class="ea-text ea-small">Spectrogram</text>
  <!-- Spectrogram grid icon -->
  <rect x="110" y="96" width="48" height="18" rx="2" style="fill:var(--accent,#4361ee);fill-opacity:0.1;stroke:var(--accent,#4361ee);stroke-width:0.5;" />
  <line x1="118" y1="96" x2="118" y2="114" style="stroke:var(--accent,#4361ee);stroke-width:0.5;opacity:0.5;" />
  <line x1="126" y1="96" x2="126" y2="114" style="stroke:var(--accent,#4361ee);stroke-width:0.5;opacity:0.5;" />
  <line x1="134" y1="96" x2="134" y2="114" style="stroke:var(--accent,#4361ee);stroke-width:0.5;opacity:0.5;" />
  <line x1="142" y1="96" x2="142" y2="114" style="stroke:var(--accent,#4361ee);stroke-width:0.5;opacity:0.5;" />
  <line x1="150" y1="96" x2="150" y2="114" style="stroke:var(--accent,#4361ee);stroke-width:0.5;opacity:0.5;" />
  <line x1="110" y1="102" x2="158" y2="102" style="stroke:var(--accent,#4361ee);stroke-width:0.5;opacity:0.5;" />
  <line x1="110" y1="108" x2="158" y2="108" style="stroke:var(--accent,#4361ee);stroke-width:0.5;opacity:0.5;" />

  <!-- Arrow -->
  <line x1="172" y1="85" x2="186" y2="85" class="ea-arrow" />

  <!-- Stage 3: Whisper Encoder -->
  <rect x="188" y="42" width="90" height="78" class="ea-box ea-encoder" />
  <text x="233" y="68" text-anchor="middle" class="ea-text ea-label" style="font-weight:600;">Whisper</text>
  <text x="233" y="82" text-anchor="middle" class="ea-text ea-label">Encoder</text>
  <text x="233" y="100" text-anchor="middle" class="ea-frozen">[FROZEN]</text>
  <text x="233" y="114" text-anchor="middle" class="ea-text ea-small">audio -> features</text>

  <!-- Arrow -->
  <line x1="280" y1="85" x2="294" y2="85" class="ea-arrow" />

  <!-- Stage 4: Adapter -->
  <rect x="296" y="42" width="90" height="78" class="ea-box ea-adapter" />
  <text x="341" y="66" text-anchor="middle" class="ea-text ea-label" style="font-weight:600;">Adapter</text>
  <text x="341" y="82" text-anchor="middle" class="ea-text ea-small">Q-Former</text>
  <text x="341" y="94" text-anchor="middle" class="ea-text ea-small">or MLP</text>
  <text x="341" y="110" text-anchor="middle" class="ea-trainable">[TRAINABLE]</text>

  <!-- Arrow -->
  <line x1="388" y1="85" x2="402" y2="85" class="ea-arrow" />

  <!-- Stage 5: LLM -->
  <rect x="404" y="38" width="90" height="86" class="ea-box ea-llm" />
  <text x="449" y="64" text-anchor="middle" class="ea-text ea-label" style="font-weight:600;">LLM</text>
  <text x="449" y="80" text-anchor="middle" class="ea-text ea-small">Qwen / LLaMA</text>
  <text x="449" y="94" text-anchor="middle" class="ea-text ea-small">reasoning</text>
  <text x="449" y="110" text-anchor="middle" class="ea-trainable">[LoRA]</text>

  <!-- Arrow -->
  <line x1="496" y1="85" x2="510" y2="85" class="ea-arrow" />

  <!-- Stage 6: Text Output -->
  <rect x="512" y="50" width="80" height="70" class="ea-box ea-output" />
  <text x="552" y="76" text-anchor="middle" class="ea-text ea-label" style="font-weight:600;">Text</text>
  <text x="552" y="90" text-anchor="middle" class="ea-text ea-small">Output</text>
  <text x="552" y="108" text-anchor="middle" class="ea-text ea-small" style="fill:#27ae60;">transcription</text>

  <!-- Bottom annotation: modality bridge explanation -->
  <rect x="80" y="148" width="440" height="30" style="fill:var(--accent,#4361ee);fill-opacity:0.05;stroke:var(--accent,#4361ee);stroke-width:1;stroke-dasharray:4,2;rx:6;" />
  <text x="300" y="168" text-anchor="middle" class="ea-text ea-small">Adapter bridges the modality gap: projects audio features (1500 frames) into LLM token space (~80 tokens)</text>

  <!-- Training stages -->
  <text x="300" y="200" text-anchor="middle" class="ea-text ea-label" style="font-weight:600;">Training: Stage 1 -- train adapter (encoder+LLM frozen) | Stage 2 -- LoRA fine-tune LLM</text>
  <text x="300" y="214" text-anchor="middle" class="ea-text ea-small">Variants: SALMONN uses dual encoder (Whisper + BEATs); Qwen-Audio uses hierarchical tags</text>
</svg>`,

  "rlhf-pipeline": `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 280" width="600" height="280" style="max-width:100%;height:auto;" role="img" aria-label="RLHF and RLVR training pipeline diagram">
  <style>
    .rl-text { font-family: 'Segoe UI', system-ui, sans-serif; fill: var(--text-primary, #1a1a2e); }
    .rl-label { font-size: 11px; }
    .rl-title { font-size: 13px; font-weight: 600; }
    .rl-small { font-size: 10px; }
    .rl-box { stroke: var(--border-color, #e0e0e8); stroke-width: 1.5; rx: 6; }
    .rl-sft { fill: var(--accent, #4361ee); fill-opacity: 0.12; }
    .rl-rm { fill: #e74c3c; fill-opacity: 0.1; stroke: #e74c3c; }
    .rl-ppo { fill: #27ae60; fill-opacity: 0.1; stroke: #27ae60; }
    .rl-step { fill: var(--accent, #4361ee); fill-opacity: 0.08; rx: 4; }
    .rl-arrow { stroke: var(--accent, #4361ee); stroke-width: 1.5; fill: none; marker-end: url(#rl-arrowhead); }
    .rl-loop { stroke: #27ae60; stroke-width: 1.5; fill: none; marker-end: url(#rl-arrowhead-g); stroke-dasharray: 6,3; }
  </style>
  <defs>
    <marker id="rl-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--accent, #4361ee)" />
    </marker>
    <marker id="rl-arrowhead-g" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#27ae60" />
    </marker>
  </defs>

  <!-- Title -->
  <text x="300" y="20" text-anchor="middle" class="rl-text rl-title">RLHF / RLVR Training Pipeline</text>

  <!-- Phase 1: SFT -->
  <rect x="20" y="38" width="140" height="72" class="rl-box rl-sft" />
  <text x="90" y="58" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">Phase 1: SFT</text>
  <text x="90" y="74" text-anchor="middle" class="rl-text rl-small">Supervised Fine-Tuning</text>
  <text x="90" y="90" text-anchor="middle" class="rl-text rl-small">on demonstration data</text>
  <text x="90" y="104" text-anchor="middle" class="rl-text rl-small" style="fill:var(--accent,#4361ee);">Base LLM -> SFT Model</text>

  <!-- Arrow SFT -> RM -->
  <line x1="162" y1="74" x2="188" y2="74" class="rl-arrow" />

  <!-- Phase 2: Reward Model -->
  <rect x="190" y="38" width="140" height="72" class="rl-box rl-rm" />
  <text x="260" y="58" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">Phase 2: Reward Model</text>
  <text x="260" y="74" text-anchor="middle" class="rl-text rl-small">Human preferences</text>
  <text x="260" y="88" text-anchor="middle" class="rl-text rl-small">(A > B rankings)</text>
  <text x="260" y="104" text-anchor="middle" class="rl-text rl-small" style="fill:#e74c3c;">or Verifiable Rewards</text>

  <!-- Arrow RM -> PPO -->
  <line x1="332" y1="74" x2="358" y2="74" class="rl-arrow" />

  <!-- Phase 3: PPO/GRPO -->
  <rect x="360" y="32" width="220" height="84" class="rl-box rl-ppo" />
  <text x="470" y="52" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">Phase 3: PPO / GRPO Loop</text>

  <!-- Sub-steps inside PPO -->
  <rect x="374" y="60" width="70" height="24" class="rl-box rl-step" />
  <text x="409" y="76" text-anchor="middle" class="rl-text rl-small">1. Rollout</text>

  <rect x="374" y="88" width="70" height="24" class="rl-box rl-step" />
  <text x="409" y="104" text-anchor="middle" class="rl-text rl-small">2. Reward</text>

  <rect x="496" y="60" width="70" height="24" class="rl-box rl-step" />
  <text x="531" y="76" text-anchor="middle" class="rl-text rl-small">3. Update</text>

  <!-- Arrows within loop -->
  <line x1="446" y1="72" x2="494" y2="72" class="rl-arrow" />
  <path d="M446,100 L478,100 L478,72 L494,72" class="rl-arrow" />

  <!-- Loop arrow back from Update to Rollout -->
  <path d="M531,86 L531,120 L380,120 L380,114" class="rl-loop" />
  <text x="456" y="134" text-anchor="middle" class="rl-text rl-small" style="fill:#27ae60;">iterate until convergence</text>

  <!-- Detailed GRPO breakdown below -->
  <text x="300" y="166" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">GRPO Detail (Phase 3 expanded):</text>

  <!-- Rollout detail -->
  <rect x="20" y="178" width="130" height="56" class="rl-box rl-step" />
  <text x="85" y="196" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">Rollout</text>
  <text x="85" y="210" text-anchor="middle" class="rl-text rl-small">Sample G responses</text>
  <text x="85" y="222" text-anchor="middle" class="rl-text rl-small">per prompt (vLLM)</text>

  <line x1="152" y1="206" x2="172" y2="206" class="rl-arrow" />

  <!-- Reward detail -->
  <rect x="174" y="178" width="130" height="56" class="rl-box rl-step" />
  <text x="239" y="196" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">Score</text>
  <text x="239" y="210" text-anchor="middle" class="rl-text rl-small">Verifiable reward</text>
  <text x="239" y="222" text-anchor="middle" class="rl-text rl-small">(code exec, math check)</text>

  <line x1="306" y1="206" x2="326" y2="206" class="rl-arrow" />

  <!-- Advantage detail -->
  <rect x="328" y="178" width="130" height="56" class="rl-box rl-step" />
  <text x="393" y="196" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">Advantage</text>
  <text x="393" y="210" text-anchor="middle" class="rl-text rl-small">Normalize within</text>
  <text x="393" y="222" text-anchor="middle" class="rl-text rl-small">group (no critic!)</text>

  <line x1="460" y1="206" x2="480" y2="206" class="rl-arrow" />

  <!-- Update detail -->
  <rect x="482" y="178" width="100" height="56" class="rl-box rl-step" />
  <text x="532" y="196" text-anchor="middle" class="rl-text rl-label" style="font-weight:600;">Update</text>
  <text x="532" y="210" text-anchor="middle" class="rl-text rl-small">PPO-clip loss</text>
  <text x="532" y="222" text-anchor="middle" class="rl-text rl-small">+ KL penalty</text>

  <!-- Bottom note -->
  <text x="300" y="260" text-anchor="middle" class="rl-text rl-small">RLHF uses learned reward model | RLVR uses verifiable rewards (no human labels needed)</text>
  <text x="300" y="274" text-anchor="middle" class="rl-text rl-small">Key engineering: weight resharding between inference (rollout) and training (update) layouts</text>
</svg>`,

  "rag-pipeline": `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 280" width="600" height="280" style="max-width:100%;height:auto;" role="img" aria-label="RAG pipeline diagram">
  <style>
    .rg-text { font-family: 'Segoe UI', system-ui, sans-serif; fill: var(--text-primary, #1a1a2e); }
    .rg-label { font-size: 11px; }
    .rg-title { font-size: 13px; font-weight: 600; }
    .rg-small { font-size: 10px; }
    .rg-box { stroke: var(--border-color, #e0e0e8); stroke-width: 1.5; rx: 6; }
    .rg-ingest { fill: var(--accent, #4361ee); fill-opacity: 0.1; }
    .rg-query { fill: #27ae60; fill-opacity: 0.1; }
    .rg-db { fill: #e74c3c; fill-opacity: 0.1; stroke: #e74c3c; }
    .rg-llm { fill: var(--accent, #4361ee); fill-opacity: 0.2; }
    .rg-arrow { stroke: var(--accent, #4361ee); stroke-width: 1.5; fill: none; marker-end: url(#rg-arrowhead); }
    .rg-arrow-g { stroke: #27ae60; stroke-width: 1.5; fill: none; marker-end: url(#rg-arrowhead-g); }
    .rg-phase-label { font-size: 12px; font-weight: 600; }
  </style>
  <defs>
    <marker id="rg-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--accent, #4361ee)" />
    </marker>
    <marker id="rg-arrowhead-g" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#27ae60" />
    </marker>
  </defs>

  <!-- Title -->
  <text x="300" y="20" text-anchor="middle" class="rg-text rg-title">Retrieval-Augmented Generation (RAG) Pipeline</text>

  <!-- Phase label: Ingestion -->
  <text x="20" y="50" class="rg-text rg-phase-label" style="fill:var(--accent,#4361ee);">Ingestion (offline)</text>

  <!-- Document -->
  <rect x="20" y="58" width="80" height="46" class="rg-box rg-ingest" />
  <text x="60" y="78" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Documents</text>
  <text x="60" y="94" text-anchor="middle" class="rg-text rg-small">PDF, HTML, ...</text>

  <line x1="102" y1="81" x2="118" y2="81" class="rg-arrow" />

  <!-- Chunk -->
  <rect x="120" y="58" width="72" height="46" class="rg-box rg-ingest" />
  <text x="156" y="78" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Chunk</text>
  <text x="156" y="94" text-anchor="middle" class="rg-text rg-small">split by size</text>

  <line x1="194" y1="81" x2="210" y2="81" class="rg-arrow" />

  <!-- Embed -->
  <rect x="212" y="58" width="76" height="46" class="rg-box rg-ingest" />
  <text x="250" y="78" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Embed</text>
  <text x="250" y="94" text-anchor="middle" class="rg-text rg-small">text -> vector</text>

  <line x1="290" y1="81" x2="306" y2="81" class="rg-arrow" />

  <!-- Vector DB -->
  <rect x="308" y="52" width="100" height="58" class="rg-box rg-db" />
  <text x="358" y="74" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Vector DB</text>
  <text x="358" y="90" text-anchor="middle" class="rg-text rg-small">FAISS / Pinecone</text>
  <text x="358" y="102" text-anchor="middle" class="rg-text rg-small">Chroma / Milvus</text>

  <!-- Phase label: Query -->
  <text x="20" y="138" class="rg-text rg-phase-label" style="fill:#27ae60;">Query (online)</text>

  <!-- Query -->
  <rect x="20" y="146" width="80" height="46" class="rg-box rg-query" />
  <text x="60" y="166" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">User Query</text>
  <text x="60" y="182" text-anchor="middle" class="rg-text rg-small">"How does...?"</text>

  <line x1="102" y1="169" x2="118" y2="169" class="rg-arrow-g" />

  <!-- Embed query -->
  <rect x="120" y="146" width="72" height="46" class="rg-box rg-query" />
  <text x="156" y="166" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Embed</text>
  <text x="156" y="182" text-anchor="middle" class="rg-text rg-small">query -> vector</text>

  <line x1="194" y1="169" x2="210" y2="169" class="rg-arrow-g" />

  <!-- Retrieve -->
  <rect x="212" y="146" width="76" height="46" class="rg-box rg-query" />
  <text x="250" y="166" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Retrieve</text>
  <text x="250" y="182" text-anchor="middle" class="rg-text rg-small">top-k similar</text>

  <!-- Arrow from Vector DB down to Retrieve -->
  <path d="M358,112 L358,130 L270,130 L270,144" class="rg-arrow" />

  <line x1="290" y1="169" x2="306" y2="169" class="rg-arrow-g" />

  <!-- Rerank -->
  <rect x="308" y="146" width="76" height="46" class="rg-box rg-query" />
  <text x="346" y="166" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Rerank</text>
  <text x="346" y="182" text-anchor="middle" class="rg-text rg-small">cross-encoder</text>

  <line x1="386" y1="169" x2="402" y2="169" class="rg-arrow-g" />

  <!-- LLM -->
  <rect x="404" y="140" width="90" height="58" class="rg-box rg-llm" />
  <text x="449" y="162" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">LLM</text>
  <text x="449" y="178" text-anchor="middle" class="rg-text rg-small">context + query</text>
  <text x="449" y="190" text-anchor="middle" class="rg-text rg-small">-> generate</text>

  <line x1="496" y1="169" x2="512" y2="169" class="rg-arrow-g" />

  <!-- Answer -->
  <rect x="514" y="146" width="76" height="46" class="rg-box rg-query" />
  <text x="552" y="166" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Answer</text>
  <text x="552" y="182" text-anchor="middle" class="rg-text rg-small">grounded text</text>

  <!-- Bottom: key decisions -->
  <rect x="20" y="212" width="560" height="58" style="fill:var(--accent,#4361ee);fill-opacity:0.04;stroke:var(--border-color,#e0e0e8);stroke-width:1;rx:6;" />
  <text x="300" y="230" text-anchor="middle" class="rg-text rg-label" style="font-weight:600;">Key Design Decisions</text>
  <text x="165" y="248" text-anchor="middle" class="rg-text rg-small">Chunk size: 256-1024 tokens</text>
  <text x="165" y="262" text-anchor="middle" class="rg-text rg-small">(overlap 10-20%)</text>
  <text x="370" y="248" text-anchor="middle" class="rg-text rg-small">Embedding model: BGE / GTE / E5</text>
  <text x="370" y="262" text-anchor="middle" class="rg-text rg-small">Hybrid search: dense + BM25 sparse</text>
</svg>`

};

/**
 * Finds placeholder divs with class "diagram" and data-diagram attribute,
 * then replaces their innerHTML with the corresponding SVG from DIAGRAMS.
 */
function injectDiagrams() {
  const placeholders = document.querySelectorAll('.diagram[data-diagram]');
  placeholders.forEach(function(el) {
    const key = el.getAttribute('data-diagram');
    if (DIAGRAMS[key]) {
      el.innerHTML = DIAGRAMS[key];
    }
  });
}

document.addEventListener('DOMContentLoaded', injectDiagrams);
