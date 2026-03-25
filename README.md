# AI Engineer Learning Toolkit

An interactive, textbook-style learning website for AI engineers, audio AI engineers, and researchers. Covers career development, skill building, and interview preparation with 200K+ words of deeply technical content.

**Live site:** [yingxuh.github.io/learning-toolkit](https://yingxuh.github.io/learning-toolkit/)

## Features

### Content
- **15 chapters** across 6 parts, **105 sections**, **200K+ words**
- Topics: Audio LLMs, Speech-to-Speech, TTS, Speculative Decoding, vLLM Serving, RL Training, ML Engineering, Agent Development, System Design, Transformers, General LLMs, Quantization, RAG, Data Structures & Algorithms, Interview Prep
- 70+ interview questions with graded answers (60/80/95-point versions)
- 240+ code blocks with working examples
- 20+ production war stories with real debugging details and methodology
- Paper citations with arXiv IDs throughout

### Interactive Features
- **Full-text search** (Ctrl+K) with contextual results and snippet highlighting
- **Text highlighting** (4 colors: yellow, green, blue, pink) with persistent storage
- **Comments** on highlighted text with edit/delete
- **Highlight + Ask AI** - select text, click the AI button, get contextual explanations
- **AI chat assistant** with persistent memory across sessions
- **Dark/light theme** toggle
- **Study plan** with 6-week reading goals and progress tracking
- **Chapter navigation** (prev/next buttons) and floating chapter indicator
- **Keyboard shortcuts**: Ctrl+K (search), Ctrl+/ (chat), Escape (close)

### Bilingual Support
- Full Chinese (simplified) UI: title, TOC, study plan, changelog, search
- Chapter 1 has complete Chinese content translation
- All chapter and section titles translated
- Language toggle button (EN/中) in navigation bar
- 89 bilingual unit tests passing

### Authentication & Backend (Firebase)
- Google OAuth login with email allowlist
- Per-user data persistence (Firestore): highlights, comments, chat history, reading progress
- Gemini AI chat via Firebase Cloud Function (secure API key handling)
- Dual-write: localStorage (offline-first) + Firestore sync when authenticated
- Data migration on first login

### Mobile
- Responsive design with 44px minimum tap targets
- Touch-based text selection and highlighting (touchend + selectionchange events)
- Collapsible sidebar with overlay dismiss and close button
- Table scroll affordance with gradient hints
- Mobile search bar

## Architecture

```
index.html              - Main entry point (single-page app)
css/
  style.css             - Core styles, responsive design, themes
  highlights.css        - Highlight toolbar and annotation styles
  chat.css              - AI chat panel styles
js/
  i18n.js               - Internationalization module (EN/ZH)
  content.js            - Base textbook structure (chapter stubs)
  content-zh.js         - Chinese content translations
  content-ch1-2.js      - Audio LLM + Speech-to-Speech (expanded)
  content-ch3-4.js      - TTS + Speculative Decoding (expanded)
  content-ch5-6.js      - vLLM + RL Training (expanded)
  content-ch7-8.js      - ML Engineering + Agents (expanded)
  content-ch11-12.js    - Transformers + General LLMs (expanded)
  content-ch13-14.js    - Quantization + RAG (expanded)
  content-ch15.js       - DS&A for AI (expanded)
  content-loader.js     - Merges expanded content into TEXTBOOK
  app.js                - Main application logic
  search.js             - Full-text search with relevance ranking
  highlights.js         - Text highlighting with touch support
  comments.js           - Comment system on highlights
  chat.js               - AI chat with Gemini/local fallback
  firebase-config.js    - Firebase client configuration
  auth.js               - Google OAuth authentication module
  user-data.js          - Dual-write localStorage + Firestore
functions/
  index.js              - Firebase Cloud Function (Gemini API proxy)
  package.json          - Cloud Function dependencies
tests/
  index.html            - Test suite runner
  test_bilingual.html   - I18n and bilingual tests (129 tests)
  test_highlights.html  - Highlight module tests (41 tests)
  test_chat.html        - Chat module tests (41 tests)
  test_comments.html    - Comment module tests (36 tests)
  test_search.html      - Search module tests (16 tests)
  test_auth.html        - Auth module tests (22 tests)
  test_userdata.html    - User data module tests (29 tests)
  test_content_loader.html - Content loader tests (82 tests)
```

## Test Suite

**396 tests across 8 test files**, all passing.

Run tests by opening `tests/index.html` in a browser, or via Playwright:

```bash
pip install playwright && python -m playwright install chromium

python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    for test in ['test_bilingual', 'test_highlights', 'test_chat', 'test_comments',
                 'test_search', 'test_auth', 'test_userdata', 'test_content_loader']:
        page = browser.new_page()
        page.goto(f'file:///path/to/tests/{test}.html')
        page.wait_for_timeout(8000)
        print(f'{test}: {page.inner_text(\"#summary\")}')
        page.close()
    browser.close()
"
```

| Test File | Tests | Coverage |
|-----------|:-----:|---------|
| test_bilingual.html | 129 | I18n module, Chinese translations, language toggle, content integrity |
| test_content_loader.html | 82 | Content merge, chapter expansion, section IDs, quality checks |
| test_highlights.html | 41 | Storage, apply/remove highlights, colors, DOM manipulation |
| test_chat.html | 41 | Context management, memory, history, API key, DOM elements |
| test_comments.html | 36 | CRUD operations, modal/tooltip, HTML escaping |
| test_userdata.html | 29 | Save/load, managed keys, complex objects, malformed JSON |
| test_auth.html | 22 | Module exports, email allowlist, initial state, DOM |
| test_search.html | 16 | Index building, query matching, case insensitivity, result structure |

## Firebase Setup (for authentication and AI chat)

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Enable **Authentication** > **Google** provider
3. Enable **Cloud Firestore**
4. Add a Web app and copy the config to `js/firebase-config.js`
5. Deploy the Cloud Function:

```bash
cd functions
npm install
firebase login
firebase deploy --only functions
firebase functions:secrets:set GEMINI_API_KEY  # paste your Gemini API key
```

6. Update `GEMINI_FUNCTION_URL` in `js/firebase-config.js` with the deployed function URL
7. Add allowed user emails to the `ALLOWED_USERS` array

Without Firebase setup, the site works fully in offline mode with localStorage persistence and local AI fallback responses.

## Content Structure

| Part | Chapters | Focus |
|------|----------|-------|
| 1. Foundations of Audio AI | Audio LLMs, Speech-to-Speech, TTS | Audio/speech AI research and systems |
| 2. LLM Inference & Optimization | Speculative Decoding, vLLM Serving | Production LLM serving |
| 3. ML Training & Infrastructure | RL Training (RLHF/RLVR), ML Engineering | Training pipelines and best practices |
| 4. Software Engineering for AI | Agent Development, System Design | Building AI applications |
| 5. Transformer & LLM Fundamentals | Transformer Deep Dive, General LLMs | Core architecture knowledge |
| 6. Advanced Topics | Quantization, RAG, DS&A for AI | Specialized topics and interview prep |

## Development

No build step required. The site is vanilla HTML/CSS/JS.

```bash
# Serve locally
python -m http.server 8000
# Open http://localhost:8000

# Deploy (auto via GitHub Actions on push to main)
git push origin main
```

## License

Personal learning resource. Content derived from research papers, technical documentation, and production experience.
