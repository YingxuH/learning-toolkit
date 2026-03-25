# Learnings Log

## [LRN-20260326-001] CRITICAL: API Key Leaked in Public Git Repo
- **Category**: Security
- **Priority**: P0
- **Status**: promoted -> secret-hygiene skill
- **Details**: Gemini API key `AIzaSy...` was hardcoded as a fallback value in `functions/index.js` and committed to a public GitHub repo. Google's secret scanning detected and revoked the key.
- **Root Cause**: The Cloud Function code used `process.env.GEMINI_API_KEY || "actual_key_here"` as a "convenience" fallback. This pattern guarantees the key ends up in git history.
- **Fix**: Removed hardcoded key, rewrote git history with `git-filter-repo`, force-pushed.
- **Generalized Principle**: NEVER use real secrets as fallback values in code. Use `process.env.VAR || ""` with a runtime error if missing. Secrets must only live in environment variables, secret managers, or `.env` files that are gitignored.
- **Prevention**: Pre-commit hook or CI check that scans for API key patterns before allowing commits.

## [LRN-20260326-002] const globals not accessible via window.X
- **Category**: JavaScript
- **Priority**: P2
- **Status**: resolved
- **Details**: `const TEXTBOOK = {...}` at top-level of a script is a global constant accessible as `TEXTBOOK` but NOT via `window.TEXTBOOK`. The content-loader used `window.CONTENT_CH1_2` checks which silently returned undefined. Spent 5 iterations debugging this.
- **Root Cause**: ES6 `const`/`let` at global scope create global bindings but do NOT add properties to the `window` object (unlike `var`).
- **Fix**: Changed all checks from `if (window.X)` to `if (typeof X !== 'undefined')`.
- **Generalized Principle**: When checking for existence of script-defined globals in vanilla JS, always use `typeof X !== 'undefined'`, never `window.X`, unless the variable was declared with `var` or explicitly assigned to `window`.

## [LRN-20260326-003] Mobile text selection requires touchend + selectionchange
- **Category**: Mobile UX
- **Priority**: P2
- **Status**: resolved
- **Details**: Highlight toolbar never appeared on mobile because only `mouseup` was listened for. On mobile Safari/Chrome, text selection fires `selectionchange` but NOT `mouseup`.
- **Fix**: Added `touchend` listener (coords from `e.changedTouches[0]`) and `selectionchange` listener (coords from `range.getBoundingClientRect()`). Used longer setTimeout (300ms vs 10ms) for touch to let browser finalize selection.
- **Generalized Principle**: Any text-selection-based UI must handle 3 events: `mouseup` (desktop), `touchend` (mobile), and `selectionchange` (cross-platform fallback). Touch coordinates come from `changedTouches[0]`, not the event itself.

## [LRN-20260326-004] GitHub Pages + Firebase: static files + backend pattern
- **Category**: Architecture
- **Priority**: P3
- **Status**: resolved
- **Details**: For a small-user app (~10), GitHub Pages serves static files while Firebase provides auth, database, and serverless functions. Firebase SDK loads via CDN script tags (no build step). The dual-write pattern (localStorage offline-first + Firestore sync) ensures the app works without internet.
- **Generalized Principle**: For small-team tools, the GitHub Pages + Firebase free tier pattern gives you auth + database + serverless at zero cost without needing a build system.

## [LRN-20260326-005] Playwright screenshots miss inner-scroll containers
- **Category**: Testing
- **Priority**: P3
- **Status**: resolved
- **Details**: The website uses `overflow-y: auto` on `#content-area` rather than body scroll. Playwright's page.scroll() and full-page screenshots only capture the viewport, not the scrolled content within the container. All "scroll to 50%" screenshots looked identical.
- **Generalized Principle**: When testing sites with CSS overflow scroll containers, use element-level scrolling (`element.scrollTop = X`) not window-level scrolling. For screenshots, scroll the specific container element.
