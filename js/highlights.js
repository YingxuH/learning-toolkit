// === Highlight Module ===
(function() {
    'use strict';

    const STORAGE_KEY = 'lt_highlights';
    const toolbar = document.getElementById('highlight-toolbar');
    let currentSelection = null;
    let currentRange = null;

    document.addEventListener('DOMContentLoaded', () => {
        setupHighlighting();
    });

    function setupHighlighting() {
        const contentArea = document.getElementById('content-area');

        // Show toolbar on text selection
        contentArea.addEventListener('mouseup', (e) => {
            setTimeout(() => {
                const selection = window.getSelection();
                if (selection.isCollapsed || selection.toString().trim().length === 0) {
                    hideToolbar();
                    return;
                }

                // Only highlight within textbook content
                const range = selection.getRangeAt(0);
                if (!document.getElementById('textbook-content').contains(range.commonAncestorContainer)) {
                    return;
                }

                currentSelection = selection.toString();
                currentRange = range.cloneRange();
                showToolbar(e.clientX, e.clientY);
            }, 10);
        });

        // Hide toolbar on click elsewhere
        document.addEventListener('mousedown', (e) => {
            if (!e.target.closest('#highlight-toolbar') && !e.target.closest('.user-highlight')) {
                hideToolbar();
            }
        });

        // Color buttons
        toolbar.querySelectorAll('[data-color]').forEach(btn => {
            btn.addEventListener('click', () => {
                if (currentRange) {
                    applyHighlight(currentRange, btn.getAttribute('data-color'));
                    hideToolbar();
                }
            });
        });

        // Ask AI button
        document.getElementById('hl-ask-ai').addEventListener('click', () => {
            if (currentSelection) {
                // Open chat with context
                const chatPanel = document.getElementById('chat-panel');
                if (chatPanel.classList.contains('hidden')) {
                    App.toggleChat();
                }
                window.ChatModule.setContext(currentSelection);
                applyHighlight(currentRange, 'blue');
                hideToolbar();
            }
        });

        // Comment button
        document.getElementById('hl-comment').addEventListener('click', () => {
            if (currentSelection && currentRange) {
                const highlightId = applyHighlight(currentRange, 'yellow');
                hideToolbar();
                window.CommentModule.showCommentModal(highlightId);
            }
        });
    }

    function showToolbar(x, y) {
        toolbar.classList.remove('hidden');
        const contentArea = document.getElementById('content-area');
        const contentRect = contentArea.getBoundingClientRect();

        // Position relative to viewport
        toolbar.style.left = Math.min(x - 80, window.innerWidth - 220) + 'px';
        toolbar.style.top = (y - 50) + 'px';
    }

    function hideToolbar() {
        toolbar.classList.add('hidden');
    }

    function applyHighlight(range, color) {
        const id = 'hl-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5);

        try {
            const span = document.createElement('span');
            span.className = `user-highlight color-${color}`;
            span.setAttribute('data-hl-id', id);
            range.surroundContents(span);

            // Save
            saveHighlight(id, color, range);

            // Clear selection
            window.getSelection().removeAllRanges();

            // Add click handler for showing comment
            span.addEventListener('click', (e) => {
                e.stopPropagation();
                window.CommentModule.showCommentTooltip(id, e.clientX, e.clientY);
            });

            return id;
        } catch (e) {
            // surroundContents fails on partial element selections
            // Fallback: wrap text nodes individually
            const fragment = range.extractContents();
            const wrapper = document.createElement('span');
            wrapper.className = `user-highlight color-${color}`;
            wrapper.setAttribute('data-hl-id', id);
            wrapper.appendChild(fragment);
            range.insertNode(wrapper);

            saveHighlight(id, color, range);
            window.getSelection().removeAllRanges();

            wrapper.addEventListener('click', (e) => {
                e.stopPropagation();
                window.CommentModule.showCommentTooltip(id, e.clientX, e.clientY);
            });

            return id;
        }
    }

    function saveHighlight(id, color) {
        const highlights = getHighlights();
        const el = document.querySelector(`[data-hl-id="${id}"]`);
        if (el) {
            highlights[id] = {
                color: color,
                text: el.textContent,
                timestamp: Date.now(),
                // Store location info for restoration
                sectionId: el.closest('section[id]')?.id || el.closest('.chapter')?.id || ''
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(highlights));
        }
    }

    function getHighlights() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
        } catch {
            return {};
        }
    }

    function removeHighlight(id) {
        const el = document.querySelector(`[data-hl-id="${id}"]`);
        if (el) {
            const parent = el.parentNode;
            while (el.firstChild) {
                parent.insertBefore(el.firstChild, el);
            }
            parent.removeChild(el);
        }
        const highlights = getHighlights();
        delete highlights[id];
        localStorage.setItem(STORAGE_KEY, JSON.stringify(highlights));

        // Also remove comment
        window.CommentModule.removeComment(id);
    }

    function restoreHighlights() {
        // Highlights are stored but restoration from plain text is complex
        // For now, highlights persist via DOM until content is re-rendered
    }

    // Export
    window.HighlightManager = {
        getHighlights,
        removeHighlight,
        restoreHighlights,
        applyHighlight
    };
})();
