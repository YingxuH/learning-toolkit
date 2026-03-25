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

        // Shared handler for both mouse and touch selection
        function handleSelectionEnd(clientX, clientY) {
            setTimeout(() => {
                const selection = window.getSelection();
                if (!selection || selection.isCollapsed || selection.toString().trim().length === 0) {
                    hideToolbar();
                    return;
                }

                const range = selection.getRangeAt(0);
                if (!document.getElementById('textbook-content').contains(range.commonAncestorContainer)) {
                    return;
                }

                currentSelection = selection.toString();
                currentRange = range.cloneRange();
                showToolbar(clientX, clientY);
            }, 300); // Longer delay for touch to let selection finalize
        }

        // Desktop: mouseup
        contentArea.addEventListener('mouseup', (e) => {
            handleSelectionEnd(e.clientX, e.clientY);
        });

        // Mobile: touchend
        contentArea.addEventListener('touchend', (e) => {
            if (e.changedTouches && e.changedTouches.length > 0) {
                const touch = e.changedTouches[0];
                handleSelectionEnd(touch.clientX, touch.clientY);
            }
        });

        // Also listen for selectionchange (works better on some mobile browsers)
        let selectionChangeTimer = null;
        document.addEventListener('selectionchange', () => {
            clearTimeout(selectionChangeTimer);
            selectionChangeTimer = setTimeout(() => {
                const selection = window.getSelection();
                if (!selection || selection.isCollapsed || selection.toString().trim().length === 0) {
                    return;
                }
                const range = selection.getRangeAt(0);
                if (!document.getElementById('textbook-content').contains(range.commonAncestorContainer)) {
                    return;
                }
                currentSelection = selection.toString();
                currentRange = range.cloneRange();
                // Position toolbar near the selection
                const rect = range.getBoundingClientRect();
                if (rect.width > 0) {
                    showToolbar(rect.left + rect.width / 2, rect.top);
                }
            }, 500);
        });

        // Hide toolbar on click/tap elsewhere
        document.addEventListener('mousedown', (e) => {
            if (!e.target.closest('#highlight-toolbar') && !e.target.closest('.user-highlight')) {
                hideToolbar();
            }
        });
        document.addEventListener('touchstart', (e) => {
            if (!e.target.closest('#highlight-toolbar') && !e.target.closest('.user-highlight')) {
                hideToolbar();
            }
        }, { passive: true });

        // Color buttons
        toolbar.querySelectorAll('[data-color]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (currentRange) {
                    applyHighlight(currentRange, btn.getAttribute('data-color'));
                    hideToolbar();
                }
            });
        });

        // Ask AI button
        document.getElementById('hl-ask-ai').addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (currentSelection) {
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
        document.getElementById('hl-comment').addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (currentSelection && currentRange) {
                const highlightId = applyHighlight(currentRange, 'yellow');
                hideToolbar();
                window.CommentModule.showCommentModal(highlightId);
            }
        });
    }

    function showToolbar(x, y) {
        toolbar.classList.remove('hidden');
        // Use fixed positioning for consistent behavior on desktop and mobile
        const toolbarWidth = 220;
        const left = Math.max(8, Math.min(x - toolbarWidth / 2, window.innerWidth - toolbarWidth - 8));
        const top = Math.max(8, y - 56);
        toolbar.style.left = left + 'px';
        toolbar.style.top = top + 'px';
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

            saveHighlight(id, color);
            window.getSelection().removeAllRanges();

            span.addEventListener('click', (e) => {
                e.stopPropagation();
                window.CommentModule.showCommentTooltip(id, e.clientX, e.clientY);
            });

            return id;
        } catch (e) {
            const fragment = range.extractContents();
            const wrapper = document.createElement('span');
            wrapper.className = `user-highlight color-${color}`;
            wrapper.setAttribute('data-hl-id', id);
            wrapper.appendChild(fragment);
            range.insertNode(wrapper);

            saveHighlight(id, color);
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
        window.CommentModule.removeComment(id);
    }

    function restoreHighlights() {
        // Highlights persist via DOM until content is re-rendered
    }

    // Export (including test helpers)
    window.HighlightManager = {
        getHighlights,
        removeHighlight,
        restoreHighlights,
        applyHighlight,
        _test_saveHighlight: saveHighlight
    };
})();
