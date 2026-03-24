// === Comments Module ===
(function() {
    'use strict';

    const STORAGE_KEY = 'lt_comments';
    const modal = document.getElementById('comment-modal');
    let currentHighlightId = null;
    let activeTooltip = null;

    document.addEventListener('DOMContentLoaded', () => {
        setupComments();
    });

    function setupComments() {
        document.getElementById('comment-save').addEventListener('click', saveComment);
        document.getElementById('comment-cancel').addEventListener('click', () => {
            modal.classList.add('hidden');
            currentHighlightId = null;
        });

        // Close tooltip on outside click
        document.addEventListener('click', (e) => {
            if (activeTooltip && !e.target.closest('.comment-tooltip') && !e.target.closest('.user-highlight')) {
                removeTooltip();
            }
        });
    }

    function showCommentModal(highlightId) {
        currentHighlightId = highlightId;
        const existing = getComments()[highlightId];
        document.getElementById('comment-text').value = existing ? existing.text : '';
        modal.classList.remove('hidden');
        document.getElementById('comment-text').focus();
    }

    function saveComment() {
        const text = document.getElementById('comment-text').value.trim();
        if (!text || !currentHighlightId) return;

        const comments = getComments();
        comments[currentHighlightId] = {
            text: text,
            timestamp: Date.now()
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(comments));

        // Mark highlight as having comment
        const el = document.querySelector(`[data-hl-id="${currentHighlightId}"]`);
        if (el) el.classList.add('has-comment');

        modal.classList.add('hidden');
        currentHighlightId = null;
    }

    function showCommentTooltip(highlightId, x, y) {
        removeTooltip();

        const comments = getComments();
        const comment = comments[highlightId];
        const highlight = document.querySelector(`[data-hl-id="${highlightId}"]`);

        const tooltip = document.createElement('div');
        tooltip.className = 'comment-tooltip';

        if (comment) {
            const date = new Date(comment.timestamp).toLocaleDateString();
            tooltip.innerHTML = `
                <div class="comment-header">
                    <span class="comment-date">${date}</span>
                </div>
                <div>${escapeHtml(comment.text)}</div>
                <div class="comment-actions">
                    <button class="edit-comment" data-id="${highlightId}">Edit</button>
                    <button class="delete-comment" data-id="${highlightId}">Remove</button>
                </div>
            `;
        } else {
            tooltip.innerHTML = `
                <div>Highlighted text</div>
                <div class="comment-actions">
                    <button class="edit-comment" data-id="${highlightId}">Add Comment</button>
                    <button class="delete-comment" data-id="${highlightId}">Remove Highlight</button>
                </div>
            `;
        }

        // Position
        tooltip.style.left = Math.min(x, window.innerWidth - 320) + 'px';
        tooltip.style.top = (y + 10) + 'px';

        document.body.appendChild(tooltip);
        activeTooltip = tooltip;

        // Actions
        tooltip.querySelector('.edit-comment').addEventListener('click', () => {
            removeTooltip();
            showCommentModal(highlightId);
        });

        tooltip.querySelector('.delete-comment').addEventListener('click', () => {
            removeTooltip();
            window.HighlightManager.removeHighlight(highlightId);
        });
    }

    function removeTooltip() {
        if (activeTooltip) {
            activeTooltip.remove();
            activeTooltip = null;
        }
    }

    function getComments() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
        } catch {
            return {};
        }
    }

    function removeComment(highlightId) {
        const comments = getComments();
        delete comments[highlightId];
        localStorage.setItem(STORAGE_KEY, JSON.stringify(comments));
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Export
    window.CommentModule = {
        showCommentModal,
        showCommentTooltip,
        removeComment,
        getComments
    };
})();
