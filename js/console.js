// === User Console Module ===
(function() {
    'use strict';

    document.addEventListener('DOMContentLoaded', () => {
        setupConsole();
    });

    function setupConsole() {
        const toggleBtn = document.getElementById('console-toggle');
        const closeBtn = document.getElementById('console-close');
        const panel = document.getElementById('console-panel');

        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const chatPanel = document.getElementById('chat-panel');
                if (!chatPanel.classList.contains('hidden')) chatPanel.classList.add('hidden');
                panel.classList.toggle('hidden');
                if (!panel.classList.contains('hidden')) {
                    panel.classList.add('flex');
                    renderConsole();
                } else {
                    panel.classList.remove('flex');
                }
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                panel.classList.add('hidden');
                panel.classList.remove('flex');
            });
        }
    }

    function renderConsole() {
        const container = document.getElementById('console-content');
        if (!container) return;

        const user = window.AuthModule ? window.AuthModule.getUser() : null;
        const progress = JSON.parse(localStorage.getItem('lt_reading_progress') || '{}');
        const highlights = JSON.parse(localStorage.getItem('lt_highlights') || '{}');
        const comments = JSON.parse(localStorage.getItem('lt_comments') || '{}');
        const chatHistory = JSON.parse(localStorage.getItem('lt_chat_history') || '[]');
        const chatMemory = JSON.parse(localStorage.getItem('lt_chat_memory') || '{}');

        const totalChapters = 15;
        const readCount = Object.keys(progress).length;
        const highlightCount = Object.keys(highlights).length;
        const commentCount = Object.keys(comments).length;
        const chatCount = chatHistory.length;

        let html = '';

        // User profile
        html += `<div class="pb-4 border-b border-gray-200 dark:border-gray-800">
            <div class="flex items-center gap-3 mb-3">
                ${user ? `<img src="${user.photoURL || ''}" class="w-10 h-10 rounded-full ring-2 ring-gray-200 dark:ring-gray-700">` : '<div class="w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-400 text-lg">?</div>'}
                <div>
                    <div class="font-semibold text-sm">${user ? user.displayName || user.email : 'Not signed in'}</div>
                    <div class="text-xs text-gray-400">${user ? user.email : 'Sign in to sync across devices'}</div>
                </div>
            </div>
        </div>`;

        // Stats cards
        html += `<div>
            <h4 class="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-3">Overview</h4>
            <div class="grid grid-cols-2 gap-3">
                <div class="p-3 bg-accent-50 dark:bg-accent-900/20 rounded-lg border border-accent-200 dark:border-accent-800">
                    <div class="text-2xl font-bold text-accent-600">${Math.round(readCount / totalChapters * 100)}%</div>
                    <div class="text-xs text-gray-500">${readCount}/${totalChapters} chapters read</div>
                </div>
                <div class="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <div class="text-2xl font-bold text-yellow-600">${highlightCount}</div>
                    <div class="text-xs text-gray-500">Highlights</div>
                </div>
                <div class="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                    <div class="text-2xl font-bold text-green-600">${commentCount}</div>
                    <div class="text-xs text-gray-500">Comments</div>
                </div>
                <div class="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                    <div class="text-2xl font-bold text-purple-600">${chatCount}</div>
                    <div class="text-xs text-gray-500">AI Messages</div>
                </div>
            </div>
        </div>`;

        // Reading progress
        html += `<div>
            <h4 class="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-3">Reading Progress</h4>
            <div class="space-y-1.5">`;
        if (typeof TEXTBOOK !== 'undefined') {
            TEXTBOOK.parts.forEach(part => {
                part.chapters.forEach(ch => {
                    const isRead = !!progress[ch.id];
                    const readDate = isRead ? new Date(progress[ch.id]).toLocaleDateString() : '';
                    html += `<div class="flex items-center gap-2 py-1.5 px-2 rounded-md ${isRead ? 'bg-green-50 dark:bg-green-900/10' : 'bg-gray-50 dark:bg-surface-800'}">
                        <span class="w-5 h-5 flex items-center justify-center rounded-full text-xs ${isRead ? 'bg-green-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-400'}">${isRead ? '&#10003;' : ''}</span>
                        <span class="text-xs flex-1 ${isRead ? 'text-gray-700 dark:text-gray-300' : 'text-gray-400'}">${ch.title}</span>
                        ${readDate ? `<span class="text-[10px] text-gray-400">${readDate}</span>` : ''}
                    </div>`;
                });
            });
        }
        html += `</div></div>`;

        // Recent highlights
        if (highlightCount > 0) {
            html += `<div>
                <h4 class="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-3">Recent Highlights</h4>
                <div class="space-y-2">`;
            const hlEntries = Object.entries(highlights).sort((a, b) => (b[1].timestamp || 0) - (a[1].timestamp || 0)).slice(0, 10);
            hlEntries.forEach(([id, hl]) => {
                const colorClass = { yellow: 'bg-yellow-100 border-yellow-300', green: 'bg-green-100 border-green-300', blue: 'bg-blue-100 border-blue-300', pink: 'bg-pink-100 border-pink-300' }[hl.color] || 'bg-gray-100 border-gray-300';
                html += `<div class="p-2 rounded-md border ${colorClass} text-xs">
                    <div class="text-gray-700 dark:text-gray-300 line-clamp-2">"${(hl.text || '').substring(0, 120)}${(hl.text || '').length > 120 ? '...' : ''}"</div>
                    ${comments[id] ? `<div class="mt-1 text-gray-500 italic">Note: ${comments[id].text.substring(0, 80)}</div>` : ''}
                </div>`;
            });
            html += `</div></div>`;
        }

        // AI Chat memory
        if (Object.keys(chatMemory).length > 0) {
            html += `<div>
                <h4 class="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-3">AI Memory</h4>
                <div class="space-y-1 text-xs">`;
            Object.entries(chatMemory).forEach(([key, val]) => {
                html += `<div class="flex justify-between py-1 border-b border-gray-100 dark:border-gray-800">
                    <span class="text-gray-500">${key}</span>
                    <span class="text-gray-700 dark:text-gray-300 text-right max-w-[200px] truncate">${val}</span>
                </div>`;
            });
            html += `</div></div>`;
        }

        // Recent chat messages
        if (chatCount > 0) {
            html += `<div>
                <h4 class="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-3">Recent AI Chats</h4>
                <div class="space-y-2 max-h-60 overflow-y-auto">`;
            chatHistory.slice(-10).reverse().forEach(msg => {
                const isUser = msg.role === 'user';
                const time = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : '';
                html += `<div class="p-2 rounded-md ${isUser ? 'bg-accent-50 dark:bg-accent-900/20' : 'bg-gray-50 dark:bg-surface-800'} text-xs">
                    <div class="flex justify-between mb-1">
                        <span class="font-medium ${isUser ? 'text-accent-600' : 'text-gray-500'}">${isUser ? 'You' : 'AI'}</span>
                        <span class="text-[10px] text-gray-400">${time}</span>
                    </div>
                    <div class="text-gray-600 dark:text-gray-400 line-clamp-3">${(msg.content || '').substring(0, 200)}</div>
                </div>`;
            });
            html += `</div></div>`;
        }

        // Data management
        html += `<div class="pt-4 border-t border-gray-200 dark:border-gray-800">
            <h4 class="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-3">Data Management</h4>
            <div class="space-y-2">
                <button onclick="ConsoleModule.syncNow()" class="w-full h-9 text-xs font-medium rounded-lg bg-accent-50 dark:bg-accent-900/20 text-accent-600 hover:bg-accent-100 dark:hover:bg-accent-900/40 border border-accent-200 dark:border-accent-800 transition">
                    Sync to Cloud
                </button>
                <button onclick="ConsoleModule.exportData()" class="w-full h-9 text-xs font-medium rounded-lg bg-gray-100 dark:bg-surface-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-700 transition">
                    Export All Data (JSON)
                </button>
                <button onclick="ConsoleModule.resetHighlights()" class="w-full h-9 text-xs font-medium rounded-lg bg-yellow-50 text-yellow-700 hover:bg-yellow-100 border border-yellow-200 transition">
                    Reset Highlights & Comments
                </button>
                <button onclick="ConsoleModule.resetChat()" class="w-full h-9 text-xs font-medium rounded-lg bg-purple-50 text-purple-700 hover:bg-purple-100 border border-purple-200 transition">
                    Reset AI Chat History & Memory
                </button>
                <button onclick="ConsoleModule.resetProgress()" class="w-full h-9 text-xs font-medium rounded-lg bg-orange-50 text-orange-700 hover:bg-orange-100 border border-orange-200 transition">
                    Reset Reading Progress
                </button>
                <button onclick="ConsoleModule.resetAll()" class="w-full h-9 text-xs font-medium rounded-lg bg-red-50 text-red-700 hover:bg-red-100 border border-red-200 transition">
                    Reset ALL Data
                </button>
            </div>
        </div>`;

        container.innerHTML = html;
    }

    // Data management functions
    function syncNow() {
        const user = window.AuthModule ? window.AuthModule.getUser() : null;
        if (!user) { alert('Please sign in first to sync data.'); return; }
        if (window.UserDataModule) {
            window.UserDataModule.syncFromFirestore(user.uid).then(() => {
                alert('Data synced successfully!');
                renderConsole();
            });
        }
    }

    function exportData() {
        const data = {};
        ['lt_highlights', 'lt_comments', 'lt_chat_memory', 'lt_chat_history', 'lt_reading_progress', 'lt_content_updates', 'theme', 'lt_language'].forEach(key => {
            const val = localStorage.getItem(key);
            if (val) { try { data[key] = JSON.parse(val); } catch { data[key] = val; } }
        });
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'learning-toolkit-data.json'; a.click();
        URL.revokeObjectURL(url);
    }

    function resetHighlights() {
        if (!confirm('Reset all highlights and comments? This cannot be undone.')) return;
        localStorage.removeItem('lt_highlights');
        localStorage.removeItem('lt_comments');
        alert('Highlights and comments cleared.');
        renderConsole();
    }

    function resetChat() {
        if (!confirm('Reset AI chat history and memory? This cannot be undone.')) return;
        localStorage.removeItem('lt_chat_history');
        localStorage.removeItem('lt_chat_memory');
        alert('Chat history and memory cleared.');
        renderConsole();
    }

    function resetProgress() {
        if (!confirm('Reset reading progress? This cannot be undone.')) return;
        localStorage.removeItem('lt_reading_progress');
        alert('Reading progress cleared.');
        renderConsole();
    }

    function resetAll() {
        if (!confirm('Reset ALL data (progress, highlights, comments, chat, preferences)? This cannot be undone.')) return;
        ['lt_highlights', 'lt_comments', 'lt_chat_memory', 'lt_chat_history', 'lt_reading_progress', 'lt_content_updates', 'lt_gemini_key'].forEach(key => {
            localStorage.removeItem(key);
        });
        alert('All data cleared. Refreshing...');
        location.reload();
    }

    window.ConsoleModule = { renderConsole, syncNow, exportData, resetHighlights, resetChat, resetProgress, resetAll };
})();
