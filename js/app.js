// === Main Application ===
(function() {
    'use strict';

    // State
    const state = {
        currentChapter: null,
        sidebarOpen: true,
        chatOpen: false,
        theme: localStorage.getItem('theme') || 'light'
    };

    // Initialize
    document.addEventListener('DOMContentLoaded', init);

    function init() {
        applyTheme(state.theme);
        createOverlay();
        renderTOC();
        renderContent();
        setupNavigation();
        setupKeyboardShortcuts();
        setupReadingProgress();

        // Collapse sidebar on mobile by default
        if (window.innerWidth < 768) {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.add('collapsed');
            state.sidebarOpen = false;
        }

        // Navigate to hash or first chapter
        const hash = window.location.hash.slice(1);
        if (hash) {
            scrollToSection(hash);
        }
    }

    function createOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'sidebar-overlay';
        document.body.appendChild(overlay);
        overlay.addEventListener('click', () => {
            if (state.sidebarOpen && window.innerWidth < 768) {
                toggleSidebar();
            }
        });
    }

    // === Theme ===
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        state.theme = theme;
        localStorage.setItem('theme', theme);
        const btn = document.getElementById('theme-toggle');
        btn.innerHTML = theme === 'dark' ? '&#9788;' : '&#9789;';
    }

    // === Table of Contents ===
    function renderTOC() {
        const toc = document.getElementById('toc');
        let html = '';

        TEXTBOOK.parts.forEach((part, pi) => {
            html += `<div class="toc-part">`;
            html += `<div class="toc-part-title">Part ${pi + 1}: ${part.title}</div>`;
            part.chapters.forEach(chapter => {
                html += `<a class="toc-item" data-chapter="${chapter.id}" href="#${chapter.id}">${chapter.title}</a>`;
                chapter.sections.forEach(section => {
                    html += `<a class="toc-item toc-section" data-section="${section.id}" href="#${section.id}">${section.title}</a>`;
                });
            });
            html += `</div>`;
        });

        toc.innerHTML = html;

        // Click handlers
        toc.querySelectorAll('.toc-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const id = item.getAttribute('data-chapter') || item.getAttribute('data-section');
                scrollToSection(id);
                // Close sidebar on mobile
                if (window.innerWidth < 768) {
                    toggleSidebar();
                }
            });
        });
    }

    // === Content Rendering ===
    function renderContent() {
        const container = document.getElementById('textbook-content');
        let html = '<div id="reading-progress"></div>';
        let chapterNum = 0;

        TEXTBOOK.parts.forEach((part) => {
            part.chapters.forEach(chapter => {
                chapterNum++;
                html += `<div class="chapter" id="${chapter.id}">`;
                html += `<div class="chapter-number">Chapter ${chapterNum}</div>`;
                html += `<h2>${chapter.title}</h2>`;

                chapter.sections.forEach(section => {
                    html += `<section id="${section.id}">`;
                    html += `<h3>${section.title}</h3>`;
                    html += section.content;
                    html += `</section>`;
                });

                html += `</div>`;
            });
        });

        container.innerHTML = html;

        // Restore highlights
        if (window.HighlightManager) {
            window.HighlightManager.restoreHighlights();
        }
    }

    // === Navigation ===
    function setupNavigation() {
        // Sidebar toggle
        document.getElementById('sidebar-toggle').addEventListener('click', toggleSidebar);

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => {
            applyTheme(state.theme === 'dark' ? 'light' : 'dark');
        });

        // Chat toggle
        document.getElementById('chat-toggle').addEventListener('click', toggleChat);
        document.getElementById('chat-close').addEventListener('click', toggleChat);

        // Scroll spy for TOC
        const contentArea = document.getElementById('content-area');
        contentArea.addEventListener('scroll', () => {
            updateReadingProgress();
            updateActiveTOC();
        });
    }

    function toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('collapsed');
        state.sidebarOpen = !state.sidebarOpen;

        // Handle overlay for mobile
        const overlay = document.getElementById('sidebar-overlay');
        if (overlay) {
            overlay.classList.toggle('visible', state.sidebarOpen && window.innerWidth < 768);
        }
    }

    function toggleChat() {
        const panel = document.getElementById('chat-panel');
        panel.classList.toggle('hidden');
        state.chatOpen = !state.chatOpen;
        if (state.chatOpen) {
            document.getElementById('chat-input').focus();
        }
    }

    function scrollToSection(id) {
        const el = document.getElementById(id);
        if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            window.location.hash = id;
        }
    }

    function updateActiveTOC() {
        const contentArea = document.getElementById('content-area');
        const scrollTop = contentArea.scrollTop;
        const sections = contentArea.querySelectorAll('.chapter, section[id]');
        let activeId = null;

        sections.forEach(section => {
            if (section.offsetTop - 100 <= scrollTop) {
                activeId = section.id;
            }
        });

        document.querySelectorAll('.toc-item').forEach(item => {
            const id = item.getAttribute('data-chapter') || item.getAttribute('data-section');
            item.classList.toggle('active', id === activeId);
        });
    }

    // === Reading Progress ===
    function setupReadingProgress() {
        const bar = document.createElement('div');
        bar.id = 'reading-progress';
        document.body.appendChild(bar);
    }

    function updateReadingProgress() {
        const content = document.getElementById('content-area');
        const scrollTop = content.scrollTop;
        const scrollHeight = content.scrollHeight - content.clientHeight;
        const progress = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
        const bar = document.getElementById('reading-progress');
        if (bar) bar.style.width = progress + '%';
    }

    // === Keyboard Shortcuts ===
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+K: Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                document.getElementById('global-search').focus();
            }
            // Ctrl+/: Toggle chat
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                toggleChat();
            }
            // Escape: Close modals/search
            if (e.key === 'Escape') {
                document.getElementById('search-results').classList.add('hidden');
                document.getElementById('comment-modal').classList.add('hidden');
                document.getElementById('global-search').blur();
            }
        });
    }

    // Export for other modules
    window.App = {
        scrollToSection,
        toggleChat,
        renderContent,
        getState: () => state
    };
})();
