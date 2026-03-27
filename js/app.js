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
        // Merge expanded content into TEXTBOOK before rendering
        if (typeof _mergeExpandedContent === 'function') {
            _mergeExpandedContent();
        }

        applyTheme(state.theme);
        createOverlay();
        renderTOC();
        renderContent();
        setupNavigation();
        setupKeyboardShortcuts();
        setupReadingProgress();

        // Collapse sidebar on mobile/tablet by default
        if (window.innerWidth < 900) {
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
        // Tailwind dark mode
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        state.theme = theme;
        localStorage.setItem('theme', theme);
    }

    // === Table of Contents ===
    function renderTOC() {
        const toc = document.getElementById('toc');
        const lang = window.I18n ? window.I18n.getLang() : 'en';
        const t = window.I18n ? window.I18n.t : () => null;
        let html = '';

        TEXTBOOK.parts.forEach((part, pi) => {
            const partTitle = t('part_' + (pi + 1)) || part.title;
            html += `<div class="toc-part">`;
            html += `<div class="toc-part-title">Part ${pi + 1}: ${partTitle}</div>`;
            part.chapters.forEach(chapter => {
                const chTitle = t(chapter.id) || chapter.title;
                html += `<a class="toc-item" data-chapter="${chapter.id}" href="#${chapter.id}">${chTitle}</a>`;
                chapter.sections.forEach(section => {
                    const secTitle = t(section.id) || section.title;
                    html += `<a class="toc-item toc-section" data-section="${section.id}" href="#${section.id}">${secTitle}</a>`;
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

        // What's New & Reading Progress hero section
        html += renderHeroSection();

        // Flatten chapters for prev/next navigation
        const allChapters = [];
        TEXTBOOK.parts.forEach(part => {
            part.chapters.forEach(ch => allChapters.push(ch));
        });

        const lang = window.I18n ? window.I18n.getLang() : 'en';
        const t = window.I18n ? window.I18n.t : () => null;

        // Build ZH content lookup
        const zhContentMap = {};
        if (lang === 'zh' && typeof TEXTBOOK_ZH !== 'undefined') {
            TEXTBOOK_ZH.parts.forEach(part => {
                part.chapters.forEach(ch => {
                    ch.sections.forEach(sec => {
                        if (sec.content) zhContentMap[sec.id] = sec.content;
                    });
                });
            });
        }

        TEXTBOOK.parts.forEach((part) => {
            part.chapters.forEach(chapter => {
                chapterNum++;
                const chIdx = allChapters.indexOf(chapter);
                const chTitle = t(chapter.id) || chapter.title;
                const chLabel = lang === 'zh' ? `第 ${chapterNum} 章` : `Chapter ${chapterNum}`;
                html += `<div class="chapter" id="${chapter.id}">`;
                html += `<div class="chapter-number">${chLabel}</div>`;
                html += `<h2>${chTitle}</h2>`;

                chapter.sections.forEach(section => {
                    const secTitle = t(section.id) || section.title;
                    const secContent = zhContentMap[section.id] || section.content;
                    html += `<section id="${section.id}">`;
                    html += `<h3>${secTitle}</h3>`;
                    html += secContent;
                    html += `</section>`;
                });

                // Chapter navigation buttons
                html += `<div class="chapter-nav">`;
                if (chIdx > 0) {
                    const prevTitle = t(allChapters[chIdx-1].id) || allChapters[chIdx-1].title;
                    html += `<a class="nav-prev" href="#${allChapters[chIdx-1].id}">&larr; ${prevTitle}</a>`;
                } else {
                    html += `<span></span>`;
                }
                if (chIdx < allChapters.length - 1) {
                    const nextTitle = t(allChapters[chIdx+1].id) || allChapters[chIdx+1].title;
                    html += `<a class="nav-next" href="#${allChapters[chIdx+1].id}">${nextTitle} &rarr;</a>`;
                }
                html += `</div>`;

                html += `</div>`;
            });
        });

        container.innerHTML = html;

        // Wrap tables for mobile scroll affordance
        if (window.innerWidth < 768) {
            container.querySelectorAll('.chapter table').forEach(table => {
                if (!table.parentElement.classList.contains('table-wrapper')) {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'table-wrapper';
                    table.parentNode.insertBefore(wrapper, table);
                    wrapper.appendChild(table);
                    // Add scroll hint if table overflows
                    if (table.scrollWidth > wrapper.clientWidth) {
                        const hint = document.createElement('div');
                        hint.className = 'scroll-hint';
                        hint.textContent = '\u2190 swipe to see more \u2192';
                        wrapper.parentNode.insertBefore(hint, wrapper);
                    }
                    wrapper.addEventListener('scroll', () => {
                        const atEnd = wrapper.scrollLeft + wrapper.clientWidth >= wrapper.scrollWidth - 5;
                        wrapper.classList.toggle('scrolled-end', atEnd);
                    });
                }
            });
        }

        // Chapter nav click handlers
        container.querySelectorAll('.chapter-nav a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const id = link.getAttribute('href').slice(1);
                scrollToSection(id);
            });
        });

        // Restore highlights
        if (window.HighlightManager) {
            window.HighlightManager.restoreHighlights();
        }
    }

    // === Navigation ===
    function setupNavigation() {
        // Sidebar toggle
        document.getElementById('sidebar-toggle').addEventListener('click', toggleSidebar);
        document.getElementById('sidebar-close').addEventListener('click', () => {
            if (state.sidebarOpen) toggleSidebar();
        });

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => {
            applyTheme(state.theme === 'dark' ? 'light' : 'dark');
        });

        // Language toggle
        const langBtn = document.getElementById('lang-toggle');
        if (langBtn) {
            langBtn.addEventListener('click', () => {
                if (window.I18n) {
                    window.I18n.toggleLang();
                    langBtn.textContent = window.I18n.getLang() === 'zh' ? '中' : 'EN';
                }
            });
            // Set initial label
            if (window.I18n) {
                langBtn.textContent = window.I18n.getLang() === 'zh' ? '中' : 'EN';
            }
        }

        // Chat toggle
        document.getElementById('chat-toggle').addEventListener('click', toggleChat);
        document.getElementById('chat-close').addEventListener('click', toggleChat);

        // Scroll spy for TOC
        const contentArea = document.getElementById('content-area');
        contentArea.addEventListener('scroll', () => {
            updateReadingProgress();
            updateActiveTOC();
            updateChapterIndicator();
        });
    }

    // Update UI text for current language
    function updateUILang() {
        const lang = window.I18n ? window.I18n.getLang() : 'en';
        const t = window.I18n ? window.I18n.t : () => null;

        document.querySelector('.nav-title').textContent = t('nav_title') || 'AI Engineer Learning Toolkit';
        document.querySelector('.sidebar-header h2').textContent = t('contents') || 'Contents';

        const searchInput = document.getElementById('global-search');
        if (searchInput) searchInput.placeholder = t('search_placeholder') || 'Search content... (Ctrl+K)';

        const langBtn = document.getElementById('lang-toggle');
        if (langBtn) langBtn.textContent = lang === 'zh' ? '中' : 'EN';
    }

    // Floating chapter indicator
    function updateChapterIndicator() {
        let indicator = document.getElementById('chapter-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'chapter-indicator';
            document.body.appendChild(indicator);
        }

        const contentArea = document.getElementById('content-area');
        const scrollTop = contentArea.scrollTop;
        const chapters = contentArea.querySelectorAll('.chapter');
        let currentIdx = 0;

        chapters.forEach((ch, i) => {
            if (ch.offsetTop - 150 <= scrollTop) currentIdx = i;
        });

        const total = chapters.length;
        indicator.textContent = `${currentIdx + 1} / ${total}`;
        indicator.style.opacity = scrollTop > 200 ? '1' : '0';
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

        // Track reading progress
        if (activeId) {
            const chapter = document.getElementById(activeId);
            if (chapter && chapter.classList.contains('chapter')) {
                markChapterRead(activeId);
            }
        }
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

    // === Hero Section ===
    function renderHeroSection() {
        const lang = window.I18n ? window.I18n.getLang() : 'en';
        const t = window.I18n ? window.I18n.t : () => null;
        const progress = getReadingProgress();
        const totalChapters = TEXTBOOK.parts.reduce((sum, p) => sum + p.chapters.length, 0);
        const readChapters = Object.keys(progress).length;
        const pct = totalChapters > 0 ? Math.round((readChapters / totalChapters) * 100) : 0;
        const chaptersWord = t('chapters_label') || 'chapters';

        let goalsHtml = '';
        if (TEXTBOOK.readingGoals) {
            const spLabel = t('study_plan') || 'Study Plan';
            goalsHtml = `<div class="reading-goals"><h4>${spLabel}</h4><div class="goals-grid">`;
            TEXTBOOK.readingGoals.forEach(goal => {
                const done = goal.chapters.every(id => progress[id]);
                const count = goal.chapters.filter(id => progress[id]).length;
                const goalLabel = t(goal.id) || goal.label;
                goalsHtml += `<div class="goal-item ${done ? 'completed' : ''}">
                    <span class="goal-check">${done ? '&#10003;' : count + '/' + goal.chapters.length}</span>
                    <span class="goal-label">${goalLabel}</span>
                </div>`;
            });
            goalsHtml += '</div></div>';
        }

        let changelogHtml = '';
        if (TEXTBOOK.changelog && TEXTBOOK.changelog.length > 0) {
            const ruLabel = t('recent_updates') || 'Recent Updates';
            changelogHtml = `<div class="changelog"><h4>${ruLabel}</h4>`;
            TEXTBOOK.changelog.slice(0, 3).forEach((entry, i) => {
                const entryText = t('changelog_' + (i + 1)) || entry.text;
                changelogHtml += `<div class="changelog-item"><span class="changelog-date">${entry.date}</span> ${entryText}</div>`;
            });
            changelogHtml += '</div>';
        }

        return `
        <div class="hero-section" id="hero">
            <div class="hero-progress">
                <div class="progress-ring">
                    <span class="progress-pct">${pct}%</span>
                    <span class="progress-label">${readChapters}/${totalChapters} ${chaptersWord}</span>
                </div>
                ${goalsHtml}
            </div>
            ${changelogHtml}
        </div>`;
    }

    function getReadingProgress() {
        try { return JSON.parse(localStorage.getItem('lt_reading_progress') || '{}'); }
        catch { return {}; }
    }

    function markChapterRead(chapterId) {
        const progress = getReadingProgress();
        progress[chapterId] = Date.now();
        localStorage.setItem('lt_reading_progress', JSON.stringify(progress));
    }

    // Export for other modules
    window.App = {
        scrollToSection,
        toggleChat,
        renderContent,
        renderTOC,
        updateUILang,
        getState: () => state
    };
})();
