// === War Stories Index ===
// Extracts all production war stories from content and creates a browsable index
(function() {
    'use strict';

    function extractWarStories() {
        const stories = [];
        const container = document.getElementById('textbook-content');
        if (!container) return stories;

        container.querySelectorAll('.callout.warning').forEach((callout, idx) => {
            const titleEl = callout.querySelector('.callout-title');
            if (!titleEl) return;
            const title = titleEl.textContent.trim();
            if (!title.toLowerCase().includes('war story') && !title.includes('踩坑')) return;

            const section = callout.closest('section[id]');
            const chapter = callout.closest('.chapter');
            const sectionId = section ? section.id : (chapter ? chapter.id : '');
            const chapterTitle = chapter ? chapter.querySelector('h2') : null;

            stories.push({
                id: 'ws-' + idx,
                title: title.replace(/^Production War Story:\s*/i, '').replace(/^生产踩坑实录[：:]\s*/i, ''),
                fullTitle: title,
                html: callout.outerHTML,
                sectionId: sectionId,
                chapterTitle: chapterTitle ? chapterTitle.textContent.trim() : '',
                text: callout.textContent.substring(0, 200) + '...'
            });
        });

        return stories;
    }

    function renderWarStoriesPage() {
        const stories = extractWarStories();
        if (stories.length === 0) return '';

        const lang = window.I18n ? window.I18n.getLang() : 'en';
        const heading = lang === 'zh' ? '生产踩坑实录集锦' : 'Production War Stories';
        const subtitle = lang === 'zh'
            ? `${stories.length} 个来自真实生产环境的踩坑故事，附带具体数据和修复方案`
            : `${stories.length} real-world production failure stories with specific data and fixes`;

        let html = `
        <div class="war-stories-page" id="war-stories-section">
            <div class="ws-header">
                <h2>${heading}</h2>
                <p class="ws-subtitle">${subtitle}</p>
            </div>
            <div class="ws-grid">`;

        stories.forEach(story => {
            html += `
            <div class="ws-card" data-section="${story.sectionId}">
                <div class="ws-card-chapter">${story.chapterTitle}</div>
                <div class="ws-card-title">${story.title}</div>
                <div class="ws-card-preview">${story.text}</div>
                <a class="ws-card-link" href="#${story.sectionId}">Read full story &rarr;</a>
            </div>`;
        });

        html += '</div></div>';
        return html;
    }

    function setupWarStoriesNav() {
        // Add "War Stories" link to the sidebar TOC
        const toc = document.getElementById('toc');
        if (!toc) return;

        const wsLink = document.createElement('div');
        wsLink.className = 'toc-part';
        wsLink.innerHTML = `
            <div class="toc-part-title" style="color: #e67e22;">War Stories</div>
            <a class="toc-item" data-section="war-stories-section" href="#war-stories-section" style="color: #e67e22;">
                All Production War Stories
            </a>
        `;
        toc.insertBefore(wsLink, toc.firstChild);

        wsLink.querySelector('.toc-item').addEventListener('click', (e) => {
            e.preventDefault();
            const el = document.getElementById('war-stories-section');
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            if (window.innerWidth < 768 && window.App) {
                // Close sidebar on mobile
                const sidebar = document.getElementById('sidebar');
                if (sidebar && !sidebar.classList.contains('collapsed')) {
                    document.getElementById('sidebar-toggle').click();
                }
            }
        });
    }

    function injectWarStories() {
        const content = document.getElementById('textbook-content');
        if (!content) return;

        // Insert war stories section after the hero section
        const hero = content.querySelector('.hero-section');
        if (hero) {
            const wsHtml = renderWarStoriesPage();
            if (wsHtml) {
                const wsDiv = document.createElement('div');
                wsDiv.innerHTML = wsHtml;
                hero.after(wsDiv.firstElementChild);
            }
        }

        setupWarStoriesNav();

        // Wire up card clicks
        content.querySelectorAll('.ws-card-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const sectionId = link.getAttribute('href').substring(1);
                if (window.App) App.scrollToSection(sectionId);
            });
        });
    }

    // Run after content is rendered
    const origRender = window.App && window.App.renderContent;
    if (origRender) {
        const _origRender = origRender;
        window.App.renderContent = function() {
            _origRender.call(window.App);
            setTimeout(injectWarStories, 100);
        };
    } else {
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(injectWarStories, 500);
        });
    }

    window.WarStories = { extractWarStories, renderWarStoriesPage, injectWarStories };
})();
