// === Search Module ===
(function() {
    'use strict';

    const searchInput = document.getElementById('global-search');
    const searchResults = document.getElementById('search-results');
    let searchIndex = [];
    let debounceTimer = null;

    document.addEventListener('DOMContentLoaded', () => {
        buildSearchIndex();
        setupSearch();
    });

    function buildSearchIndex() {
        searchIndex = [];
        TEXTBOOK.parts.forEach(part => {
            part.chapters.forEach(chapter => {
                chapter.sections.forEach(section => {
                    // Strip HTML tags for plain text search
                    const text = section.content.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
                    searchIndex.push({
                        chapterTitle: chapter.title,
                        sectionTitle: section.title,
                        sectionId: section.id,
                        chapterId: chapter.id,
                        text: text,
                        textLower: text.toLowerCase()
                    });
                });
            });
        });
    }

    function setupSearch() {
        searchInput.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => performSearch(searchInput.value), 150);
        });

        searchInput.addEventListener('focus', () => {
            if (searchInput.value.trim().length >= 2) {
                performSearch(searchInput.value);
            }
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.search-container')) {
                searchResults.classList.add('hidden');
            }
        });
    }

    function performSearch(query) {
        query = query.trim().toLowerCase();
        if (query.length < 2) {
            searchResults.classList.add('hidden');
            return;
        }

        const results = [];
        const queryWords = query.split(/\s+/);

        searchIndex.forEach(item => {
            // Check if all query words are present
            const allMatch = queryWords.every(w => item.textLower.includes(w));
            if (!allMatch) return;

            // Find best snippet
            const idx = item.textLower.indexOf(queryWords[0]);
            const snippetStart = Math.max(0, idx - 60);
            const snippetEnd = Math.min(item.text.length, idx + 120);
            let snippet = item.text.substring(snippetStart, snippetEnd);
            if (snippetStart > 0) snippet = '...' + snippet;
            if (snippetEnd < item.text.length) snippet += '...';

            // Highlight query words in snippet
            let highlightedSnippet = snippet;
            queryWords.forEach(w => {
                const regex = new RegExp(`(${escapeRegex(w)})`, 'gi');
                highlightedSnippet = highlightedSnippet.replace(regex, '<mark>$1</mark>');
            });

            results.push({
                chapterTitle: item.chapterTitle,
                sectionTitle: item.sectionTitle,
                sectionId: item.sectionId,
                snippet: highlightedSnippet,
                relevance: queryWords.reduce((sum, w) => {
                    return sum + (item.textLower.split(w).length - 1);
                }, 0)
            });
        });

        // Sort by relevance
        results.sort((a, b) => b.relevance - a.relevance);

        renderSearchResults(results.slice(0, 10));
    }

    function renderSearchResults(results) {
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="search-result-item"><div class="search-result-text">No results found</div></div>';
        } else {
            searchResults.innerHTML = results.map(r => `
                <div class="search-result-item" data-section="${r.sectionId}">
                    <div class="search-result-chapter">${r.chapterTitle} &rsaquo; ${r.sectionTitle}</div>
                    <div class="search-result-text">${r.snippet}</div>
                </div>
            `).join('');
        }

        searchResults.classList.remove('hidden');

        // Click handlers
        searchResults.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                const sectionId = item.getAttribute('data-section');
                if (sectionId) {
                    App.scrollToSection(sectionId);
                    searchResults.classList.add('hidden');
                    searchInput.value = '';
                    searchInput.blur();
                }
            });
        });
    }

    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
})();
