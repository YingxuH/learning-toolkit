// Content Loader - merges expanded chapter content into TEXTBOOK
(function() {
    'use strict';

    // Map of chapter IDs to their expanded content sources
    const contentSources = [
        { obj: 'CONTENT_CH1_2', chapters: { ch1: 'ch1_sections', ch2: 'ch2_sections' }, chapterIds: ['audio-llm-landscape', 'speech-to-speech'] },
        { obj: 'CONTENT_CH3_4', chapters: { ch3: 'ch3_sections', ch4: 'ch4_sections' }, chapterIds: ['tts-technology', 'speculative-decoding'] },
        { obj: 'CONTENT_CH5_6', chapters: { ch5: 'ch5_sections', ch6: 'ch6_sections' }, chapterIds: ['vllm-serving', 'rl-training'] },
        { obj: 'CONTENT_CH7_8', chapters: { ch7: 'ch7_sections', ch8: 'ch8_sections' }, chapterIds: ['ml-engineering', 'agent-development'] },
        { obj: 'CONTENT_CH9_10', chapters: { ch9: 'ch9_sections', ch10: 'ch10_sections' }, chapterIds: ['system-design', 'interview-prep'] }
    ];

    function mergeContent() {
        if (!window.TEXTBOOK) return;

        const chapterMap = {};
        TEXTBOOK.parts.forEach(part => {
            part.chapters.forEach(ch => {
                chapterMap[ch.id] = ch;
            });
        });

        contentSources.forEach(source => {
            const obj = window[source.obj];
            if (!obj) return;

            const keys = Object.keys(source.chapters);
            keys.forEach((key, idx) => {
                const sectionsKey = source.chapters[key];
                const chapterId = source.chapterIds[idx];
                const sections = obj[sectionsKey];

                if (!sections || !chapterMap[chapterId]) return;

                // Replace sections in the chapter
                const chapter = chapterMap[chapterId];
                const newSections = sections.map(s => ({
                    id: s.id,
                    title: s.title,
                    content: s.content
                }));

                // Merge: keep existing sections that aren't replaced, add new ones
                const existingIds = new Set(chapter.sections.map(s => s.id));
                const newIds = new Set(newSections.map(s => s.id));

                // Replace existing sections with expanded versions
                const merged = [];
                const processedIds = new Set();

                // First add all new sections in order
                newSections.forEach(ns => {
                    merged.push(ns);
                    processedIds.add(ns.id);
                });

                // Then add any existing sections not covered by new content
                chapter.sections.forEach(es => {
                    if (!processedIds.has(es.id)) {
                        merged.push(es);
                    }
                });

                chapter.sections = merged;
            });
        });
    }

    // Run immediately - all content scripts are already loaded (synchronous script tags)
    mergeContent();

    window.ContentLoader = { mergeContent };
})();
