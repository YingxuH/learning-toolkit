// Content Loader - merges expanded chapter content into TEXTBOOK
// Runs synchronously before app.js init
(function() {
    'use strict';

    function mergeContent() {
        if (!window.TEXTBOOK) return;

        // Build chapter lookup
        const chapterMap = {};
        TEXTBOOK.parts.forEach(function(part) {
            part.chapters.forEach(function(ch) {
                chapterMap[ch.id] = ch;
            });
        });

        // Merge function
        function mergeChapter(chapterId, newSections) {
            if (!newSections || !chapterMap[chapterId]) return;
            var chapter = chapterMap[chapterId];
            var merged = [];
            var processedIds = {};

            // Add all new sections first
            for (var i = 0; i < newSections.length; i++) {
                merged.push({
                    id: newSections[i].id,
                    title: newSections[i].title,
                    content: newSections[i].content
                });
                processedIds[newSections[i].id] = true;
            }

            // Add existing sections not in new content
            for (var j = 0; j < chapter.sections.length; j++) {
                if (!processedIds[chapter.sections[j].id]) {
                    merged.push(chapter.sections[j]);
                }
            }

            chapter.sections = merged;
        }

        // Merge each expanded content file
        if (window.CONTENT_CH1_2) {
            mergeChapter('audio-llm-landscape', window.CONTENT_CH1_2.ch1_sections);
            mergeChapter('speech-to-speech', window.CONTENT_CH1_2.ch2_sections);
        }
        if (window.CONTENT_CH3_4) {
            mergeChapter('tts-technology', window.CONTENT_CH3_4.ch3_sections);
            mergeChapter('speculative-decoding', window.CONTENT_CH3_4.ch4_sections);
        }
        if (window.CONTENT_CH5_6) {
            mergeChapter('vllm-serving', window.CONTENT_CH5_6.ch5_sections);
            mergeChapter('rl-training', window.CONTENT_CH5_6.ch6_sections);
        }
        if (window.CONTENT_CH7_8) {
            mergeChapter('ml-engineering', window.CONTENT_CH7_8.ch7_sections);
            mergeChapter('agent-development', window.CONTENT_CH7_8.ch8_sections);
        }
        if (window.CONTENT_CH9_10) {
            mergeChapter('system-design', window.CONTENT_CH9_10.ch9_sections);
            mergeChapter('interview-prep', window.CONTENT_CH9_10.ch10_sections);
        }
    }

    // Run immediately
    mergeContent();

    window.ContentLoader = { mergeContent: mergeContent };
})();
