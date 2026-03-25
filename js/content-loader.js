// Content Loader - merges expanded chapter content into TEXTBOOK
// Must run synchronously after all content-ch*.js files load

function _mergeExpandedContent() {
    if (typeof TEXTBOOK === 'undefined') return;

    var chapterMap = {};
    TEXTBOOK.parts.forEach(function(part) {
        part.chapters.forEach(function(ch) {
            chapterMap[ch.id] = ch;
        });
    });

    function merge(chapterId, newSections) {
        if (!newSections || !chapterMap[chapterId]) return;
        var chapter = chapterMap[chapterId];
        var merged = [];
        var seen = {};
        for (var i = 0; i < newSections.length; i++) {
            merged.push({ id: newSections[i].id, title: newSections[i].title, content: newSections[i].content });
            seen[newSections[i].id] = true;
        }
        for (var j = 0; j < chapter.sections.length; j++) {
            if (!seen[chapter.sections[j].id]) {
                merged.push(chapter.sections[j]);
            }
        }
        chapter.sections = merged;
    }

    try {
        if (typeof CONTENT_CH1_2 !== 'undefined') {
            merge('audio-llm-landscape', CONTENT_CH1_2.ch1_sections);
            merge('speech-to-speech', CONTENT_CH1_2.ch2_sections);
        }
    } catch(e) {}
    try {
        if (typeof CONTENT_CH3_4 !== 'undefined') {
            merge('tts-technology', CONTENT_CH3_4.ch3_sections);
            merge('speculative-decoding', CONTENT_CH3_4.ch4_sections);
        }
    } catch(e) {}
    try {
        if (typeof CONTENT_CH5_6 !== 'undefined') {
            merge('vllm-serving', CONTENT_CH5_6.ch5_sections);
            merge('rl-training', CONTENT_CH5_6.ch6_sections);
        }
    } catch(e) {}
    try {
        if (typeof CONTENT_CH7_8 !== 'undefined') {
            merge('ml-engineering', CONTENT_CH7_8.ch7_sections);
            merge('agent-development', CONTENT_CH7_8.ch8_sections);
        }
    } catch(e) {}
    try {
        if (typeof CONTENT_CH9_10 !== 'undefined') {
            merge('system-design', CONTENT_CH9_10.ch9_sections);
            merge('interview-prep', CONTENT_CH9_10.ch10_sections);
        }
    } catch(e) {}
}

_mergeExpandedContent();
