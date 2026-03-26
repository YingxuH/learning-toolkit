// === App Configuration ===
// API keys are injected at deploy time via GitHub Secrets
// NEVER commit actual keys to this file

const APP_CONFIG = {
    geminiModel: "gemini-3.1-pro-preview",
    geminiApiBase: "https://generativelanguage.googleapis.com/v1beta/models/",
    allowedUsers: [
        "yingxu.he1998@gmail.com",
        "lewis.won@gmail.com",
        "hesirui00@gmail.com"
    ]
};

// Gemini API key management (stored in localStorage)
function getGeminiKey() {
    return localStorage.getItem('lt_gemini_key');
}

function setGeminiKey(key) {
    if (key) localStorage.setItem('lt_gemini_key', key);
    else localStorage.removeItem('lt_gemini_key');
}

function promptGeminiKey() {
    const key = prompt(
        'Enter your Gemini API key to enable AI chat.\n\n' +
        'Get one free at: https://aistudio.google.com/apikey\n\n' +
        'The key is stored locally in your browser only.'
    );
    if (key && key.trim().startsWith('AIza')) {
        setGeminiKey(key.trim());
        return key.trim();
    }
    return null;
}

// Firebase is initialized by config.local.js (injected at deploy time)
// If not available, the app works in localStorage-only mode
