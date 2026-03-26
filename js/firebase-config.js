// === App Configuration ===
// Gemini API key - loaded from localStorage or prompted on first use
// NEVER commit actual API keys to git

const APP_CONFIG = {
    // Gemini model to use
    geminiModel: "gemini-3.1-pro-preview",
    geminiApiBase: "https://generativelanguage.googleapis.com/v1beta/models/",

    // Firebase (optional - for multi-device sync with <10 users)
    // Set to null to disable Firebase entirely (localStorage-only mode)
    firebase: null,

    // Allowed users for Firebase auth (if enabled)
    allowedUsers: ["yingxu.he1998@gmail.com"]
};

// Gemini API key management
function getGeminiKey() {
    return localStorage.getItem('lt_gemini_key');
}

function setGeminiKey(key) {
    localStorage.setItem('lt_gemini_key', key);
}

function promptGeminiKey() {
    const key = prompt(
        'Enter your Gemini API key to enable AI chat.\n\n' +
        'Get one free at: https://aistudio.google.com/apikey\n\n' +
        'The key is stored locally in your browser only (never sent to our servers).'
    );
    if (key && key.trim().startsWith('AIza')) {
        setGeminiKey(key.trim());
        return key.trim();
    }
    return null;
}

// Initialize Firebase only if config is provided
if (APP_CONFIG.firebase && typeof firebase !== 'undefined') {
    try {
        firebase.initializeApp(APP_CONFIG.firebase);
    } catch(e) {
        console.log('[Firebase] Init failed:', e.message);
    }
}
