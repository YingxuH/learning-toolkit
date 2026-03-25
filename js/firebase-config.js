// Firebase Configuration
// NOTE: These are public client-side keys (safe to expose in source code).
// The actual Firebase project must be created at https://console.firebase.google.com
// After creating the project, replace these placeholder values with real ones.
//
// Setup steps:
// 1. Go to https://console.firebase.google.com
// 2. Create a new project (e.g., "learning-toolkit-ai")
// 3. Enable Authentication > Google provider
// 4. Enable Cloud Firestore
// 5. Add a Web app and copy the config below
// 6. Deploy Cloud Functions (see functions/ directory)

const FIREBASE_CONFIG = {
    // REPLACE these with your actual Firebase project config
    apiKey: "REPLACE_WITH_YOUR_FIREBASE_API_KEY",
    authDomain: "learning-toolkit-ai.firebaseapp.com",
    projectId: "learning-toolkit-ai",
    storageBucket: "learning-toolkit-ai.appspot.com",
    messagingSenderId: "REPLACE",
    appId: "REPLACE"
};

// Gemini Cloud Function URL (deployed via Firebase Functions)
const GEMINI_FUNCTION_URL = "https://us-central1-learning-toolkit-ai.cloudfunctions.net/geminiChat";

// Allowed users (email allowlist)
const ALLOWED_USERS = [
    "yingxu.he1998@gmail.com"
];

// Initialize Firebase (only if SDK is loaded)
if (typeof firebase !== 'undefined' && FIREBASE_CONFIG.apiKey !== 'REPLACE_WITH_YOUR_FIREBASE_API_KEY') {
    firebase.initializeApp(FIREBASE_CONFIG);
} else {
    console.log('[Firebase] Not initialized - configure firebase-config.js with real project credentials');
}
