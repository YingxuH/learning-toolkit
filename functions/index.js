const { onRequest } = require("firebase-functions/v2/https");
const admin = require("firebase-admin");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const cors = require("cors")({ origin: true });

admin.initializeApp();

// Gemini API Key - store as Firebase secret in production:
// firebase functions:secrets:set GEMINI_API_KEY
// API key MUST be set via environment variable or Firebase secret. Never hardcode.
// Deploy with: firebase functions:secrets:set GEMINI_API_KEY
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
    console.error("GEMINI_API_KEY environment variable is not set");
}

// Rate limiting: max requests per user per minute
const RATE_LIMIT = 10;
const rateLimitMap = new Map();

function checkRateLimit(uid) {
    const now = Date.now();
    const key = uid;
    const entry = rateLimitMap.get(key) || { count: 0, resetAt: now + 60000 };

    if (now > entry.resetAt) {
        entry.count = 0;
        entry.resetAt = now + 60000;
    }

    entry.count++;
    rateLimitMap.set(key, entry);

    return entry.count <= RATE_LIMIT;
}

exports.geminiChat = onRequest({ cors: true, secrets: ["GEMINI_API_KEY"] }, async (req, res) => {
    // Handle CORS preflight
    if (req.method === "OPTIONS") {
        res.set("Access-Control-Allow-Origin", "*");
        res.set("Access-Control-Allow-Methods", "POST");
        res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status(204).send("");
        return;
    }

    if (req.method !== "POST") {
        res.status(405).json({ error: "Method not allowed" });
        return;
    }

    // Verify Firebase Auth token
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        res.status(401).json({ error: "Missing or invalid authorization header" });
        return;
    }

    const idToken = authHeader.split("Bearer ")[1];
    let uid;
    try {
        const decoded = await admin.auth().verifyIdToken(idToken);
        uid = decoded.uid;
    } catch (error) {
        res.status(401).json({ error: "Invalid auth token" });
        return;
    }

    // Rate limiting
    if (!checkRateLimit(uid)) {
        res.status(429).json({ error: "Rate limit exceeded. Please wait a moment." });
        return;
    }

    // Extract request body
    const { messages, systemPrompt, context } = req.body;
    if (!messages || !Array.isArray(messages)) {
        res.status(400).json({ error: "messages array is required" });
        return;
    }

    try {
        const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
        const model = genAI.getGenerativeModel({
            model: "gemini-2.5-flash-preview-05-20",
            systemInstruction: systemPrompt || "You are an AI tutor for an AI Engineering textbook."
        });

        // Build conversation history for Gemini
        const history = [];
        for (const msg of messages.slice(0, -1)) {
            history.push({
                role: msg.role === "assistant" ? "model" : "user",
                parts: [{ text: msg.content }]
            });
        }

        const chat = model.startChat({ history });
        const lastMessage = messages[messages.length - 1];
        let prompt = lastMessage.content;

        // Add context if provided
        if (context) {
            prompt = `[The user highlighted this text from the textbook: "${context}"]\n\n${prompt}`;
        }

        const result = await chat.sendMessage(prompt);
        const response = result.response.text();

        res.json({ response });
    } catch (error) {
        console.error("[Gemini] Error:", error);
        res.status(500).json({ error: "AI service error: " + error.message });
    }
});
