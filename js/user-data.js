// === User Data Module ===
// Dual-write abstraction over localStorage + Firestore
(function() {
    'use strict';

    const MIGRATION_PREFIX = 'lt_migrated_';

    // All localStorage keys managed by the app
    const MANAGED_KEYS = {
        highlights: 'lt_highlights',
        comments: 'lt_comments',
        chatMemory: 'lt_chat_memory',
        chatHistory: 'lt_chat_history',
        readingProgress: 'lt_reading_progress',
        contentUpdates: 'lt_content_updates',
        theme: 'theme',
        language: 'lt_language'
    };

    function getFirestore() {
        if (typeof firebase !== 'undefined' && firebase.apps && firebase.apps.length) {
            return firebase.firestore();
        }
        return null;
    }

    function getUserId() {
        if (window.AuthModule) {
            const user = window.AuthModule.getUser();
            return user ? user.uid : null;
        }
        return null;
    }

    // === Core Operations ===

    function save(key, data) {
        // Always save to localStorage (offline-first)
        try {
            localStorage.setItem(key, typeof data === 'string' ? data : JSON.stringify(data));
        } catch (e) {
            console.warn('[UserData] localStorage save failed:', e);
        }

        // Also save to Firestore if authenticated
        const uid = getUserId();
        const db = getFirestore();
        if (uid && db) {
            const docKey = getFirestoreKey(key);
            if (docKey) {
                db.collection('users').doc(uid).set({
                    [docKey]: data,
                    lastUpdated: firebase.firestore.FieldValue.serverTimestamp()
                }, { merge: true }).catch(e => {
                    console.warn('[UserData] Firestore save failed:', e);
                });
            }
        }
    }

    function load(key) {
        try {
            const val = localStorage.getItem(key);
            if (val === null) return null;
            try { return JSON.parse(val); } catch { return val; }
        } catch {
            return null;
        }
    }

    function getFirestoreKey(localKey) {
        for (const [name, lsKey] of Object.entries(MANAGED_KEYS)) {
            if (lsKey === localKey) return name;
        }
        return null;
    }

    // === Migration ===

    async function migrateLocalToFirestore(uid) {
        const db = getFirestore();
        if (!db || !uid) return;

        const migrationFlag = MIGRATION_PREFIX + uid;
        if (localStorage.getItem(migrationFlag)) return; // Already migrated

        const data = {};
        for (const [name, lsKey] of Object.entries(MANAGED_KEYS)) {
            const val = load(lsKey);
            if (val !== null) {
                data[name] = val;
            }
        }

        if (Object.keys(data).length > 0) {
            try {
                await db.collection('users').doc(uid).set(data, { merge: true });
                localStorage.setItem(migrationFlag, Date.now().toString());
                console.log('[UserData] Migration complete:', Object.keys(data).length, 'keys');
            } catch (e) {
                console.warn('[UserData] Migration failed:', e);
            }
        }
    }

    // === Sync from Firestore ===

    async function syncFromFirestore(uid) {
        const db = getFirestore();
        if (!db || !uid) return;

        // First, migrate local data if not done
        await migrateLocalToFirestore(uid);

        try {
            const doc = await db.collection('users').doc(uid).get();
            if (doc.exists) {
                const data = doc.data();
                for (const [name, lsKey] of Object.entries(MANAGED_KEYS)) {
                    if (data[name] !== undefined) {
                        const val = data[name];
                        localStorage.setItem(lsKey, typeof val === 'string' ? val : JSON.stringify(val));
                    }
                }
                console.log('[UserData] Synced from Firestore');
            }
        } catch (e) {
            console.warn('[UserData] Sync failed:', e);
        }
    }

    // === Profile ===

    async function saveProfile(user) {
        const db = getFirestore();
        if (!db || !user) return;

        try {
            await db.collection('users').doc(user.uid).set({
                profile: {
                    email: user.email,
                    displayName: user.displayName,
                    photoURL: user.photoURL,
                    lastLogin: firebase.firestore.FieldValue.serverTimestamp()
                }
            }, { merge: true });
        } catch (e) {
            console.warn('[UserData] Profile save failed:', e);
        }
    }

    // Export
    window.UserDataModule = {
        save,
        load,
        migrateLocalToFirestore,
        syncFromFirestore,
        saveProfile,
        MANAGED_KEYS
    };
})();
