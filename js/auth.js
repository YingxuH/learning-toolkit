// === Authentication Module ===
(function() {
    'use strict';

    let currentUser = null;
    let authReadyCallbacks = [];
    let authReady = false;

    document.addEventListener('DOMContentLoaded', () => {
        setupAuth();
    });

    function setupAuth() {
        // If Firebase is not initialized, show offline mode
        if (typeof firebase === 'undefined' || !firebase.apps || !firebase.apps.length) {
            setOfflineMode();
            return;
        }

        const auth = firebase.auth();

        // Listen for auth state changes
        auth.onAuthStateChanged((user) => {
            authReady = true;
            if (user) {
                // Check allowlist
                if (isAllowed(user.email)) {
                    currentUser = user;
                    updateUI(user);
                    // Sync data from Firestore
                    if (window.UserDataModule) {
                        window.UserDataModule.syncFromFirestore(user.uid);
                    }
                } else {
                    // Not in allowlist - sign out
                    auth.signOut();
                    currentUser = null;
                    updateUI(null);
                    alert('Access restricted. Your email is not in the allowlist.');
                }
            } else {
                currentUser = null;
                updateUI(null);
            }
            authReadyCallbacks.forEach(cb => cb(currentUser));
            authReadyCallbacks = [];
        });

        // Setup sign-in button
        const signInBtn = document.getElementById('auth-signin');
        if (signInBtn) {
            signInBtn.addEventListener('click', signIn);
        }

        // Setup sign-out button
        const signOutBtn = document.getElementById('auth-signout');
        if (signOutBtn) {
            signOutBtn.addEventListener('click', signOut);
        }
    }

    function signIn() {
        if (typeof firebase === 'undefined' || !firebase.apps.length) {
            alert('Firebase not configured. See js/firebase-config.js');
            return;
        }
        const provider = new firebase.auth.GoogleAuthProvider();
        provider.addScope('email');
        firebase.auth().signInWithPopup(provider).catch((error) => {
            console.error('[Auth] Sign-in error:', error);
            if (error.code === 'auth/popup-blocked') {
                alert('Popup blocked. Please allow popups for this site.');
            }
        });
    }

    function signOut() {
        if (typeof firebase !== 'undefined' && firebase.apps.length) {
            firebase.auth().signOut();
        }
        currentUser = null;
        updateUI(null);
    }

    function isAllowed(email) {
        if (typeof ALLOWED_USERS === 'undefined') return true;
        return ALLOWED_USERS.includes(email);
    }

    function updateUI(user) {
        const signInBtn = document.getElementById('auth-signin');
        const userInfo = document.getElementById('auth-user-info');
        const userName = document.getElementById('auth-user-name');
        const userAvatar = document.getElementById('auth-user-avatar');
        const signOutBtn = document.getElementById('auth-signout');

        if (!signInBtn) return;

        if (user) {
            signInBtn.classList.add('hidden');
            if (userInfo) userInfo.classList.remove('hidden');
            if (userName) userName.textContent = user.displayName || user.email.split('@')[0];
            if (userAvatar) {
                userAvatar.src = user.photoURL || '';
                userAvatar.style.display = user.photoURL ? 'block' : 'none';
            }
        } else {
            signInBtn.classList.remove('hidden');
            if (userInfo) userInfo.classList.add('hidden');
        }
    }

    function setOfflineMode() {
        authReady = true;
        const signInBtn = document.getElementById('auth-signin');
        if (signInBtn) {
            signInBtn.textContent = 'Offline';
            signInBtn.disabled = true;
            signInBtn.style.opacity = '0.5';
        }
    }

    function getUser() {
        return currentUser;
    }

    function getIdToken() {
        if (!currentUser) return Promise.resolve(null);
        return currentUser.getIdToken();
    }

    function onAuthReady(callback) {
        if (authReady) {
            callback(currentUser);
        } else {
            authReadyCallbacks.push(callback);
        }
    }

    // Export
    window.AuthModule = {
        signIn,
        signOut,
        getUser,
        getIdToken,
        isAllowed,
        onAuthReady
    };
})();
