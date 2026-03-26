# Firebase Setup (2 minutes)

## Step 1: Create Project
1. Go to https://console.firebase.google.com
2. Click "Create a project"
3. Name: `learning-toolkit` (or anything)
4. Disable Google Analytics (not needed)
5. Click "Create project"

## Step 2: Enable Google Auth
1. In the Firebase console, go to **Authentication** (left sidebar)
2. Click **Get started**
3. Click **Google** provider
4. Toggle **Enable**
5. Set support email to your email
6. Click **Save**

## Step 3: Enable Firestore
1. Go to **Firestore Database** (left sidebar)
2. Click **Create database**
3. Choose **Start in production mode**
4. Select region (asia-southeast1 for Singapore)
5. Click **Enable**

## Step 4: Get Web App Config
1. Go to **Project Settings** (gear icon top-left)
2. Scroll to "Your apps" section
3. Click the **Web** icon (`</>`)
4. Register app name: `learning-toolkit-web`
5. **Copy the firebaseConfig object** - you'll need it

## Step 5: Update the Site
Run this command, replacing the values with your actual Firebase config:

```bash
# In the repo directory:
gh secret set FIREBASE_CONFIG --repo YingxuH/learning-toolkit --body '{
  "apiKey": "YOUR_API_KEY",
  "authDomain": "YOUR_PROJECT.firebaseapp.com",
  "projectId": "YOUR_PROJECT_ID",
  "storageBucket": "YOUR_PROJECT.firebasestorage.app",
  "messagingSenderId": "YOUR_SENDER_ID",
  "appId": "YOUR_APP_ID"
}'
```

## Step 6: Deploy Firestore Rules
In the Firebase console > Firestore > Rules tab, paste:
```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId}/{document=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```
Click **Publish**.

## Step 7: Add Authorized Domain
1. Go to **Authentication** > **Settings** > **Authorized domains**
2. Add `yingxuh.github.io`

That's it! Push any commit to trigger redeployment and the login will work.
