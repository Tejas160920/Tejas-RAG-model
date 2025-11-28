# RAG Chatbot Deployment Instructions

## Overview
This guide will help you deploy your RAG chatbot to Hugging Face Spaces and integrate it with your portfolio website.

---

## Part 1: Deploy to Hugging Face Spaces

### Step 1: Create a Hugging Face Account
1. Go to https://huggingface.co
2. Click "Sign Up" and create an account
3. Verify your email

### Step 2: Create a New Space
1. Click on your profile picture (top right)
2. Click "New Space"
3. Fill in the details:
   - **Space name:** `tejas-portfolio-rag`
   - **License:** MIT
   - **Select SDK:** Docker
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
4. Click "Create Space"

### Step 3: Upload Your Files
You need to upload these files from the `huggingface-api` folder:

```
huggingface-api/
├── app.py          (FastAPI backend)
├── requirements.txt (Python dependencies)
├── Dockerfile      (Container configuration)
├── DATA.pdf        (Your portfolio data)
└── README.md       (Space description)
```

**Option A: Upload via Web Interface**
1. In your new Space, click "Files" tab
2. Click "Add file" → "Upload files"
3. Drag and drop all 5 files
4. Click "Commit changes"

**Option B: Upload via Git (Recommended)**
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR-USERNAME/tejas-portfolio-rag

# Copy files to the cloned folder
cd tejas-portfolio-rag
# Copy all files from huggingface-api folder here

# Push to Hugging Face
git add .
git commit -m "Initial deployment"
git push
```

### Step 4: Set Environment Variable (REQUIRED)
1. Go to your Space → Settings
2. Scroll to "Repository secrets"
3. Add new secret:
   - **Name:** `GROQ_API_KEY`
   - **Value:** Your Groq API key (get one from https://console.groq.com)
4. Click "Save"

### Step 5: Wait for Build
1. Go to your Space page
2. Watch the "Building" status in the logs
3. Wait until it shows "Running" (usually 3-5 minutes)
4. Your API is now live!

### Step 6: Test Your API
Your API URL will be:
```
https://YOUR-USERNAME-tejas-portfolio-rag.hf.space
```

Test it:
1. Open the URL in browser - should show: `{"status": "online", "message": "Tejas Portfolio RAG API is running!"}`
2. Test the chat endpoint using curl or Postman:
```bash
curl -X POST "https://YOUR-USERNAME-tejas-portfolio-rag.hf.space/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Tejas skills?"}'
```

---

## Part 2: Update Your Portfolio Website

### Step 1: Update the API URL in Chatbot.js
Open `Portfolio/src/components/Chatbot.js` and update line 18:

```javascript
// Replace this:
const API_URL = 'https://YOUR-USERNAME-tejas-portfolio-rag.hf.space/chat';

// With your actual URL:
const API_URL = 'https://tejas160920-tejas-portfolio-rag.hf.space/chat';
```

### Step 2: Test Locally
```bash
cd Portfolio
npm start
```
- Click the green chat bubble in the bottom right
- Try asking questions

### Step 3: Deploy to Vercel
```bash
git add .
git commit -m "Add AI chatbot feature"
git push
```
Vercel will automatically deploy the changes.

---

## Troubleshooting

### Issue: Space shows "Error" or "Building Failed"
1. Check the logs in Hugging Face Space
2. Common issues:
   - Missing DATA.pdf file
   - Typo in requirements.txt
   - Dockerfile syntax error

### Issue: CORS Error in browser
The API already has CORS enabled for all origins. If you still see errors:
1. Check browser console for actual error
2. Make sure the Space is "Running" not "Building"

### Issue: Slow first response
This is normal! The first request takes 10-30 seconds because:
- The model needs to load
- FAISS index needs to be built
Subsequent requests will be faster (2-5 seconds).

### Issue: "API request failed" in chat
1. Check if Space is running: visit the API URL directly
2. Check browser Network tab for actual error
3. Space might have gone to sleep (free tier) - visit the URL to wake it up

---

## File Structure Summary

```
Portfolio/
├── RAG/
│   ├── huggingface-api/        ← Upload this folder to HF Spaces
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   ├── DATA.pdf
│   │   └── README.md
│   ├── RAG_final.ipynb
│   └── DATA.pdf
│
└── Portfolio/
    └── src/
        └── components/
            ├── Chatbot.js      ← React chat component
            ├── Chatbot.css     ← Chat styling
            └── ...
```

---

## Quick Checklist

- [ ] Created Hugging Face account
- [ ] Created new Space (Docker SDK)
- [ ] Uploaded all 5 files to Space
- [ ] Space shows "Running" status
- [ ] Tested API endpoint in browser
- [ ] Updated API_URL in Chatbot.js
- [ ] Tested chatbot locally
- [ ] Pushed changes to GitHub
- [ ] Verified on live Vercel site

---

## Need Help?

- Hugging Face Spaces docs: https://huggingface.co/docs/hub/spaces
- FastAPI docs: https://fastapi.tiangolo.com/
- Groq API docs: https://console.groq.com/docs

Your chatbot will appear as a green chat bubble in the bottom-right corner of your portfolio!
