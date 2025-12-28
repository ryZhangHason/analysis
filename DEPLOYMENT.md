# Deployment Guide

## 🌐 Your Website is Live!

**Frontend URL**: https://ryzhanghason-stock.github.io/

The frontend is automatically deployed via GitHub Pages. Any push to the `main` branch will update the website.

## 🔧 Deploy the API Backend

To make the predictions work, you need to deploy the Flask API. Here are 3 free options:

### Option 1: Render.com (Recommended - Easiest)

1. **Sign up** at https://render.com (use your GitHub account)

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select repository: `ryzhanghason-stock.github.io`

3. **Configure**:
   - **Name**: `stock-predictor-api`
   - **Root Directory**: `api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r ../requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Free

4. **Deploy**: Click "Create Web Service"

5. **Copy your API URL**: Will be like `https://stock-predictor-api-xxxx.onrender.com`

6. **Update Frontend**:
   ```bash
   cd /Users/h/Downloads/stock-predictor-web
   # Edit js/app.js - change line 3:
   # const API_URL = 'https://YOUR-RENDER-URL.onrender.com/api';
   git add js/app.js
   git commit -m "Update API URL to Render deployment"
   git push
   ```

### Option 2: Railway.app (Fast Setup)

1. **Sign up** at https://railway.app

2. **New Project** → "Deploy from GitHub repo"

3. Select `ryzhanghason-stock.github.io`

4. Railway auto-detects Flask and deploys

5. Get your URL from Railway dashboard

6. Update `js/app.js` with your Railway URL

### Option 3: PythonAnywhere (Traditional)

1. **Sign up** at https://www.pythonanywhere.com (free account)

2. **Upload Files**:
   - Go to "Files" tab
   - Upload entire `api` folder

3. **Create Web App**:
   - Go to "Web" tab
   - Add a new web app
   - Choose Flask
   - Python 3.9

4. **Configure**:
   - Edit WSGI configuration file
   - Point to your app.py

5. **Install Dependencies**:
   - Open Bash console
   - `pip install --user -r requirements.txt`

6. **Get URL**: `https://yourusername.pythonanywhere.com`

## 📝 After Deploying API

1. **Update API URL** in `js/app.js`:
   ```javascript
   const API_URL = 'https://YOUR-API-URL.com/api';
   ```

2. **Test the API**:
   ```bash
   curl https://YOUR-API-URL.com/api/health
   ```
   Should return: `{"status":"ok","message":"Stock Predictor API is running"}`

3. **Push changes**:
   ```bash
   git add js/app.js
   git commit -m "Connect to deployed API"
   git push
   ```

4. **Visit your website**: https://ryzhanghason-stock.github.io/

## 🎯 Quick Test

Before deploying, test locally:

```bash
# Terminal 1 - Start API
cd /Users/h/Downloads/stock-predictor-web/api
python3 app.py

# Terminal 2 - Test API
curl http://localhost:5000/api/health

# Browser - Open
open /Users/h/Downloads/stock-predictor-web/index.html
```

## 🔄 Future Updates

To update your website:

```bash
cd /Users/h/Downloads/stock-predictor-web
# Make changes
git add .
git commit -m "Your update message"
git push
```

GitHub Pages will automatically rebuild and deploy!

## 🆘 Troubleshooting

**Website shows but predictions don't work**:
- API isn't deployed yet
- API URL in `js/app.js` is incorrect
- Check browser console for errors (F12)

**CORS errors**:
- Make sure Flask-CORS is installed
- API must be deployed and accessible

**API deployment fails**:
- Check Python version (needs 3.9+)
- Verify requirements.txt is correct
- Check deployment logs

## 📊 What's Deployed

**Frontend (GitHub Pages)**:
- ✅ HTML/CSS/JavaScript
- ✅ Automatically updates on push
- ✅ Free forever
- ✅ Custom domain support

**Backend (Render/Railway/PythonAnywhere)**:
- 🔄 Needs manual deployment
- 💰 Free tier available
- 📈 May sleep after inactivity (free tier)
- 🚀 First request may be slow (waking up)

Enjoy your live stock predictor! 🎉
