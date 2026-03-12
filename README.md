# Stock Price Predictor Web App

🌐 **Live Demo**: https://ryzhanghason.github.io/analysis/

A beautiful web-based stock prediction application with **Factor Analysis** and **Meta-Learning** capabilities.

## 🌟 Features

- **🚀 No Server Needed**: Python runs entirely in your browser via WebAssembly
- **🔒 Privacy-First**: All computations happen locally - no data sent to servers
- **📊 Real-time Stock Data**: Fetches live data from Yahoo Finance
- **📈 Technical Analysis**: Calculates MA, RSI, MACD, Bollinger Bands, ADX, CCI, MFI, OBV
- **🎯 Smart Predictions**: Multi-indicator scoring system for UP/DOWN predictions
- **🔍 Factor Analysis**: 
  - Value factors (PE, PB, PS, Dividend Yield)
  - Profitability factors (ROE, ROA, Gross Margin, Operating Margin)
  - Growth factors (Earnings Growth, Revenue Growth)
  - Leverage factors (Debt/Equity, Current Ratio)
  - Composite factor scores (0-100 scale)
- **🧠 Meta-Learning Optimizer**: Smart strategy optimization with regime detection & ensemble methods
- **📱 Responsive Design**: Works on desktop and mobile devices
- **⚡ One-Time Load**: ~100MB download on first visit, then cached by browser

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone/Download this repository**

2. **Install Python dependencies**:
   ```bash
   cd stock-predictor-web
   pip install -r requirements.txt
   ```

3. **Start the API server**:
   ```bash
   cd api
   python app.py
   ```

   The API will start at `http://localhost:5000`

4. **Open the web interface**:
   - Simply open `index.html` in your browser
   - Or use a local server:
     ```bash
     # Python 3
     python -m http.server 8000
     ```
   - Then visit: `http://localhost:8000`

5. **Enter a stock symbol** (e.g., AAPL, TSLA, MSFT) and click "Fetch & Predict"!

### Option 2: Deploy to GitHub Pages + Cloud API

#### Frontend (GitHub Pages):

1. Create a new repository on GitHub (e.g., `stock-predictor`)
2. Push this code to your repo:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/stock-predictor.git
   git push -u origin main
   ```
3. Go to Settings → Pages → Source → Select "main" branch
4. Your frontend will be live at: `https://YOUR_USERNAME.github.io/stock-predictor/`

#### Backend API (Deploy to Render/Railway/PythonAnywhere):

**Option A: Deploy to Render.com (Free)**

1. Create account at https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Configure:
   - **Root Directory**: `api`
   - **Build Command**: `pip install -r ../requirements.txt`
   - **Start Command**: `python app.py`
5. Deploy! You'll get a URL like `https://your-api.onrender.com`

**Option B: Deploy to Railway.app (Free)**

1. Create account at https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repo
4. Railway will auto-detect Flask and deploy

**Option C: Deploy to PythonAnywhere (Free)**

1. Create account at https://www.pythonanywhere.com
2. Upload your `api` folder
3. Create a new web app with Flask
4. Configure WSGI file to point to your app

6. **Update the frontend**:
   - Edit `js/app.js`
   - Change `const API_URL = 'http://localhost:5000/api'`
   - To: `const API_URL = 'https://YOUR-API-URL.com/api'`
   - Commit and push

## 📁 Project Structure

```
stock-predictor-web/
├── index.html              # Main HTML page
├── css/
│   └── style.css          # Styling
├── js/
│   └── app.js             # Frontend logic
├── api/
│   ├── app.py             # Flask API server
│   ├── data_fetcher.py    # Stock data fetching
│   ├── feature_engineering.py  # Technical indicators
│   ├── model.py           # XGBoost model
│   └── strategy_optimizer.py   # Strategy optimization
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 Screenshots

The web app features:
- Clean, modern interface
- Real-time loading states
- Color-coded predictions (green for UP, red for DOWN)
- Interactive Chart.js visualizations
- Mobile-responsive design

## 🔧 Configuration

### API Endpoint
Edit `js/app.js` to change the API endpoint:
```javascript
const API_URL = 'YOUR_API_URL/api';
```

### Stock Settings
Users can configure:
- Stock symbol
- Time period (6mo, 1y, 2y, 5y, max)
- Strategy optimization (on/off)

## 📊 How It Works

1. User enters stock symbol
2. Frontend sends request to Flask API
3. API fetches historical data from Yahoo Finance
4. Calculates 100+ technical indicators
5. Trains/loads XGBoost model
6. Makes prediction and optimizes trading strategy
7. Returns results with charts to frontend

## 🔬 Factor Analysis API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/factors` | POST | Get fundamental factors for a stock |
| `/api/factors/compare` | POST | Compare factors across multiple stocks |
| `/api/factors/compare-full` | POST | Full analysis combining technical + fundamental |

### Example Request

```bash
curl -X POST http://localhost:5000/api/factors \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

### Response (Factor Scores)

```json
{
  "symbol": "AAPL",
  "factor_scores": {
    "value_score": 45.2,
    "profitability_score": 92.5,
    "growth_score": 65.0,
    "leverage_score": 78.3,
    "momentum_score": 55.0,
    "overall_score": 67.2
  },
  "fundamentals": {
    "trailingPE": 28.5,
    "priceToBook": 45.0,
    "returnOnEquity": "150.00%",
    ...
  }
}
```

### Factor Categories

- **Value Score**: PE, PB, PS ratios, dividend yield
- **Profitability Score**: ROE, ROA, gross margin, operating margin
- **Growth Score**: Earnings growth, revenue growth
- **Leverage Score**: Debt/Equity, current ratio, quick ratio
- **Momentum Score**: 52-week change, beta

## 🛠️ Development

### Run in development mode:
```bash
# Terminal 1 - API Server
cd api
python app.py

# Terminal 2 - Frontend (optional)
python -m http.server 8000
```

### Build for production:
- Minify CSS/JS files
- Deploy API to cloud service
- Deploy frontend to GitHub Pages

## 📝 License

Private project - All rights reserved

## 👨‍💻 Author

Built with ❤️ by [ryZhangHason](https://github.com/ryZhangHason)
