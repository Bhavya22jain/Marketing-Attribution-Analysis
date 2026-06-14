## 📌 About This Project

This project performs a complete comparative analysis of marketing 
attribution models to determine how conversion credit should be 
distributed across marketing channels.

Built as a portfolio project during my 3rd year of studies, it covers 
the full data analytics pipeline — from raw data cleaning to machine 
learning to an interactive web dashboard.

## 🎯 Problem Statement

In digital marketing, multiple channels (Search, Email, Social, etc.) 
contribute to a single conversion. The key question is:

> **Which channel actually deserves credit for the sale?**

Different attribution models answer this differently. This project 
implements and compares all major models side by side.

## 📊 Dataset

- 10,000 marketing campaign records
- 5 channels: Search, Email, Influencer, Social, Display
- Features: Impressions, Clicks, Leads, Conversions, Cost, Revenue, ROI
- Date range: Jan 2025 — Jan 2026

## ⚙️ What's Implemented

### Data Preprocessing
- Missing value handling
- Duplicate removal
- Feature engineering: CTR, CVR, CPC, CPA, ROAS, Profit

### Attribution Models (built from scratch)
| Model | Logic |
|---|---|
| First Touch | 100% credit to first channel |
| Last Touch | 100% credit to last channel |
| Linear | Equal credit to all channels |
| Time Decay | More credit to channels closer to conversion |
| Position Based | 40% first, 40% last, 20% middle |

### Machine Learning
- **K-Means Clustering** — segments 10,000 campaigns into 4 tiers
- **Logistic Regression** — data-driven attribution proxy
- **Random Forest Classifier** — predicts high-performing campaigns
- Live prediction tool with probability scoring

### Interactive Dashboard (Streamlit)
- 6 pages: Overview, Attribution, Channels, ML Segmentation, 
  Predictive ML, Insights
- Swiss editorial design — dark sidebar, light content
- Live model selector, filters, date range, CSV export

## 🔑 Key Findings

- **Search** = highest ROAS (2.013x) → best channel for budget scaling
- **Influencer** = most conversions (2.1M+) → best for awareness
- **Social** = only underperforming channel (ROI < 100%)
- **Linear Attribution** recommended for balanced reporting
- Model divergence across all 5 models = under 3%

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas / NumPy | Data processing |
| scikit-learn | ML models |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |
| Jupyter | Analysis notebook |

## 🚀 How to Run

1. Clone the repository
   git clone https://github.com/yourusername/marketing-attribution-analysis

2. Install dependencies
   pip install -r requirements.txt

3. Run the dashboard
   streamlit run streamlit_dashboard.py

4. Open in browser
   http://localhost:8501

## 📁 Project Structure

├── streamlit_dashboard.py        ← Interactive web dashboard
├── attribution_engine.py         ← Core attribution engine
├── marketing_attribution.ipynb   ← Full analysis notebook  
├── requirements.txt              ← Dependencies
├── README.md                     ← This file
└── marketing_campaign_           ← Dataset
    performance_10000.csv         

## 👤 Author

[BHAVYA JAIN]



