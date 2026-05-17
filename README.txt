# 📊 Olist Sales Intelligence & Inventory Suite

An enterprise-grade forecasting dashboard designed to help e-commerce managers minimize stockouts and optimize inventory turnover. Built using the **Olist Brazilian E-commerce dataset**.

## 🚀 Live Demo
[https://olist-demand-forecastin-dash.streamlit.app/]

## 🛠️ The Challenge
Managing inventory for a large-scale marketplace is complex. Overstocking leads to tied-up capital, while stockouts lead to lost revenue. This project provides a **Decision Support System (DSS)** to automate reorder points based on machine learning predictions.

## 🧠 Technical Highlights
- **Forecasting Engine:** Facebook Prophet with custom hyperparameter tuning (`changepoint_prior_scale=0.5`).
- **Data Engineering:** Handled log-transformations (`np.log1p` / `np.expm1`) to stabilize high-variance e-commerce sales data.
- **Inventory Logic:** Integrated Safety Stock and Reorder Point (ROP) formulas using statistical Z-scores and supplier lead-time variables.
- **Interactive UI:** A sleek Dark-themed dashboard built with Streamlit and Plotly for real-time sensitivity analysis.

## 💻 Tech Stack
- **Language:** Python 3.11
- **ML Model:** FB Prophet
- **Dashboard:** Streamlit
- **Visualization:** Plotly (Interactive Charts)
- **Deployment:** Streamlit Cloud

## 📦 Installation & Setup
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/your-repo-name.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---
**Developed by Ali Khalil** *Data Science & AI Student | Portfolio Project*
