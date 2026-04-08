# 🍕 Dominos Automated Decision Making System

A geo-spatial, stochastic simulation system for optimizing last-mile food delivery using real road networks and probabilistic decision-making.

---

## 🚀 Overview

This project simulates a city-scale delivery network where:

- 📦 Demand is generated dynamically (Poisson-based)
- 🏬 Stores operate under capacity constraints
- 🚚 Delivery time is stochastic (real-world uncertainty)
- 🧠 Orders are assigned using intelligent strategies

---

## 🧩 Assignment Strategies

### 1. Nearest Store
- Assigns orders based on shortest distance

### 2. Optimized Strategy
Minimizes: Cost + λ × (1 – P(on-time))
Subject to SLA constraints: P(on-time) ≥ threshold


✅ Balances cost & reliability  
✅ Reduces overload  
✅ Improves SLA performance  

---

## 🖥️ Interactive UI

Built with Streamlit for real-time decision support:

- 📍 Add stores dynamically on map  
- ⚙️ Configure demand & SLA  
- 📊 Live simulation updates  
- 🗺️ Visual delivery tracking  

### Map Legend
- 🔵 Created  
- 🟢 < 25 min  
- 🟡 25–30 min  
- 🔴 > 30 min  

---

## ⚡ System Design

- 🗺️ Road network via OSMnx  
- ⚙️ Event-driven simulation engine  
- 🧵 Thread-safe UI updates (queue-based)  
- 💾 Cached graph & distance computations  

---

## 📊 Outputs
Strategy wise
- SLA success rate  
- Average delivery time  
- Store utilization  
- Queue delays   

---

## ⚙️ Run Locally

pip install -r requirements.txt
streamlit run app.py

☁️ Deployment

Ready for Streamlit Cloud:
- Preloaded graph (no API calls)
- Cached computations
- Fast & reproducible runs
