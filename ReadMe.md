🍕 Domino’s Geo-Spatial Delivery Optimization
Automated Decision Making System for Last-Mile Operations

A stochastic simulation + interactive UI system for optimizing food delivery operations under uncertainty using real road networks.

🚀 Overview

This project simulates a city-scale delivery network where:

Demand is generated dynamically (not user-input)
Stores operate under capacity constraints
Deliveries are stochastic (uncertain travel time)
Orders are assigned using:
Nearest Store (baseline)
Probabilistic Optimization (cost + SLA aware)

The system enables real time store-order assignment with better load balancing, and SLA performance.

🧠 Key Features

📍 Geo-Spatial Engine
Road network via OSMnx
Orders sampled on real road graph
Network-based shortest path distances

⚙️ Simulation Engine
Event-driven architecture
Order lifecycle:
Creation → Assignment → Queue → Prep → Delivery
Poisson-based demand generation

🏬 Store Model
Item-based capacity:
Main items (oven constrained)
Side items (parallelizable)
Queue-driven delays (no order rejection)

🚚 Delivery Model
Distance-based travel time
Stochastic variability (traffic, delays)
SLA target: 30 minutes

🧩 Assignment Strategies
1. Nearest Store
Assign based on shortest distance
2. Optimized Strategy

Minimizes:
Cost + λ × (1 – P(on-time))

Subject to:
P(on-time) ≥ threshold

✅ Balances cost & reliability
✅ Prevents store overload
✅ Improves SLA adherence

🖥️ Streamlit UI

Interactive decision interface built with Streamlit

Features
Configure:
Demand scenarios
SLA constraints
Simulation horizon
Add stores dynamically on map

Visualization:
Orders
Assignments
Delivery outcomes
Map Color Coding
🔵 Created
🟢 < 25 min
🟡 25–30 min
🔴 > 30 min

⚡ Performance Design
Graph Handling
Preloaded .graphml (no API calls)
Cached using st.cache_resource
Hybrid Optimization
Store distances computed lazily per store
Cached per node → avoids repeated Dijkstra
Event Architecture
Engine emits events
UI consumes via queue.Queue
Thread-safe updates

📊 Outputs & Insights
SLA success rate
Average delivery time
Store utilization
Queue lengths
Cost per order
Insights Generated
Overloaded stores
SLA failure regions
Store placement effectiveness
Cost vs service trade-offs

🧪 Use Cases
Automatic Decision Making System
Store placement optimization
Demand surge simulation
Capacity planning

⚙️ Setup
pip install -r requirements.txt
streamlit run app.py

☁️ Deployment
Deployed via Streamlit Cloud
Graph file shipped with repo
No external API dependency
Fully reproducible simulation

🧭 Future Enhancements
Dine in/ Takeaway amalgamation
Multi-city simulation
Fleet constraints
Advanced visualization (heatmaps, analytics)

🏁 Summary
This project is a:

Geo-spatial, stochastic decision-support system
for optimizing last-mile delivery under uncertainty

Combining:

Simulation
Optimization
Real-world road networks
Interactive visualization