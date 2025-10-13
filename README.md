# Electro-Thermal Lifetime Assessment for Inverter Semiconductors (IGBTs and Diodes)

### Author  
**Anirudh Katoch**  
PhD Candidate, Technical University of Munich (TUM)  
Chair of Renewable and Sustainable Energy Systems (ENS), CoSES Team  
Supervisor: **Prof. Thomas Hamacher**  
📧 ge26cih@mytum.de  

---

## 🌍 Project Overview  

This repository presents a **Python-based framework** for **electro-thermal lifetime assessment** of inverter semiconductor devices  primarily **IGBTs** and **diodes** under realistic mission profiles.  
The model evaluates the thermal behavior, degradation and lifetime of  grid forming inverters.  (Can also be used for other kinds of inverters.)  

The methodology combines:
- **Thermal and electrical loss modeling** from inverter operation, and  
- **Empirical power cycling lifetime models** derived from laboratory studies.  

---

## 📚 Research Basis  

This work builds on the following two foundational studies:

1. **U. Scheuermann et al. (2014)** — *Power Cycling Testing with Different Load Pulse Durations*,  
   *Proceedings of PEMD 2014*.  
   → Introduces the empirical **lifetime model** for IGBTs and diodes based on power cycling tests, incorporating the effect of temperature swing, mean temperature, pulse duration, and device parameters.

2. **Yunting Liu et al. (2020)** — *Aging Effect Analysis of PV Inverter Semiconductors for Ancillary Services Support*,  
   *IEEE Open Journal of Industry Applications*.  
   → Provides the electro-thermal coupling model for PV inverters under reactive power operation and defines how temperature swings relate to inverter control and mission profiles.


Together, these studies form the **scientific backbone** of this Python implementation.

---

## ⚙️ Model Description  

The model estimates **semiconductor lifetime degradation** in two main stages:

### 1️⃣ Thermal-Electrical Modeling  
- Calculates semiconductor losses (conduction + switching) from inverter operating conditions.  
- Derives **junction temperature (Tj)** and **temperature swing (ΔTj)** using thermal impedance networks and mission profiles.  
- Implements temperature-dependent behavior using device data and ambient conditions.

### 2️⃣ Lifetime Estimation  
- Uses **Scheuermann’s power cycling lifetime model** as discussed in  **U. Scheuermann et al. (2014)**  which is updated LESIT model. 


## 🧩 Repository Structure  

Electro-Thermal-Lifetime-Assessment/
│
├── data/ # Input mission profiles and inverter parameters
├── models/ # Python modules for electro-thermal & lifetime modeling
├── results/ # Output figures and computed lifetimes
├── notebooks/ # Example Jupyter notebooks for analysis
├── requirements.txt # Python dependencies
├── LICENSE
└── README.md

---

## 💻 Environment Setup  

Developed and tested with **Python 3.13.7**.

To set up the environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate     # (on Windows)
# or: source venv/bin/activate (on macOS/Linux)

pip install -r requirements.txt