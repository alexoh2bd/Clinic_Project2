# Clinic Project – Predicting Clinical Trial Success

This project aims to **predict the probability of success of clinical trials** using data from ClinicalTrials.gov. Following course requirements, our group experimented with multiple data sources, labeling strategies, feature pipelines, and model families.

---

## 📂 Repository Structure

```

.
├── Baseline.ipynb               # Text-based TF-IDF + Logistic Regression baseline
├── structured_model.ipynb       # Structured-feature Random Forest model
├── trials_data_students.ipynb   # XML parsing pipeline for extracting trial info
├── phases.npy                   # Phase category mapping
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore

```

---

# 1. Project Objective

**Goal:**  
Build a machine learning system that predicts whether a clinical trial will succeed.

**Definition of success:**  
Trials labeled as **COMPLETED** are treated as positive outcomes; all others are negative.

---

# 2. Data Sources

We explored the two official project data options:

### **1. ClinicalTrials.gov XML Dataset**
- Parsed XML fields such as baseline characteristics, participant flow, arms/interventions, metrics, and analyzed results.  
- Implemented in `trials_data_students.ipynb`.

### **2. Processed CSV Dataset**
- A curated structured dataset containing ~559k trials.  
- Includes fields like:  
  `NCT ID`, `Study Status`, `Phase`, `Enrollment`, `Conditions`, `Interventions`, `Dates`, etc.  
- Used in `structured_model.ipynb`.

---

# 3. Labeling Approach

We used an existing outcome label derived from **Study Status**:

```

Success (1) = "COMPLETED"
Failure (0) = all other statuses

```

This approach ensures consistency across text, structured, and XML pipelines.

---

# 4. Data Pipeline

### **Text Processing (Baseline.ipynb)**
- Concatenate textual fields (brief summary, description, conditions)
- Clean and normalize text
- Convert to TF-IDF sparse features

### **Structured Pipeline (structured_model.ipynb)**
- Parse attributes such as enrollment, phase, type, sponsor  
- Encode categorical fields  
- Generate numeric features (duration, number of conditions/interventions, location count)  
- Train/test split

### **XML Extraction (trials_data_students.ipynb)**
- Convert ClinicalTrials.gov XML into flat rows with:
  - Participant flow  
  - Baseline characteristics  
  - Outcome measures  
  - Milestones and drop/withdraw details  
- Generate structured tables for ML

---

# 5. Models Implemented

### **1. Text Baseline – Logistic Regression**
- TF-IDF vectorization  
- Fast & interpretable  
- Notebook: `Baseline.ipynb`

### **2. Structured Model – Random Forest**
- Handles heterogeneous feature types  
- Captures non-linear feature interactions  
- Notebook: `structured_model.ipynb`

### **3. XML Pipeline – Feature Extraction**
- Extracts metadata and patient flow  
- Prepares raw XML for future modeling  
- Notebook: `trials_data_students.ipynb`

---

# 6. Evaluation

**Metrics used:**
- Accuracy  
- Precision / Recall / F1 Score  
- ROC-AUC  

All models share the same train-test split for fair comparison.

---

# 7. Key Findings

- Text-only models capture high-level semantic signals.  
- Structured features provide strong predictive power (duration, phase, sample size).  
- XML parsing enables richer feature options for future extensions.  
- Combining structured + text features is a promising direction.

---

# 8. How to Run

Install dependencies:

```

pip install -r requirements.txt

```

Launch notebooks:

```

jupyter notebook Baseline.ipynb
jupyter notebook structured_model.ipynb
jupyter notebook trials_data_students.ipynb

```

---

# 9. Team Contributions

### **Baseline Model**
- Text cleaning  
- TF-IDF feature construction  
- Logistic Regression baseline

### **Structured Model**
- CSV preprocessing  
- Feature engineering  
- Random Forest classifier

### **XML Pipeline**
- Raw XML parsing  
- Metadata extraction  
- Conversion to analysis-ready format

---

# 10. Future Work

- Combine structured + text features  
- Train Gradient Boosting models  
- Explore time-to-event or survival modeling  
- Build an integrated “probability of success” predictor using all available signals

