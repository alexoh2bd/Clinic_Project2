# Clinic Project – Predicting Clinical Trial Success

This project aims to **predict the probability of success of clinical trials** using data from ClinicalTrials.gov. We experimented with multiple data sources, labeling strategies, feature pipelines, and model families, focusing on three primary approaches: a text-based baseline, a structured data model, and a deep learning ClinicalBERT model.

---

## 📂 Repository Structure

```
.
├── Base.ipynb                   # Text-based TF-IDF + Logistic Regression/LightGBM baseline
├── ClinicalBert.ipynb           # Deep learning model using ClinicalBERT embeddings
├── structured_model.ipynb       # Random Forest model using structured features
├── trials_data_students.ipynb   # XML parsing pipeline helper
├── phases.npy                   # Phase category mapping
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore
```

---

# 1. Project Objective

**Goal:**
Build a machine learning system that predicts whether a clinical trial will succeed based on its characteristics and description.

**Definition of Success:**
*   **Base & ClinicalBert**: Trials labeled as **COMPLETED**, having **Clinical Results**, and in **Phase 2/3, 3, or 4**.
*   **Structured Model**: Trials labeled as **COMPLETED** and having **Published Results**.

---

# 2. Data Sources & Pipelines

### **1. XML Dataset (Used in `Base.ipynb` & `ClinicalBert.ipynb`)**
*   **Source**: Raw XML files from ClinicalTrials.gov.
*   **Pipeline**:
    *   Parses XML to extract text fields: `brief_summary`, `detailed_description`, `interventions`, `primary_outcomes`.
    *   Filters for trials with valid phases.
    *   **Base**: Converts text to TF-IDF sparse features (up to 50k features).
    *   **ClinicalBert**: Tokenizes text for BERT input.

### **2. Processed CSV Dataset (Used in `structured_model.ipynb`)**
*   **Source**: `raw_data/ctg-studies.csv` containing ~559k trials.
*   **Pipeline**:
    *   Loads CSV and performs feature engineering.
    *   Encodes categorical variables (Phase, Study Type, Sex).
    *   Handles missing values and generates numeric counts.

---

# 3. Features

| Model | Feature Type | Key Features |
|-------|--------------|--------------|
| **Base** | Textual (Sparse) | TF-IDF vectors from summaries, descriptions, interventions, and outcomes. |
| **Structured** | Numeric & Categorical | `enrollment`, `num_conditions`, `num_interventions`, `num_locations`, `num_collaborators`, `phase`, `study_type`, `sex`. |
| **ClinicalBert** | Embeddings (Dense) | Contextual embeddings learned from `medicalai/ClinicalBERT` on trial descriptions. |

---

# 4. Models & Evaluation

We evaluated models using **Accuracy**, **ROC-AUC**, **Precision**, **Recall**, and **F1-Score**.

### **1. Text Baseline (`Base.ipynb`)**
*   **Models**: Logistic Regression, LightGBM (with SVD and raw TF-IDF).
*   **Performance**:
    *   **Accuracy**: ~0.806
    *   **Precision**: ~0.611
    *   **Recall**: ~0.449
    *   **F1 Score**: ~0.52
    *   **ROC-AUC**: ~0.85 (LightGBM)
*   **Key Insight**: High-dimensional text features provide a strong baseline, capturing semantic signals about the trial's focus.

### **2. Structured Model (`structured_model.ipynb`)**
*   **Model**: Random Forest Classifier (with balanced class weights).
*   **Performance**:
    *   **Accuracy**: ~0.687
    *   **Precision**: ~0.282
    *   **Recall**: ~0.928
    *   **F1 Score**: ~0.432
    *   **ROC-AUC**: ~0.877
*   **Key Insight**: Structured features like enrollment size and number of locations are predictive but less discriminative than full text descriptions.

### **3. ClinicalBERT (`ClinicalBert.ipynb`)**
*   **Model**: Fine-tuned `medicalai/ClinicalBERT` (Transformer).
    *   Fine-tuned top 4 layers of BERT + Classification Head.
*   **Performance**:
    *   **Accuracy**: ~0.76
    *   **ROC-AUC**: ~0.72
    *   **Precision**: ~0.46
    *   **Recall**: ~0.28
    *   **F1 Score**: ~0.35
*   **Key Insight**: Deep learning captures complex semantic relationships but requires significant compute and careful tuning.

---

# 5. How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Launch Notebooks**:
    *   For the baseline: `jupyter notebook Base.ipynb`
    *   For structured data: `jupyter notebook structured_model.ipynb`
    *   For ClinicalBERT: `jupyter notebook ClinicalBert.ipynb` (Requires GPU recommended)

---

# 6. Future Work

*   **Ensemble Approaches**: Combine the strong text baseline with structured features.
*   **Advanced NLP**: Experiment with larger Transformer models or different fine-tuning strategies.
*   **Survival Analysis**: Model the time-to-completion rather than just binary success.
