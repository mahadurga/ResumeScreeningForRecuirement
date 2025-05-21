# AI Resume Screening

## Overview

This project implements a machine learning system to automate resume screening, helping recruiters efficiently decide whether to **Hire** or **Reject** candidates. The model combines structured candidate data (experience, education, certifications, etc.) with unstructured textual data extracted from skills using TF-IDF vectorization. A Random Forest classifier predicts recruiter decisions based on this combined information.

---

## Features

- Preprocessing of candidate data including label encoding of categorical features.
- TF-IDF vectorization of skills text for feature extraction.
- Random Forest classification with tunable complexity.
- Baseline comparison using a Dummy classifier predicting the majority class.
- Upload and process resumes in PDF or TXT format for live predictions.
- Evaluation includes accuracy, classification report, and confusion matrix.
- Handles imbalanced data and explores trade-offs between model simplicity and accuracy.

---

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `PyMuPDF`

Install dependencies via pip:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn PyMuPDF
