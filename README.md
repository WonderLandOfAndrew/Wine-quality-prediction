# Wine Quality Prediction using Deep Learning

This repository contains a deep learning project focused on predicting red wine quality
based on physicochemical properties using a Multilayer Perceptron (MLP).

## Repository Structure

- **model-mlp branch**
  - `wine_quality_mlp.ipynb`
  - Data preprocessing, model training, evaluation, and experiments

- **frontend-streamlit branch**
  - `app.py`
  - `requirements.txt`
  - Streamlit-based user interface for model inference

## Dataset

The project uses the Wine Quality Red dataset introduced by Cortez et al. (2009),
sourced from Kaggle:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

## Model Overview

- Custom Multilayer Perceptron (MLP)
- Binary classification:
  - Quality ≥ 7 → Good
  - Quality < 7 → Not Good
- Feature normalization and class balancing applied

## Frontend

A Streamlit application allows users to input physicochemical wine attributes
and obtain a quality prediction along with a confidence score.

## How to Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py

