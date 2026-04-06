# 🌍 Air Quality Index (AQI) Forecasting

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

## 📝 Overview

This repository contains a deep learning-based model designed to forecast the **Air Quality Index (AQI)** using historical environmental and pollutant data. The project leverages **TensorFlow/Keras** to build predictive models that can help in environmental monitoring and public health awareness.

## 🚀 Getting Started

### 🛠 Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 🐍 Environment Setup

It is highly recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source .venv/bin/activate
```

### 📦 Installation

To ensure you have exactly the same environment as the rest of the team:

```bash
# Recommended: Install from the exact lock file
pip install -r requirements-lock.txt

# Or, if you want just the top-level requirements:
pip install -r requirements.txt
```

## 📊 Project Structure

- `data.ipynb`: 🧪 The main Jupyter notebook used for data preprocessing, model exploration, and visualization.
- `requirements.txt`: 📄 Top-level libraries required (Pinned to secure versions).
- `requirements-lock.txt`: 🔒 Exact snapshot of all dependencies for reproducibility.
- `pyproject.toml`: ⚙️ Configuration for Python tools (Ruff).
- `run-venv.txt`: 📜 Quick reference for virtual environment setup instructions.
- `data/`: 📁 _(Git Ignored)_ Directory for local datasets (CSV, JSON, etc.).

## 🛠 Development & Status Checks

To maintain high code quality and consistency across the team, we use several automated "status check" tools.

### 📦 Setup Developer Tools

Since we use **pnpm** for managing developer tools:

```bash
pnpm install
```

### ✨ Code Quality Tools

- **Ruff**: 🐍 Lightning-fast Python linting and formatting (Handles `.py` and `.ipynb` files).
- **Prettier**: 🎨 Formats code (Markdown, JSON, etc.) automatically.
- **cspell**: 🔤 Checks for spelling errors in comments and documentation.
- **Lefthook**: 🛡️ Orchestrates pre-commit hooks. It will automatically run checks when you `git commit`.

### ⚡ Manual Commands
For manual checks, you can run:
```bash
pnpm format:py   # Formats Python code
pnpm lint:py     # Lints Python code
pnpm format      # Formats Markdown/JSON
pnpm spellcheck  # Checks spelling
```

### 🧪 Running Tests

If you're building a web interface or API tests, use **Playwright**:

```bash
pnpm test
```

## 📜 Citation
Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49.
