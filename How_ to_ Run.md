# Phishing Data Detection Project

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate: `.\venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

## Project Structure
- `data/`: Contains raw CSV data.
- `src/`: Modular Python scripts for preprocessing, modeling, and evaluation.
- `models/`: Stored trained `.joblib` files.
- `run_pipeline.py`: The main entry point to clean, train, and evaluate.

## How to Run
Run the full pipeline using:
```bash
python run_pipeline.py