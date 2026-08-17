# CardioAI Prototype

A small local Streamlit + scikit-learn school-project prototype.

## What it does

1. Loads a synthetic CSV dataset from `data/health_data.csv`.
2. Trains a Random Forest classifier.
3. Takes six user inputs:
   - Age
   - Total cholesterol
   - Systolic blood pressure
   - Blood glucose
   - BMI
   - Resting heart rate
4. Predicts a demo risk class: Low / Moderate / High.
5. Loads a second CSV (`data/recommendations.csv`) and displays personalized educational prevention suggestions.
6. Generates a matplotlib chart.

## No server required

The CSV files are acting as your local data source. Nothing needs to be hosted online.

## Run in Visual Studio Code

Open the project folder in VS Code and use the terminal:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will open the app in your browser.

You can also double-click `run_app.bat` on Windows after Python is installed.

## Important scientific limitation

The included health rows are synthetic demonstration data. The model is for demonstrating a machine-learning pipeline, not for medical diagnosis.

The project also avoids claiming that Neem, Peepal, or another plant can directly lower blood pressure, cholesterol, glucose, or heart rate. Plant suggestions are presented as environmental/biodiversity ideas.

For a stronger science-project version, replace the synthetic dataset with a properly documented, validated cardiovascular dataset and cite the data source and clinical definitions.
