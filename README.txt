# Sales Prediction using XGBoost

A Streamlit web application that uses a saved XGBoost regression model to estimate sales from six business inputs.

## Included files
- `app.py` — Streamlit web application
- `sales_prediction_dataset.csv` — synthetic dataset (1,000 rows, 8 columns)
- `sales_xgboost_model.pkl` — trained model/pipeline
- `requirements.txt` — Python dependencies

## Run locally (Windows / VS Code)
Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud
1. Create a **public** GitHub repository.
2. Upload all four files listed above plus this README.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
5. Deploy. If deployment reports a dependency or model-loading error, use the full error message to troubleshoot.

## Reported model evaluation
- MAE: 2527.35
- RMSE: 3202.70
- R²: 0.9540

These metrics are the supplied project results. The dataset is synthetic, so they should not be interpreted as verified real-world forecasting performance.
