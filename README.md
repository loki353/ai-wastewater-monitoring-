# WATERWISE: AI-Assisted Wastewater Monitoring and Anomaly Detection

Isolation Forest screening of the UCI Water Treatment Plant dataset (ID 106), with an interactive Streamlit dashboard.

## Run locally
```
pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```
On Windows, double-click `START_WATERWISE.bat`. Do not start the dashboard with `python app.py`.

## Deploy on Streamlit Community Cloud
Main file path: `app.py`. The dashboard reads the files in `outputs/`, so keep that folder in the repository.

## Data
Place the original `water-treatment.data` from the UCI page (dataset 106) in `data/` and re-run `python main.py`. `main.py` warns if pH values are physically impossible, which means the file is not the original dataset.

Dataset: Poch, M. (1993). Water Treatment Plant. UCI Machine Learning Repository. https://doi.org/10.24432/C5FS4C (CC BY 4.0).

Anomaly flags are screening aids, not proof of pollution, equipment faults or regulatory violations.
