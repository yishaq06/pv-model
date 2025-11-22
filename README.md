# :earth_americas: GDP dashboard template

A simple Streamlit app showing the GDP of different countries in the world.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

```
pv-model
├─ .devcontainer
│  └─ devcontainer.json
├─ LICENSE
├─ README.md
├─ data
│  ├─ MODEL_DEV_DATA_SHEET.xlsx
│  └─ gdp_data.csv
├─ models
│  ├─ carbon_model.py
│  ├─ lcoe_model.py
│  ├─ savings_model.py
│  └─ system_size_model.py
├─ pv_model_outputs
│  ├─ cleaned_pv_dataset.csv
│  ├─ models_summary.csv
│  ├─ rf_25yr_savings.pkl
│  ├─ rf_capacity.pkl
│  ├─ rf_co2.pkl
│  └─ rf_lcoe.pkl
├─ requirements.txt
├─ streamlit_app.py
├─ train_pv_models.py
└─ utils
   ├─ charts.py
   └─ preprocessing.py

```