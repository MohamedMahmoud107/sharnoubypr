# LEO / El-Sharnouby Business Analytics & Optimization Dashboard

A Streamlit dashboard that brings together sales analytics, customer
segmentation, SKU-level demand forecasts, inventory planning, and
production-optimization outputs for the LEO / El-Sharnouby case study.

> **Data note:** This project uses a mixture of public business facts
> and modeled/synthetic or anonymized data for analysis and
> demonstration. Results shown by the dashboard should not be
> interpreted as verified actual company performance.

## Features

### 1. Executive Insights

-   Presents high-level business KPIs from the available sales data.
-   Displays sales and profit metrics only when the required source
    columns are available.
-   Provides visual summaries of business performance.

### 2. Customer Segmentation

-   Summarizes customer clusters using median monetary value, frequency,
    and recency.
-   Reports customer counts and each cluster's percentage of the
    customer dataset.
-   Visualizes average monetary value by cluster.
-   Allows the segmentation statistics to be downloaded.
-   Supports an optional CSV upload for segment prediction when a
    compatible saved clustering model and preprocessing objects are
    available.

### 3. Demand Forecasting

-   Explores historical monthly demand by SKU.
-   Lets the user select a SKU and forecast month.
-   Uses the saved SKU forecasting model when the model's expected
    feature pipeline is available.
-   Can display the prepared monthly forecast outputs from
    `Monthly_Forecasts.csv`.

### 4. Inventory Planning

-   Displays the prepared inventory-planning dataset.
-   Supports SKU/month filtering and highlights forecast demand and
    inventory-shortage information where those columns exist.
-   Allows the displayed planning data to be downloaded.

### 5. Production Optimization

-   Displays the production-optimization results and summary.
-   Helps review recommended production quantities, requirements, and
    capacity-related metrics available in the prepared output files.
-   Allows the results to be downloaded.

## Project structure

The application expects the data and saved models to be organized like
this:

``` text
project/
├── app.py
├── streamlit_data/
│   ├── Sales_Transactions.csv
│   ├── Customer_Segmentation.csv
│   ├── Monthly_Forecasts.csv
│   ├── Inventory_Planning.csv
│   ├── Optimization_Summary.csv
│   └── Production_Optimization.csv
└── saved_models/
    ├── final_sku_xgboost_model.joblib
    └── ... preprocessing objects and clustering model, if used
```

Keep the filenames consistent with the paths configured in `app.py`. The
app can only display sections for which the corresponding files are
available.

## Requirements

-   Python 3.10 or newer is recommended.
-   Streamlit
-   pandas
-   NumPy
-   Plotly
-   joblib
-   scikit-learn / XGBoost, if required by the saved models

Install the main dashboard dependencies:

``` bash
pip install streamlit pandas numpy plotly joblib scikit-learn xgboost
```

## Run the dashboard

Open a terminal in the project directory and run:

``` bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually
`http://localhost:8501`. Open that URL in your browser.

## Expected input files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `Sales_Transactions.csv`            Historical sales transactions used
                                      for business insights and
                                      historical demand exploration.

  `Customer_Segmentation.csv`         Existing customer-level segment
                                      assignments and features used for
                                      cluster summaries.

  `Monthly_Forecasts.csv`             Prepared monthly SKU-level forecast
                                      results.

  `Inventory_Planning.csv`            Prepared inventory-planning results
                                      combining demand and
                                      inventory-related information.

  `Optimization_Summary.csv`          High-level summary of the
                                      production-optimization run.

  `Production_Optimization.csv`       SKU-level production-optimization
                                      recommendations.
  -----------------------------------------------------------------------

The saved SKU forecasting model is expected at:

``` text
saved_models/final_sku_xgboost_model.joblib
```

Additional clustering models and preprocessing objects may be needed for
predicting segments for newly uploaded customer records. Uploaded
customer data must follow the feature schema and preprocessing used when
the clustering model was trained.

## Important limitations

-   The data is for a case study/demo and may include synthetic or
    modeled records. Do not present the outputs as audited company
    results.
-   Forecast and optimization results depend on the quality, time
    coverage, assumptions, and feature pipeline of their source data and
    models.
-   The inventory snapshot timing and some inventory reconciliation
    differences have not been fully resolved. These are data-quality
    limitations and should be documented rather than silently corrected.
-   `Purchase_Orders.csv` is at the raw-material level (`Material_ID`);
    its quantities should not be added directly to finished-product SKU
    inventory.
-   A saved model alone may not be sufficient to generate a new
    forecast: the application must provide the same feature engineering
    and preprocessing used during training.
-   Profit metrics should only be labeled **net profit** if a valid
    net-profit field or a defensible net-profit calculation is present.
    Gross profit must not be relabeled as net profit.

## Suggested next steps

-   Verify that the configured file paths and column names match the
    actual CSV files.
-   Keep model training, preprocessing, and inference features
    consistent.
-   Validate forecasts against a held-out period and track suitable
    error metrics.
-   Review production and inventory recommendations with the assumptions
    and capacity constraints documented.
