
from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LEO Business Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path("streamlit_data")
MODEL_DIR = Path("saved_models")

SALES_PATH = DATA_DIR / "Sales_Transactions.csv"
CUSTOMER_SEGMENTATION_PATH = DATA_DIR / "Customer_Segmentation.csv"
FORECAST_PATH = DATA_DIR / "Monthly_Forecasts.csv"
INVENTORY_PATH = DATA_DIR / "Inventory_Planning.csv"
OPTIMIZATION_SUMMARY_PATH = DATA_DIR / "Optimization_Summary.csv"
PRODUCTION_OPTIMIZATION_PATH = DATA_DIR / "Production_Optimization.csv"

FORECAST_MODEL_PATH = MODEL_DIR / "final_sku_xgboost_model.joblib"

KMEANS_MODEL_PATHS = [
    MODEL_DIR / "KMeans_model.pkl",
    MODEL_DIR / "KMeans_model.pickle",
    MODEL_DIR / "KMeans_model",
]

SCALER_PATHS = [
    MODEL_DIR / "scaler.pkl",
    MODEL_DIR / "scaler.pickle",
    MODEL_DIR / "scaler.joblib",
    MODEL_DIR / "standard_scaler.pkl",
    MODEL_DIR / "standard_scaler.joblib",
]

CUSTOMER_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Average_Order_Value",
    "Purchase_Regularity",
]

SEGMENT_NAMES = {
    0: "High-Value, Frequent Customers",
    1: "One-Time, Low-Spending Customers",
    2: "Occasional, Moderate-Spending Customers",
    3: "Recently Active Customers",
}


# =========================================================
# 2. SHARED FUNCTIONS
# =========================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_resource
def load_model(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        if path.suffix.lower() in [".pkl", ".pickle"]:
            with open(path, "rb") as file:
                return pickle.load(file)

        return joblib.load(path)

    except Exception:
        return None


def load_first_existing(paths):
    for path in paths:
        if path.exists():
            return load_model(str(path)), path

    return None, None


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(col).strip().replace(" ", "_")
        for col in df.columns
    ]
    return df


def find_column(df, candidates):
    if df is None:
        return None

    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


def parse_month_column(df):
    df = normalize_columns(df)

    month_col = find_column(
        df,
        ["Forecast_Month", "Month", "Snapshot_Month"],
    )

    if month_col:
        df[month_col] = pd.to_datetime(
            df[month_col],
            errors="coerce",
        )

    return df


def format_egp(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f} EGP"


def show_missing_file(path):
    st.error(f"File not found: `{path}`")


# =========================================================
# 3. LOAD PROJECT DATA
# =========================================================

sales = load_csv(str(SALES_PATH))
customers = load_csv(str(CUSTOMER_SEGMENTATION_PATH))
monthly_forecasts = load_csv(str(FORECAST_PATH))
inventory = load_csv(str(INVENTORY_PATH))
optimization_summary = load_csv(str(OPTIMIZATION_SUMMARY_PATH))
production_optimization = load_csv(str(PRODUCTION_OPTIMIZATION_PATH))

if sales is not None:
    sales = normalize_columns(sales)

    sales_date_col = find_column(
        sales,
        ["Order_Date", "Invoice_Date", "Date"],
    )

    if sales_date_col:
        sales[sales_date_col] = pd.to_datetime(
            sales[sales_date_col],
            errors="coerce",
        )

if customers is not None:
    customers = normalize_columns(customers)

if monthly_forecasts is not None:
    monthly_forecasts = parse_month_column(monthly_forecasts)

if inventory is not None:
    inventory = parse_month_column(inventory)

if production_optimization is not None:
    production_optimization = parse_month_column(
        production_optimization
    )

if optimization_summary is not None:
    optimization_summary = normalize_columns(
        optimization_summary
    )


# =========================================================
# 4. SIDEBAR
# =========================================================

st.sidebar.title("LEO Business Analytics")

page = st.sidebar.radio(
    "Navigate to",
    [
        "Insights",
        "Customer Segmentation",
        "Demand Forecasting",
        "Inventory Planning",
        "Production Optimization",
    ],
)

st.sidebar.divider()
st.sidebar.caption("LEO Business Decision Support System")


# =========================================================
# PAGE 1: BUSINESS INSIGHTS
# =========================================================

if page == "Insights":

    st.title("Business Insights")
    st.caption("Sales performance and profitability")

    if sales is None:
        show_missing_file(SALES_PATH)
        st.stop()

    date_col = find_column(
        sales,
        ["Order_Date", "Invoice_Date", "Date"],
    )

    invoice_col = find_column(
        sales,
        ["Invoice_ID", "Order_ID", "Transaction_ID"],
    )

    sku_col = find_column(
        sales,
        ["SKU", "Product_ID", "Product_Name"],
    )

    location_col = find_column(
        sales,
        ["Location_ID", "Location_Name", "Branch"],
    )

    channel_col = find_column(
        sales,
        ["Channel", "Sales_Channel"],
    )

    quantity_col = find_column(
        sales,
        ["Quantity", "Sold_Qty", "Sales_Quantity"],
    )

    profit_col = find_column(
        sales,
        ["Gross_Profit_EGP", "Gross_Profit", "Profit"],
    )

    net_sales_col = find_column(
        sales,
        ["Net_Sales_EGP", "Net_Sales"],
    )

    unique_orders = (
        sales[invoice_col].nunique()
        if invoice_col else 0
    )

    unique_products = (
        sales[sku_col].nunique()
        if sku_col else 0
    )

    total_profit = (
        pd.to_numeric(sales[profit_col], errors="coerce").sum()
        if profit_col else 0
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Unique Orders", f"{unique_orders:,}")
    k2.metric("Unique Products", f"{unique_products:,}")
    k3.metric("Business Locations", "4")
    k4.metric("Total Gross Profit", format_egp(total_profit))

    st.divider()

    # Annual profitability
    if date_col and profit_col and net_sales_col:

        annual = sales.copy()

        annual["Year"] = annual[date_col].dt.year

        annual[profit_col] = pd.to_numeric(
            annual[profit_col],
            errors="coerce",
        )

        annual[net_sales_col] = pd.to_numeric(
            annual[net_sales_col],
            errors="coerce",
        )

        annual_summary = (
            annual.groupby("Year", as_index=False)
            .agg(
                Gross_Profit=(profit_col, "sum"),
                Net_Sales=(net_sales_col, "sum"),
            )
        )

        annual_summary["Gross_Margin_Pct"] = np.where(
            annual_summary["Net_Sales"] != 0,
            annual_summary["Gross_Profit"]
            / annual_summary["Net_Sales"] * 100,
            np.nan,
        )

        st.subheader("Annual Profitability")

        st.dataframe(
            annual_summary.round(2),
            use_container_width=True,
            hide_index=True,
        )

    # Gross profit by channel
    if channel_col and profit_col:

        channel_data = sales.copy()

        channel_data[profit_col] = pd.to_numeric(
            channel_data[profit_col],
            errors="coerce",
        )

        channel_profit = (
            channel_data.groupby(channel_col, as_index=False)
            .agg(Gross_Profit=(profit_col, "sum"))
            .sort_values("Gross_Profit", ascending=False)
        )

        fig = px.bar(
            channel_profit,
            x=channel_col,
            y="Gross_Profit",
            title="Gross Profit by Channel",
            labels={
                channel_col: "Sales Channel",
                "Gross_Profit": "Gross Profit (EGP)",
            },
            text_auto=".2s",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Seasonal pattern across latest available years
    if date_col and quantity_col:

        seasonal_data = sales.dropna(
            subset=[date_col]
        ).copy()

        seasonal_data[quantity_col] = pd.to_numeric(
            seasonal_data[quantity_col],
            errors="coerce",
        )

        seasonal_data = seasonal_data.dropna(
            subset=[quantity_col]
        )

        seasonal_data["Year"] = (
            seasonal_data[date_col].dt.year
        )

        seasonal_data["Month_Number"] = (
            seasonal_data[date_col].dt.month
        )

        years = sorted(
            seasonal_data["Year"].dropna().unique()
        )[-3:]

        seasonal_data = seasonal_data[
            seasonal_data["Year"].isin(years)
        ]

        monthly = (
            seasonal_data.groupby(
                ["Year", "Month_Number"],
                as_index=False,
            )
            .agg(Quantity=(quantity_col, "sum"))
        )

        index = pd.MultiIndex.from_product(
            [years, range(1, 13)],
            names=["Year", "Month_Number"],
        )

        monthly = (
            monthly.set_index(["Year", "Month_Number"])
            .reindex(index)
            .reset_index()
        )

        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        fig = px.line(
            monthly,
            x="Month_Number",
            y="Quantity",
            color="Year",
            markers=True,
            title="Monthly Sales Quantity Across the Latest 3 Years",
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=month_names,
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Quantity Sold",
        )

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE 2: CUSTOMER SEGMENTATION
# =========================================================

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")
    st.caption(
        "Segment statistics, customer distribution, "
        "and monetary contribution"
    )

    # Load existing segmentation results automatically.
    if customers is None:
        show_missing_file(CUSTOMER_SEGMENTATION_PATH)
        st.stop()

    segment_col = find_column(
        customers,
        ["Segment", "Segment_Name", "Customer_Segment", "Cluster_Name"],
    )

    cluster_col = find_column(
        customers,
        ["Cluster", "Cluster_ID", "Cluster_Label", "Segment_ID"],
    )

    monetary_col = find_column(
        customers,
        ["Monetary", "Total_Spending", "Total_Spend", "Customer_Value"],
    )

    customer_id_col = find_column(
        customers,
        ["Customer_ID", "CustomerID", "Customer_Code"],
    )

    if segment_col is None and cluster_col is None:
        st.error(
            "The segmentation file must contain a Segment or Cluster column."
        )
        st.write("Available columns:", customers.columns.tolist())
        st.stop()

    if segment_col is None:
        customers["Segment"] = customers[cluster_col].map(
            SEGMENT_NAMES
        ).fillna(
            customers[cluster_col].astype(str).map(
                lambda value: f"Cluster {value}"
            )
        )
        segment_col = "Segment"

    # Use a customer identifier for counts when available.
    if customer_id_col:
        customer_count = (
            customers.groupby(segment_col)[customer_id_col]
            .nunique()
            .rename("Customer_Count")
        )
    else:
        customer_count = (
            customers.groupby(segment_col)
            .size()
            .rename("Customer_Count")
        )

    total_customers = int(customer_count.sum())

    segment_distribution = customer_count.reset_index()

    segment_distribution["Percentage"] = (
        segment_distribution["Customer_Count"]
        / total_customers * 100
        if total_customers else 0
    )

    # Main KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric(
        "Number of Segments",
        f"{customers[segment_col].nunique():,}",
    )

    if monetary_col:
        total_monetary = pd.to_numeric(
            customers[monetary_col],
            errors="coerce",
        ).sum()

        col3.metric(
            "Total Monetary Value",
            format_egp(total_monetary),
        )
    else:
        col3.metric("Total Monetary Value", "N/A")

    st.divider()

    # Segment size and percentage
    st.subheader("Customer Distribution by Segment")

    st.dataframe(
        segment_distribution.round(2),
        use_container_width=True,
        hide_index=True,
    )

    distribution_fig = px.bar(
        segment_distribution,
        x=segment_col,
        y="Percentage",
        text="Percentage",
        title="Percentage of Customers in Each Segment",
        labels={
            segment_col: "Customer Segment",
            "Percentage": "Customers (%)",
        },
    )

    distribution_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    st.plotly_chart(
        distribution_fig,
        use_container_width=True,
    )

    # Statistics and percentiles for available numerical features
    st.subheader("Statistics and Percentiles by Segment")

    numerical_candidates = [
        "Recency",
        "Frequency",
        "Monetary",
        "Average_Order_Value",
        "Purchase_Regularity",
    ]

    numerical_cols = [
        col for col in numerical_candidates
        if col in customers.columns
    ]

    if not numerical_cols:
        numerical_cols = [
            col for col in customers.select_dtypes(
                include=np.number
            ).columns
            if col != cluster_col
        ]

    if numerical_cols:

        stat_rows = []

        for segment_name, group in customers.groupby(
            segment_col,
            dropna=False,
        ):

            for metric in numerical_cols:

                values = pd.to_numeric(
                    group[metric],
                    errors="coerce",
                ).dropna()

                if values.empty:
                    continue

                stat_rows.append({
                    "Segment": segment_name,
                    "Metric": metric,
                    "Count": int(values.count()),
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "Std": values.std(),
                    "Min": values.min(),
                    "P25": values.quantile(0.25),
                    "P50": values.quantile(0.50),
                    "P75": values.quantile(0.75),
                    "P90": values.quantile(0.90),
                    "P95": values.quantile(0.95),
                    "Max": values.max(),
                })

        statistics_table = pd.DataFrame(stat_rows)

        st.dataframe(
            statistics_table.round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "P25, P50, P75, P90, and P95 are distribution percentiles. "
            "Mean, median, standard deviation, minimum, and maximum "
            "are descriptive statistics."
        )

    # Total monetary contribution by segment
    if monetary_col:

        monetary_data = customers.copy()

        monetary_data[monetary_col] = pd.to_numeric(
            monetary_data[monetary_col],
            errors="coerce",
        )

        monetary_summary = (
            monetary_data.groupby(segment_col, as_index=False)
            .agg(
                Total_Monetary=(monetary_col, "sum"),
                Average_Monetary=(monetary_col, "mean"),
            )
            .sort_values("Total_Monetary", ascending=False)
        )

        st.subheader("Monetary Contribution by Segment")

        monetary_fig = px.bar(
            monetary_summary,
            x=segment_col,
            y="Total_Monetary",
            text="Total_Monetary",
            title="Total Monetary Value by Customer Segment",
            labels={
                segment_col: "Customer Segment",
                "Total_Monetary": "Total Monetary Value (EGP)",
            },
        )

        monetary_fig.update_traces(
            texttemplate="%{text:,.2f}",
            textposition="outside",
        )

        st.plotly_chart(
            monetary_fig,
            use_container_width=True,
        )

        monetary_summary["Monetary_Percentage"] = (
            monetary_summary["Total_Monetary"]
            / monetary_summary["Total_Monetary"].sum() * 100
            if monetary_summary["Total_Monetary"].sum() != 0
            else 0
        )

        st.dataframe(
            monetary_summary.round(2),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("View Existing Segmented Customers"):
        st.dataframe(
            customers,
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download Segmentation Statistics",
        data=customers.to_csv(index=False).encode("utf-8"),
        file_name="Customer_Segmentation_View.csv",
        mime="text/csv",
    )

    # Optional prediction for newly uploaded customers
    st.divider()
    st.subheader("Optional: Predict Segments for New Customers")

    uploaded_customers = st.file_uploader(
        "Upload a new customer feature CSV",
        type=["csv"],
        key="new_customer_upload",
    )

    if uploaded_customers is not None:

        new_customers = normalize_columns(
            pd.read_csv(uploaded_customers)
        )

        missing_features = [
            feature for feature in CUSTOMER_FEATURES
            if feature not in new_customers.columns
        ]

        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        else:

            kmeans_model, kmeans_path = load_first_existing(
                KMEANS_MODEL_PATHS
            )

            scaler, scaler_path = load_first_existing(
                SCALER_PATHS
            )

            if kmeans_model is None:

                st.error(
                    "The saved KMeans model was not found. "
                    "Check the model filename and extension."
                )

            else:

                try:

                    X_new = new_customers[
                        CUSTOMER_FEATURES
                    ].copy()

                    X_new = X_new.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )

                    if X_new.isna().any().any():
                        raise ValueError(
                            "The uploaded features contain missing "
                            "or infinite values."
                        )

                    # Use the same preprocessing as training.
                    # If training used log1p or another transform,
                    # reproduce that transform here before scaling.
                    if scaler is not None:
                        X_transformed = scaler.transform(X_new)
                    else:
                        X_transformed = X_new

                    new_customers["Cluster"] = (
                        kmeans_model.predict(X_transformed)
                    )

                    new_customers["Segment"] = (
                        new_customers["Cluster"]
                        .map(SEGMENT_NAMES)
                        .fillna(
                            new_customers["Cluster"].map(
                                lambda value: f"Cluster {value}"
                            )
                        )
                    )

                    st.success(
                        f"Predictions generated using {kmeans_path.name}."
                    )

                    st.dataframe(
                        new_customers,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download New Customer Predictions",
                        data=new_customers.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="New_Customer_Segments.csv",
                        mime="text/csv",
                    )

                except Exception as exc:
                    st.error(
                        "Prediction failed. Confirm that the saved "
                        "model and preprocessing match the training "
                        f"notebook. Details: {exc}"
                    )


# =========================================================
# PAGE 3: DEMAND FORECASTING
# =========================================================

elif page == "Demand Forecasting":

    st.title("Demand Forecasting")
    st.caption("Historical demand and SKU-level forecasting")

    if sales is None:
        show_missing_file(SALES_PATH)
        st.stop()

    date_col = find_column(
        sales,
        ["Order_Date", "Invoice_Date", "Date"],
    )

    quantity_col = find_column(
        sales,
        ["Quantity", "Sold_Qty", "Sales_Quantity"],
    )

    sku_col = find_column(
        sales,
        ["SKU", "Product_ID", "Product_Code"],
    )

    if not all([date_col, quantity_col, sku_col]):
        st.error(
            "Sales data needs date, quantity, and SKU columns."
        )
        st.stop()

    history = sales.dropna(
        subset=[date_col, sku_col, quantity_col]
    ).copy()

    history[quantity_col] = pd.to_numeric(
        history[quantity_col],
        errors="coerce",
    )

    history = history.dropna(subset=[quantity_col])

    history["Year"] = history[date_col].dt.year
    history["Month_Number"] = history[date_col].dt.month
    history["Month_Start"] = (
        history[date_col].dt.to_period("M").dt.to_timestamp()
    )

    years = sorted(history["Year"].dropna().unique())[-3:]

    historical = history[history["Year"].isin(years)]

    monthly = (
        historical.groupby(
            ["Year", "Month_Number"],
            as_index=False,
        )
        .agg(Quantity=(quantity_col, "sum"))
    )

    index = pd.MultiIndex.from_product(
        [years, range(1, 13)],
        names=["Year", "Month_Number"],
    )

    monthly = (
        monthly.set_index(["Year", "Month_Number"])
        .reindex(index)
        .reset_index()
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig = px.line(
        monthly,
        x="Month_Number",
        y="Quantity",
        color="Year",
        markers=True,
        title="Monthly Sales Quantity Across the Latest 3 Available Years",
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=month_names,
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Quantity Sold",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("SKU Forecast Engine")

    sku_options = sorted(
        history[sku_col].dropna().astype(str).unique()
    )

    selected_sku = st.selectbox(
        "Select SKU",
        sku_options,
    )

    latest_month = history["Month_Start"].max()

    if pd.isna(latest_month):
        st.error("No valid historical month found.")
        st.stop()

    next_month = (
        latest_month + pd.offsets.MonthBegin(1)
    )

    selected_month = st.date_input(
        "Forecast month",
        value=next_month.date(),
    )

    selected_month = pd.Timestamp(
        selected_month
    ).to_period("M").to_timestamp()

    sku_history = history[
        history[sku_col].astype(str) == selected_sku
    ].copy()

    sku_monthly = (
        sku_history.groupby(
            "Month_Start",
            as_index=False,
        )
        .agg(Quantity=(quantity_col, "sum"))
        .sort_values("Month_Start")
    )

    forecast_model = load_model(str(FORECAST_MODEL_PATH))

    if st.button("Predict Selected SKU", type="primary"):

        if forecast_model is None:

            st.error(
                f"Forecast model not found: {FORECAST_MODEL_PATH}"
            )

        elif selected_month <= latest_month:

            st.error(
                "The forecast month must be after the last "
                "observed sales month."
            )

        else:

            st.error(
                "The model is loaded, but a valid prediction requires "
                "the exact feature-engineering code used during training. "
                "The app will not fabricate lag or rolling features. "
                "Connect the original training feature builder here."
            )

            st.info(
                "The saved Monthly_Forecasts.csv can still be used "
                "to display the existing forecasts for this SKU."
            )

    if monthly_forecasts is not None:

        fc_sku_col = find_column(
            monthly_forecasts,
            ["SKU", "Product_ID", "Product_Code"],
        )

        fc_month_col = find_column(
            monthly_forecasts,
            ["Forecast_Month", "Month"],
        )

        fc_quantity_col = find_column(
            monthly_forecasts,
            [
                "Forecast_Demand",
                "Predicted_Quantity",
                "Forecast_Quantity",
            ],
        )

        if all([fc_sku_col, fc_month_col, fc_quantity_col]):

            sku_forecasts = monthly_forecasts[
                monthly_forecasts[fc_sku_col].astype(str)
                == selected_sku
            ].copy()

            sku_forecasts[fc_quantity_col] = pd.to_numeric(
                sku_forecasts[fc_quantity_col],
                errors="coerce",
            )

            matching_month = sku_forecasts[
                sku_forecasts[fc_month_col] == selected_month
            ]

            if not matching_month.empty:

                forecast_value = matching_month[
                    fc_quantity_col
                ].sum()

                st.metric(
                    f"Saved Forecast — {selected_sku}",
                    f"{forecast_value:,.2f} units",
                    help=selected_month.strftime("%B %Y"),
                )

            st.subheader("Saved Forecasts for Selected SKU")

            st.dataframe(
                sku_forecasts,
                use_container_width=True,
                hide_index=True,
            )

    # Historical SKU demand chart
    st.subheader(f"Historical Monthly Demand — {selected_sku}")

    if not sku_monthly.empty:

        sku_fig = px.line(
            sku_monthly,
            x="Month_Start",
            y="Quantity",
            markers=True,
            title=f"Monthly Quantity for {selected_sku}",
        )

        sku_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Quantity Sold",
        )

        st.plotly_chart(
            sku_fig,
            use_container_width=True,
        )


# =========================================================
# PAGE 4: INVENTORY PLANNING
# =========================================================

elif page == "Inventory Planning":

    st.title("Inventory Planning")
    st.caption("Stock position, demand shortage, and production needs")

    if inventory is None:
        show_missing_file(INVENTORY_PATH)
        st.stop()

    inv = normalize_columns(inventory)

    sku_col = find_column(
        inv,
        ["SKU", "Product_ID", "Product_Code"],
    )

    month_col = find_column(
        inv,
        ["Forecast_Month", "Month"],
    )

    stock_col = find_column(
        inv,
        ["Current_Stock", "Closing_Stock", "Available_Stock"],
    )

    demand_col = find_column(
        inv,
        ["Forecast_Demand", "Predicted_Quantity"],
    )

    safety_col = find_column(
        inv,
        ["Safety_Stock"],
    )

    shortage_col = find_column(
        inv,
        ["Demand_Shortage", "Stock_Shortage", "Shortage"],
    )

    production_col = find_column(
        inv,
        ["Recommended_Production", "Production_Required"],
    )

    if month_col:
        inv[month_col] = pd.to_datetime(
            inv[month_col],
            errors="coerce",
        )

        months = sorted(inv[month_col].dropna().unique())

        if months:

            selected_month = st.selectbox(
                "Planning month",
                [pd.Timestamp(x) for x in months],
                format_func=lambda x: x.strftime("%B %Y"),
            )

            view = inv[inv[month_col] == selected_month].copy()

        else:
            view = inv.copy()

    else:
        view = inv.copy()

    if sku_col:

        sku_options = sorted(
            view[sku_col].dropna().astype(str).unique()
        )

        selected_sku = st.selectbox(
            "SKU filter",
            ["All SKUs"] + sku_options,
            key="inventory_sku",
        )

        if selected_sku != "All SKUs":
            view = view[
                view[sku_col].astype(str) == selected_sku
            ]

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "SKUs in View",
        f"{view[sku_col].nunique() if sku_col else len(view):,}",
    )

    if shortage_col:
        shortage_total = pd.to_numeric(
            view[shortage_col],
            errors="coerce",
        ).fillna(0).sum()

        k2.metric(
            "Demand Shortage",
            f"{shortage_total:,.2f}",
        )
    else:
        k2.metric("Demand Shortage", "N/A")

    if production_col:
        production_total = pd.to_numeric(
            view[production_col],
            errors="coerce",
        ).fillna(0).sum()

        k3.metric(
            "Recommended Production",
            f"{production_total:,.2f}",
        )
    else:
        k3.metric("Recommended Production", "N/A")

    st.subheader("Inventory Planning Table")

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
    )

    if shortage_col and sku_col:

        chart_data = view.copy()

        chart_data[shortage_col] = pd.to_numeric(
            chart_data[shortage_col],
            errors="coerce",
        ).fillna(0)

        chart_data = chart_data[
            chart_data[shortage_col] > 0
        ].sort_values(shortage_col, ascending=False)

        if chart_data.empty:
            st.success("No shortage is recorded in this view.")
        else:

            fig = px.bar(
                chart_data,
                x=sku_col,
                y=shortage_col,
                title="Demand Shortage by SKU",
                labels={
                    sku_col: "SKU",
                    shortage_col: "Shortage",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Inventory Planning View",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="Inventory_Planning_View.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 5: PRODUCTION OPTIMIZATION
# =========================================================

elif page == "Production Optimization":

    st.title("Production Optimization")
    st.caption("Optimized production quantities and capacity results")

    if production_optimization is None:
        show_missing_file(PRODUCTION_OPTIMIZATION_PATH)
        st.stop()

    opt = normalize_columns(production_optimization)

    sku_col = find_column(
        opt,
        ["SKU", "Product_ID", "Product_Code"],
    )

    month_col = find_column(
        opt,
        ["Forecast_Month", "Month", "Optimization_Month"],
    )

    production_col = find_column(
        opt,
        [
            "Recommended_Production",
            "Optimized_Production",
            "Production_Quantity",
            "Production_Qty",
        ],
    )

    shortage_col = find_column(
        opt,
        ["Remaining_Shortage", "Shortage", "Unmet_Demand"],
    )

    demand_col = find_column(
        opt,
        ["Net_Requirement", "Forecast_Demand", "Demand"],
    )

    if month_col:

        opt[month_col] = pd.to_datetime(
            opt[month_col],
            errors="coerce",
        )

        months = sorted(opt[month_col].dropna().unique())

        if months:

            selected_month = st.selectbox(
                "Optimization month",
                [pd.Timestamp(x) for x in months],
                format_func=lambda x: x.strftime("%B %Y"),
            )

            view = opt[opt[month_col] == selected_month].copy()

        else:
            view = opt.copy()

    else:
        view = opt.copy()

    if sku_col:

        sku_options = sorted(
            view[sku_col].dropna().astype(str).unique()
        )

        selected_sku = st.selectbox(
            "SKU filter",
            ["All SKUs"] + sku_options,
            key="optimization_sku",
        )

        if selected_sku != "All SKUs":
            view = view[
                view[sku_col].astype(str) == selected_sku
            ]

    st.subheader("Optimization Summary")

    if optimization_summary is not None:

        st.dataframe(
            optimization_summary,
            use_container_width=True,
            hide_index=True,
        )

    k1, k2, k3 = st.columns(3)

    if production_col:

        production_total = pd.to_numeric(
            view[production_col],
            errors="coerce",
        ).fillna(0).sum()

        k1.metric(
            "Recommended Production",
            f"{production_total:,.2f}",
        )
    else:
        k1.metric("Recommended Production", "N/A")

    if demand_col:

        demand_total = pd.to_numeric(
            view[demand_col],
            errors="coerce",
        ).fillna(0).sum()

        k2.metric(
            "Net Requirement",
            f"{demand_total:,.2f}",
        )
    else:
        k2.metric("Net Requirement", "N/A")

    if shortage_col:

        shortage_total = pd.to_numeric(
            view[shortage_col],
            errors="coerce",
        ).fillna(0).sum()

        k3.metric(
            "Remaining Shortage",
            f"{shortage_total:,.2f}",
        )
    else:
        k3.metric("Remaining Shortage", "N/A")

    st.subheader("Production Decisions by SKU")

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
    )

    if sku_col and production_col:

        chart_data = view.copy()

        chart_data[production_col] = pd.to_numeric(
            chart_data[production_col],
            errors="coerce",
        ).fillna(0)

        chart_data = chart_data.sort_values(
            production_col,
            ascending=False,
        )

        fig = px.bar(
            chart_data,
            x=sku_col,
            y=production_col,
            title="Recommended Production Quantity by SKU",
            labels={
                sku_col: "SKU",
                production_col: "Recommended Production",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

    if shortage_col:

        shortage_total = pd.to_numeric(
            view[shortage_col],
            errors="coerce",
        ).fillna(0).sum()

        if shortage_total > 0:
            st.warning(
                f"Remaining shortage: {shortage_total:,.2f} units."
            )
        else:
            st.success(
                "The selected optimization output reports no remaining shortage."
            )

    st.download_button(
        "Download Production Optimization View",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="Production_Optimization_View.csv",
        mime="text/csv",
    )
