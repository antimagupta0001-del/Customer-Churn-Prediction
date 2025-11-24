import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

st.title("📊 Customer Churn Analysis Dashboard")

# Sidebar - upload & sample
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if st.sidebar.button("Download sample CSV"):
    # create a small sample csv in-memory
    sample_df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "age": [25, 34, 45],
        "monthly_charges": [29.99, 49.9, 99.0],
        "contract_type": ["month-to-month", "one-year", "two-year"],
        "churn": [0, 1, 0]
    })
    buffer = io.StringIO()
    sample_df.to_csv(buffer, index=False)
    st.sidebar.download_button("Download sample.csv", buffer.getvalue(), file_name="sample_churn.csv", mime="text/csv")

@st.cache_data
def load_csv(file_obj):
    # try common encodings if necessary
    try:
        return pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding='latin-1')

if uploaded_file is not None:
    # load
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📌 Dataset Summary (Numeric columns)")
    st.write(df.describe(include='number'))

    # Basic info
    st.markdown("**Dataset shape:** " + str(df.shape))
    with st.expander("Show column data types"):
        st.write(df.dtypes)

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.subheader("⚠️ Missing values")
        st.dataframe(missing[missing > 0].sort_values(ascending=False))
    else:
        st.info("No missing values detected.")

    # Column selection logic
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    st.sidebar.subheader("Visualization Filters")

    selected_num_col = None
    selected_cat_col = None

    if numeric_cols:
        selected_num_col = st.sidebar.selectbox("Select Numerical Column for Histogram", ["-- None --"] + numeric_cols)
        if selected_num_col == "-- None --":
            selected_num_col = None
    else:
        st.sidebar.info("No numeric columns detected.")

    if cat_cols:
        selected_cat_col = st.sidebar.selectbox("Select Categorical Column for Countplot", ["-- None --"] + cat_cols)
        if selected_cat_col == "-- None --":
            selected_cat_col = None
    else:
        st.sidebar.info("No categorical columns detected.")

    # Histogram
    if selected_num_col:
        st.subheader(f"📈 Histogram of `{selected_num_col}`")
        try:
            fig, ax = plt.subplots()
            sns.histplot(df[selected_num_col].dropna(), kde=True, ax=ax)
            ax.set_xlabel(selected_num_col)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Could not plot histogram: {e}")

    # Countplot
    if selected_cat_col:
        st.subheader(f"🟦 Countplot of `{selected_cat_col}`")
        try:
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x=selected_cat_col, order=df[selected_cat_col].value_counts().index, ax=ax2)
            plt.xticks(rotation=45, ha="right")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as e:
            st.error(f"Could not plot countplot: {e}")

        # show top categories value counts table
        with st.expander(f"Show value counts for `{selected_cat_col}`"):
            st.dataframe(df[selected_cat_col].value_counts())

    # Correlation heatmap (numeric)
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap (numeric columns)")
        try:
            corr = df[numeric_cols].corr()
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        except Exception as e:
            st.error(f"Could not plot correlation heatmap: {e}")
    else:
        st.info("Need at least two numeric columns to show correlation heatmap.")

    # Helpful suggestions & quick insights
    st.subheader("💡 Quick Insights & Next Steps")
    suggestions = []
    if df.shape[0] < 1000:
        suggestions.append("Dataset is small — consider adding more rows for robust modeling.")
    if missing.sum() > 0:
        suggestions.append("There are missing values. Consider imputing or dropping missing rows/columns.")
    if "churn" in df.columns.str.lower():
        suggestions.append("Detected a 'churn' column name (case-insensitive). You can treat this as target for classification.")
    else:
        suggestions.append("If you want to build a churn model, add a binary target column like `churn` (0/1).")

    for s in suggestions:
        st.write("- " + s)

    # Option to download cleaned subset (example)
    if st.button("Download numeric summary (CSV)"):
        numeric_summary = df[numeric_cols].describe().transpose()
        buf = io.StringIO()
        numeric_summary.to_csv(buf)
        st.download_button("Download numeric_summary.csv", buf.getvalue(), file_name="numeric_summary.csv", mime="text/csv")

else:
    st.info("📂 Please upload a CSV file to start analysis.")
    st.caption("Tip: Put the target column name as `churn` (0 or 1) if you want to later build a classifier.")