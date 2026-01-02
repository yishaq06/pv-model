import streamlit as st
import tempfile
import os
import pandas as pd
from train_pv_models import train_from_excel

st.set_page_config(page_title="Train PV Models", layout="wide")

st.title("🧠 Train PV Machine Learning Models")

st.markdown("""
Upload your **prepared dataset CSV file** to train PV forecasting models
The system will automatically clean the data, detect targets, train models, and save outputs.
""")

uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("File uploaded successfully.")

        # ---------------- Preview Uploaded File ----------------
        try:
            preview_df = pd.read_excel(file_path, header=None)
            st.subheader("📄 Dataset Preview (First 10 Rows)")
            st.caption("Raw preview before any cleaning or preprocessing.")
            st.dataframe(preview_df.head(10), width='stretch')
            st.caption(f"Shape: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")

            # df_raw = pd.read_excel(file_path, header=None)
            # detected_headers = df_raw.iloc[0].fillna("").astype(str).tolist()

            # st.subheader("🧠 Header Row Used for Training")
            # st.write(detected_headers)
        except Exception as e:
            st.warning("Unable to preview the uploaded file.")
            st.exception(e)

        st.markdown("---")

        # ---------------- Training Trigger ----------------
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training models. Please wait..."):
                try:
                    summary_df = train_from_excel(
                        input_path=file_path,
                        output_dir="pv_model_outputs"
                    )

                    st.success("✅ Training completed successfully!")

                    st.subheader("📊 Training Summary")
                    st.dataframe(summary_df, width='stretch')

                except Exception as e:
                    st.error("❌ Training failed")
                    st.exception(e)
