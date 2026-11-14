import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Price Prediction", page_icon="💻", layout="centered")


@st.cache_resource
def load_model(path: str = "Price_prediction_model.pkl"):
    return joblib.load(path)


FEATURES = ["speed", "hd", "ram", "screen"]

st.title("💻 Price Prediction")
st.write("Enter the specifications to predict the price.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model 'Price_prediction_model.pkl': {e}")
    st.stop()

tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV)"])

# ---------- Single prediction ----------
with tab_single:
    c1, c2 = st.columns(2)
    speed = c1.number_input("Speed (MHz)", min_value=0.0, value=66.0, step=1.0)
    hd = c2.number_input("HD (MB)", min_value=0.0, value=500.0, step=10.0)
    ram = c1.number_input("RAM (MB)", min_value=0.0, value=8.0, step=1.0)
    screen = c2.number_input("Screen (inches)", min_value=0.0, value=14.0, step=0.5)

    if st.button("Predict Price", type="primary"):
        X = pd.DataFrame([[speed, hd, ram, screen]], columns=FEATURES)
        value = float(model.predict(X).ravel()[0])
        st.success(f"Predicted Price: **{value:,.2f}**")

    with st.expander("Model details"):
        if hasattr(model, "coef_"):
            coefs = pd.DataFrame(
                {"feature": FEATURES, "coefficient": list(model.coef_.ravel())}
            )
            st.dataframe(coefs, hide_index=True)
            st.write(f"Intercept: {float(pd.Series(model.intercept_).iloc[0]):.4f}")

# ---------- Batch prediction ----------
with tab_batch:
    st.write(f"Upload a CSV with columns: `{', '.join(FEATURES)}`")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["predicted_price"] = model.predict(df[FEATURES]).ravel()
            st.dataframe(df)
            st.download_button(
                "Download predictions",
                df.to_csv(index=False).encode("utf-8"),
                file_name="price_predictions.csv",
                mime="text/csv",
            )
