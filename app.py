# app.py - Predictive Delivery Optimizer (Streamlit)
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import time

# --- Page setup ---
st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

# --- Load & show logo in sidebar ---
LOGO_FILENAME = "logo.png"
try:
    if os.path.exists(LOGO_FILENAME):
        logo = Image.open(LOGO_FILENAME)
        st.sidebar.image(logo, use_container_width=True)
    else:
        st.sidebar.markdown("")  # no logo file found
except Exception:
    st.sidebar.markdown("")  # ignore logo load issues

# --- Title and description ---
st.title("Predictive Delivery Optimizer")
st.write("Use this app to predict delivery delay status based on shipment features.")

# --- Load model and data ---
DATA_CSV = "cleaned_master_dataset.csv"
MODEL_FILE = "rf_model.joblib"
LABEL_ENCODER_FILE = "target_label_encoder.joblib"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_CSV)

@st.cache_data
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_target_encoder():
    return joblib.load(LABEL_ENCODER_FILE)

# load safely
try:
    data = load_data()
    model = load_model()
    target_le = load_target_encoder()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- Animated Gradient Summary Dashboard ---
st.markdown("---")
st.markdown("### Quick Performance Summary")

try:
    total_orders = len(data)
    avg_cost = round(data["Delivery_Cost_INR"].mean(), 2)
    avg_distance = round(data["Distance_KM"].mean(), 2)
    if "Delay_Days" in data.columns:
        avg_delay = round(data["Delay_Days"].mean(), 2)
    else:
        avg_delay = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div style='background: linear-gradient(135deg, #FFD700, #FFB300); padding: 20px; border-radius: 12px; text-align:center; color:black;'>📦<br><b>Total Orders</b><br>{total_orders}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='background: linear-gradient(135deg, #00C9A7, #92FE9D); padding: 20px; border-radius: 12px; text-align:center; color:black;'>💰<br><b>Avg Delivery Cost</b><br>₹{avg_cost}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='background: linear-gradient(135deg, #4facfe, #00f2fe); padding: 20px; border-radius: 12px; text-align:center; color:black;'>🛣️<br><b>Avg Distance</b><br>{avg_distance} km</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div style='background: linear-gradient(135deg, #FF5F6D, #FFC371); padding: 20px; border-radius: 12px; text-align:center; color:black;'>⏰<br><b>Avg Delay</b><br>{avg_delay} days</div>", unsafe_allow_html=True)

    time.sleep(0.5)
    st.success("Dashboard metrics updated successfully!")
except Exception as e:
    st.warning(f"Could not calculate summary metrics: {e}")

# --- Build encoders for categorical columns ---
encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(data[col].astype(str))
        encoders[col] = le

# --- Sidebar Inputs ---
st.sidebar.header("Input Shipment Details")
user_input = {}
for col in data.columns:
    if col in ["Delivery_Status", "Delay_Status_auto", "Delay_Days"]:
        continue
    if data[col].dtype == 'object':
        options = list(encoders[col].classes_)
        user_input[col] = st.sidebar.selectbox(f"{col}", options, index=0)
    else:
        user_input[col] = st.sidebar.number_input(
            f"{col}",
            min_value=float(data[col].min()),
            max_value=float(data[col].max()),
            value=float(data[col].mean())
        )

df_input = None
pred_label = None
probs = None

# --- Add Predict Button ---
if st.sidebar.button("Predict Delay Status"):
    with st.spinner("AI Model Analyzing Your Shipment... Please wait..."):
        progress = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.15)
            progress.progress(percent)
        time.sleep(0.2)

    df_input = pd.DataFrame([user_input])

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df_input.columns:
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except Exception:
                df_input[col] = df_input[col].astype(str).apply(
                    lambda x: le.transform([le.classes_[0]])[0]
                )

    # Align columns
    if hasattr(model, "feature_names_in_"):
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        pred_encoded = model.predict(df_input)[0]
        pred_label = target_le.inverse_transform([pred_encoded])[0]
        probs = model.predict_proba(df_input)[0]

        st.balloons()
        st.success("Prediction complete! Here's your result ")

        # Display result
        st.subheader("Prediction Result")
        st.write(f"**Predicted Delivery Status:** `{pred_label}`")

        st.subheader("Prediction Confidence")
        st.dataframe(
            pd.DataFrame(
                {"Status": list(target_le.classes_), "Probability": list(probs)}
            )
        )

        # --- Feedback ---
        st.markdown("---")
        st.write("### How accurate does this feel?")
        feedback = st.radio(
            "Share your opinion:",
            ["Excellent", "Good", "Average", "Needs Improvement"],
            horizontal=True,
        )
        if feedback:
            st.info(f"Thanks for your feedback! You rated this as: **{feedback}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # --- Prediction ---
    try:
        pred_encoded = model.predict(df_input)[0]
        pred_label = target_le.inverse_transform([pred_encoded])[0]
        probs = model.predict_proba(df_input)[0]

        st.balloons()
        st.success("Prediction complete! Here's your result ")

        # Display Results
        st.subheader("Prediction Result")
        st.write(f"**Predicted Delivery Status:** `{pred_label}`")

        st.subheader("Prediction Confidence")
        st.dataframe(
            pd.DataFrame(
                {"Status": list(target_le.classes_), "Probability": list(probs)}
            )
        )

        # Optional feedback prompt
        st.markdown("---")
        st.write("###  How accurate does this feel?")
        feedback = st.radio(
            "Share your opinion:",
            ["Excellent", "Good", "Average", "Needs Improvement"],
            horizontal=True,
        )
        if feedback:
            st.info(f"Thanks for your feedback! You rated this as: **{feedback}** ")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Dashboard Section ---
st.markdown("---")
st.header("Delivery Performance Dashboard")

# --- Segment-wise Delivery Analysis ---
st.markdown("---")
st.header("Segment-wise Delivery Analysis")

col1, col2 = st.columns(2)
segment_choice = col1.selectbox("Select Customer Segment", sorted(data["Customer_Segment"].unique()))
priority_choice = col2.selectbox("Select Priority", sorted(data["Priority"].unique()))

filtered_data = data[
    (data["Customer_Segment"] == segment_choice) &
    (data["Priority"] == priority_choice)
]

st.write(f"Showing data for **{segment_choice}** segment with **{priority_choice}** priority")

col3, col4, col5 = st.columns(3)
col3.metric("Avg Delivery Cost (INR)", round(filtered_data["Delivery_Cost_INR"].mean(), 2))
col4.metric("Avg Distance (KM)", round(filtered_data["Distance_KM"].mean(), 2))
if "Delay_Days" in filtered_data.columns:
    col5.metric("Avg Delay (Days)", round(filtered_data["Delay_Days"].mean(), 2))
else:
    col5.metric("Avg Delay (Days)", "N/A")

st.subheader("Product Category vs Avg Delivery Cost")
st.bar_chart(filtered_data.groupby("Product_Category")["Delivery_Cost_INR"].mean())

if "Delay_Days" in filtered_data.columns:
    st.subheader("Product Category vs Avg Delay")
    st.bar_chart(filtered_data.groupby("Product_Category")["Delay_Days"].mean())

# --- AI-Generated Insights Section ---
st.markdown("---")
st.subheader("AI Insights")

try:
    selected_segment = segment_choice
    selected_priority = priority_choice
    avg_cost = round(data[data["Customer_Segment"] == selected_segment]["Delivery_Cost_INR"].mean(), 2)
    avg_distance = round(data[data["Customer_Segment"] == selected_segment]["Distance_KM"].mean(), 2)

    st.write(f"**Insight:** For the **{selected_segment}** customer segment with **{selected_priority}** priority,")
    st.write(f"- The average delivery cost is approximately **₹{avg_cost}**,")
    st.write(f"- With an average delivery distance of **{avg_distance} km.**")
except Exception as e:
    st.warning("Unable to generate insights at the moment. Please check selected filters.")

# --- Export & Download Section ---
st.markdown("---")
st.header("Export and Download")

csv_data = filtered_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv_data,
    file_name=f"filtered_data_{segment_choice}_{priority_choice}.csv",
    mime="text/csv",
)

# --- PDF Export Section ---
from fpdf import FPDF

st.markdown("---")
st.header("Export Full Report as PDF")

def generate_pdf(pred_status, avg_cost, avg_dist, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Predictive Delivery Optimizer Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Status: {pred_status}", ln=True)
    pdf.cell(200, 10, txt=f"Average Delivery Cost: Rs. {avg_cost}", ln=True)
    pdf.cell(200, 10, txt=f"Average Distance: {avg_dist} km", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"AI Insights:\n{insights}")
    pdf.output("Delivery_Report.pdf")
    return "Delivery_Report.pdf"

if st.button("Generate PDF Report"):
    try:
        insights = (
            f"For {segment_choice} customers with {priority_choice} priority, "
            f"average cost is Rs. {round(filtered_data['Delivery_Cost_INR'].mean(), 2)} "
            f"and average distance is {round(filtered_data['Distance_KM'].mean(), 2)} km."
        )
        pdf_file = generate_pdf(
            pred_label if pred_label else "Not Predicted Yet",
            round(filtered_data["Delivery_Cost_INR"].mean(), 2),
            round(filtered_data["Distance_KM"].mean(), 2),
            insights,
        )
        with open(pdf_file, "rb") as file:
            st.download_button(
                label="Download PDF Report",
                data=file,
                file_name="Delivery_Report.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")


# --- AI Insight Cards ---
st.markdown("---")
st.header("AI Insight Cards & Smart Recommendations")

try:
    avg_cost = round(filtered_data["Delivery_Cost_INR"].mean(), 2)
    avg_dist = round(filtered_data["Distance_KM"].mean(), 2)
    avg_delay = round(filtered_data["Delay_Days"].mean(), 2) if "Delay_Days" in filtered_data.columns else 0

    st.subheader("Automated Insight Summary")

    # Layout for four colorful cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="background-color:#FFD700;padding:20px;border-radius:15px;text-align:center;">
                <h4>Total Orders</h4>
                <h2>{len(filtered_data)}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color:#7CFC00;padding:20px;border-radius:15px;text-align:center;">
                <h4>Avg Delivery Cost</h4>
                <h2>₹{avg_cost}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color:#1E90FF;padding:20px;border-radius:15px;text-align:center;color:white;">
                <h4>Avg Distance</h4>
                <h2>{avg_dist} km</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style="background-color:#FF6347;padding:20px;border-radius:15px;text-align:center;color:white;">
                <h4>Avg Delay</h4>
                <h2>{avg_delay} days</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Smart Recommendations")

    # Recommendation Logic
    if avg_delay > 3:
        st.warning("High delay detected — optimize carrier performance and routing paths.")
    elif avg_cost > 800:
        st.info(" Delivery cost is above average — consider optimizing shipment consolidation.")
    elif avg_dist > 1200:
        st.info("Deliveries cover long distances — use regional hubs to reduce costs.")
    else:
        st.success("Logistics performance looks optimized — keep monitoring weekly.")

except Exception as e:
    st.error(f"Unable to generate AI Insight Cards: {e}")


# --- Export AI Insights as PDF (robust to unicode) ---
from fpdf import FPDF

st.markdown("---")
st.header("Export Full AI Report as PDF (Unicode-safe)")

def create_ai_report_pdf_safe(segment, priority, avg_cost, avg_dist, avg_delay, recommendation):
    # Create PDF and ensure any non-latin-1 chars are replaced
    def clean_text(text):
        # convert to str, replace unsupported chars with '?'
        return str(text).encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=clean_text("Predictive Delivery Optimizer - AI Report"), ln=True, align='C')
    pdf.ln(8)

    # Summary block
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=clean_text("Shipment Summary:"), ln=True)
    pdf.set_font("Arial", '', 11)
    summary = (
        f"Customer Segment: {segment}\n"
        f"Priority: {priority}\n"
        f"Average Delivery Cost: Rs. {avg_cost}\n"
        f"Average Distance: {avg_dist} km\n"
        f"Average Delay: {avg_delay} days\n"
    )
    pdf.multi_cell(0, 7, clean_text(summary))
    pdf.ln(6)

    # AI Insights
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=clean_text("AI Insights:"), ln=True)
    pdf.set_font("Arial", '', 11)
    insight_text = (
        f"For {segment} customers with {priority} priority, the average delivery cost is Rs. {avg_cost}, "
        f"and the average delivery distance is {avg_dist} km."
    )
    pdf.multi_cell(0, 7, clean_text(insight_text))
    pdf.ln(6)

    # Recommendation
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=clean_text("Recommendation:"), ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, clean_text(recommendation))
    pdf.ln(8)

    # Footer
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, txt=clean_text("Generated by Predictive Delivery Optimizer | © 2025 Stuti Rao"), ln=True, align='C')

    out_path = "AI_Delivery_Report.pdf"
    pdf.output(out_path)
    return out_path

# Button and generation
if st.button("Generate AI PDF Report"):
    try:
        # define metrics used for recommendation (ensure these variables exist)
        avg_cost = round(filtered_data["Delivery_Cost_INR"].mean(), 2)
        avg_dist = round(filtered_data["Distance_KM"].mean(), 2)
        avg_delay = round(filtered_data["Delay_Days"].mean(), 2) if "Delay_Days" in filtered_data.columns else 0

        if avg_delay > 3:
            recommendation = "High delays detected — optimize routes and delivery schedules."
        elif avg_cost > 800:
            recommendation = "High delivery cost — consider consolidating shipments."
        elif avg_dist > 1200:
            recommendation = "Deliveries cover long distances — explore regional distribution hubs."
        else:
            recommendation = "Logistics performance looks optimized — keep monitoring performance weekly."

        pdf_file = create_ai_report_pdf_safe(segment_choice, priority_choice, avg_cost, avg_dist, avg_delay, recommendation)

        # serve download
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="⬇ Download AI Report (PDF)",
                data=f,
                file_name="AI_Delivery_Report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        # show full exception so we can debug if needed
        st.error(f"Error while generating AI Report: {e}")
        import traceback
        st.text(traceback.format_exc())


# --- Live KPI Trend Analysis ---
st.markdown("---")
st.header("Live KPI Trend Analysis")

try:
    if "Order_Date" in data.columns:
        data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")
        cols = ["Delivery_Cost_INR", "Distance_KM"]
        if "Delay_Days" in data.columns:
            cols.append("Delay_Days")

        trend_df = data.groupby("Order_Date")[cols].mean().reset_index()

        st.subheader("Average Delivery Cost Over Time")
        st.line_chart(trend_df, x="Order_Date", y="Delivery_Cost_INR")

        st.subheader("Average Distance Over Time")
        st.line_chart(trend_df, x="Order_Date", y="Distance_KM")

        if "Delay_Days" in trend_df.columns:
            st.subheader("Average Delay Days Over Time")
            st.line_chart(trend_df, x="Order_Date", y="Delay_Days")
        else:
            st.info("Delay_Days data not available, skipping this chart.")
    else:
        st.warning("'Order_Date' column not found in dataset.")
except Exception as e:
    st.error(f"Error generating trend analysis: {e}")


# --- Footer ---
st.markdown("---")
st.caption("Made by **Stuti Rao** | Data Science Project | Predictive Logistics Optimization")
