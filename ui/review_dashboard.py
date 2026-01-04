# ui/review_dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import glob
from datetime import datetime

# --- Configuration ---
LOG_DIR = "data/logs"
st.set_page_config(page_title="Seizure Detection Audit", layout="wide")

def load_logs():
    """Reads all JSON log files and merges them into a DataFrame"""
    all_events = []
    
    # Find all json files in log directory
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        return pd.DataFrame()

    log_files = glob.glob(os.path.join(LOG_DIR, "*.json"))
    
    for file_path in log_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Flatten the metadata for the table
                    flat_event = {
                        "Timestamp": data["timestamp"],
                        "Risk Score": f"{data['risk_score'] * 100:.1f}%",
                        "Status": data["status"],
                        "Duration Frames": data["metadata"].get("counter", 0)
                    }
                    all_events.append(flat_event)
                except json.JSONDecodeError:
                    continue
                    
    df = pd.DataFrame(all_events)
    if not df.empty:
        # Convert timestamp to datetime object for sorting
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp', ascending=False)
    return df

# --- UI Layout ---
st.title("🏥 Seizure Detection Event Log")
st.markdown("Real-time audit trail of detected convulsive events.")

# 1. Metrics Row
df = load_logs()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Events Detected", len(df) if not df.empty else 0)
with col2:
    if not df.empty:
        last_event = df.iloc[0]['Timestamp'].strftime("%H:%M:%S")
    else:
        last_event = "--:--:--"
    st.metric("Last Detection Time", last_event)
with col3:
    st.metric("System Status", "Active", delta="OK")

st.divider()

# 2. Data Table
st.subheader("Event History")

if df.empty:
    st.info("No seizure events detected yet. System is monitoring...")
else:
    # Highlight high risk events
    st.dataframe(
        df,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(format="D MMM YYYY, h:mm a"),
        },
        use_container_width=True
    )

    # 3. Simple Chart (Risk over time)
    st.subheader("Risk Intensity Analysis")
    # Clean risk score string "85.5%" -> float 85.5 for charting
    chart_df = df.copy()
    chart_df['Risk Value'] = chart_df['Risk Score'].str.rstrip('%').astype(float)
    st.line_chart(chart_df, x='Timestamp', y='Risk Value')

# Auto-refresh button
if st.button('Refresh Logs'):
    st.rerun()