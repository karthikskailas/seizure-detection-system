# ui/review_dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import glob
from datetime import datetime

# Import Alert Configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.alert_system import AlertConfig

# --- Configuration ---
LOG_DIR = "data/logs"
st.set_page_config(page_title="Seizure Detection Dashboard", layout="wide")

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
                        "Timestamp": data.get("timestamp", "Unknown"),
                        "Risk Score": f"{data.get('risk_score', 0) * 100:.1f}%",
                        "Status": data.get("status", "UNKNOWN"),
                        "Duration Frames": data.get("metadata", {}).get("counter", 0)
                    }
                    all_events.append(flat_event)
                except Exception:
                    # Skip any problematic entries
                    continue
                    
    df = pd.DataFrame(all_events)
    if not df.empty:
        # Convert timestamp to datetime object for sorting
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp', ascending=False)
    return df

def load_alert_logs():
    """Load alert-specific logs"""
    alert_log_path = os.path.join(LOG_DIR, "alerts.json")
    alerts = []
    if os.path.exists(alert_log_path):
        with open(alert_log_path, 'r') as f:
            for line in f:
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return alerts

# =============================================================================
# SIDEBAR: ALERT CONFIGURATION
# =============================================================================
st.sidebar.title("⚙️ Alert Configuration")

# Load current config
config = AlertConfig()

with st.sidebar.expander("� Emergency Contact Email", expanded=True):
    email = st.text_input(
        "Email Address",
        value=config.get("emergency_contact_email", ""),
        placeholder="emergency@example.com",
        help="Alert emails will be sent to this address"
    )
    
    if st.button("💾 Save Contact"):
        config.set("emergency_contact_email", email)
        st.success("✅ Contact saved!")

with st.sidebar.expander("� Mailtrap SMTP Settings", expanded=False):
    st.markdown("Get credentials from [Mailtrap](https://mailtrap.io)")
    
    use_email = st.checkbox(
        "Enable Email Alerts",
        value=config.get("use_email", True)
    )
    
    smtp_host = st.text_input(
        "SMTP Host",
        value=config.get("smtp_host", "sandbox.smtp.mailtrap.io")
    )
    smtp_port = st.number_input(
        "SMTP Port",
        value=config.get("smtp_port", 2525),
        min_value=1,
        max_value=65535
    )
    smtp_user = st.text_input(
        "SMTP Username",
        value=config.get("smtp_username", ""),
        type="password"
    )
    smtp_pass = st.text_input(
        "SMTP Password",
        value=config.get("smtp_password", ""),
        type="password"
    )
    
    if st.button("💾 Save Email Settings"):
        config.set("use_email", use_email)
        config.set("smtp_host", smtp_host)
        config.set("smtp_port", smtp_port)
        config.set("smtp_username", smtp_user)
        config.set("smtp_password", smtp_pass)
        st.success("✅ Email settings saved!")

with st.sidebar.expander("🔊 Alert Message", expanded=False):
    alert_msg = st.text_area(
        "Custom Alert Message",
        value=config.get("alert_message", "⚠️ ALERT: Possible seizure detected!"),
        height=80
    )
    location_enabled = st.checkbox(
        "Include Location in Alerts",
        value=config.get("location_enabled", True)
    )
    
    if st.button("💾 Save Message Settings"):
        config.set("alert_message", alert_msg)
        config.set("location_enabled", location_enabled)
        st.success("✅ Message settings saved!")

st.sidebar.divider()
st.sidebar.caption("Config saved to: `data/alert_config.json`")

# =============================================================================
# MAIN CONTENT: EVENT LOGS
# =============================================================================
st.title("🏥 Seizure Detection Dashboard")
st.markdown("Real-time monitoring and alert configuration.")

# Tab layout for different views
tab1, tab2 = st.tabs(["📊 Event Log", "🔔 Alert History"])

with tab1:
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
            width='stretch'
        )

        # 3. Simple Chart (Risk over time)
        st.subheader("Risk Intensity Analysis")
        # Clean risk score string "85.5%" -> float 85.5 for charting
        chart_df = df.copy()
        chart_df['Risk Value'] = chart_df['Risk Score'].str.rstrip('%').astype(float)
        st.line_chart(chart_df, x='Timestamp', y='Risk Value')

with tab2:
    st.subheader("Alert History")
    
    alerts = load_alert_logs()
    
    if not alerts:
        st.info("No alerts sent yet. Alerts will appear here when seizures are detected.")
    else:
        for alert in reversed(alerts[-20:]):  # Show last 20 alerts
            alert_type = alert.get("alert_type", "unknown")
            success = "✅" if alert.get("success", False) else "❌"
            timestamp = alert.get("timestamp", "Unknown")
            confidence = alert.get("confidence_score", 0)
            location = alert.get("location", {})
            
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{timestamp}**")
                with col2:
                    st.write(f"Type: `{alert_type}`")
                with col3:
                    st.write(f"{success} | Conf: {int(confidence*100)}%")
                
                if location.get("city", "Unknown") != "Unknown":
                    st.caption(f"📍 {location.get('city')}, {location.get('region')}")
                st.divider()

# Auto-refresh button
if st.button('🔄 Refresh'):
    st.rerun()
