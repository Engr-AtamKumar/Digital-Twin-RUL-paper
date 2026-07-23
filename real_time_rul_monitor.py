import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timezone

# === CONFIGURATION ===
THRESHOLD = 0  # RUL threshold for alerts
ALERT_LIMIT = 20  # Max alerts to include in one email
DATABASE_URI = "postgresql+psycopg2://postgres:atam123@localhost:5432/printer"
EMAIL_SENDER = "atamkumar2018@gmail.com"
EMAIL_PASSWORD = "wzdhawsdwjhzecjl"
EMAIL_RECEIVER = "jessaniatamkumar@gmail.com"

# === STEP 1: Connect to TimescaleDB and Load Data ===
engine = create_engine(DATABASE_URI)
query = """
SELECT timestamp, "Predicted_RUL" 
FROM predictions_data 
ORDER BY timestamp DESC 
LIMIT 100;
"""
df = pd.read_sql(query, engine)

# === STEP 2: Detect Negative RUL Entries ===
alerts = df[df["Predicted_RUL"] < THRESHOLD]

if alerts.empty:
    print("✅ No critical RUL violations detected.")
else:
    print("⚠️ CRITICAL: Low RUL Detected!")
    print(alerts[["timestamp", "Predicted_RUL"]])

    # === STEP 3: Log Alerts to New Table ===
    alerts = alerts.copy()  # avoids SettingWithCopyWarning
    alerts.loc[:, "alert_time"] = datetime.now(timezone.utc)
    alerts.to_sql("rul_anomalies", engine, if_exists="append", index=False)
    print("📦 Anomalies logged to: rul_anomalies")

    # === STEP 4: Optional RUL Reset (For Display) ===
    df["Predicted_RUL"] = df["Predicted_RUL"].clip(lower=0)

    # === STEP 5: Send Email Alert ===
    body_lines = ["⚠️ LOW RUL ALERT - Maintenance Required\n"]
    for _, row in alerts.head(ALERT_LIMIT).iterrows():
        body_lines.append(f"• RUL = {int(row['Predicted_RUL'])} at {row['timestamp']}")

    msg = MIMEText("\n".join(body_lines))
    msg["Subject"] = "🚨 LOW RUL ALERT – Immediate Action Needed"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("✅ Email alert sent successfully.")
    except Exception as e:
        print("❌ Failed to send email:", e)