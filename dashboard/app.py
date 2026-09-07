#!/usr/bin/env python3
"""VigilEye fleet dashboard.

    streamlit run dashboard/app.py

Reads the SQLite event store written by ``vigileye.logger.SessionLogger``
and turns it into the fleet-level view described in the abstract: which
drivers are accumulating fatigue events, at what hours, and on which
trips. This is the part that makes the system valuable to an operator
rather than only to the driver in the seat.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="VigilEye Fleet Dashboard", page_icon="👁", layout="wide")

DB_DEFAULT = "logs/vigileye.db"


@st.cache_data(ttl=15)
def load(db_path: str):
    if not Path(db_path).exists():
        return None, None
    con = sqlite3.connect(db_path)
    sessions = pd.read_sql_query("SELECT * FROM sessions", con)
    events = pd.read_sql_query("SELECT * FROM events", con)
    con.close()

    for df, col in ((sessions, "started_at"), (events, "wall_time")):
        if not df.empty:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return sessions, events


st.title("👁 VigilEye — Fleet Safety Analytics")

db_path = st.sidebar.text_input("Event database", DB_DEFAULT)
sessions, events = load(db_path)

if sessions is None:
    st.warning(f"No database at `{db_path}`. Run `python run.py` to record a session first.")
    st.stop()
if sessions.empty:
    st.info("Database exists but contains no sessions yet.")
    st.stop()

# -- filters ---------------------------------------------------------------
drivers = ["All"] + sorted(sessions["driver_id"].dropna().unique().tolist())
driver = st.sidebar.selectbox("Driver", drivers)
if driver != "All":
    sessions = sessions[sessions["driver_id"] == driver]
    events = events[events["session_id"].isin(sessions["session_id"])]

alerts = events[events["kind"] == "alert"].copy()
transitions = events[events["kind"] == "state_change"].copy()

# -- KPIs ------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sessions", len(sessions))
c2.metric("Total frames", f"{int(sessions['frames'].fillna(0).sum()):,}")
c3.metric("Drowsy alerts", int((alerts["state"] == "DROWSY").sum()))
c4.metric("Distraction alerts", int((alerts["state"] == "DISTRACTED").sum()))
c5.metric("Critical (level 2)", int((alerts["level"] == 2).sum()))

st.divider()

# -- charts ----------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Alerts by type")
    if alerts.empty:
        st.caption("No alerts recorded.")
    else:
        st.bar_chart(alerts.groupby("state").size().rename("alerts"))

with right:
    st.subheader("Alerts by hour of day")
    if alerts.empty or alerts["wall_time"].isna().all():
        st.caption("No timestamped alerts.")
    else:
        by_hour = (
            alerts.assign(hour=alerts["wall_time"].dt.hour)
            .groupby("hour").size().reindex(range(24), fill_value=0).rename("alerts")
        )
        st.bar_chart(by_hour)
        peak = int(by_hour.idxmax())
        if by_hour.max() > 0:
            st.caption(f"Peak risk hour: {peak:02d}:00–{peak+1:02d}:00 "
                       f"({int(by_hour.max())} alerts)")

st.subheader("Risk score over time")
if transitions.empty:
    st.caption("No state changes recorded.")
else:
    trend = transitions.set_index("wall_time")[["drowsy_score", "distract_score"]].sort_index()
    st.line_chart(trend)

# -- driver leaderboard ----------------------------------------------------
st.subheader("Driver risk ranking")
merged = alerts.merge(
    sessions[["session_id", "driver_id", "vehicle_id"]], on="session_id", how="left"
)
if merged.empty:
    st.caption("No alerts to rank.")
else:
    ranking = (
        merged.groupby("driver_id")
        .agg(
            total_alerts=("id", "count"),
            drowsy=("state", lambda s: int((s == "DROWSY").sum())),
            distracted=("state", lambda s: int((s == "DISTRACTED").sum())),
            critical=("level", lambda s: int((s == 2).sum())),
            peak_drowsy_score=("drowsy_score", "max"),
        )
        .sort_values("total_alerts", ascending=False)
    )
    frames_per_driver = sessions.groupby("driver_id")["frames"].sum()
    hours = (frames_per_driver / (30 * 3600)).replace(0, pd.NA)
    ranking["alerts_per_hour"] = (ranking["total_alerts"] / hours).round(2)
    st.dataframe(ranking, use_container_width=True)

# -- session log -----------------------------------------------------------
st.subheader("Sessions")
st.dataframe(
    sessions[["session_id", "driver_id", "vehicle_id", "started_at",
              "ended_at", "frames", "mean_fps"]].sort_values("started_at", ascending=False),
    use_container_width=True,
)

with st.expander("Raw alert log"):
    st.dataframe(
        alerts[["wall_time", "session_id", "state", "level",
                "drowsy_score", "distract_score", "perclos", "message"]]
        .sort_values("wall_time", ascending=False),
        use_container_width=True,
    )
