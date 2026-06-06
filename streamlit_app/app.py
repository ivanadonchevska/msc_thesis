import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="Bulgarian News Timeline")

REPO_ROOT = Path(__file__).parent.parent

DOT_SIZE = 14
MAX_PREV_DOTS = 5
MIN_ARTICLES = 1
MAX_STORIES_DEFAULT = 8


@st.cache_data
def load_data():
    df = pd.read_parquet(REPO_ROOT / "data/processed/articles_clean.parquet")
    df["published_at_dt"] = pd.to_datetime(df["published_at_dt"], utc=True)
    with open(REPO_ROOT / "data/processed_system/stories.pkl", "rb") as f:
        stories = pickle.load(f)
    with open(REPO_ROOT / "data/processed_system/dots_louvain.pkl", "rb") as f:
        all_dots = pickle.load(f)
    embs_all = np.load(REPO_ROOT / "data/processed_system/embeddings.npy")

    names_path = REPO_ROOT / "data/processed_system/story_names.pkl"
    story_names = pickle.load(open(names_path, "rb")) if names_path.exists() else {}

    summaries_path = REPO_ROOT / "data/processed_system/dot_summaries.pkl"
    dot_summaries = (
        pickle.load(open(summaries_path, "rb")) if summaries_path.exists() else {}
    )

    return df, stories, all_dots, embs_all, story_names, dot_summaries


def day_name(today_dots, embs_all, df):
    indices = [idx for dot in today_dots for idx in dot["indices"]]
    if not indices:
        return None
    embs = embs_all[indices]
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embs)[0]
    return df.iloc[indices[sims.argmax()]]["title"]


@st.cache_data
def build_story_meta(_all_dots):
    meta = {}
    for sid, dots in _all_dots.items():
        dates = [d["effective_start"].date() for d in dots]
        meta[sid] = {"start": min(dates), "end": max(dates)}
    return meta


@st.cache_data
def build_date_index(_all_dots):
    index = {}
    for sid, dots in _all_dots.items():
        for dot in dots:
            d = dot["effective_start"].date()
            index.setdefault(d, set()).add(sid)
    return {d: list(sids) for d, sids in sorted(index.items())}


df, stories, all_dots, embs_all, story_names, dot_summaries = load_data()
story_meta = build_story_meta(all_dots)
date_index = build_date_index(all_dots)
sorted_dates = sorted(date_index.keys())
last_date = sorted_dates[-1]

if "selected_date" not in st.session_state:
    st.session_state.selected_date = last_date

# date navigation
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("← Previous"):
        idx = sorted_dates.index(st.session_state.selected_date)
        if idx > 0:
            st.session_state.selected_date = sorted_dates[idx - 1]
with col2:
    st.markdown(
        f"<h2 style='text-align:center'>"
        f"{st.session_state.selected_date.strftime('%d %B %Y')}"
        f"</h2>",
        unsafe_allow_html=True,
    )
with col3:
    if st.button("Next →"):
        idx = sorted_dates.index(st.session_state.selected_date)
        if idx < len(sorted_dates) - 1:
            st.session_state.selected_date = sorted_dates[idx + 1]

selected_date = st.session_state.selected_date

# filter stories active on selected date
active_sids = [
    sid
    for sid in date_index.get(selected_date, [])
    if len(stories.get(sid, [])) >= MIN_ARTICLES
]


def latest_dot_time(sid):
    dots = all_dots.get(sid, [])
    today = [
        d["effective_start"]
        for d in dots
        if d["effective_start"].date() == selected_date
    ]
    return max(today) if today else pd.Timestamp.min.tz_localize("UTC")


all_active_sids = sorted(active_sids, key=latest_dot_time, reverse=True)
total_active = len(all_active_sids)

if not all_active_sids:
    st.info("No stories found for this date.")
    st.stop()

if "show_all" not in st.session_state:
    st.session_state.show_all = False

# reset show_all when date changes
if st.session_state.get("last_date") != selected_date:
    st.session_state.show_all = False
    st.session_state.last_date = selected_date
    st.session_state.pop("story_select", None)
    st.session_state.pop("dot_select", None)
    st.session_state.pop("last_event_key", None)

active_sids = (
    all_active_sids
    if st.session_state.show_all
    else all_active_sids[:MAX_STORIES_DEFAULT]
)
cap_col, btn_col = st.columns([3, 1])
with cap_col:
    st.caption(
        f"Showing {len(active_sids)} of {total_active} stories active on this date"
    )
with btn_col:
    if not st.session_state.show_all and total_active > MAX_STORIES_DEFAULT:
        if st.button(f"Show all {total_active} stories"):
            st.session_state.show_all = True
            st.rerun()
    elif st.session_state.show_all:
        if st.button("Show less"):
            st.session_state.show_all = False
            st.rerun()

# collect visible dots per story — prev: last MAX_PREV_DOTS before today, today: all on selected date
story_visible = {}
for sid in active_sids:
    dots = all_dots.get(sid, [])
    today_dots = [d for d in dots if d["effective_start"].date() == selected_date]
    prev_dots = sorted(
        [d for d in dots if d["effective_start"].date() < selected_date],
        key=lambda d: d["effective_start"],
    )[-MAX_PREV_DOTS:]
    story_visible[sid] = {"today": today_dots, "prev": prev_dots}

sid_labels = {
    sid: (
        story_names.get((sid, selected_date))
        or day_name(vis["today"], embs_all, df)
        or str(sid)
    )
    for sid, vis in story_visible.items()
}

# positional x-axis: prev dots at -MAX_PREV_DOTS..-1, red line at 0, today dots at 1..N
RED_LINE_X = 0
PREV_START = -MAX_PREV_DOTS  # leftmost position

# compute x range
max_today = max((len(v["today"]) for v in story_visible.values()), default=1)
x_left = -(MAX_PREV_DOTS + 0.5)
x_right = max_today + 0.5

# build figure
fig = go.Figure()

for i, sid in enumerate(active_sids):
    meta = story_meta.get(sid, {})
    start_str = meta["start"].strftime("%d %b") if meta else ""
    end_str = meta["end"].strftime("%d %b") if meta else ""
    label = f"{sid_labels[sid]}  |  {start_str} – {end_str}"

    vis = story_visible[sid]
    today_dots = vis["today"]
    prev_dots = vis["prev"]

    if not today_dots and not prev_dots:
        continue

    # positional x: prev dots at -len(prev)..-1, today dots at 1..N
    n_prev = len(prev_dots)
    prev_x = list(range(-n_prev, 0))  # e.g. [-3, -2, -1]
    today_x = list(range(1, len(today_dots) + 1))  # e.g. [1, 2, 3]

    line_x_start = prev_x[0] if prev_x else 0
    line_x_end = today_x[-1] if today_x else 0

    # connecting line
    fig.add_trace(
        go.Scatter(
            x=[line_x_start, line_x_end],
            y=[label, label],
            mode="lines",
            line=dict(color="#aec7e8", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # previous dots — lighter
    if prev_dots:
        fig.add_trace(
            go.Scatter(
                x=prev_x,
                y=[label] * n_prev,
                mode="markers",
                marker=dict(
                    size=DOT_SIZE, color="#7C98AC", line=dict(width=1, color="white")
                ),
                hovertext=[
                    f"<b>{d['effective_start'].strftime('%d %b %H:%M')}</b><br>"
                    f"Articles: {d['size']}<br>"
                    f"Sources: {', '.join(sorted(set(d['sources'])))}"
                    for d in prev_dots
                ],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name=label,
            )
        )

    # today dots — highlighted
    if today_dots:
        fig.add_trace(
            go.Scatter(
                x=today_x,
                y=[label] * len(today_dots),
                mode="markers",
                marker=dict(
                    size=DOT_SIZE,
                    color="#34557d",
                    line=dict(width=1, color="white"),
                ),
                hovertext=[
                    f"<b>{d['effective_start'].strftime('%d %b %H:%M')}</b><br>"
                    f"Articles: {d['size']}<br>"
                    f"Sources: {', '.join(sorted(set(d['sources'])))}"
                    for d in today_dots
                ],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name=label,
            )
        )

# red line at x=0 = today boundary
fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)

fig.update_layout(
    height=max(300, len(active_sids) * 55 + 60),
    margin=dict(l=10, r=10, t=20, b=20),
    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    plot_bgcolor="#d1e7f1",
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        range=[x_left, x_right],
        zeroline=False,
    ),
    clickmode="event+select",
)

event = st.plotly_chart(
    fig, use_container_width=True, on_select="rerun", key="timeline"
)

# sync chart click → selectboxes (only process new clicks)
event_key = (
    str(event.selection.points[0])
    if event and event.selection and event.selection.points
    else None
)
if event_key and event_key != st.session_state.get("last_event_key"):
    st.session_state["last_event_key"] = event_key
    point = event.selection.points[0]
    curve_idx = point["curve_number"]
    point_idx = point["point_number"]

    # rebuild trace map: per story → line, prev (optional), today (optional)
    trace_map = []
    for s in active_sids:
        vis = story_visible[s]
        if not vis["today"] and not vis["prev"]:
            continue
        trace_map.append(None)  # line trace
        if vis["prev"]:
            trace_map.append((s, vis["prev"]))
        if vis["today"]:
            trace_map.append((s, vis["today"]))

    if curve_idx < len(trace_map) and trace_map[curve_idx] is not None:
        clicked_sid, dot_group = trace_map[curve_idx]
        if point_idx < len(dot_group):
            clicked_dot = dot_group[point_idx]
            # update story selectbox
            clicked_story_label = (
                f"{sid_labels[clicked_sid]}  |  "
                f"{story_meta[clicked_sid]['start'].strftime('%d %b')} – "
                f"{story_meta[clicked_sid]['end'].strftime('%d %b')}"
            )
            st.session_state["story_select"] = clicked_story_label
            st.session_state["last_story_select"] = clicked_sid
            # update dot selectbox
            clicked_dots_all = sorted(
                all_dots.get(clicked_sid, []), key=lambda d: d["effective_start"]
            )
            clicked_dot_idx = clicked_dots_all.index(clicked_dot)
            clicked_dot_label = (
                f"{'★ ' if clicked_dot['effective_start'].date() == selected_date else ''}"
                f"{clicked_dot['effective_start'].strftime('%d %b %H:%M')}  "
                f"({clicked_dot['size']} articles)"
            )
            st.session_state["dot_select"] = clicked_dot_label

# dot detail panel via selectboxes
st.markdown("---")
col_s, col_d = st.columns([2, 2])

with col_s:
    story_options = {
        f"{sid_labels[sid]}  |  {story_meta[sid]['start'].strftime('%d %b')} – {story_meta[sid]['end'].strftime('%d %b')}": sid
        for sid in active_sids
    }
    selected_label = st.selectbox(
        "Select story", list(story_options.keys()), key="story_select"
    )
    selected_sid = story_options[selected_label]

# show ALL dots for the selected story (full history), not just the window
all_story_dots = sorted(
    all_dots.get(selected_sid, []), key=lambda d: d["effective_start"]
)

with col_d:
    dot_options = {
        f"{'★ ' if d['effective_start'].date() == selected_date else ''}"
        f"{d['effective_start'].strftime('%d %b %H:%M')}  ({d['size']} articles)": i
        for i, d in enumerate(all_story_dots)
    }
    dot_labels = list(dot_options.keys())
    # reset dot selection when story changes
    if st.session_state.get("last_story_select") != selected_sid:
        today_indices = [
            i
            for i, d in enumerate(all_story_dots)
            if d["effective_start"].date() == selected_date
        ]
        default_dot = today_indices[0] if today_indices else 0
        st.session_state["dot_select"] = dot_labels[default_dot]
        st.session_state["last_story_select"] = selected_sid
    selected_dot_label = st.selectbox("Select dot", dot_labels, key="dot_select")
    selected_dot_idx = dot_options[selected_dot_label]

dot = all_story_dots[selected_dot_idx]

summary = dot_summaries.get((selected_sid, selected_dot_idx))
if summary:
    st.info(summary)

st.markdown(f"**Sources:** {', '.join(sorted(set(dot['sources'])))}")
st.markdown("**Articles:**")
for idx in sorted(dot["indices"], key=lambda i: df.iloc[i]["published_at_dt"]):
    row = df.iloc[idx]
    pub = row["published_at_dt"].strftime("%d %b %H:%M")
    st.markdown(f"- **{pub}** [{row['source']}] — [{row['title']}]({row['url']})")
