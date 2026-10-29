import os
import html
import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HDFC Bank AI Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "reports" / "api_test_report.html"

try:
    API_BASE_URL = st.secrets.get(
        "API_BASE_URL", "http://127.0.0.1:8000"
    ).rstrip("/")
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")
except Exception:
    API_BASE_URL = os.getenv(
        "API_BASE_URL", "http://127.0.0.1:8000"
    ).rstrip("/")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


# =========================================================
# 2. SESSION STATE
# =========================================================

for key, default in {
    "admin_authenticated": False,
    "messages": [],
    "last_answer": "",
    "last_question": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# 3. STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .hero {
        padding: 27px;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e5eef5;
        margin-bottom: 0;
    }

    .answer-box {
        background: white;
        border: 1px solid #d9e2ec;
        border-left: 5px solid #087f8c;
        border-radius: 12px;
        padding: 22px;
        line-height: 1.8;
        overflow-wrap: anywhere;
        color: #243b53;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 16px;
    }

    h1, h2, h3 {
        color: #102a43;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. API FUNCTIONS
# =========================================================

def api_get(endpoint, timeout=12):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, "Backend connection failed. Start FastAPI first."

    except requests.exceptions.Timeout:
        return None, "Backend request timed out."

    except requests.exceptions.RequestException as exc:
        return None, f"API request failed: {exc}"

    except ValueError:
        return None, "API did not return valid JSON."


def api_ask(question):
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=90,
        )

        try:
            data = response.json()
        except ValueError:
            data = {}

        if not response.ok:
            return None, str(
                data.get("detail", f"HTTP {response.status_code}")
            )

        return data, None

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to FastAPI."

    except requests.exceptions.Timeout:
        return None, "The request timed out. Please try again."

    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def number(data, keys, default=0):
    if not isinstance(data, dict):
        return default

    for key in keys:
        if data.get(key) is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                pass

    return default


def extract_rows(data, keys=()):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def normalise_dataframe(rows):
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    rename = {
        "created_at": "timestamp",
        "recorded_at": "timestamp",
        "request_time": "timestamp",
        "latency_seconds": "latency",
        "elapsed_seconds": "latency",
    }

    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


# =========================================================
# 5. SIDEBAR AND NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("## 🏦 HDFC Bank")
    st.caption("AI Assistant Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Employee Assistant",
            "Admin Dashboard",
            "System Status",
        ],
    )

    st.divider()

    if st.button("🔄 Refresh current page", use_container_width=True):
        st.rerun()

    health, health_error = api_get("/health", timeout=5)

    if health_error:
        st.error("Backend disconnected")
    else:
        st.success("Backend connected")

    st.caption("HDFC Custom LLM Pipeline API")


# =========================================================
# 6. EMPLOYEE ASSISTANT
# =========================================================

if page == "Employee Assistant":

    st.markdown(
        """
        <div class="hero">
            <h1>HDFC Bank AI Assistant</h1>
            <p>Internal knowledge assistant for bank employees</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Ask banking FAQ or process-related questions. "
        "Verify important decisions using official bank policies."
    )

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder="Enter your banking-related question...",
            height=110,
            max_chars=8000,
        )

        submitted = st.form_submit_button(
            "Get Answer",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        question = question.strip()

        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Preparing your answer..."):
                data, error = api_ask(question)

            if error:
                st.error(error)

            else:
                status = str(data.get("status", "")).lower()
                answer = data.get("answer", "")

                if isinstance(answer, dict):
                    answer = answer.get("text", "")

                answer = str(answer or "").strip()

                if status in ("failure", "error") or not answer:
                    st.error(
                        "The assistant could not provide an answer. "
                        "Please try again."
                    )

                elif status == "blocked":
                    st.warning(answer)

                else:
                    st.session_state.last_question = question
                    st.session_state.last_answer = answer
                    st.session_state.messages.append({
                        "question": question,
                        "answer": answer,
                    })

    if st.session_state.last_answer:
        st.markdown("### Assistant Answer")

        st.markdown(
            '<div class="answer-box">'
            + html.escape(st.session_state.last_answer)
            + "</div>",
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Download answer",
            data=st.session_state.last_answer,
            file_name="hdfc_answer.txt",
            mime="text/plain",
        )

    if st.session_state.messages:
        with st.expander("Conversation history"):
            for item in reversed(st.session_state.messages[-10:]):
                st.markdown("**Question**")
                st.write(item["question"])
                st.markdown("**Answer**")
                st.write(item["answer"])
                st.divider()

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_answer = ""
            st.session_state.last_question = ""
            st.rerun()


# =========================================================
# 7. SYSTEM STATUS
# =========================================================

elif page == "System Status":

    st.title("⚙️ System Status")
    st.caption("Live connectivity and backend health")

    health, health_error = api_get("/health")
    metrics, metrics_error = api_get("/metrics")

    left, right = st.columns(2)

    with left:
        st.subheader("Backend Health")

        if health_error:
            st.error(health_error)
        else:
            st.success("FastAPI is responding")

            st.metric(
                "FAQ Count",
                number(health, ["faq_count", "total_faqs"]),
            )

            if "intent_classifier_loaded" in health:
                st.metric(
                    "Intent Classifier",
                    "Loaded" if health["intent_classifier_loaded"]
                    else "Not loaded",
                )

    with right:
        st.subheader("API Metrics")

        if metrics_error:
            st.error(metrics_error)
        else:
            st.metric(
                "Recorded Requests",
                int(number(metrics, ["total_requests"])),
            )

            st.metric(
                "Average Latency",
                f"{number(metrics, ['average_latency_seconds']):.2f} s",
            )

    st.markdown("### API Configuration")
    st.code(API_BASE_URL)


# =========================================================
# 8. ADMIN DASHBOARD - LOGIN
# =========================================================

elif page == "Admin Dashboard":

    st.title("📊 Admin Monitoring Dashboard")
    st.caption("Operational analytics and test results")

    if not ADMIN_PASSWORD:
        st.warning(
            "Set ADMIN_PASSWORD in .streamlit/secrets.toml "
            "before using the dashboard."
        )
        st.stop()

    if not st.session_state.admin_authenticated:

        with st.form("admin_login"):
            password = st.text_input(
                "Admin Password",
                type="password",
            )
            login = st.form_submit_button(
                "Login",
                type="primary",
            )

        if login:
            if password and password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")

        st.stop()

    top_left, top_right = st.columns([5, 1])

    with top_left:
        st.success("Admin session active")

    with top_right:
        if st.button("Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

    # -----------------------------------------------------
    # LOAD BACKEND DATA
    # -----------------------------------------------------

    metrics, metrics_error = api_get("/metrics")
    history, history_error = api_get("/metrics/history")
    trend, trend_error = api_get("/metrics/latency-trend")
    daily, daily_error = api_get("/metrics/daily")

    if metrics_error:
        st.error(f"Could not load metrics: {metrics_error}")
        st.stop()

    if not isinstance(metrics, dict):
        st.error("Invalid metrics response.")
        st.stop()

    total = int(number(metrics, ["total_requests", "total_questions"]))
    success = int(number(
        metrics, ["successful_requests", "success_requests"]
    ))
    failed = int(number(
        metrics, ["failed_requests", "failure_requests"]
    ))
    blocked = int(number(metrics, ["blocked_requests"]))
    latency = number(
        metrics, ["average_latency_seconds", "avg_latency_seconds"]
    )

    # -----------------------------------------------------
    # FILTER CONTROLS
    # -----------------------------------------------------

    st.divider()

    control1, control2 = st.columns([2, 2])

    with control1:
        show_charts = st.toggle("Show analytics charts", value=True)

    with control2:
        st.caption("Refresh this page to fetch the latest backend data.")

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    st.markdown("### Key Performance Indicators")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Questions", f"{total:,}")
    c2.metric(
        "Successful",
        f"{success:,}",
        delta=f"{success / total * 100:.1f}% of total" if total else None,
    )
    c3.metric(
        "Failed",
        f"{failed:,}",
        delta=f"{failed / total * 100:.1f}% of total" if total else None,
        delta_color="inverse",
    )
    c4.metric("Blocked", f"{blocked:,}")
    c5.metric("Average Latency", f"{latency:.2f} s")

    if total == 0:
        st.info("No requests recorded yet. Ask a question in Employee Assistant.")

    # -----------------------------------------------------
    # REQUEST ANALYTICS
    # -----------------------------------------------------

    if show_charts:
        st.divider()
        st.subheader("Request Analytics")

        chart1, chart2 = st.columns(2)

        outcome_df = pd.DataFrame({
            "Outcome": ["Successful", "Failed", "Blocked"],
            "Count": [success, failed, blocked],
        })

        with chart1:
            st.markdown("**Requests by outcome**")
            visible_df = outcome_df[outcome_df["Count"] > 0]

            if not visible_df.empty:
                st.bar_chart(
                    visible_df.set_index("Outcome"),
                    y="Count",
                )
            else:
                st.caption("No outcome data available.")

        with chart2:
            st.markdown("**Outcome breakdown**")

            if total:
                visible_df = outcome_df.copy()
                visible_df["Percentage"] = (
                    visible_df["Count"] / total * 100
                ).round(2)

                st.dataframe(
                    visible_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No recorded requests.")

    # -----------------------------------------------------
    # ERROR CATEGORIES
    # -----------------------------------------------------

    st.divider()
    st.subheader("Error Category Analysis")

    error_data = metrics.get(
        "error_categories",
        metrics.get("errors_by_category", {}),
    )

    if isinstance(error_data, dict) and error_data:
        error_df = pd.DataFrame(
            list(error_data.items()),
            columns=["Category", "Count"],
        )
    elif isinstance(error_data, list) and error_data:
        error_df = pd.DataFrame(error_data)
    else:
        error_df = pd.DataFrame()

    if not error_df.empty:
        if "Category" not in error_df.columns:
            if "category" in error_df.columns:
                error_df.rename(columns={"category": "Category"}, inplace=True)
            elif "error_category" in error_df.columns:
                error_df.rename(
                    columns={"error_category": "Category"},
                    inplace=True,
                )

        if "Count" not in error_df.columns:
            if "count" in error_df.columns:
                error_df.rename(columns={"count": "Count"}, inplace=True)
            elif "total" in error_df.columns:
                error_df.rename(columns={"total": "Count"}, inplace=True)

        if {"Category", "Count"}.issubset(error_df.columns):
            error_df["Count"] = pd.to_numeric(
                error_df["Count"], errors="coerce"
            ).fillna(0)

            col1, col2 = st.columns([3, 2])

            with col1:
                st.dataframe(
                    error_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.bar_chart(error_df.set_index("Category"), y="Count")
        else:
            st.json(error_data)
    else:
        st.info(
            "No error categories returned. The backend must record and "
            "return error_categories for this chart to populate."
        )

    # -----------------------------------------------------
    # LATENCY TREND
    # -----------------------------------------------------

    st.divider()
    st.subheader("Latency Trend")

    trend_rows = []

    if not trend_error:
        trend_rows = extract_rows(
            trend,
            ("trend", "latency_trend", "data", "points", "history"),
        )

    trend_df = normalise_dataframe(trend_rows)

    if not trend_df.empty:
        time_col = next(
            (c for c in ("timestamp", "date", "day", "time")
             if c in trend_df.columns),
            None,
        )
        latency_col = next(
            (c for c in (
                "latency", "latency_seconds",
                "average_latency_seconds", "avg_latency_seconds"
            ) if c in trend_df.columns),
            None,
        )

        if time_col and latency_col:
            trend_df[latency_col] = pd.to_numeric(
                trend_df[latency_col], errors="coerce"
            )
            trend_df = trend_df.dropna(subset=[latency_col])

            if not trend_df.empty:
                st.line_chart(
                    trend_df.set_index(time_col)[[latency_col]]
                )
            else:
                st.info("No numeric latency data.")
        else:
            st.dataframe(trend_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No latency trend returned by the backend yet."
            if not trend_error
            else f"Latency trend unavailable: {trend_error}"
        )

    # -----------------------------------------------------
    # DAILY METRICS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Daily Metrics")

    daily_rows = []

    if not daily_error:
        daily_rows = extract_rows(
            daily,
            ("daily", "daily_metrics", "data", "records"),
        )

    daily_df = normalise_dataframe(daily_rows)

    if not daily_df.empty:
        st.dataframe(daily_df, use_container_width=True, hide_index=True)

        date_col = next(
            (c for c in ("date", "day", "timestamp")
             if c in daily_df.columns),
            None,
        )

        count_cols = [
            c for c in (
                "successful_requests", "success_requests",
                "failed_requests", "failure_requests",
                "blocked_requests", "total_requests"
            ) if c in daily_df.columns
        ]

        if show_charts and date_col and count_cols:
            plot_df = daily_df.set_index(date_col).copy()

            for col in count_cols:
                plot_df[col] = pd.to_numeric(
                    plot_df[col], errors="coerce"
                )

            st.line_chart(plot_df[count_cols])
    else:
        st.info(
            "Daily metrics not available."
            if not daily_error
            else f"Daily metrics unavailable: {daily_error}"
        )

    # -----------------------------------------------------
    # REQUEST HISTORY + FILTERS + EXPORT
    # -----------------------------------------------------

    st.divider()
    st.subheader("Request History")

    history_rows = []

    if not history_error:
        history_rows = extract_rows(
            history,
            ("history", "requests", "records", "data", "items"),
        )

    history_df = normalise_dataframe(history_rows)

    if not history_df.empty:

        f1, f2 = st.columns([2, 2])

        with f1:
            search = st.text_input(
                "Search history",
                placeholder="Search status, category or request ID",
            )

        with f2:
            if "status" in history_df.columns:
                statuses = sorted(
                    history_df["status"].dropna().astype(str).unique().tolist()
                )
                selected_statuses = st.multiselect(
                    "Filter by status",
                    options=statuses,
                    default=statuses,
                )
            else:
                selected_statuses = []

        filtered_df = history_df.copy()

        if search:
            mask = filtered_df.astype(str).apply(
                lambda col: col.str.contains(
                    search, case=False, na=False
                )
            ).any(axis=1)
            filtered_df = filtered_df[mask]

        if selected_statuses and "status" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["status"].astype(str).isin(selected_statuses)
            ]

        st.caption(f"Showing {len(filtered_df)} of {len(history_df)} records")

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Export filtered history to CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="hdfc_request_history.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No history data available."
            if not history_error
            else f"Request history unavailable: {history_error}"
        )

    # -----------------------------------------------------
    # API TEST REPORT - RENDER REAL HTML
    # -----------------------------------------------------

    st.divider()
    st.subheader("API Test Report")

    if REPORT_PATH.exists():
        st.success("API test report found")

        st.caption("reports/api_test_report.html")

        try:
            report_html = REPORT_PATH.read_text(
                encoding="utf-8",
                errors="replace",
            )

            # Render the actual pytest-html page instead of displaying
            # the HTML markup as plain text.
            components.html(
                report_html,
                height=850,
                scrolling=True,
            )

        except OSError as exc:
            st.error(f"Unable to read test report: {exc}")

    else:
        st.warning(
            "Report not found at reports/api_test_report.html"
        )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.divider()
    st.caption(
        "Dashboard metrics reflect backend-recorded data. "
        "Charts remain empty when the corresponding data is unavailable."
    )