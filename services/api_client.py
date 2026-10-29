import os
import requests
import streamlit as st


def get_api_base_url():
    try:
        return st.secrets["API_BASE_URL"].rstrip("/")
    except Exception:
        return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def ask_question(question: str):
    url = f"{get_api_base_url()}/ask"

    response = requests.post(
        url,
        json={"question": question},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_health():
    response = requests.get(
        f"{get_api_base_url()}/health",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_metrics():
    response = requests.get(
        f"{get_api_base_url()}/metrics",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()