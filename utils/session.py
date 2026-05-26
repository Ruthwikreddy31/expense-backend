import streamlit as st

DEFAULT_USER_ID = "local-user"
DEFAULT_USERNAME = "Local User"


def ensure_default_session() -> str:
    """Use a local default user so the app works without authentication."""
    st.session_state.logged_in = True
    st.session_state.user_id = st.session_state.get("user_id") or DEFAULT_USER_ID
    st.session_state.username = st.session_state.get("username") or DEFAULT_USERNAME
    return st.session_state.user_id
