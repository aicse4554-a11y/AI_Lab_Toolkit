"""Small shared UI helpers used by every page."""
from pathlib import Path

import streamlit as st

from core.theme import CHOICES, theme_css

ROOT = Path(__file__).resolve().parent
ORIGINALS = ROOT / "original_scripts"


def theme_switcher() -> None:
    """Sidebar Auto / Light / Dark switch. The choice is kept while you move between pages."""
    with st.sidebar:
        choice = st.radio(
            "🌗 Theme",
            CHOICES,
            key="theme_choice",
            horizontal=True,
            help="Auto follows your device's light/dark setting. Light and Dark override it. "
                 "If you picked a theme in Streamlit's ⋮ → Settings menu, leave this on Auto.",
        )
    css = theme_css(choice)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def setup(title: str, icon: str, subtitle: str = "") -> None:
    """Page config + theme switch + title. Must be the first Streamlit call on a page."""
    st.set_page_config(page_title=f"{title} | AI Lab Toolkit", page_icon=icon, layout="wide")
    theme_switcher()
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)


def show_originals(*filenames: str) -> None:
    """Expander that shows the original lab program(s) a page is built from."""
    with st.expander("📄 Original lab program(s) behind this page"):
        for name in filenames:
            path = ORIGINALS / name
            if path.exists():
                st.markdown(f"**{name}**")
                st.code(path.read_text(encoding="utf-8"), language="python")
            else:
                st.caption(f"{name} not found in original_scripts/")
