"""CSS for the in-app Auto / Light / Dark switch.

Streamlit follows the device's colour scheme by default (``prefers-color-scheme``). To force the opposite
look we invert the whole page (including pop-up menus, tables and charts) and flip hues back so colours stay
natural, then re-invert images so photos and digits look normal.
"""

CHOICES = ["Auto", "Light", "Dark"]

_FLIP = "filter: invert(1) hue-rotate(180deg);"


def theme_css(choice: str) -> str:
    """Return CSS for the chosen mode ("" for Auto).

    Dark  -> flip the page only when the device is in light mode.
    Light -> flip the page only when the device is in dark mode.
    """
    if choice not in ("Light", "Dark"):
        return ""
    device = "light" if choice == "Dark" else "dark"
    return f"@media (prefers-color-scheme: {device}) {{ html {{ {_FLIP} }} img, video {{ {_FLIP} }} }}"
