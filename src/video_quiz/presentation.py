"""Safe presentation-boundary helpers."""

from html import escape


def escape_html(value: object) -> str:
    """Escape untrusted content before interpolating it into an HTML fragment."""
    return escape(str(value), quote=True)
