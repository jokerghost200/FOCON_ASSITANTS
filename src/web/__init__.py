from src.web.state import WebState

try:
    from src.web.app import create_app
except ImportError:
    create_app = None

__all__ = ["WebState", "create_app"]

