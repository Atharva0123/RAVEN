"""Root Streamlit entry point; delegates UI implementation to dashboard/app.py."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "dashboard" / "app.py"),
                   run_name="__main__")
