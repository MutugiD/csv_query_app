import streamlit.web.cli as stcli
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "chat_interface.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost"
    ]
    sys.exit(stcli.main())