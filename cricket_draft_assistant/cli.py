from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    web_app = Path(__file__).with_name("web.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(web_app), *sys.argv[1:]])
