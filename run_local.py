#!/usr/bin/env python3
"""
Local runner for the Form Processing System
This file helps resolve import issues when running locally on Windows
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import and run the main app
if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "5000"])