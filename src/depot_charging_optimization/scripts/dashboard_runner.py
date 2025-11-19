import subprocess
import sys


def run_dashboard():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/depot_charging_optimization/dashboard.py"])
