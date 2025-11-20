import subprocess
import sys


def main():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/depot_charging_optimization/dashboard.py"])
