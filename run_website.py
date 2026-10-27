"""One-command launcher for the AdaptiveLearn website.

    python run_website.py              # set up if needed, then open the site
    python run_website.py --quick      # faster first-time training (smaller model search)
    python run_website.py --retrain    # rebuild the models even if they exist
    python run_website.py --port 8600  # use another port

What it does:
  1. checks the Python version and installs any missing packages from requirements.txt
  2. trains the models once if results/models/ is empty
     (run_all.py when the raw OULAD files are present, otherwise train_from_processed.py)
  3. starts Streamlit and opens the site in your browser
"""
import argparse
import importlib.util
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "streamlit_app.py"
MODEL_DIR = ROOT / "results" / "models"
RAW_CLICKS = ROOT / "data" / "oulad_raw" / "studentVle.csv"
NEEDED_MODELS = ("classifier", "regressor", "clusterer", "recommender")

# pip package name -> import name
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn", "scipy": "scipy",
            "matplotlib": "matplotlib", "seaborn": "seaborn", "joblib": "joblib",
            "streamlit": "streamlit"}


def say(step, msg):
    print(f"[{step}] {msg}", flush=True)


def check_python():
    if sys.version_info < (3, 10):
        sys.exit(f"[X] Python 3.10 or newer is needed (you have {sys.version.split()[0]}). "
                 "Get it from https://python.org")


def install_packages():
    missing = [pip for pip, mod in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        say("1/3", "Python packages are already installed.")
        return
    say("1/3", f"Installing {', '.join(missing)} (first run takes a few minutes)...")
    # no -q: pip's own download progress shows, so a slow connection doesn't look like a freeze
    # --prefer-binary: take ready-made wheels, never a source archive that needs a C compiler
    # --timeout/--retries: survive slow or flaky connections (VPNs such as Cloudflare WARP)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--progress-bar", "on",
                             "--prefer-binary", "--only-binary", "numpy,scipy,pandas,scikit-learn,matplotlib",
                             "--timeout", "60", "--retries", "10",
                             "-r", str(ROOT / "requirements.txt")])
    if result.returncode != 0:
        sys.exit("[X] Package install failed. Check your internet connection (a VPN can slow "
                 "downloads a lot), then run start.bat again. It picks up where it stopped.")


def models_ready():
    return all((MODEL_DIR / f"{name}.joblib").exists() for name in NEEDED_MODELS)


def train(quick, force):
    if models_ready() and not force:
        say("2/3", "Trained models found, skipping training.")
        return
    script = "run_all.py" if RAW_CLICKS.exists() else "train_from_processed.py"
    note = "full pipeline on the raw OULAD data" if script == "run_all.py" else \
        "from the processed data in the repo, no recommender"
    say("2/3", f"Training models with {script} ({note}).")
    say("2/3", "This happens once and takes about 5-20 minutes. Leave this window open.")
    cmd = [sys.executable, str(ROOT / script)] + (["--quick"] if quick else [])
    if subprocess.run(cmd, cwd=ROOT).returncode != 0 or not models_ready():
        sys.exit("[X] Training did not finish. Scroll up for the error message.")


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def pick_port(preferred):
    for port in range(preferred, preferred + 20):
        if port_free(port):
            return port
    sys.exit(f"[X] Ports {preferred}-{preferred + 19} are all busy. Try --port with another number.")


def open_when_up(url, port, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        if not port_free(port):
            time.sleep(1.5)        # give Streamlit a moment to finish its first render
            webbrowser.open(url)
            return
        time.sleep(0.5)


def launch(port):
    port = pick_port(port)
    url = f"http://localhost:{port}"
    say("3/3", f"Starting the website at {url}")
    print("      Your browser will open by itself. Press Ctrl+C here to stop the site.\n", flush=True)
    threading.Thread(target=open_when_up, args=(url, port), daemon=True).start()
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP),
           "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    try:
        subprocess.run(cmd, cwd=ROOT)      # cwd=ROOT so .streamlit/config.toml (the theme) is used
    except KeyboardInterrupt:
        print("\nWebsite stopped.")


def main():
    parser = argparse.ArgumentParser(description="Set up and open the AdaptiveLearn website.")
    parser.add_argument("--quick", action="store_true", help="faster first-time training")
    parser.add_argument("--retrain", action="store_true", help="rebuild the models")
    parser.add_argument("--port", type=int, default=8501, help="port to serve on (default 8501)")
    args = parser.parse_args()

    print("=== AdaptiveLearn ===\n", flush=True)
    check_python()
    install_packages()
    train(args.quick, args.retrain)
    launch(args.port)


if __name__ == "__main__":
    main()
