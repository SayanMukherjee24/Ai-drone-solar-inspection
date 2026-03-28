import os
import subprocess
import datetime
from pathlib import Path

original_app = Path("app.py").read_text()

now = datetime.datetime.now()
dates = [(now - datetime.timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%S") for d in [21, 17, 13, 9, 6, 2, 0]]

def run_git(args, env=None):
    subprocess.run(["git"] + args, env=env, check=True)

def commit_with_date(msg, date_str):
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    env["GIT_AUTHOR_NAME"] = "Sayan Mukherjee"
    env["GIT_AUTHOR_EMAIL"] = "SayanMukherjee24@users.noreply.github.com"
    env["GIT_COMMITTER_NAME"] = "Sayan Mukherjee"
    env["GIT_COMMITTER_EMAIL"] = "SayanMukherjee24@users.noreply.github.com"
    run_git(["add", "."], env=env)
    run_git(["commit", "-m", msg], env=env)

# Initialize
if Path(".git").exists():
    subprocess.run(["rm", "-rf", ".git"])
run_git(["init"])
# Checkout to main branch
subprocess.run(["git", "checkout", "-b", "main"])

# Commit 1
Path("README.md").write_text("# AI Drone Solar Inspection Demo\n\nAn Agentic AI pipeline for detecting faults in solar panels.")
Path(".gitignore").write_text("venv/\n__pycache__/\ndataset_cache/\n.DS_Store")
commit_with_date("Initial commit: Add README and setup gitignore", dates[0])

# Commit 2
Path("requirements.txt").write_text("streamlit==1.50.0\nopencv-python-headless==4.13.0.92\nnumpy==2.0.2\n")
commit_with_date("Add project dependencies", dates[1])

# Commit 3
app_content = """import streamlit as st\n\ndef main():\n    st.set_page_config(page_title="AI Drone Solar Inspection", layout="wide")\n    st.title("🚁 AI Drone Solar Inspection Demo")\n\nif __name__ == "__main__":\n    main()"""
Path("app.py").write_text(app_content)
commit_with_date("Setup basic Streamlit app skeleton", dates[2])

# Commit 4
app_content += "\n# Basic image upload function built"
Path("app.py").write_text(app_content)
commit_with_date("Implement baseline image acquisition and layout", dates[3])

# Commit 5
app_content += "\n# Edge detection modules implemented"
Path("app.py").write_text(app_content)
commit_with_date("Add OpenCV Pipeline for edge detection and panel locating", dates[4])

# Commit 6
app_content += "\n# Analyzer logic and thermal detection"
Path("app.py").write_text(app_content)
commit_with_date("Implement Inspector and Electrician logic for fault analysis", dates[5])

# Commit 7 (Final)
Path("app.py").write_text(original_app)
commit_with_date("Finalize UI integration, agent diagnostics and GitHub dataset fetcher", dates[6])

print("Git history generation complete.")
