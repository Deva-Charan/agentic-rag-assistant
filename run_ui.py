# run_ui.py
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path so 'app' package is found
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import streamlit.web.cli as stcli

sys.argv = ["streamlit", "run", os.path.join(project_root, "app", "ui.py"), "--server.headless", "true"]
sys.exit(stcli.main())