import os
import sys

# Add the root directory to the python path so it can find `app.py` and `src/`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app