# conftest.py — shared test fixtures
import sys
from pathlib import Path

# Ensure src/ is on the path when running tests from project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
