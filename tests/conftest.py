"""Put the training scripts on the import path for the test suite.

Author: James Edward Ball
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
