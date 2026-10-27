"""Traffic flow analysis pipeline."""

import os

# ultralytics pip-installs missing packages at run time unless this is off (CLAUDE.md rule 6).
# It is read when ultralytics is first imported, so it must be set before that import.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
