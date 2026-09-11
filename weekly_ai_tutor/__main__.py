"""Allows `python3 -m weekly_ai_tutor run --transcripts-dir DIR --minutes N`.

See `cli.py`'s module docstring for why this isn't yet a real installed
`weekly-ai-tutor` console script.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
