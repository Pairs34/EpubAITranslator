import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from epub_translator.app import main

if __name__ == "__main__":
    main()
