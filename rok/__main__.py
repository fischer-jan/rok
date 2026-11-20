"""Start Rok: `python -m rok [FILE.vox]`. Rok.app runs this file directly.

`python -m rok run SCRIPT.py [FILE.vox]` runs a script with no window: the
script finds a headless `session` (rok/headless.py) ready, with FILE open.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if len(sys.argv) > 1 and sys.argv[1] == "run":
    from rok.headless import main as run_main  # noqa: E402

    sys.exit(run_main(sys.argv[2:]))

from rok.app.app import main  # noqa: E402

main()
