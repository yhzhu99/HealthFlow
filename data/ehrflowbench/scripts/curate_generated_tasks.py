from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ehrflowbench.scripts.prepare_tasks.curate_generated_tasks import main
from data.ehrflowbench.scripts.prepare_tasks.curate_generated_tasks import run_curation


if __name__ == "__main__":
    main()
