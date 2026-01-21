# intervention_faithfulness/__main__.py
"""
Package entrypoint so users can run:

  python -m intervention_faithfulness <cmd> [args...]

Delegates to intervention_faithfulness.cli.main().
"""

from __future__ import annotations

from intervention_faithfulness.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
