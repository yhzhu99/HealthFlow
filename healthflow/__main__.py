"""
Entry point for running HealthFlow as a module
python -m healthflow
"""

from .cli import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())