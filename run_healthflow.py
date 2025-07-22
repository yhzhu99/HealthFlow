#!/usr/bin/env python3
"""
Simple script to run HealthFlow system
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.cli import main

if __name__ == "__main__":
    asyncio.run(main())