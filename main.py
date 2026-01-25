#!/usr/bin/env python3
"""
Main launcher for Solana Copy Trading Bot
Run this file to start the system
"""

import sys
from pathlib import Path

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'infrastructure'))

# Import and run master
from master_v2 import main

if __name__ == "__main__":
    main()
