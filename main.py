#!/usr/bin/env python3
"""
BIBLOS v2 - Main Entry Point

Scripture Data Extraction System (SDES) Pipeline.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli.main import main

if __name__ == "__main__":
    main()
