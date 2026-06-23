#!/usr/bin/env python3
"""
Launch the Measure GUI application.

Usage:
    python run_gui.py
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_gui.app import main

if __name__ == "__main__":
    main()
