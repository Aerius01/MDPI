#!/usr/bin/env python3
"""
Command line interface for common utilities module.
Usage: python -m modules.common [options]
"""

import argparse
import sys
from .constants import CONSTANTS, ProcessingConstants

def main():
    parser = argparse.ArgumentParser(
        description='Display processing constants and utility information.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.common --show-constants
  python -m modules.common --show-all
        """
    )
    
    parser.add_argument('--show-constants', action='store_true',
                        help='Display all processing constants')
    parser.add_argument('--show-all', action='store_true',
                        help='Display all available information')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    try:
        if args.show_constants or args.show_all:
            print("Processing Constants:")
            print("=" * 50)
            for field in CONSTANTS.__dataclass_fields__:
                value = getattr(CONSTANTS, field)
                print(f"{field:25}: {value}")
        
        if args.show_all:
            print("\nAvailable Modules:")
            print("=" * 50)
            print("- modules.duplicate_detection")
            print("- modules.depth_profiling")
            print("- modules.flatfielding")
            print("- modules.object_detection")
            print("- modules.object_classification")
            print("\nRun 'python -m modules.MODULE_NAME --help' for module-specific options")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 