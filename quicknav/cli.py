import sys
import os
import argparse
import re
from datetime import datetime
from . import __version__

def print_and_exit(msg):
    print(msg)
    sys.exit(0)

def validate_proj_num(arg):
    if not re.fullmatch(r"\d{5}", arg):
        print_and_exit("ERROR:Invalid argument (must be 5-digit project number)")
    return arg

def resolve_project_path(project_arg, custom_root=None):
    """Resolve project number or search term to project path(s)."""
    try:
        if custom_root:
            os.environ['QUICKNAV_ROOT'] = custom_root

        # Import here to use updated environment
        from .find_project_path import (
            get_onedrive_folder, get_project_folders, get_range_folder,
            search_project_dirs, search_by_name
        )

        onedrive_folder = get_onedrive_folder()
        pfolder = get_project_folders(onedrive_folder)

        # Check if it's a 5-digit project number
        if re.fullmatch(r"\d{5}", project_arg):
            proj_num = project_arg
            range_folder = get_range_folder(proj_num, pfolder)
            matches = search_project_dirs(proj_num, range_folder)
        else:
            # Search by name
            matches = search_by_name(project_arg, pfolder)

        return matches

    except Exception as e:
        print_and_exit(f"ERROR:{str(e)}")

def cmd_navigate_project(args):
    """Handle basic project navigation command."""
    matches = resolve_project_path(args.project, args.root)

    if not matches:
        if re.fullmatch(r"\d{5}", args.project):
            print_and_exit(f"ERROR:No project folder found for number {args.project}")
        else:
            print_and_exit(f"ERROR:No project folders found containing '{args.project}'")
    elif len(matches) == 1:
        print_and_exit(f"SUCCESS:{matches[0]}")
    else:
        if re.fullmatch(r"\d{5}", args.project):
            print_and_exit("SELECT:" + "|".join(matches))
        else:
            print_and_exit("SEARCH:" + "|".join(matches))

def cmd_navigate_document(args):
    """Handle document navigation command."""
    try:
        from .doc_navigator import navigate_to_document

        # Resolve project path first
        matches = resolve_project_path(args.project, args.root)

        if not matches:
            print_and_exit(f"ERROR:No project found for '{args.project}'")
        elif len(matches) > 1:
            print_and_exit("ERROR:Multiple projects found, please be more specific")

        project_path = matches[0]

        # Extract project code from path if available
        project_code = None
        project_name = os.path.basename(project_path)
        match = re.match(r"^(\d{5}) - ", project_name)
        if match:
            project_code = match.group(1)

        # Determine selection mode
        selection_mode = 'auto'
        if args.latest:
            selection_mode = 'latest'
        elif args.choose:
            selection_mode = 'choose'

        # Apply filters
        room_filter = args.room
        co_filter = args.co
        exclude_archive = not args.include_archive

        # Navigate to document
        result = navigate_to_document(
            project_path=project_path,
            doc_type=args.type,
            selection_mode=selection_mode,
            project_code=project_code,
            room_filter=room_filter,
            co_filter=co_filter,
            exclude_archive=exclude_archive
        )

        print_and_exit(result)

    except ImportError:
        print_and_exit("ERROR:Document navigation not available")
    except Exception as e:
        print_and_exit(f"ERROR:{str(e)}")

def create_parser():
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='quicknav',
        description='Navigate to project folders and documents'
    )

    parser.add_argument('--version', '-V', action='version',
                       version=f'quicknav {__version__}')

    # Add global options
    parser.add_argument('--root', help='Custom root directory path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Project navigation command (default)
    project_parser = subparsers.add_parser(
        'project',
        help='Navigate to project folder (default command)'
    )
    project_parser.add_argument(
        'project',
        help='5-digit project number or search term'
    )
    project_parser.set_defaults(func=cmd_navigate_project)

    # Document navigation command
    doc_parser = subparsers.add_parser(
        'doc',
        help='Navigate to project documents'
    )
    doc_parser.add_argument(
        'project',
        help='5-digit project number or search term'
    )
    doc_parser.add_argument(
        '--type', '-t',
        choices=[
            'lld', 'hld', 'change_order', 'sales_po', 'floor_plans',
            'scope', 'qa_itp', 'swms', 'supplier_quotes', 'photos'
        ],
        required=True,
        help='Type of document to find'
    )

    # Selection mode options (mutually exclusive)
    selection_group = doc_parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        '--latest',
        action='store_true',
        help='Return latest version of each document series'
    )
    selection_group.add_argument(
        '--choose',
        action='store_true',
        help='Return all matching documents for user selection'
    )

    # Filter options
    doc_parser.add_argument(
        '--room',
        type=int,
        help='Filter by room number'
    )
    doc_parser.add_argument(
        '--co',
        type=int,
        help='Filter by change order number'
    )
    doc_parser.add_argument(
        '--include-archive',
        action='store_true',
        help='Include archived/old documents'
    )
    doc_parser.set_defaults(func=cmd_navigate_document)

    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()

    # Handle legacy single argument case for backwards compatibility
    if len(sys.argv) == 2 and not sys.argv[1].startswith('-'):
        # Check if it's a version request
        if sys.argv[1] in ("--version", "-V"):
            print(f"quicknav {__version__}")
            sys.exit(0)

        # Treat as legacy project navigation
        sys.argv = ['quicknav', 'project', sys.argv[1]]

    args = parser.parse_args()

    # If no subcommand specified, show help
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    # Execute the subcommand
    args.func(args)

if __name__ == "__main__":
    main()