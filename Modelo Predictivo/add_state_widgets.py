#!/usr/bin/env python3
"""
Add missing "state" keys to metadata.widgets in one or more .ipynb files.

Usage:
  python3 add_state_widgets.py notebook.ipynb
  python3 add_state_widgets.py *.ipynb
  python3 add_state_widgets.py --remove notebook.ipynb   # optional: remove metadata.widgets entirely

The script creates a timestamped backup next to each notebook before writing changes.
"""
import nbformat
from pathlib import Path
import shutil
import datetime
import sys
import argparse

def backup(path: Path):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_suffix(path.suffix + f".backup.{ts}")
    shutil.copy2(path, dest)
    return dest

def ensure_state_in_obj(obj):
    """
    Recursively walk obj (dict/list) and add 'state': {} to widget dicts that lack it.
    Returns (changed_bool).
    """
    changed = False
    if isinstance(obj, dict):
        # If this dict looks like a widget entry (has model_module/model_name or _model_name etc.)
        # and is missing 'state', add empty state.
        widget_keys = {'model_module','model_name','model_module_version','_model_name','_model_module'}
        if 'state' not in obj and any(k in obj for k in widget_keys):
            obj['state'] = {}
            changed = True
        # Recurse into values
        for v in obj.values():
            if isinstance(v, (dict, list)):
                if ensure_state_in_obj(v):
                    changed = True
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                if ensure_state_in_obj(item):
                    changed = True
    return changed

def fix_notebook(path: Path, remove=False):
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return False
    print(f"Processing {path} ...")
    bak = backup(path)
    print(f"  backup: {bak}")
    nb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
    meta = nb.get('metadata', {})
    if 'widgets' not in meta:
        print("  metadata.widgets not present — nothing to change.")
        return False
    if remove:
        del meta['widgets']
        nb['metadata'] = meta
        nbformat.write(nb, str(path))
        print("  metadata.widgets removed and notebook written.")
        return True
    changed = ensure_state_in_obj(meta['widgets'])
    if changed:
        nb['metadata'] = meta
        nbformat.write(nb, str(path))
        print("  added missing 'state' entries and wrote notebook.")
    else:
        print("  no missing 'state' entries found.")
    return changed

def main():
    p = argparse.ArgumentParser(description="Add missing 'state' keys to metadata.widgets in notebooks.")
    p.add_argument('notebooks', nargs='+', help='One or more .ipynb files')
    p.add_argument('--remove', action='store_true', help='Remove metadata.widgets instead of adding state')
    args = p.parse_args()
    any_changed = False
    for nb in args.notebooks:
        try:
            changed = fix_notebook(Path(nb), remove=args.remove)
            any_changed = any_changed or changed
        except Exception as e:
            print(f"ERROR processing {nb}: {e}", file=sys.stderr)
    sys.exit(0 if any_changed else 0)

if __name__ == "__main__":
    main()