#!/usr/bin/env python3
"""
jsonl_sanitize_check.py

Checks a JSONL (JSON-per-line) file for common issues like:
- trailing commas at end of line
- trailing backslashes at end of line
- invalid escape sequences (via json.loads)
- invisible characters (NBSP, CR, etc.)

Optionally fixes simple issues and writes a cleaned output file.
Creates a .bak backup before writing.

Usage:
  python jsonl_sanitize_check.py path/to/file.jsonl
  python jsonl_sanitize_check.py path/to/file.jsonl --fix
  python jsonl_sanitize_check.py path/to/file.jsonl --fix --normalize-quotes
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path
import re

VALID_ESCAPES = set(list(r'\/bfnrt"u'))

SMARTS = {
    '\u2018': "'",  # ‘
    '\u2019': "'",  # ’
    '\u201C': '"',  # “
    '\u201D': '"',  # ”
    '\u2013': '-',  # – en dash
    '\u2014': '-',  # — em dash
    '\u00A0': ' ',  # NBSP
}

def show_context(s: str, pos: int, width: int = 30) -> str:
    start = max(0, pos - width)
    end = min(len(s), pos + width)
    caret = " " * (pos - start) + "^"
    return s[start:end] + "\n" + caret

def guess_invalid_escape_positions(s: str):
    # Find backslashes that are not followed by a valid escape char
    # Note: this is a heuristic (JSON can contain \\ and \uXXXX which are valid)
    # We'll still warn if we see \ followed by an invalid char.
    issues = []
    i = 0
    while i < len(s) - 1:
        if s[i] == "\\":
            nxt = s[i+1]
            if nxt not in VALID_ESCAPES:
                issues.append(i)
        i += 1
    return issues

def normalize_quotes(s: str) -> str:
    return "".join(SMARTS.get(ch, ch) for ch in s)

def has_trailing_comma(line: str) -> bool:
    # true if after trimming whitespace/newlines, the line ends with a comma
    return line.rstrip("\r\n \t").endswith(",")

def has_trailing_backslash(line: str) -> bool:
    return line.rstrip("\r\n").endswith("\\")

def find_invisible_chars(s: str):
    # Look for control characters (except tab/newline) and NBSP
    inv = []
    for i, ch in enumerate(s):
        cat = unicodedata.category(ch)
        if ch == '\t' or ch == '\n' or ch == '\r':
            continue
        if ch == '\u00A0':  # NBSP
            inv.append((i, 'NBSP'))
        elif cat.startswith('C'):  # other control chars
            inv.append((i, f'CTRL({ord(ch)})'))
    return inv

def process_file(path: Path, do_fix: bool, do_norm: bool):
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines(True)  # keep line endings
    problems = 0
    fixed_lines = []

    for idx, line in enumerate(lines, start=1):
        original_line = line
        line_work = line

        trailing_comma = has_trailing_comma(line_work)
        trailing_backslash = has_trailing_backslash(line_work)
        invisibles = find_invisible_chars(line_work)

        # Warn about suspicious escapes (heuristic)
        escape_positions = guess_invalid_escape_positions(line_work)

        parse_error = None
        try:
            json.loads(line_work)
        except json.JSONDecodeError as e:
            parse_error = e

        if parse_error or trailing_comma or trailing_backslash or invisibles or escape_positions:
            problems += 1
            print(f"[Line {idx}] Potential issue(s):")

            if trailing_comma:
                print("  - Trailing comma at end of line")

            if trailing_backslash:
                print("  - Trailing backslash at end of line")

            if invisibles:
                spots = ", ".join([f"{pos}:{kind}" for pos, kind in invisibles])
                print(f"  - Invisible/control characters detected at {spots}")

            # Only show escape warnings if json.loads failed to avoid noise
            if parse_error:
                if escape_positions:
                    # Filter positions near error pointer
                    near = [p for p in escape_positions if abs(p - parse_error.pos) <= 5]
                    if near:
                        print(f"  - Suspicious backslash near position(s): {near}")
                print("  - json error:", parse_error)
                print(show_context(line_work, parse_error.pos))

        # Fixes
        if do_fix:
            changed = False
            lw = line_work

            # Remove trailing comma/backslash at EOL
            if trailing_comma:
                lw = lw.rstrip("\r\n \t")
                if lw.endswith(","):
                    lw = lw[:-1]
                    changed = True
                lw += "\n" if not original_line.endswith("\n") else "\n"
            if trailing_backslash:
                lw = lw.rstrip("\r\n")
                if lw.endswith("\\"):
                    lw = lw[:-1]
                    changed = True
                lw += "\n"

            # Normalize smart quotes/NBSP if requested
            if do_norm:
                new_lw = normalize_quotes(lw)
                if new_lw != lw:
                    lw = new_lw
                    changed = True

            # If still not parseable, try a conservative escape for stray backslashes not forming valid sequences
            if changed:
                try:
                    json.loads(lw)
                except json.JSONDecodeError as e:
                    # attempt to replace backslash+space or backslash+newline with backslash-escaped form (rare)
                    lw2 = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', lw)
                    try:
                        json.loads(lw2)
                        lw = lw2
                        changed = True
                    except json.JSONDecodeError:
                        pass

            if changed:
                fixed_lines.append(lw)
            else:
                fixed_lines.append(original_line)
        else:
            fixed_lines.append(original_line)

    if do_fix:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text("".join(lines), encoding="utf-8")
            print(f"\nBackup written to: {backup}")
        out_path = path
        out_path.write_text("".join(fixed_lines), encoding="utf-8")
        print(f"Cleaned file written to: {out_path}")

    if problems == 0:
        print("No issues detected. ✅")
    else:
        print(f"\nDetected potential issues on {problems} line(s).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path", type=Path, help="Path to JSONL file")
    ap.add_argument("--fix", action="store_true", help="Write fixes in place (creates .bak backup)")
    ap.add_argument("--normalize-quotes", action="store_true", help="Normalize smart quotes and NBSP to ASCII")
    args = ap.parse_args()

    if not args.jsonl_path.exists():
        print(f"File not found: {args.jsonl_path}", file=sys.stderr)
        sys.exit(1)

    process_file(args.jsonl_path, do_fix=args.fix, do_norm=args.normalize_quotes)

if __name__ == "__main__":
    main()
