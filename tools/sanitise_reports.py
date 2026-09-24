"""
sanitise_reports.py - strip dates, clock times, session ids and weekday names
from a printed report, and withhold the example-row tables that carry
timestamps. Prints how many replacements were made per pattern.

Usage: python sanitise_reports.py <in.txt> <out.txt> [--withhold "section title" ...]
                                                     [--drop-block "line prefix" ...]
A withheld section runs from a line containing the title to the next
"=====" rule; it is replaced by one note. A dropped block runs from a line
starting with the prefix to the next blank line.
"""
import argparse
import re

MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
PATTERNS = [
    # a path runs to the next colon or line end (paths here may contain spaces)
    ("file path", re.compile(r"(?:[A-Za-z]:[\\/]|/[a-z]/|\.\./)[^:\n]*"), "[path]"),
    ("iso datetime", re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"), "[date]"),
    ("d/m/y date", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[date]"),
    ("day month", re.compile(r"\b\d{1,2}\s*" + MONTHS + r"(?:\s+\d{4})?\b"), "[date]"),
    ("month day", re.compile(r"\b" + MONTHS + r"\s+\d{1,2}\b(?!:)"), "[date]"),
    ("month year", re.compile(r"\b" + MONTHS + r"\s+\d{4}\b"), "[month]"),
    ("day range before [date]", re.compile(r"\b\d{1,2}\s*-\s*\[date\]"), "[date]"),
    ("clock time", re.compile(r"\b\d{1,2}:\d{2}\b"), "[time]"),
    ("session id", re.compile(r"\bS\d{3}(?:c\d)?\b"), "[id]"),
    ("weekday", re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day|sday|nesday|rsday|urday)?\b"), "[weekday]"),
]
RULE = "=" * 20
NOTE = "example rows withheld (they carry timestamps)"


def sanitise(text, withhold=(), drop_blocks=(), remove=()):
    lines = text.split("\n")
    out, i, n_withheld, n_dropped = [], 0, 0, 0
    while i < len(lines):
        line = lines[i]
        if any(r in line for r in remove):
            # delete the section (title, rule and body) up to the rule that opens the next one
            if out and out[-1].startswith(RULE):
                out.pop()
            i += 1
            while i < len(lines) and not lines[i].startswith(RULE):
                i += 1
            n_withheld += 1
            continue
        if any(w in line for w in withhold):
            out.append(line)
            i += 1
            if i < len(lines) and lines[i].startswith(RULE):      # the rule under the title
                out.append(lines[i]); i += 1
            out.append(NOTE)
            n_withheld += 1
            while i < len(lines) and not lines[i].startswith(RULE):
                i += 1
            # keep the rule that opens the next section (the blank line before it too)
            if out and out[-1] != "":
                out.append("")
            continue
        if any(line.startswith(d) for d in drop_blocks):
            out.append("(block withheld: it carries clock times or weekdays)")
            n_dropped += 1
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    counts = {}
    for name, rx, rep in PATTERNS:
        text, k = rx.subn(rep, text)
        counts[name] = k
    counts["sections withheld"] = n_withheld
    counts["blocks dropped"] = n_dropped
    return text, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--withhold", action="append", default=[])
    ap.add_argument("--drop-block", action="append", default=[])
    ap.add_argument("--remove", action="append", default=[], help="delete a whole section by title")
    ap.add_argument("--header", default="")
    args = ap.parse_args()
    text = open(args.src, encoding="utf-8").read()
    text, counts = sanitise(text, args.withhold, args.drop_block, args.remove)
    if args.header:
        text = args.header.rstrip("\n") + "\n\n" + text
    with open(args.dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    total = sum(v for k, v in counts.items() if k not in ("sections withheld", "blocks dropped"))
    print(f"{args.dst}: {total} replacements " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
