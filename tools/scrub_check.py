"""
scrub_check.py - grep the repository tree for anything that must not be
published. Exit code 1 if anything matches. Run before every commit.

Built-in checks (every text file):
  - absolute Windows / cloud-drive / home paths
  - email addresses
  - a patient-id pattern of the form eu-west-1-<word>-<word>-<digits>
  - export filename fragments and vendor hostnames (generic patterns)
Date and clock-time checks:
  - results/ and results/reports/: no calendar date, clock time, session id
    or weekday name at all
  - data/: dates and times allowed only in the placeholder year 2000
  - README.md and docs/: dates allowed only from an explicit whitelist
--private-list <file>: extra literal patterns (one regex per line, # comments),
kept OUTSIDE the repository.
"""
import argparse
import re
import sys
from pathlib import Path

TEXT_EXT = {".py", ".md", ".txt", ".csv", ".json", ".cfg", ".toml", ".yml", ".yaml", ".gitignore", ""}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "processed"}   # processed/ is gitignored everywhere
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
DATE_RX = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{1,2}\s*" + MONTHS + r"\b"),
    re.compile(r"\b" + MONTHS + r"\s+\d{1,2}\b(?!:)"),
    re.compile(r"\b" + MONTHS + r"\s+\d{4}\b"),
]
TIME_RX = re.compile(r"\b\d{1,2}:\d{2}\b")
SID_RX = re.compile(r"\bS\d{3}(?:c\d)?\b")
WEEKDAY_RX = re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day|sday|nesday|rsday|urday)?\b")
GENERIC = [
    ("absolute path", re.compile(r"\b[A-Za-z]:\\|\b[A-Za-z]:/|/[a-z]/My Drive|My Drive|/Users/|\\Users\\|AppData|/home/[a-z]", re.I)),
    ("windows path fragment", re.compile(r"[A-Za-z][A-Za-z0-9 ]*\\[A-Za-z][A-Za-z0-9 ]*\\[A-Za-z]")),
    # any email address except a noreply address (git commits carry one by necessity)
    ("email", re.compile(r"(?<![A-Za-z0-9._%+-])(?!noreply@)[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("patient id", re.compile(r"eu-west-1-[a-z]+-[a-z]+-[0-9]+", re.I)),
    ("export filename / hostname", re.compile(r"Clarity_Export|export_Nick|Tomlin_Nick|Apple Health Data|Apple Health export\.zip|apple_health_export|glooko\.com|my\.glooko|api\.glooko|api/v3|graph/data|export_csv|devices_and_settings", re.I)),
    ("event log", re.compile(r"Event_Log|glucose_log|alcohol|illness", re.I)),
]
README_DATE_WHITELIST = ["December 2025", "July 2026", "31 December 2026", "January to March 2027",
                         "March 2027", "April 2027", "September 2026", "2026 Nick Tomlin", "Martinsson et al. (2020)",
                         "Cuya (2025)", "Grinsztajn et al. 2022"]


def iter_files(root):
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts) or p.name == "scrub_check.py":
            continue
        if p.is_file() and p.suffix.lower() in TEXT_EXT:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--private-list")
    ap.add_argument("--extra-text", help="extra text file to check with the generic patterns (e.g. a git log dump)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    private = []
    if args.private_list:
        for line in open(args.private_list, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                private.append(("private: " + line[:40], re.compile(line, re.I)))
    problems = []

    def check_generic(text, label):
        for name, rx in GENERIC + private:
            for m in rx.finditer(text):
                s = max(0, m.start() - 30); e = min(len(text), m.end() + 30)
                problems.append(f"{label}: {name}: ...{text[s:e]!r}...")

    for p in iter_files(root):
        rel = p.relative_to(root).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        check_generic(text, rel)
        top = rel.split("/")[0]
        if rel.startswith("results/"):
            for rx in DATE_RX + [TIME_RX, SID_RX, WEEKDAY_RX]:
                for m in rx.finditer(text):
                    s = max(0, m.start() - 30); e = min(len(text), m.end() + 30)
                    problems.append(f"{rel}: date/time/id/weekday: ...{text[s:e]!r}...")
        elif top == "data":
            for rx in DATE_RX:
                for m in rx.finditer(text):
                    if not m.group(0).startswith("2000-"):
                        s = max(0, m.start() - 30); e = min(len(text), m.end() + 30)
                        problems.append(f"{rel}: non-placeholder date: ...{text[s:e]!r}...")
        elif rel in ("README.md",) or top == "docs":
            for rx in DATE_RX:
                for m in rx.finditer(text):
                    ctx = text[max(0, m.start() - 20): m.end() + 20]
                    if not any(w in ctx for w in README_DATE_WHITELIST):
                        problems.append(f"{rel}: date outside the whitelist: ...{ctx!r}...")
            for m in TIME_RX.finditer(text):
                ctx = text[max(0, m.start() - 30): m.end() + 30]
                if "07:00 to 21:00" not in ctx and "07:00 and 21:00" not in ctx:
                    problems.append(f"{rel}: clock time: ...{ctx!r}...")
    if args.extra_text:
        check_generic(Path(args.extra_text).read_text(encoding="utf-8", errors="replace"), args.extra_text)

    if problems:
        print(f"SCRUB CHECK FAILED: {len(problems)} match(es)")
        for pr in problems[:200]:
            print("  " + pr)
        sys.exit(1)
    print(f"scrub check passed: 0 matches over {sum(1 for _ in iter_files(root))} text files"
          + (f" and {args.extra_text}" if args.extra_text else ""))


if __name__ == "__main__":
    main()
