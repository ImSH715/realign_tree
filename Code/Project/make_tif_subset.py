import argparse
import csv
import os
import re
from pathlib import Path


DATE_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)")
FOLDER_RE = re.compile(r"20\d{2}[-_](\d{2})")


def parse_month(path: str):
    name = os.path.basename(path)
    match = DATE_RE.search(name)
    if match:
        day, month, year = match.groups()
        month_i = int(month)
        if 1 <= month_i <= 12:
            return month_i, f"{year}-{month}-{day}", "filename_ddmmyyyy"

    for part in Path(path).parts:
        match = FOLDER_RE.search(part)
        if match:
            month_i = int(match.group(1))
            if 1 <= month_i <= 12:
                return month_i, f"{part}", "folder"

    return None, "", ""


def find_tifs(roots):
    paths = []
    for root in roots:
        root = Path(root)
        for pattern in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
            paths.extend(root.rglob(pattern))
    return sorted(set(p.resolve() for p in paths))


def safe_link_name(path: Path, index: int):
    stem = path.stem
    suffix = path.suffix.lower()
    parent = path.parent.name.replace(" ", "_").replace("/", "_")
    return f"{index:06d}_{parent}_{stem}{suffix}"


def main():
    p = argparse.ArgumentParser(description="Build a symlinked TIFF subset by parsed month/season.")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--include_months", default="")
    p.add_argument("--exclude_tokens", nargs="*", default=["Raw_photographs", ".files", "vis_output", "labels"])
    p.add_argument("--manifest_csv", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    include_months = {
        int(x) for x in re.split(r"[, ]+", args.include_months.strip()) if x
    }

    output_root = Path(args.output_root)
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else output_root / "tif_subset_manifest.csv"

    rows = []
    selected = []
    for path in find_tifs(args.roots):
        path_str = str(path)
        if any(token in path_str for token in args.exclude_tokens):
            continue
        month, parsed_date, source = parse_month(path_str)
        keep = month is not None and (not include_months or month in include_months)
        rows.append({
            "path": path_str,
            "month": month if month is not None else "",
            "parsed_date": parsed_date,
            "date_source": source,
            "keep": int(keep),
        })
        if keep:
            selected.append(path)

    print(f"Found TIFFs     : {len(rows)}")
    print(f"Selected TIFFs  : {len(selected)}")
    if include_months:
        print(f"Included months : {sorted(include_months)}")

    if args.dry_run:
        for row in rows[:20]:
            print(row)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    link_dir = output_root / "tifs"
    link_dir.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "month", "parsed_date", "date_source", "keep", "link"])
        writer.writeheader()
        for i, row in enumerate(rows):
            link = ""
            if row["keep"]:
                src = Path(row["path"])
                dst = link_dir / safe_link_name(src, i)
                if not dst.exists():
                    os.symlink(src, dst)
                link = str(dst)
            row = dict(row)
            row["link"] = link
            writer.writerow(row)

    print(f"Subset root     : {output_root}")
    print(f"Symlink TIFF dir: {link_dir}")
    print(f"Manifest        : {manifest_csv}")


if __name__ == "__main__":
    main()
