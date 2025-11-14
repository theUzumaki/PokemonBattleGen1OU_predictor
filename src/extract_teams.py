#!/usr/bin/env python3
"""
extract_teams.py

Scan a JSONL of battles and extract all unique p1 team configurations
(using the first Pokémon in `p1_team_details` as the lead). Writes a
JSON file with teams grouped by lead and occurrence counts.

Usage:
  python -m src.extract_teams --input data/train.jsonl --out data/meta_teams.json --max-records 200000

Output format (example):
{
  "by_lead": {
    "alakazam": [
      {"team": ["alakazam", "snorlax", "gengar", ...], "count": 42},
      ...
    ],
    ...
  },
  "all_teams": [
    {"team": [...], "count": 123},
    ...
  ],
  "n_records_processed": 12345
}

This script is defensive about missing keys and normalizes species names
to lowercase.
"""

from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter
from typing import List, Dict, Any


def _normalize_species(name: str) -> str:
    if not isinstance(name, str):
        return "unknown"
    return name.strip().lower()


def _extract_species_list(team_details: Any) -> List[str]:
    """Given `p1_team_details` (usually a list), return ordered list of species names.
    Supports elements that are dicts with keys 'species' or 'name', or plain strings.
    """
    species = []
    if not isinstance(team_details, list):
        return species
    for entry in team_details:
        if isinstance(entry, dict):
            sp = entry.get("species") or entry.get("name") or entry.get("species_name")
            if sp is None:
                # Some datasets nest under 'pokemon' or similar
                sp = entry.get("pokemon")
        else:
            sp = entry if isinstance(entry, str) else None
        species.append(_normalize_species(sp) if sp is not None else "unknown")
    return species


def main():
    parser = argparse.ArgumentParser(description="Extract p1 team configurations from JSONL and group by lead")
    parser.add_argument("--input", "-i", default="data/train.jsonl", help="Path to input JSONL")
    parser.add_argument("--out", "-o", default="data/meta_teams.json", help="Path to output JSON file")
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit of records to process")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    by_lead: Dict[str, Counter] = defaultdict(Counter)
    all_teams_counter: Counter = Counter()
    n_proc = 0

    with in_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if args.max_records and i >= args.max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # skip malformed
                continue

            team_details = rec.get("p1_team_details") or rec.get("p1_team") or rec.get("p1_team_info")
            species_list = _extract_species_list(team_details)
            if not species_list:
                continue
            lead = species_list[0]
            rest = species_list[1:]
            # Canonicalize non-lead members by sorting so different orders of the
            # same remaining Pokémon are treated as the same team configuration.
            canonical_rest = tuple(sorted(rest))
            # Represent canonical team as (lead, *sorted_rest)
            team_tuple = (lead,) + canonical_rest
            by_lead[lead][team_tuple] += 1
            all_teams_counter[team_tuple] += 1
            n_proc += 1

    # Convert counters to sorted lists
    out_obj = {"by_lead": {}, "all_teams": [], "n_records_processed": n_proc}

    for lead, counter in by_lead.items():
        items = []
        for team_tuple, cnt in counter.most_common():
            items.append({"team": list(team_tuple), "count": int(cnt)})
        out_obj["by_lead"][lead] = items

    for team_tuple, cnt in all_teams_counter.most_common():
        out_obj["all_teams"].append({"team": list(team_tuple), "count": int(cnt)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out_obj, fh, indent=2)

    print(f"Wrote {out_path} (processed {n_proc} records)")


if __name__ == "__main__":
    main()
