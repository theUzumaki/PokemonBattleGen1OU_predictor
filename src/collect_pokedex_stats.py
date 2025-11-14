#!/usr/bin/env python3
"""
collect_pokedex_stats.py

Scan data/train.jsonl and collect all species seen along with any base
stat fields present in the records. Outputs a json mapping species -> {
  "count": <n observations>,
  "bases": {"hp": mean, "atk": mean, ...},
  "examples": [<some representative entries>]
}

Usage:
  python -m src.collect_pokedex_stats --input data/train.jsonl --out data/pokedex_stats.json --max-records 200000
"""

from pathlib import Path
import argparse
import json
from collections import defaultdict
from typing import Dict, Any, List

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe", "speed"]
ALT_KEYS = {
    "atk": ["attack", "base_atk"],
    "def": ["defense", "base_def"],
    "spa": ["sp_atk", "spatk"],
    "spd": ["sp_def", "spdef"],
    "spe": ["speed", "base_spd", "base_speed"],
    "hp": ["hp", "base_hp"],
}


def _normalize_species(name: Any) -> str:
    if not isinstance(name, str):
        return "unknown"
    return name.strip().lower()


def _extract_stats_from_member(member: Any) -> Dict[str, float]:
    """Try to extract base stats from a team member dict or similar structure."""
    out = {}
    if not isinstance(member, dict):
        return out
    # Direct keys
    for k in STAT_KEYS:
        if k in member and isinstance(member[k], (int, float)):
            out[k] = float(member[k])
    # Nested stats dict
    stats = member.get("base_stats") or member.get("stats") or member.get("base")
    if isinstance(stats, dict):
        for k, v in stats.items():
            key = k.lower()
            if key in STAT_KEYS and isinstance(v, (int, float)):
                out[key] = float(v)
            # try alt canonicalization
            for canon, alts in ALT_KEYS.items():
                if key in alts and isinstance(v, (int, float)):
                    out[canon] = float(v)
    # Try other alt keys directly on member
    for canon, alts in ALT_KEYS.items():
        for alt in alts:
            if alt in member and isinstance(member[alt], (int, float)):
                out[canon] = float(member[alt])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/train.jsonl")
    parser.add_argument("--out", "-o", default="data/pokedex_stats.json")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    species_data: Dict[str, Dict[str, Any]] = {}
    # accumulator: species -> stat -> [vals]
    accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, int] = defaultdict(int)
    examples: Dict[str, List[Any]] = defaultdict(list)

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
                continue

            # Look for team details (both p1 and p2)
            for side in ("p1", "p2"):
                team = rec.get(f"{side}_team_details") or rec.get(f"{side}_team") or []
                if not isinstance(team, list):
                    continue
                for member in team:
                    species = None
                    if isinstance(member, dict):
                        species = member.get("species") or member.get("name")
                    elif isinstance(member, str):
                        species = member
                    species = _normalize_species(species)
                    if not species:
                        continue
                    counts[species] += 1
                    if len(examples[species]) < 3:
                        examples[species].append(member)
                    stats = _extract_stats_from_member(member)
                    for k, v in stats.items():
                        accum[species][k].append(v)

            timeline = rec.get("battle_timeline") or []
            p2_pkmns = set()
            for turn in timeline:
                p2_pkmns.add(turn.get("p2_pokemon_state").get("name"))
            
            for pkmn in p2_pkmns:
                species = _normalize_species(pkmn)
                if not species:
                    continue
                counts[species] += 1
                if len(examples[species]) < 3:
                    examples[species].append(pkmn)
                stats = _extract_stats_from_member(turn.get("p2_pokemon_state") or {})
                for k, v in stats.items():
                    accum[species][k].append(v)

    # summarize
    out = {}
    for sp, cnt in counts.items():
        entry = {"count": int(cnt), "bases": {}, "examples": examples.get(sp, [])}
        stats_map = accum.get(sp, {})
        for k, vals in stats_map.items():
            if vals:
                entry["bases"][k] = float(sum(vals) / len(vals))
        out[sp] = entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"Wrote {out_path} (species: {len(out)})")


if __name__ == '__main__':
    main()
