#!/usr/bin/env python3
"""
feature_extractor.py

Extracts 20 specified features from a battle JSONL dataset and writes a CSV
with one row per battle. The script follows the exact per-feature logic
supplied in the specification. This implementation uses only the Python
standard library so it can run without installing pandas.

Usage example:
  python -m src.feature_extractor --input data/train.jsonl --output out_features --max-records 200

Notes/assumptions:
- Move and species names are normalized to lower-case for matching against
  HEALING_MOVES and BOOST_MOVES; POKEDEX_TYPES keys are stored in lower-case.
- TYPE_CHART is a Gen 1 style effectiveness table. If a Pokémon name is not
  present in `POKEDEX_TYPES`, that Pokémon is treated as having unknown types
  and neutral effectiveness (1.0) for coverage calcs.
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Set

# ---------- Generation 1 constants (partial but extensible) ----------

MAJOR_STATUS_LIST = {"slp", "par", "frz", "brn", "psn"}

# Updated based on moves_found.txt
HEALING_MOVES = {"softboiled", "recover", "rest"}

BOOST_MOVES = {"swordsdance", "amnesia", "agility"}
# Note: "barrier" and "doubleteam" are not present in moves_found.txt, so removed for now.

STATUS_MOVES = {"toxic", "confuseray", "stunpowder", "poisonpowder", "sleeppowder", "thunderwave", "will-o-wisp", "leechseed", "hypnosis", "stunspore", "sing", "lovelykiss"}

LOW_ACC_STATUS_MOVES = {"sing", "sleeppowder", "thunderwave", "lovelykiss", "stunspore", "hypnosis"}
# Added "hypnosis" from moves_found.txt; "will-o-wisp" is not present, so removed.

# Minimal POKEDEX_TYPES mapping for common Gen 1 Pokémon seen in dataset sample.
# Keys are lowercased for robust lookup. Extend this dictionary as needed.
POKEDEX_TYPES: Dict[str, List[str]] = {
    "alakazam": ["PSYCHIC"],
    "snorlax": ["NORMAL"],
    "tauros": ["NORMAL"],
    "exeggutor": ["GRASS", "PSYCHIC"],
    "chansey": ["NORMAL"],
    "gengar": ["GHOST", "POISON"],
    "starmie": ["WATER", "PSYCHIC"],
    "zapdos": ["ELECTRIC", "FLYING"],
    "jolteon": ["ELECTRIC"],
    "slowbro": ["WATER", "PSYCHIC"],
    "jynx": ["ICE", "PSYCHIC"],
    "victreebel": ["GRASS", "POISON"],
    "dragonite": ["DRAGON", "FLYING"],
    "cloyster": ["WATER", "ICE"],
    "lapras": ["WATER", "ICE"],
    "alakazam": ["PSYCHIC"],
    "jolteon": ["ELECTRIC"],
    "zapdos": ["ELECTRIC", "FLYING"],
    "rhydon": ["GROUND", "ROCK"],
    "golem": ["ROCK", "GROUND"],
    "articuno": ["ICE", "FLYING"],
    "persian": ["NORMAL"],
    "charizard": ["FIRE", "FLYING"],
}

# Minimal POKEDEX base speed mapping (Gen 1 approximate base Speed stat).
# Keys are lowercased species names. Extend as needed.
POKEDEX_BASE_SPEED: Dict[str, int] = {
    "alakazam": 120,
    "snorlax": 30,
    "tauros": 110,
    "exeggutor": 55,
    "chansey": 50,
    "gengar": 110,
    "starmie": 115,
    "zapdos": 100,
    "jolteon": 130,
    "slowbro": 30,
    "jynx": 95,
    "victreebel": 70,
    "dragonite": 80,
    "cloyster": 70,
    "lapras": 60,
    "rhydon": 40,
    "golem": 45,
    "alakazam": 120,
    "articuno": 85,
    "persian": 115,
    "charizard": 100,
}

POKEDEX_BASE_HP: Dict[str, int] = {
    "snorlax": 160,
    "tauros": 75,
    "exeggutor": 95,
    "chansey": 250,
    "gengar": 60,
    "starmie": 60,
    "zapdos": 90,
    "jolteon": 65,
    "slowbro": 95,
    "jynx": 65,
    "victreebel": 80,
    "dragonite": 91,
    "cloyster": 50,
    "lapras": 130,
    "rhydon": 106,
    "golem": 80,
    "alakazam": 55,
    "articuno": 90,
    "persian": 65,
    "charizard": 78,
}

# Auto-augment POKEDEX_BASE_SPEED with species discovered in data/meta_teams.json
# by assigning the median of known speeds to unknown species. This helps avoid
# entirely-empty p2 speed columns which produce NaNs in correlation computations.
try:
    import statistics
    meta_path = Path("data/meta_teams.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        # collect species seen in meta_teams (by_lead keys and team members)
        seen = set()
        by_lead = meta.get("by_lead", {}) if isinstance(meta, dict) else {}
        for lead, teams in (by_lead or {}).items():
            lead_norm = lead.strip().lower() if isinstance(lead, str) else "unknown"
            seen.add(lead_norm)
            for t in teams:
                team = t.get("team") if isinstance(t, dict) else None
                if isinstance(team, list):
                    for s in team:
                        if isinstance(s, str):
                            seen.add(s.strip().lower())

        # compute median of known speeds
        known_speeds = [v for v in POKEDEX_BASE_SPEED.values() if isinstance(v, (int, float))]
        if known_speeds:
            try:
                med = int(statistics.median(known_speeds))
            except Exception:
                med = int(sum(known_speeds) / len(known_speeds))
            for s in sorted(seen):
                if s not in POKEDEX_BASE_SPEED or POKEDEX_BASE_SPEED.get(s) is None:
                    POKEDEX_BASE_SPEED[s] = med
except Exception:
    # non-fatal: if meta file missing or malformed, continue with original mapping
    pass

# Simplified Gen 1 TYPE_CHART. If an attack/defender mapping is missing, default 1.0
# Values: 2.0 super-effective, 0.5 not very effective, 0 immune (0.0), otherwise 1.0
TYPE_CHART: Dict[str, Dict[str, float]] = {
    "NORMAL": {},
    "FIRE": {"GRASS": 2.0, "ICE": 2.0, "BUG": 2.0, "ROCK": 0.5, "FIRE": 0.5, "WATER": 0.5},
    "WATER": {"FIRE": 2.0, "ROCK": 2.0, "GROUND": 2.0, "WATER": 0.5, "GRASS": 0.5},
    "GRASS": {"WATER": 2.0, "GROUND": 2.0, "ROCK": 2.0, "FIRE": 0.5, "GRASS": 0.5, "POISON": 0.5, "FLYING": 0.5, "BUG": 0.5},
    "ELECTRIC": {"WATER": 2.0, "FLYING": 2.0, "GRASS": 0.5, "ELECTRIC": 0.5, "GROUND": 0.0},
    "ICE": {"GRASS": 2.0, "GROUND": 2.0, "FLYING": 2.0, "DRAGON": 2.0, "FIRE": 0.5, "WATER": 0.5, "ICE": 0.5},
    "FIGHTING": {"NORMAL": 2.0, "ROCK": 2.0, "ICE": 2.0, "PSYCHIC": 0.5, "FLYING": 0.5},
    "POISON": {"GRASS": 2.0, "POISON": 0.5, "GROUND": 0.5, "ROCK": 0.5, "GHOST": 0.5},
    "GROUND": {"FIRE": 2.0, "ELECTRIC": 2.0, "POISON": 2.0, "ROCK": 2.0, "GRASS": 0.5, "BUG": 0.5, "FLYING": 0.0},
    "FLYING": {"GRASS": 2.0, "FIGHTING": 2.0, "BUG": 2.0, "ELECTRIC": 0.5, "ROCK": 0.5},
    "PSYCHIC": {"FIGHTING": 2.0, "POISON": 2.0, "PSYCHIC": 0.5},
    "BUG": {"GRASS": 2.0, "PSYCHIC": 0.5, "FIRE": 0.5, "FLYING": 0.5, "ROCK": 0.5},
    "ROCK": {"FIRE": 2.0, "ICE": 2.0, "FLYING": 2.0, "BUG": 2.0, "FIGHTING": 0.5},
    "GHOST": {"PSYCHIC": 0.0, "GHOST": 2.0},
    "DRAGON": {"DRAGON": 2.0},
}

# ---------- Helper functions ----------

def _normalize_move_name(name: str) -> str:
    return name.strip().lower() if isinstance(name, str) else ""


def _normalize_species(name: str) -> str:
    return name.strip().lower() if isinstance(name, str) else ""


def get_pokemon_types(name: str) -> List[str]:
    if not name:
        return []
    return POKEDEX_TYPES.get(_normalize_species(name), [])


def get_pokedex_speed(name: str):
    """Return base Speed for a species if known, otherwise None."""
    if not name:
        return None
    return POKEDEX_BASE_SPEED.get(_normalize_species(name))


def effectiveness_of_move_vs(move_type: str, defender_types: List[str]) -> float:
    """Return effectiveness multiplier of an attacking move type vs a list of defender types.
    If a chart entry is missing, default to 1.0 for that pair.
    The total multiplier is the product across defender types (standard Gen 1 behavior).
    """
    if not move_type or not defender_types:
        return 1.0
    atk = move_type.strip().upper()
    multiplier = 1.0
    for d in defender_types:
        d_up = d.strip().upper()
        val = TYPE_CHART.get(atk, {}).get(d_up)
        if val is None:
            val = 1.0
        multiplier *= val
    return multiplier

def pokemon_type_matchup(pkmn1, pkmn2) -> float:
    """Compute type effectiveness multiplier of pkmn1 attacking pkmn2.
    Both pkmn1 and pkmn2 are species names (strings).
    If types are unknown, assume neutral effectiveness (1.0).
    """
    types1 = get_pokemon_types(pkmn1.get("name"))
    types2 = get_pokemon_types(pkmn2.get("name"))

    if not types1 or not types2:
        return 1.0
    # For each type of pkmn1, compute effectiveness vs pkmn2's types, then take max
    max_effectiveness = 1.0
    for t1 in types1:
        eff = effectiveness_of_move_vs(t1, types2)
        if eff > max_effectiveness:
            max_effectiveness = eff
    return max_effectiveness

# ---------- Feature extraction logic per specification ----------

def extract_20_features(record: Dict[str, Any]) -> Dict[str, Any]:
    timeline = record.get("battle_timeline")
    if not isinstance(timeline, list):
        timeline = []

    features: Dict[str, Any] = {}

    # --- New Features ---

    # Attack_move_ratio_delta removed (low importance)

    # Amount_of_misses_delta removed (low importance / unreliable)
    # timeline and features already initialized above; remove duplicate

    # Compute status-inflicted counts for each player, then fold into deltas
    p1_sleep_successful = 0
    p1_frz_successful = 0
    p1_par_successful = 0
    p2_sleep_successful = 0
    p2_frz_successful = 0
    p2_par_successful = 0
    p1_sleep_inflicted = 0
    p1_frz_inflicted = 0
    p1_par_inflicted = 0
    p2_sleep_inflicted = 0
    p2_frz_inflicted = 0
    p2_par_inflicted = 0

    for i in range(1, len(timeline)):
        cur = timeline[i]
        prev = timeline[i - 1]
        # status changes applied to the opponent are recorded in the opponent's pokemon_state
        cur_status_on_p2 = (cur.get("p2_pokemon_state") or {}).get("status")
        prev_status_on_p2 = (prev.get("p2_pokemon_state") or {}).get("status")
        if isinstance(cur_status_on_p2, str):
            status = cur_status_on_p2.lower()
            if status == "slp":
                p1_sleep_inflicted += 1
            elif status == "frz":
                p1_frz_inflicted += 1
            elif status == "par":
                p1_par_inflicted += 1

        cur_status_on_p1 = (cur.get("p1_pokemon_state") or {}).get("status")
        prev_status_on_p1 = (prev.get("p1_pokemon_state") or {}).get("status")
        if isinstance(cur_status_on_p1, str):
            status = cur_status_on_p1.lower()
            if status == "slp":
                p2_sleep_inflicted += 1
            elif status == "frz":
                p2_frz_inflicted += 1
            elif status == "par":
                p2_par_inflicted += 1

    # Store deltas (p1 minus p2) instead of separate features
    features["sleep_inflicted_delta"] = p1_sleep_inflicted - p2_sleep_inflicted
    features["frz_inflicted_delta"] = p1_frz_inflicted - p2_frz_inflicted
    features["par_inflicted_delta"] = p1_par_inflicted - p2_par_inflicted
    #features["total_status_inflicted_delta"] = (p1_sleep_inflicted + p1_frz_inflicted + p1_par_inflicted) - (p2_sleep_inflicted + p2_frz_inflicted + p2_par_inflicted)

    # (p1_healing_moves_used removed)

    # 4/5. blended_faints_before_first_ko
    # Positive: P1 faints inflicted before suffering a faint
    # Negative: P2 faints inflicted before suffering a faint
    p1_fainted_mons_count = 0
    p2_fainted_mons_count = 0
    draw = 0
    for t in timeline:
        p1_state = t.get("p1_pokemon_state") or {}
        p2_state = t.get("p2_pokemon_state") or {}
        p1_status = p1_state.get("status")
        p2_status = p2_state.get("status")

        if p1_fainted_mons_count != 0 and p2_fainted_mons_count != 0:
            #print("OUT")
            break

        if p1_status == p2_status == "fnt":
            #print("DRAW")
            draw += 1
            break

        if p1_status == "fnt":
            p1_fainted_mons_count += 1
        elif p2_status == "fnt":
            p2_fainted_mons_count += 1
        #print(f"Status 1: {p1_status}, Status 2: {p2_status}, partial: {p1_fainted_mons_count}-{p2_fainted_mons_count}")
        
    # Blend into a single feature
    #print (f"P1 faints before first KO: {p1_fainted_mons_count}, P2 faints before first KO: {p2_fainted_mons_count}")
    blended = p1_fainted_mons_count if p1_fainted_mons_count > 0 else -p2_fainted_mons_count if p2_fainted_mons_count > 0 else 0
    #print (f"Blended faints before first KO: {blended}")
    features["blended_faints_before_first_ko"] = blended
    #features["both_kos"] = draw

    # p1_low_acc_status_hit_rate removed (low importance)

    # MISSED MOVES
    p1_missed_moves = 0
    p2_missed_moves = 0

    # 7 & 8. strategic switches per player: count voluntary switches (not caused by faint)
    p1_strategic = 0
    p2_strategic = 0
    for i in range(1, len(timeline)):
        p1_switch = False
        p2_switch = False

        # P1 voluntary switches
        p1_cur_name = (timeline[i].get("p1_pokemon_state") or {}).get("name")
        p1_prev_name = (timeline[i - 1].get("p1_pokemon_state") or {}).get("name")
        p1_prev_status = (timeline[i - 1].get("p1_pokemon_state") or {}).get("status")
        if p1_cur_name != p1_prev_name and p1_prev_name is not None:
            p1_switch = True
            # count as strategic if previous pokemon did not faint
            if not (isinstance(p1_prev_status, str) and p1_prev_status == "fnt"):
                p1_strategic += 1

        # P2 voluntary switches
        p2_cur_name = (timeline[i].get("p2_pokemon_state") or {}).get("name")
        p2_prev_name = (timeline[i - 1].get("p2_pokemon_state") or {}).get("name")
        p2_prev_status = (timeline[i - 1].get("p2_pokemon_state") or {}).get("status")
        if record.get("battle_id") == 105:
            print(f"P2 Switch Check: Cur: {p2_cur_name}, Prev: {p2_prev_name}, Prev Status: {p2_prev_status}")
        if p2_cur_name != p2_prev_name and p2_prev_name is not None:
            p2_switch = True
            if isinstance(p2_prev_status, str) and p2_prev_status != "fnt":
                if record.get("battle_id") == 105:
                    print(f"P2 Strategic Switch Detected at turn {i}")
                p2_strategic += 1

        # missed moves counting
        p1_move = timeline[i].get("p1_move_details")
        p2_move = timeline[i].get("p2_move_details")
        if p1_move and p2_move:
            old_health_p2 = (timeline[i - 1].get("p2_pokemon_state") or {}).get("hp_pct")
            new_health_p2 = (timeline[i].get("p2_pokemon_state") or {}).get("hp_pct")
            if p2_move.get("name") not in HEALING_MOVES and p1_move.get("name") not in HEALING_MOVES and p1_move.get("name") not in BOOST_MOVES and p1_move.get("name") not in STATUS_MOVES:
                if not p2_switch and new_health_p2 >= old_health_p2:
                    p1_missed_moves += 1

            old_health_p1 = (timeline[i - 1].get("p1_pokemon_state") or {}).get("hp_pct")
            new_health_p1 = (timeline[i].get("p1_pokemon_state") or {}).get("hp_pct")
            if p1_move.get("name") not in HEALING_MOVES and p2_move.get("name") not in HEALING_MOVES and p2_move.get("name") not in BOOST_MOVES and p2_move.get("name") not in STATUS_MOVES:
                if not p1_switch and new_health_p1 >= old_health_p1:
                    p2_missed_moves += 1

    # Emit only p2 strategic switches; p1_strategic_switches removed per request
    features["p2_strategic_switches"] = p2_strategic
    features["delta_missed_moves"] = p2_missed_moves - p1_missed_moves

    # early_game_hp_gradient removed (low importance)

    # momentum_shift_last_5_turns removed

    # p1_revealed_attacking_types_count removed (low importance)
    p1_types_seen: Set[str] = set()
    for t in timeline:
        m = t.get("p1_move_details")
        if m and m.get("category") in ("PHYSICAL", "SPECIAL") and isinstance(m.get("type"), str):
            p1_types_seen.add(m.get("type"))


    # 12. p2_revealed_attacking_types_count

    # p1_revealed_coverage_ratio removed (low importance)

    # 14. p2_revealed_coverage_ratio

    # 15 & 16. p1_resisted_hits_on_switch, p1_super_effective_hits_on_switch

    # p1_boost_moves_used removed (low importance)

    # p2_boost_moves_used removed



    # 20. p1_first_status_inflicted_turn
    # Interpretation: first turn P1 *landed* a major status on P2 (status change on P2)

    # include winner flag (player_won). Do NOT include record identifiers to avoid leakage.
    if record.get("player_won"):
        features["player_won"] = 1
    else:
        features["player_won"] = 0

    # New speed-related features
    # 1) p1_avg_speed: average of base_spd values present in p1_team_details entries
    # 2) p2_avg_speed: average base speed inferred from POKEDEX_BASE_SPEED for p2 team
    # 3) p2_spd_affidability: how many of the six p2 pokemon had known base speed values
    def _extract_base_spd_from_member(member: Any):
        # Accept dicts with 'base_spd', 'base_speed', or nested 'stats':{'spd':...}
        if isinstance(member, dict):
            if "base_spd" in member and isinstance(member["base_spd"], (int, float)):
                return float(member["base_spd"])
            if "base_speed" in member and isinstance(member["base_speed"], (int, float)):
                return float(member["base_speed"])
            stats = member.get("base_stats") or member.get("stats") or {}
            if isinstance(stats, dict):
                spd = stats.get("spd") or stats.get("speed")
                if isinstance(spd, (int, float)):
                    return float(spd)
            # sometimes species is provided and we can look it up
            species = member.get("species") or member.get("name")
            if isinstance(species, str):
                spd = get_pokedex_speed(species)
                if spd is not None:
                    return float(spd)
        return None

    """
    # p1_avg_speed
    p1_team = record.get("p1_team_details") or record.get("p1_team") or []
    p1_speeds = []
    if isinstance(p1_team, list):
        for m in p1_team:
            spd = _extract_base_spd_from_member(m)
            if spd is not None:
                p1_speeds.append(spd)
    features["p1_avg_speed"] = float(sum(p1_speeds) / len(p1_speeds)) if p1_speeds else 0.0

    # p2_avg_speed and p2_spd_affidability
    p2_team = set()
    p2_team_spds = []
    for t in timeline:
        pokemon = t.get("p2_pokemon_state")
        p2_team.add(pokemon.get("name"))

    for m in p2_team:
        spd = POKEDEX_BASE_SPEED.get(_normalize_species(m))
        if spd is not None:
            p2_team_spds.append(spd)

    features["p2_avg_speed"] = float(sum(p2_team_spds) / len(p2_team_spds)) if p2_team_spds else 0.0
    features["p2_spd_affidability"] = len(p2_team_spds) / 6.0 if p2_team_spds else 0.0
    """

    # lead_vel_delta
    features["lead_spd_delta"] = 0
    p1_pokemon = (record.get("p1_team_details")[0])
    p2_pokemon = (record.get("p2_lead_details"))

    p1_lead_spe = p1_pokemon.get("base_spe")
    p2_lead_spe = p2_pokemon.get("base_spe")
    features["lead_spd_delta"] = p1_lead_spe - p2_lead_spe

    """
    # lead_type_matchup
    features["lead_type_matchup"] = 0
    features["lead_type_matchup"] = pokemon_type_matchup(p1_pokemon, p2_pokemon)
    """

    #first_sleep
    features["first_sleep"] = 0

    for t in timeline:

        cur_status_on_p1 = (t.get("p1_pokemon_state") or {}).get("status")
        cur_status_on_p2 = (t.get("p2_pokemon_state") or {}).get("status")

        if isinstance(cur_status_on_p2, str) and cur_status_on_p2.lower() == "slp":
            features["first_sleep"] = t.get("turn")
            break
        elif isinstance(cur_status_on_p1, str) and cur_status_on_p1.lower() == "slp":
            features["first_sleep"] = -t.get("turn")
            break

        cur_status_on_p1 = (t.get("p1_pokemon_state").get("status"))
        cur_status_on_p2 = (t.get("p2_pokemon_state").get("status"))

    # avg_remaining_health
    features["p1_avg_remaining_health"] = 0.0
    name_pkmn1 = set()
    hp_pkmn1 = dict()
    features["p2_avg_remaining_health"] = 0.0
    name_pkmn2 = set()
    hp_pkmn2 = dict()

    # explosion/self-destruct causes faint
    p1_exp_faints = 0
    p1_exp_not_faints = 0
    p2_exp_faints = 0
    p2_exp_not_faints = 0

    # pokemon left alive p1 FOR TAUROS
    list_pkmn1_alive = [pkmn.get("name") for pkmn in record.get("p1_team_details")]

    list_pkmn2_known = set()

    # tauros status
    tauros_status = "nostatus"

    # longest hyperbeam streak p1
    hyperbeam_streak_p1 = 0

    for t in timeline:

        cur_status_on_p1 = (t.get("p1_pokemon_state").get("status"))
        cur_status_on_p2 = (t.get("p2_pokemon_state").get("status"))
        move_p1 = (t.get("p1_move_details"))
        move_p2 = (t.get("p2_move_details"))
        if move_p1 is not None:
            move_p1 = move_p1.get("name")
        if move_p2 is not None:
            move_p2 = move_p2.get("name")

        if t.get("p1_pokemon_state").get("name") == "tauros":
            tauros_status = cur_status_on_p1

        # avg_remaining_health
        p1_health = (t.get("p1_pokemon_state").get("hp_pct"))
        p2_health = (t.get("p2_pokemon_state").get("hp_pct"))
        p1_name = (t.get("p1_pokemon_state").get("name"))
        p2_name = (t.get("p2_pokemon_state").get("name"))

        list_pkmn2_known.add(p2_name)

        hp_pkmn1[p1_name] = p1_health
        hp_pkmn2[p2_name] = p2_health

        # EXPLOSION / SELF-DESTRUCT CAUSES FAINT
        if move_p1 is not None:
            
            move_p1_norm = _normalize_move_name(move_p1)
            if (record.get("battle_id") == 106):
                print(f"Turn {t.get('turn')}, Move P1: {move_p1_norm}, Status on P2: {cur_status_on_p2}")
            if move_p1_norm in {"explosion", "selfdestruct"} and cur_status_on_p2.lower() == "fnt":
                #print("FAINT] " + t.get("p1_pokemon_state").get("name") + " used " + move_p1_norm + " on " + t.get("p2_pokemon_state").get("name") + " and caused " + t.get("p2_pokemon_state").get("status"))
                #print()
                p1_exp_faints += 1
            elif move_p1_norm in {"explosion", "selfdestruct"} and cur_status_on_p2.lower() != "fnt":
                #print("UNFAINT] " + t.get("p1_pokemon_state").get("name") + " used " + move_p1_norm + " on " + t.get("p2_pokemon_state").get("name") + " and caused " + t.get("p2_pokemon_state").get("status")   )
                #print()
                p1_exp_not_faints += 1

            if move_p1_norm == "hyperbeam":
                #print("HYPERBEAM used by " + t.get("p1_pokemon_state").get("name") + " on turn " + str(t.get("turn")) + " in battle " + str(record.get("battle_id")))
                if cur_status_on_p2.lower() == "fnt":
                    #print("Current status of target " + t.get("p2_pokemon_state").get("name") + ": " + str(cur_status_on_p2))
                    hyperbeam_streak_p1 += 1
        if move_p2 is not None:
            move_p2_norm = _normalize_move_name(move_p2)
            if move_p2_norm in {"explosion", "selfdestruct"} and cur_status_on_p1.lower() == "fnt":
                #print("FAINT] " + t.get("p2_pokemon_state").get("name") + " used " + move_p2_norm + " on " + t.get("p1_pokemon_state").get("name") + " and caused " + t.get("p1_pokemon_state").get("status"))
                p2_exp_faints += 1
            elif move_p2_norm in {"explosion", "selfdestruct"} and cur_status_on_p1.lower() != "fnt":
                #print("UNFAINT] " + t.get("p2_pokemon_state").get("name") + " used " + move_p2_norm + " on " + t.get("p1_pokemon_state").get("name") + " and caused " + t.get("p1_pokemon_state").get("status"))
                p2_exp_not_faints += 1

        

    total_sum1 = 0
    total_sum2 = 0
    count1 = 0
    count2 = 0
    dead_pkmn1 = 0
    dead_pkmn2 = 0

    if record.get("battle_id") == 106:
        print(f"HP Pkmn1: {hp_pkmn1}")

    for _, hp in hp_pkmn1.items():
        if hp != 0:
            total_sum1 += hp
        else:
            dead_pkmn1 += 1
    for hp in hp_pkmn2.values():
        if hp != 0:
            total_sum2 += hp
        else:
            dead_pkmn2 += 1

    count1 = 6 - dead_pkmn1
    for _ in range(6 - len(hp_pkmn1)):
        total_sum1 += 1
    count2 = 6 - dead_pkmn2
    for _ in range(6 - len(hp_pkmn2)):
        total_sum2 += 1

    features["p1_avg_remaining_health"] = total_sum1 / 6
    #features["p1_pkmns_alive"] = count1
    features["p2_avg_remaining_health"] = total_sum2 / 6

    #print(f"ROW: {record.get('battle_id')}, P1 avg health: {features['p1_avg_remaining_health']:.2f}, P2 avg health: {features['p2_avg_remaining_health']:.2f}, P1 alive: {count1}, P2 alive: {count2}")
    #features["p2_pkmns_alive"] = count2
    #features["pkmns_alive_delta"] = count2 - count1


    # EXPLOSION / SELF-DESTRUCT CAUSES FAINT
    #features["p1_explosion_faints_adv"] = p1_exp_faints
    #features["p1_explosion_does_not_faint_adv"] = p1_exp_not_faints
    #features["p2_explosion_faints_adv"] = p2_exp_faints
    #features["p2_explosion_does_not_faint_adv"] = p2_exp_not_faints

    # TAUROS ALIVE
    dead_pkmn_list = [name for name, pkmn in hp_pkmn1.items() if pkmn == 0]
    if record.get("battle_id") == 106:
        print(f"Dead pkmn list: {dead_pkmn_list}")
        print(hp_pkmn1.items())
    list_pkmn1_alive = [name for name in list_pkmn1_alive if name not in dead_pkmn_list]

    score_move_roaster = 0
    for pkmn in list_pkmn1_alive:
        with open("data/moves.json", "r", encoding="utf-8") as f:
            with open("data/move_presence_differences.json", "r", encoding="utf-8") as f2:
                move_scores = json.load(f2)
                pkmn_moves = json.load(f)
                for move in pkmn_moves.get(pkmn, []):
                    score_move_roaster += move_scores[move[0]] * move[1]

    if record.get("battle_id") == 106:
        print(f"Pkmns alive list: {list_pkmn1_alive}")
        print(1.0 if "tauros" not in hp_pkmn1.keys() else hp_pkmn1["tauros"])

    if "tauros" in list_pkmn1_alive:
        features["p1_tauros_hp_pct"] = 1.0 if "tauros" not in hp_pkmn1.keys() else hp_pkmn1["tauros"]
        if record.get("battle_id") == 106:
            print("TAUROS ALIVE in battle " + str(record.get("battle_id")) + " with HP%: " + str(features["p1_tauros_hp_pct"]))
    else:
        features["p1_tauros_hp_pct"] = 0.0
        if record.get("battle_id") == 106:
            print("TAUROS NOT ALIVE in battle " + str(record.get("battle_id")))

    if "chansey" in list_pkmn1_alive:
        features["p1_chansey_hp_pct"] = 1.0 if "chansey" not in hp_pkmn1.keys() else hp_pkmn1["chansey"]
    else:
        features["p1_chansey_hp_pct"] = 0.0

    # Tauros in team1
    # features["p1_has_tauros"] = 1 if any(m.get("name") == "tauros" for m in record.get("p1_team_details", [])) else 0

    # Tauros status
    """
    if tauros_status != "nostatus":
        features["p1_tauros_status"] = 2
    elif tauros_status == "par":
        features["p1_tauros_status"] = 3
    elif tauros_status == "frz":
        features["p1_tauros_status"] = 1
    else:
        features["p1_tauros_status"] = 0
    """

    # HYPERBEAM STREAK
    # features["p1_hyperbeam_streak"] = hyperbeam_streak_p1
    # print("HYPERBEAM STREAK in battle " + str(record.get("battle_id")) + ": " + str(hyperbeam_streak_p1))

    return features


# ---------- CLI / file processing ----------

def process_jsonl(input_path: str, output_dir: str, max_records: int = None) -> None:
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "features.csv"

    written = 0
    with in_path.open("r", encoding="utf-8") as fh, out_csv.open("w", encoding="utf-8", newline="") as out_fh:
        writer = None
        for i, line in enumerate(fh):
            if max_records and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # skip malformed lines
                continue
            feats = extract_20_features(rec)
            if writer is None:
                # Dynamically use only present keys for header
                fieldnames = list(feats.keys())
                writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
                writer.writeheader()
            # ensure only these fields are written (others ignored)
            # convert any non-primitive to string safely if needed
            row = {k: feats.get(k) for k in writer.fieldnames}
            writer.writerow(row)
            written += 1
            if written == (106 + 1):  # print example from battle_id offset right
                print(f"Example from row{rec.get('battle_id')}: {row}")
    print(f"Wrote {written} feature rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract 20 features from JSONL battles dataset")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for CSV")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of records for quick tests")
    args = parser.parse_args()

    process_jsonl(args.input, args.output, args.max_records)


if __name__ == "__main__":
    main()
