import json
import argparse
from pathlib import Path
from typing import Dict, List

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

def process_record(rec: dict):
    
    """
    print()
    print("TEAM 1")
    for pkmn in rec.get("p1_team_details", []):
        pkmn_name = pkmn.get("name", "").lower().replace("-", " ")
        print(pkmn_name)
    
    print()
    print("TEAM 2")
    print(rec.get("p2_lead_details", {}).get("name", "").lower().replace("-", " "))

    print()
    print("BATTLE LOG")
    """

    i = 0
    for turn in rec.get("battle_timeline", []):

        i+= 1
        
        pkmn1 = turn.get("p1_pokemon_state", {})
        pkmn1_name = pkmn1.get("name", "").lower().replace("-", " ")
        pkmn1_health = pkmn1.get("hp_pct", 0)
        pkmn1_status = pkmn1.get("status")
        pkmn1_effects = pkmn1.get("effects", {})
        pkmn1_boosts = pkmn1.get("boosts", {})
        pkmn2 = turn.get("p2_pokemon_state", {})
        pkmn2_name = pkmn2.get("name", "").lower().replace("-", " ")
        pkmn2_health = pkmn2.get("hp_pct", 0)
        pkmn2_status = pkmn2.get("status")
        pkmn2_effects = pkmn2.get("effects", {})
        pkmn2_boosts = pkmn2.get("boosts", {})

        try:
            move1 = turn.get("p1_move_details", {}).get("name", "").lower().replace("-", " ")
            move1_accuracy = turn.get("p1_move_details", {}).get("accuracy", 1)
            move1_priority = turn.get("p1_move_details", {}).get("priority", 0)
        except Exception:
            move1 = "null"
            move1_accuracy = 0
            move1_priority = 0
        try:
            move2 = turn.get("p2_move_details", {}).get("name", "").lower().replace("-", " ")
            move2_accuracy = turn.get("p2_move_details", {}).get("accuracy", 1)
            move2_priority = turn.get("p2_move_details", {}).get("priority", 0)
        except Exception:
            move2 = "null"
            move2_accuracy = 0
            move2_priority = 0

        """
        print("TURN" + f" {i}")
        print("Boosts p1: " + ", ".join([f"{k}: {v}" for k, v in pkmn1_boosts.items() if v != 0]))
        print("Boosts p2: " + ", ".join([f"{k}: {v}" for k, v in pkmn2_boosts.items() if v != 0]))
        print(f"{pkmn1_name} ({pkmn1_health:.2f}%) [{pkmn1_status}] vs {pkmn2_name} ({pkmn2_health:.2f}%) [{pkmn2_status}]")
        print(f"\n  {move1} (acc: {move1_accuracy}, priority: {move1_priority}) vs {move2} (acc: {move2_accuracy}, priority: {move2_priority})")  
        print()
        """

def process_live_pkmns(rec: dict):
    base_squad = {}
    for pkmn in rec.get("p1_team_details", []):
        pkmn_name = pkmn.get("name", "").lower().replace("-", " ")
        base_squad[pkmn_name] = 1.0
    
    for turn in rec.get("battle_timeline", []):
        pkmn1 = turn.get("p1_pokemon_state", {})
        pkmn1_name = pkmn1.get("name", "").lower().replace("-", " ")
        pkmn1_health = pkmn1.get("hp_pct", 0)
        if pkmn1_name in base_squad.keys():
            base_squad[pkmn1_name] = pkmn1_health
            """
            """
            if pkmn1_health <= 0:
                base_squad.pop(pkmn1_name)

    return base_squad

def process_type_coverage(moves) -> dict:
    type_coverage = {}
    
    return type_coverage

def process_moves(rec: dict) -> dict:
    moves = dict()
    for turn in rec.get("battle_timeline", []):
        move1 = turn.get("p1_move_details", {})
        if move1:
            move1 = move1.get("name", "").lower().replace("-", " ")
            move1_user = turn.get("p1_pokemon_state", {}).get("name", "").lower().replace("-", " ")
        move2 = turn.get("p2_move_details", {})
        if move2:
            move2 = move2.get("name", "").lower().replace("-", " ")
            move2_user = turn.get("p2_pokemon_state", {}).get("name", "").lower().replace("-", " ")

        if move1:
            if move1_user not in moves:
                moves[move1_user] = {move1: 1}
            elif move1 not in moves.get(move1_user, {}):
                moves[move1_user][move1] = 1
            else:
                moves[move1_user][move1] += 1
        if move2:
            if move2_user not in moves:
                moves[move2_user] = {move2: 1}
            elif move2 not in moves.get(move2_user, {}):
                moves[move2_user][move2] = 1
            else:
                moves[move2_user][move2] += 1
    return moves

def process_jsonl(input_path: str, max_records: int = None, only_rec: int = None) -> None:
    in_path = Path(input_path)

    written = 0
    with in_path.open("r", encoding="utf-8") as fh:
        writer = None

        avg_lasting_pkmn_win = {}
        avg_lasting_pkmn_lose = {}
        avg_pokemons_alive_win = {}
        avg_pokemons_alive_lose = {}
        moves = dict()
        collected_moves = set()

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

            if only_rec is not None and i != only_rec:
                continue
            process_record(rec)

            lasting_pkmns = process_live_pkmns(rec)
            for pkmn_name, health in lasting_pkmns.items():
                if rec.get("player_won", True):
                    if pkmn_name not in avg_lasting_pkmn_win.keys():
                        avg_lasting_pkmn_win[pkmn_name] = []
                    avg_lasting_pkmn_win[pkmn_name].append(health)

                    if pkmn_name not in avg_pokemons_alive_win.keys():
                        avg_pokemons_alive_win[pkmn_name] = 0
                    avg_pokemons_alive_win[pkmn_name] += 1
                else:
                    if pkmn_name not in avg_lasting_pkmn_lose.keys():
                        avg_lasting_pkmn_lose[pkmn_name] = []
                    avg_lasting_pkmn_lose[pkmn_name].append(health)

                    if pkmn_name not in avg_pokemons_alive_lose.keys():
                        avg_pokemons_alive_lose[pkmn_name] = 0
                    avg_pokemons_alive_lose[pkmn_name] += 1

            new_moves = process_moves(rec)
            for pkmn_name, move_list in new_moves.items():
                for move, amount in move_list.items():
                    if (pkmn_name, move) not in collected_moves:
                        collected_moves.add((pkmn_name, move))
                        if pkmn_name not in moves:
                            moves[pkmn_name] = dict()
                        if move not in moves[pkmn_name]:
                            moves[pkmn_name][move] = 0
                    moves[pkmn_name][move] += amount

        avg_moves_presence_win = {}
        avg_moves_presence_lose = {}
        total_alives = sum(avg_pokemons_alive_win.values()) + 1e-9
        weighted_moves_by_pkmn = {}
        for pkmn, presence in avg_pokemons_alive_win.items():
            for move, amt in moves.get(pkmn, {}).items():
                norm_amount = amt / (sum([amount for _, amount in moves.get(pkmn, {}).items()]) + 1e-9)
                weighted_moves_by_pkmn.setdefault(pkmn, {})[move] = norm_amount
                if move not in avg_moves_presence_win:
                    avg_moves_presence_win[move] = 0
                avg_moves_presence_win[move] += presence / total_alives * norm_amount

        for pkmn, presence in avg_pokemons_alive_lose.items():
            for move, amt in moves.get(pkmn, {}).items():
                norm_amount = amt / (sum([amount for _, amount in moves.get(pkmn, {}).items()]) + 1e-9)
                weighted_moves_by_pkmn.setdefault(pkmn, {})[move] = norm_amount
                if move not in avg_moves_presence_lose:
                    avg_moves_presence_lose[move] = 0
                avg_moves_presence_lose[move] += presence / total_alives * norm_amount

        differences = {}
        for move, presence in avg_moves_presence_win.items():
            win_presence = presence
            lose_presence = avg_moves_presence_lose.get(move, 0)
            differences[move] = win_presence - lose_presence

        with open("data/move_presence_differences.json", "w", encoding="utf-8") as diff_file:
            json.dump(differences, diff_file, indent=4)

        with open("data/moves.json", "w", encoding="utf-8") as moves_file:
            json.dump({k: list(v.items()) for k, v in weighted_moves_by_pkmn.items()}, moves_file, indent=4)

        win_avg_healths = {}
        win_avg_pkmns_alive = {}
        lose_avg_healths = {}
        for pkmn_name, healths in avg_lasting_pkmn_win.items():
            win_avg_healths[pkmn_name] = [sum(healths) / len(healths), len(healths)]
            win_avg_pkmns_alive[pkmn_name] = len(healths)

        for pkmn_name, healths in avg_lasting_pkmn_lose.items():
            lose_avg_healths[pkmn_name] = [sum(healths) / len(healths), len(healths)]

        for pkmn_name in set(list(win_avg_healths.keys()) + list(lose_avg_healths.keys())):
            win_health = win_avg_healths.get(pkmn_name, [0.0, 0])[0]
            weight = win_avg_healths.get(pkmn_name, [0.0, 0])[1] + lose_avg_healths.get(pkmn_name, [0.0, 0])[1]
            lose_health = lose_avg_healths.get(pkmn_name, [0.0, 0])[0]

            """
            print(f"{pkmn_name}: Win Avg Health = {win_health:.4f}, Lose Avg Health = {lose_health:.4f}")
            print(f"DIFFERENCE: {win_health - lose_health:.4f}")
            print(f"Weighted DIFFERENCE: {(win_health - lose_health) * weight:.4f}")
            print()
             """

        for pkmn_name, move_list in moves.items():
            moves[pkmn_name] = list(move_list)
            if pkmn_name in avg_lasting_pkmn_win:
                pass

        

def main():
    parser = argparse.ArgumentParser(description="Extract 20 features from JSONL battles dataset")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", "-o", required=False, help="Output directory for CSV")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of records for quick tests")
    parser.add_argument("--only-rec", type=int, default=None, help="Process only a specific record index")
    args = parser.parse_args()

    process_jsonl(args.input, args.max_records, args.only_rec if 'only_rec' in args else None)


if __name__ == "__main__":
    main()