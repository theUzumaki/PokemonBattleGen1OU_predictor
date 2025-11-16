"""
Feature extraction module copied into `ensemble_three/` to make the
package self-contained. This is a direct copy of the top-level
`feature_extractor.py` so the ensemble can be moved as a single folder.
"""

import pandas as pd
import json
import numpy as np

# Import dataclasses for type hints and data structures
from .variables import pkmn, move
from .variables import team, stats, adv_team
from .variables import battle as Battle
from .variables import battleline as Battleline


# ============================================================================
# PART 1: BATTLELINE EXTRACTION (from battleline_extractor.py)
# ============================================================================

def create_final_turn_feature(data: list[dict], is_train: bool = True) -> Battleline:
    """Create a battleline struct from raw JSON battle records.

    Args:
        data: list of battle dicts (parsed from JSONL)
        is_train: if True, expects each record to contain 'player_won' and will
                  populate the battle.win label. If False, the label will be set
                  to -1 to indicate unknown (useful for test data).

    Returns:
        Battleline dataclass instance containing parsed battles.
    """

    battleline = Battleline(battles={})
    for i, battle in enumerate(data):
        # If training data, read the label; otherwise set to -1 (unknown)
        if is_train:
            win_label = battle['player_won']
        else:
            win_label = -1

        battle_ = Battle(team1=None, team2=None, win=win_label)
        party1_details, party2_details = battle['p1_team_details'], battle['p2_lead_details']
        team1 = init_team_1(party1_details)
        team2 = init_adv_team()
        
        found_in_starting_team = set()
        found_in_adv_team = set()

        # moves 
        team1_moves = set()
        team2_moves = set()

        # status 
        team1_statusses = set()
        team2_statusses = set()
        
        # explore turns 
        for turn in battle['battle_timeline']:
            poke1 = turn['p1_pokemon_state']
            poke2 = turn['p2_pokemon_state']
            
            hp1, hp2 = return_turn_pokemon_hp_perc(poke1, poke2)
            idx_team = returnPokemonFromName(team1.pkmns, poke1['name'])
            team1.pkmns[idx_team].hps = hp1 

            # POKEMON ALIVE
            if poke2['name'] not in found_in_adv_team:
                team2.pkmn_dscvrd_alive += 1
            if hp2 == 0.0:
                team2.pkmn_dscvrd_alive -= 1
                team2.pkmn_alive -= 1
            if poke1['name'] not in found_in_starting_team:
                found_in_starting_team.add(poke1['name'])

            # POKEMON MOVES
            move_ = parseTurnMovesTeam1(turn)
            if move_ is not None:
                if move_.name not in team1_moves:
                    team1_moves.add(move_.name)
                    team1.pkmns[idx_team].moves.append(move(
                        name=move_.name,
                        cat=1 if move_.cat == 'physical' else 0,
                        type=move_.type,
                        base_pwr=move_.base_pwr,
                        accuracy=move_.accuracy,
                        priority=move_.priority
                    ))
            move2_ = parseTurnMovesTeam2(turn)
            if move2_ is not None:
                if move2_.name not in team2_moves:
                    team2_moves.add(move2_.name)
                    team2.types.append(move(
                        name=move2_.name,
                        cat=1 if move2_.cat == 'physical' else 0,
                        type=move2_.type,
                        base_pwr=move2_.base_pwr,
                        accuracy=move2_.accuracy,
                        priority=move2_.priority
                    ))

        lead_hp_2 = battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']
        team2.hp_leader = lead_hp_2
       
        team1.revealed = len(found_in_starting_team)
        battle_.team1 = team1
        battle_.team2 = team2
        battleline.battles[i] = battle_
    return battleline


def parse_turn_status_team1(poke1_state):
    """Parse status from team1 pokemon state."""
    if poke1_state['status'] == 'nostatus':
        return None
    return poke1_state['status']


def parse_turn_status_team2(poke2_state):
    """Parse status from team2 pokemon state."""
    pass


def parseTurnMovesTeam2(turn):
    """Parse move details for team2 from a turn."""
    if turn['p2_move_details'] is None:  # skips no move 
        return None
   
    move_ = turn['p2_move_details']
    return move(
        name=move_['name'],
        cat=move_['category'],
        type=move_['type'],
        base_pwr=move_['base_power'],
        accuracy=move_['accuracy'],
        priority=move_['priority']
    )


def parseTurnMovesTeam1(turn):
    """Parse move details for team1 from a turn."""
    if turn['p1_move_details'] is None:  # skips no move 
        return None
   
    move_ = turn['p1_move_details']
    return move(
        name=move_['name'],
        cat=move_['category'],
        type=move_['type'],
        base_pwr=move_['base_power'],
        accuracy=move_['accuracy'],
        priority=move_['priority']
    )


def returnPokemonFromName(team, pokemon_name):
    """Find pokemon index in team by name."""
    for i, pkm in enumerate(team):
        if pkm.id == pokemon_name:
            return i 
    return None


def return_turn_pokemon_hp_perc(pokemon_state1, pokemon_state2):
    """Returns the hp percentage of p1 and p2 at current turn."""
    hp1 = pokemon_state1['hp_pct']
    hp2 = pokemon_state2['hp_pct']
    return hp1, hp2


def init_adv_team():
    """Initialize adversary team structure."""
    team2 = adv_team(
        pkmn_alive=6,
        pkmn_dscvrd_alive=1,
        types=[],
        statuses=[],
        hp_leader=1.0
    )
    return team2


def init_team_1(p1_details):
    """Initialize team1 structure from p1_team_details."""
    team1 = team(pkmns=[], revealed=0)
    poke_list = []
    # p1_details is already a list
    for poke in p1_details:
        key = poke['name']
        poke_obj = pkmn(
            id=key, 
            hps=1.0,
            type1=poke['types'][0],
            type2=poke['types'][1],
            base_stats=stats(
                atk=poke['base_atk'],
                def_=poke['base_def'],
                spa=poke['base_spa'],
                spd=poke['base_spd'],
                spe=poke['base_spe'],
                hp=poke['base_hp']
            ),
            moves=[],
            status=[],
            effects=[],
            boosts=stats(
                atk=0, 
                def_=0, 
                spa=0, 
                spd=0, 
                spe=0, 
                hp=0
            )
        )
        poke_list.append(poke_obj)

    team1.pkmns = poke_list
    return team1


# ============================================================================
# PART 2: FEATURE EXTRACTION (from extractor.py)
# ============================================================================

def extract_battle_features(battleline: Battleline, max_moves: int = 4) -> np.ndarray:
    """
    Extract battle-level features that combine information from both teams.
    Each row represents one complete battle.
    
    Args:
        battleline: The battleline struct containing battles
        max_moves: Maximum number of moves per Pokemon (default: 4)
    
    Returns:
        numpy array where each row represents a battle's features
        
    Team1 individual Pokemon features (for each of 6 Pokemon):
        - HP percentage
        - Base stats (6 values)
        - Boosts (6 values: atk, def, spa, spd, spe, hp)
        - Status (one-hot, 8 values)
        - Effects (one-hot, 8 values)
        - Types (one-hot, 19 values for type1 and 19 for type2)
        - Move features (max_moves * (4 base features + 19 type features))
          - base_pwr, accuracy, priority, cat
          - type (one-hot, 19 values)
        
    Team2 (adversary) features:
        - pkmn_alive, pkmn_dscvrd_alive, hp_leader
        - Type presence (one-hot encoding)
        - Status presence (one-hot encoding)
    """
    
    battle_features_list = []
    
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    effects_list = ['firespin', 'wrap', 'substitute', 'clamp', 'confusion', 'typechange', 'noeffect', 'reflect']
    type_list = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
                 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy', 'notype']
    
    for battle_id, battle in battleline.battles.items():
        battle_features = []
        
        # Extract features for each Pokemon individually (up to 6 Pokemon)
        team1_pokemon = battle.team1.pkmns
        
        # Pad or truncate to exactly 6 Pokemon
        max_pokemon = 6
        for i in range(max_pokemon):
            if i < len(team1_pokemon):
                pkmn = team1_pokemon[i]
                
                # HP percentage
                battle_features.append(pkmn.hps)
                
                # Base stats
                battle_features.extend([
                    pkmn.base_stats.atk,
                    pkmn.base_stats.def_,
                    pkmn.base_stats.spa,
                    pkmn.base_stats.spd,
                    pkmn.base_stats.spe,
                    pkmn.base_stats.hp
                ])
                
                # Boosts
                battle_features.extend([
                    pkmn.boosts.atk,
                    pkmn.boosts.def_,
                    pkmn.boosts.spa,
                    pkmn.boosts.spd,
                    pkmn.boosts.spe,
                    pkmn.boosts.hp
                ])
                
                # Status (one-hot)
                for status in status_list:
                    battle_features.append(1 if pkmn.status == status else 0)
                
                # Effects (one-hot)
                for effect in effects_list:
                    battle_features.append(1 if effect in pkmn.effects else 0)

                # Types (one-hot) - type1 first, then type2
                for type_name in type_list:
                    battle_features.append(1 if type_name in pkmn.type1 else 0)
                for type_name in type_list:
                    battle_features.append(1 if type_name in pkmn.type2 else 0)
                
                # Move features
                for j in range(max_moves):
                    if j < len(pkmn.moves):
                        move = pkmn.moves[j]
                        battle_features.extend([
                            move.base_pwr,
                            move.accuracy,
                            move.priority,
                            move.cat
                        ])
                        # Move type (one-hot encoding)
                        for type_name in type_list:
                            battle_features.append(1 if move.type == type_name else 0)
                    else:
                        # Padding for missing moves: 4 base features + 19 type features
                        battle_features.extend([0] * (4 + len(type_list)))
            else:
                # Padding for missing Pokemon (all zeros)
                # HP + 6 base stats + 6 boosts + 8 status + 8 effects + 38 types (19*2) + (max_moves * (4 + 19)) moves
                padding_size = 1 + 6 + 6 + 8 + 8 + 38 + (max_moves * (4 + 19))
                battle_features.extend([0] * padding_size)
        
        # ===== TEAM2 (ADVERSARY) FEATURES =====
        battle_features.append(battle.team2.pkmn_alive)
        battle_features.append(battle.team2.pkmn_dscvrd_alive)
        battle_features.append(battle.team2.hp_leader)
        
        # Type presence (one-hot)
        for type_name in type_list:
            battle_features.append(1 if type_name in battle.team2.types else 0)
        
        # Status one-hot encoding for alive Pokemon
        # Check if each status is present in the alive Pokemon
        for status in status_list:
            battle_features.append(1 if status in battle.team2.statuses else 0)
        
        battle_features_list.append(battle_features)
    
    return np.array(battle_features_list)


def get_labels_from_battleline(battleline: Battleline) -> np.ndarray:
    """Extract win/loss labels from battleline."""
    labels = []
    for battle_id, battle in battleline.battles.items():
        labels.append(battle.win)
    return np.array(labels)


# ============================================================================
# MAIN - For testing
# ============================================================================

if __name__ == "__main__":
    train_data = []

    with open("data/train.jsonl", 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    print(f"Successfully loaded {len(train_data)} battles.")
    
    # Test battleline creation
    battleline = create_final_turn_feature(train_data[:10])  # Test with first 10 battles
    print(f"Created battleline with {len(battleline.battles)} battles")
    
    # Test feature extraction
    features = extract_battle_features(battleline, max_moves=4)
    print(f"Extracted features with shape: {features.shape}")
    
    # Test label extraction
    labels = get_labels_from_battleline(battleline)
    print(f"Extracted labels with shape: {labels.shape}")
    print(f"Wins: {np.sum(labels)}, Losses: {len(labels) - np.sum(labels)}")

"""
Feature Extractor for Pokémon Battle Prediction

This module extracts comprehensive features from battle JSON data including:
- Team composition statistics
- Opponent lead matchup analysis
- Battle dynamics from timeline (first 30 turns)
- Derived ratios and momentum indicators
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple


# Type effectiveness chart (simplified for common matchups)
TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0, 'steel': 0.5},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5, 'steel': 2},
    'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5},
    'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5, 'steel': 0.5},
    'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2, 'steel': 0.5},
    'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0, 'dark': 2, 'steel': 2, 'fairy': 0.5},
    'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2},
    'ground': {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2, 'steel': 2},
    'flying': {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'steel': 0.5},
    'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'dark': 0, 'steel': 0.5},
    'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5, 'dark': 2, 'steel': 0.5, 'fairy': 0.5},
    'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'steel': 0.5},
    'ghost': {'normal': 0, 'psychic': 2, 'ghost': 2, 'dark': 0.5},
    'dragon': {'dragon': 2, 'steel': 0.5, 'fairy': 0},
    'dark': {'fighting': 0.5, 'psychic': 2, 'ghost': 2, 'dark': 0.5, 'fairy': 0.5},
    'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2, 'rock': 2, 'steel': 0.5, 'fairy': 2},
    'fairy': {'fire': 0.5, 'fighting': 2, 'poison': 0.5, 'dragon': 2, 'dark': 2, 'steel': 0.5},
}


def get_type_effectiveness(attacking_type: str, defending_types: List[str]) -> float:
    """
    Calculate type effectiveness multiplier for an attack.
    
    Args:
        attacking_type: The type of the attacking move
        defending_types: List of types of the defending Pokémon
        
    Returns:
        Effectiveness multiplier (0, 0.25, 0.5, 1, 2, or 4)
    """
    if attacking_type == 'notype' or attacking_type not in TYPE_CHART:
        return 1.0
    
    multiplier = 1.0
    for def_type in defending_types:
        if def_type != 'notype' and def_type in TYPE_CHART[attacking_type]:
            multiplier *= TYPE_CHART[attacking_type][def_type]
    
    return multiplier


def extract_team_composition_features(team_details: List[Dict]) -> Dict[str, float]:
    """
    Extract aggregate statistics from the player's team.
    
    Features include (REFACTORED - removed redundant/low-importance features):
    - Mean, std, max of key base stats (hp, atk, spa, spe)
    - Type coverage metrics
    - Physical attacker distribution
    - Tank count
    
    Args:
        team_details: List of 6 Pokémon dictionaries with stats and types
        
    Returns:
        Dictionary of team composition features
    """
    features = {}
    
    # Extract base stats (ONLY keep high-importance stat aggregations)
    stats = {
        'hp': [p['base_hp'] for p in team_details],
        'atk': [p['base_atk'] for p in team_details],
        'def': [p['base_def'] for p in team_details],
        'spa': [p['base_spa'] for p in team_details],
        'spe': [p['base_spe'] for p in team_details],
    }
    
    # Keep only high-signal aggregations, remove redundant spd stats
    features['team_hp_mean'] = np.mean(stats['hp'])
    features['team_hp_std'] = np.std(stats['hp'])
    features['team_hp_max'] = np.max(stats['hp'])
    
    features['team_atk_mean'] = np.mean(stats['atk'])
    features['team_atk_min'] = np.min(stats['atk'])  # High importance
    
    features['team_def_min'] = np.min(stats['def'])  # High importance
    
    features['team_spa_mean'] = np.mean(stats['spa'])
    features['team_spa_max'] = np.max(stats['spa'])
    
    features['team_spe_mean'] = np.mean(stats['spe'])
    features['team_spe_std'] = np.std(stats['spe'])
    
    # Type coverage
    all_types = []
    for pokemon in team_details:
        for ptype in pokemon['types']:
            if ptype != 'notype':
                all_types.append(ptype)
    
    type_counts = Counter(all_types)
    features['team_unique_types'] = len(type_counts)
    
    # Physical attackers ONLY (removed special_attackers - perfectly correlated)
    physical_count = sum(1 for p in team_details if p['base_atk'] > p['base_spa'])
    features['team_physical_attackers'] = physical_count
    
    # Tank count (high HP and defense)
    tank_count = sum(1 for p in team_details 
                     if p['base_hp'] > 100 and (p['base_def'] + p['base_spa']) / 2 > 80)
    features['team_tank_count'] = tank_count
    
    # Speed tier - fast pokemon only (removed slow_pokemon - low importance)
    fast_count = sum(1 for p in team_details if p['base_spe'] > 100)
    features['team_fast_pokemon'] = fast_count
    
    # Overall team quality (sum of total base stats)
    total_stats = [sum([p['base_hp'], p['base_atk'], p['base_def'], 
                       p['base_spa'], p['base_spd'], p['base_spe']]) 
                   for p in team_details]
    features['team_avg_total_stats'] = np.mean(total_stats)
    features['team_min_total_stats'] = np.min(total_stats)
    
    return features


def extract_opponent_lead_features(p2_lead: Dict, p1_team: List[Dict]) -> Dict[str, float]:
    """
    Extract features about the opponent's lead Pokémon and matchup vs player team.
    REFACTORED: Removed redundant (spd) and low-importance (hp, atk, spe) features.
    
    Args:
        p2_lead: Opponent's lead Pokémon details
        p1_team: Player's team details
        
    Returns:
        Dictionary of opponent lead features
    """
    features = {}
    
    # Opponent lead base stats (KEEP only high-importance stats)
    features['opp_lead_def'] = p2_lead['base_def']
    features['opp_lead_spa'] = p2_lead['base_spa']
    # Removed: opp_lead_spd (perfect correlation with spa)
    # Removed: opp_lead_hp, opp_lead_atk, opp_lead_spe (low importance)
    
    features['opp_lead_total_stats'] = sum([
        p2_lead['base_hp'], p2_lead['base_atk'], p2_lead['base_def'],
        p2_lead['base_spa'], p2_lead['base_spd'], p2_lead['base_spe']
    ])
    
    # Type matchup advantages
    opp_types = [t for t in p2_lead['types'] if t != 'notype']
    
    # Count how many player Pokémon have type advantage vs opponent lead
    advantaged_count = 0
    disadvantaged_count = 0
    
    for p1_pokemon in p1_team:
        p1_types = [t for t in p1_pokemon['types'] if t != 'notype']
        
        # Check if player has advantage
        for p1_type in p1_types:
            effectiveness = get_type_effectiveness(p1_type, opp_types)
            if effectiveness > 1.0:
                advantaged_count += 1
                break
        
        # Check if opponent has advantage vs this Pokémon
        for opp_type in opp_types:
            effectiveness = get_type_effectiveness(opp_type, p1_types)
            if effectiveness > 1.0:
                disadvantaged_count += 1
                break
    
    features['team_vs_lead_advantage_count'] = advantaged_count
    features['team_vs_lead_disadvantage_count'] = disadvantaged_count
    features['team_vs_lead_net_advantage'] = advantaged_count - disadvantaged_count
    
    # Lead is physical attacker (removed is_special - perfect inverse correlation)
    features['opp_lead_is_physical'] = 1 if p2_lead['base_atk'] > p2_lead['base_spa'] else 0
    
    return features


def extract_battle_timeline_features(timeline: List[Dict], p1_team: List[Dict]) -> Dict[str, float]:
    """
    Extract dynamic features from the first 30 turns of battle.
    
    REFACTORED: Removed low-importance features, added advanced battle dynamics.
    
    Features include:
    - Damage differentials and HP trajectories
    - KO counts and timing
    - Move usage patterns and effectiveness
    - Status effect success rates
    - HP volatility and momentum swings
    - Turn-by-turn pressure metrics
    - Early/mid/late game performance
    
    Args:
        timeline: List of turn dictionaries (up to 30 turns)
        p1_team: Player's team for context
        
    Returns:
        Dictionary of battle timeline features
    """
    features = {}
    
    if not timeline:
        # Return zero features if no timeline
        return _get_empty_timeline_features()
    
    # Limit to first 30 turns
    timeline = timeline[:30]
    num_turns = len(timeline)
    features['num_turns'] = num_turns
    
    # Track HP changes and trajectories
    p1_hp_changes = []
    p2_hp_changes = []
    p1_hp_trajectory = []  # NEW: Track HP over time
    p2_hp_trajectory = []  # NEW
    p1_previous_hp = {}
    p2_previous_hp = {}
    
    # Track KOs and their timing
    p1_kos = 0
    p2_kos = 0
    p1_ko_turns = []  # NEW: When KOs happened
    p2_ko_turns = []  # NEW
    
    # Track move statistics
    p1_moves_power = []
    p2_moves_power = []
    p1_move_categories = {'PHYSICAL': 0, 'SPECIAL': 0, 'STATUS': 0}
    p2_move_categories = {'PHYSICAL': 0, 'SPECIAL': 0, 'STATUS': 0}
    
    # NEW: Track move effectiveness
    p1_type_effectiveness = []
    p2_type_effectiveness = []
    super_effective_count_p1 = 0
    super_effective_count_p2 = 0
    
    # Track status effects
    p1_status_inflicted = 0
    p2_status_inflicted = 0
    p1_status_turns = 0  # NEW: Duration of status
    p2_status_turns = 0  # NEW
    burn_poison_turns = 0  # NEW: Damaging status turns
    
    # Track who moves first
    p1_moved_first = 0
    
    # Track unique Pokémon revealed
    p1_pokemon_seen = set()
    p2_pokemon_seen = set()
    
    # Track boosts
    p1_total_boosts = []
    p2_total_boosts = []
    p1_max_boost = 0  # NEW: Peak boost achieved
    p2_max_boost = 0  # NEW
    boost_turns = 0  # NEW: Turns spent boosting
    
    # NEW: Track switches
    p1_switches = 0
    p2_switches = 0
    forced_switches = 0
    
    # NEW: Track HP pressure
    p1_turns_below_50 = 0
    p1_turns_critical = 0  # Below 25%
    comeback_hp_recovered = 0
    lowest_p1_hp = 100
    
    # NEW: Track momentum (advantage switching)
    current_advantage = None  # 'p1', 'p2', or None
    momentum_swings = 0
    favorable_turn_streak = 0
    max_favorable_streak = 0
    
    last_p1_active = None
    last_p2_active = None
    
    for turn_idx, turn in enumerate(timeline):
        turn_num = turn.get('turn', 0)
        
        # Player 1 state
        p1_state = turn.get('p1_pokemon_state', {})
        p1_name = p1_state.get('name', 'unknown')
        p1_hp = p1_state.get('hp_pct', 0)
        p1_status = p1_state.get('status', 'nostatus')
        p1_boosts = p1_state.get('boosts', {})
        
        p1_pokemon_seen.add(p1_name)
        p1_hp_trajectory.append(p1_hp)
        
        # Detect switches
        if last_p1_active and last_p1_active != p1_name:
            p1_switches += 1
        last_p1_active = p1_name
        
        # Player 2 state
        p2_state = turn.get('p2_pokemon_state', {})
        p2_name = p2_state.get('name', 'unknown')
        p2_hp = p2_state.get('hp_pct', 0)
        p2_status = p2_state.get('status', 'nostatus')
        p2_boosts = p2_state.get('boosts', {})
        
        p2_pokemon_seen.add(p2_name)
        p2_hp_trajectory.append(p2_hp)
        
        # Detect switches
        if last_p2_active and last_p2_active != p2_name:
            p2_switches += 1
        last_p2_active = p2_name
        
        # Track HP changes
        if p1_name in p1_previous_hp:
            hp_change = p1_hp - p1_previous_hp[p1_name]
            p1_hp_changes.append(hp_change)
            if p1_hp == 0 and p1_previous_hp[p1_name] > 0:
                p1_kos += 1  # P1 was KO'd
                p1_ko_turns.append(turn_num)
                forced_switches += 1
        p1_previous_hp[p1_name] = p1_hp
        
        if p2_name in p2_previous_hp:
            hp_change = p2_hp - p2_previous_hp[p2_name]
            p2_hp_changes.append(hp_change)
            if p2_hp == 0 and p2_previous_hp[p2_name] > 0:
                p2_kos += 1  # P2 was KO'd
                p2_ko_turns.append(turn_num)
        p2_previous_hp[p2_name] = p2_hp
        
        # Track HP pressure
        if p1_hp < 50:
            p1_turns_below_50 += 1
        if p1_hp < 25 and p1_hp > 0:
            p1_turns_critical += 1
        if p1_hp < lowest_p1_hp:
            lowest_p1_hp = p1_hp
        
        # Comeback detection
        if len(p1_hp_trajectory) > 1 and lowest_p1_hp < 30:
            if p1_hp > lowest_p1_hp + 20:  # Recovered 20+ HP
                comeback_hp_recovered = p1_hp - lowest_p1_hp
        
        # Move details
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        
        if p1_move:
            power = p1_move.get('base_power', 0)
            category = p1_move.get('category', 'STATUS')
            move_type = p1_move.get('type', 'normal')
            
            if power > 0:
                p1_moves_power.append(power)
                
                # Calculate type effectiveness
                p2_types = [t for t in p2_state.get('types', ['normal']) if t != 'notype']
                effectiveness = get_type_effectiveness(move_type, p2_types)
                p1_type_effectiveness.append(effectiveness)
                if effectiveness > 1.0:
                    super_effective_count_p1 += 1
            
            if category in p1_move_categories:
                p1_move_categories[category] += 1
                if category == 'STATUS':
                    boost_turns += 1
        
        if p2_move:
            power = p2_move.get('base_power', 0)
            category = p2_move.get('category', 'STATUS')
            move_type = p2_move.get('type', 'normal')
            
            if power > 0:
                p2_moves_power.append(power)
                
                # Calculate type effectiveness
                p1_types = [t for t in p1_state.get('types', ['normal']) if t != 'notype']
                effectiveness = get_type_effectiveness(move_type, p1_types)
                p2_type_effectiveness.append(effectiveness)
                if effectiveness > 1.0:
                    super_effective_count_p2 += 1
            
            if category in p2_move_categories:
                p2_move_categories[category] += 1
        
        # Determine who moved first
        if p1_move and p2_move:
            p1_moved_first += 1
        elif p1_move and not p2_move:
            p1_moved_first += 1
        
        # Track status inflictions and duration
        if p2_status != 'nostatus':
            p1_status_inflicted += 1
            p2_status_turns += 1
            if p2_status in ['brn', 'psn', 'tox']:
                burn_poison_turns += 1
        if p1_status != 'nostatus':
            p2_status_inflicted += 1
            p1_status_turns += 1
        
        # Track total boosts and peak
        p1_boost_sum = sum(p1_boosts.values()) if p1_boosts else 0
        p2_boost_sum = sum(p2_boosts.values()) if p2_boosts else 0
        p1_total_boosts.append(p1_boost_sum)
        p2_total_boosts.append(p2_boost_sum)
        
        if p1_boost_sum > p1_max_boost:
            p1_max_boost = p1_boost_sum
        if p2_boost_sum > p2_max_boost:
            p2_max_boost = p2_boost_sum
        
        # Momentum tracking
        turn_damage_dealt = -p2_hp_changes[-1] if p2_hp_changes else 0
        turn_damage_taken = -p1_hp_changes[-1] if p1_hp_changes else 0
        
        new_advantage = None
        if turn_damage_dealt > turn_damage_taken + 10:
            new_advantage = 'p1'
            favorable_turn_streak += 1
        elif turn_damage_taken > turn_damage_dealt + 10:
            new_advantage = 'p2'
            favorable_turn_streak = 0
        
        if new_advantage and current_advantage and new_advantage != current_advantage:
            momentum_swings += 1
        
        if new_advantage:
            current_advantage = new_advantage
        
        if favorable_turn_streak > max_favorable_streak:
            max_favorable_streak = favorable_turn_streak
    
    # Compute aggregate features
    
    # HP change features
    features['p1_mean_hp_change'] = np.mean(p1_hp_changes) if p1_hp_changes else 0
    features['p2_mean_hp_change'] = np.mean(p2_hp_changes) if p2_hp_changes else 0
    features['hp_change_differential'] = features['p1_mean_hp_change'] - features['p2_mean_hp_change']
    
    # NEW: HP volatility (standard deviation of HP changes)
    features['p1_hp_volatility'] = np.std(p1_hp_changes) if len(p1_hp_changes) > 1 else 0
    features['p2_hp_volatility'] = np.std(p2_hp_changes) if len(p2_hp_changes) > 1 else 0
    
    # NEW: HP trajectory slopes (linear regression)
    if len(p1_hp_trajectory) > 2:
        turns_array = np.arange(len(p1_hp_trajectory))
        features['p1_hp_slope'] = np.polyfit(turns_array, p1_hp_trajectory, 1)[0]
    else:
        features['p1_hp_slope'] = 0
    
    if len(p2_hp_trajectory) > 2:
        turns_array = np.arange(len(p2_hp_trajectory))
        features['p2_hp_slope'] = np.polyfit(turns_array, p2_hp_trajectory, 1)[0]
    else:
        features['p2_hp_slope'] = 0
    
    # NEW: Maximum single-turn damage
    features['p1_max_single_turn_damage'] = max([-c for c in p1_hp_changes if c < 0], default=0)
    features['p2_max_single_turn_damage'] = max([-c for c in p2_hp_changes if c < 0], default=0)
    
    # KO features
    features['p1_kos'] = p1_kos
    features['p2_kos'] = p2_kos
    features['ko_differential'] = p2_kos - p1_kos  # Positive if we KO'd more of theirs
    
    # NEW: Early/mid/late game KO differential
    early_ko_p1 = sum(1 for t in p1_ko_turns if t <= 10)
    early_ko_p2 = sum(1 for t in p2_ko_turns if t <= 10)
    mid_ko_p1 = sum(1 for t in p1_ko_turns if 10 < t <= 20)
    mid_ko_p2 = sum(1 for t in p2_ko_turns if 10 < t <= 20)
    late_ko_p1 = sum(1 for t in p1_ko_turns if t > 20)
    late_ko_p2 = sum(1 for t in p2_ko_turns if t > 20)
    
    features['early_game_ko_diff'] = early_ko_p2 - early_ko_p1
    features['mid_game_ko_diff'] = mid_ko_p2 - mid_ko_p1
    features['late_game_ko_diff'] = late_ko_p2 - late_ko_p1
    
    # Move power features
    features['p1_avg_move_power'] = np.mean(p1_moves_power) if p1_moves_power else 0
    features['p2_avg_move_power'] = np.mean(p2_moves_power) if p2_moves_power else 0
    features['move_power_differential'] = features['p1_avg_move_power'] - features['p2_avg_move_power']
    
    # NEW: Damage consistency (coefficient of variation)
    if features['p1_avg_move_power'] > 0:
        damage_std = np.std(p1_moves_power) if p1_moves_power else 0
        features['damage_consistency'] = damage_std / features['p1_avg_move_power']
    else:
        features['damage_consistency'] = 0
    
    # Move category distribution (KEEP only P1 - removed P2 as low importance)
    total_p1_moves = sum(p1_move_categories.values())
    
    features['p1_physical_move_pct'] = p1_move_categories['PHYSICAL'] / total_p1_moves if total_p1_moves > 0 else 0
    features['p1_special_move_pct'] = p1_move_categories['SPECIAL'] / total_p1_moves if total_p1_moves > 0 else 0
    features['p1_status_move_pct'] = p1_move_categories['STATUS'] / total_p1_moves if total_p1_moves > 0 else 0
    
    # NEW: Move effectiveness features
    features['avg_type_effectiveness_p1'] = np.mean(p1_type_effectiveness) if p1_type_effectiveness else 1.0
    features['avg_type_effectiveness_p2'] = np.mean(p2_type_effectiveness) if p2_type_effectiveness else 1.0
    features['super_effective_move_count'] = super_effective_count_p1
    
    # Status effect features
    features['p1_status_inflicted'] = p1_status_inflicted
    features['p2_status_inflicted'] = p2_status_inflicted
    features['status_differential'] = p1_status_inflicted - p2_status_inflicted
    
    # NEW: Status duration features
    features['status_turns_p1'] = p1_status_turns
    features['status_turns_p2'] = p2_status_turns
    features['burn_poison_turns'] = burn_poison_turns
    
    # Speed advantage (keep p1_moved_first_pct, removed speed_advantage_score - duplicate)
    features['p1_moved_first_pct'] = p1_moved_first / num_turns if num_turns > 0 else 0
    
    # Pokémon diversity
    features['p1_pokemon_used'] = len(p1_pokemon_seen)
    features['p2_pokemon_revealed'] = len(p2_pokemon_seen)
    
    # Final HP states
    last_turn = timeline[-1]
    features['p1_final_hp'] = last_turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
    features['p2_final_hp'] = last_turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
    
    # Boost features
    features['p1_avg_boost'] = np.mean(p1_total_boosts) if p1_total_boosts else 0
    features['p2_avg_boost'] = np.mean(p2_total_boosts) if p2_total_boosts else 0
    features['boost_differential'] = features['p1_avg_boost'] - features['p2_avg_boost']
    
    # NEW: Peak boost features
    features['p1_max_boost'] = p1_max_boost
    features['p2_max_boost'] = p2_max_boost
    features['boost_turns'] = boost_turns
    
    # NEW: Switch features
    features['p1_switch_frequency'] = p1_switches / num_turns if num_turns > 0 else 0
    features['p2_switch_frequency'] = p2_switches / num_turns if num_turns > 0 else 0
    features['forced_switches'] = forced_switches
    
    # NEW: HP pressure features
    features['turns_below_50_hp_pct'] = p1_turns_below_50
    features['turns_critical_hp'] = p1_turns_critical
    features['comeback_indicator'] = comeback_hp_recovered
    
    # NEW: Momentum features
    features['momentum_swings'] = momentum_swings
    features['max_consecutive_favorable_turns'] = max_favorable_streak
    
    # REMOVED: p1_priority_moves, p2_priority_moves (low importance)
    
    return features


def _get_empty_timeline_features() -> Dict[str, float]:
    """Return zero-filled timeline features for edge cases."""
    return {
        'num_turns': 0,
        'p1_mean_hp_change': 0,
        'p2_mean_hp_change': 0,
        'hp_change_differential': 0,
        'p1_hp_volatility': 0,
        'p2_hp_volatility': 0,
        'p1_hp_slope': 0,
        'p2_hp_slope': 0,
        'p1_max_single_turn_damage': 0,
        'p2_max_single_turn_damage': 0,
        'p1_kos': 0,
        'p2_kos': 0,
        'ko_differential': 0,
        'early_game_ko_diff': 0,
        'mid_game_ko_diff': 0,
        'late_game_ko_diff': 0,
        'p1_avg_move_power': 0,
        'p2_avg_move_power': 0,
        'move_power_differential': 0,
        'damage_consistency': 0,
        'p1_physical_move_pct': 0,
        'p1_special_move_pct': 0,
        'p1_status_move_pct': 0,
        'avg_type_effectiveness_p1': 1.0,
        'avg_type_effectiveness_p2': 1.0,
        'super_effective_move_count': 0,
        'p1_status_inflicted': 0,
        'p2_status_inflicted': 0,
        'status_differential': 0,
        'status_turns_p1': 0,
        'status_turns_p2': 0,
        'burn_poison_turns': 0,
        'p1_moved_first_pct': 0,
        'p1_pokemon_used': 0,
        'p2_pokemon_revealed': 0,
        'p1_final_hp': 0,
        'p2_final_hp': 0,
        'p1_avg_boost': 0,
        'p2_avg_boost': 0,
        'boost_differential': 0,
        'p1_max_boost': 0,
        'p2_max_boost': 0,
        'boost_turns': 0,
        'p1_switch_frequency': 0,
        'p2_switch_frequency': 0,
        'forced_switches': 0,
        'turns_below_50_hp_pct': 0,
        'turns_critical_hp': 0,
        'comeback_indicator': 0,
        'momentum_swings': 0,
        'max_consecutive_favorable_turns': 0,
    }


def extract_derived_ratios(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute derived ratios and momentum indicators.
    REFACTORED: Removed redundant features (momentum_index, survivability_ratio, speed_advantage_score)
    
    Args:
        features: Dictionary containing all previously extracted features
        
    Returns:
        Dictionary of derived ratio features
    """
    derived = {}
    
    # Offensive efficiency: damage dealt vs damage taken
    p1_damage_dealt = -features.get('p2_mean_hp_change', 0)  # Negative change = damage
    p2_damage_dealt = -features.get('p1_mean_hp_change', 0)
    
    if p2_damage_dealt != 0:
        derived['offensive_efficiency'] = p1_damage_dealt / abs(p2_damage_dealt)
    else:
        derived['offensive_efficiency'] = p1_damage_dealt if p1_damage_dealt > 0 else 0
    
    # REMOVED: survivability_ratio (perfect correlation with p1_final_hp)
    
    # REMOVED: momentum_index (r=0.991 with status_differential, keep status_differential instead)
    
    # Team strength vs opponent lead
    team_avg_stats = features.get('team_avg_total_stats', 0)
    opp_lead_stats = features.get('opp_lead_total_stats', 1)
    if opp_lead_stats > 0:
        derived['team_vs_lead_stat_ratio'] = team_avg_stats / opp_lead_stats
    else:
        derived['team_vs_lead_stat_ratio'] = 1.0
    
    # REMOVED: speed_advantage_score (perfect correlation with p1_moved_first_pct)
    
    # Overall battle control (composite metric)
    ko_control = features.get('ko_differential', 0) / 6  # Normalize by max possible
    hp_control = (features.get('p1_final_hp', 0) - features.get('p2_final_hp', 0)) / 2
    move_control = features.get('move_power_differential', 0) / 100
    
    derived['battle_control_score'] = (ko_control + hp_control + move_control) / 3
    
    return derived


def extract_all_features(battle_json: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract all features from a battle JSON record.
    
    This is the main entry point for feature extraction.
    
    Args:
        battle_json: Complete battle record dictionary
        
    Returns:
        Dictionary of all extracted features
    """
    features = {}
    
    # 1. Team composition features
    team_features = extract_team_composition_features(battle_json['p1_team_details'])
    features.update(team_features)
    
    # 2. Opponent lead features
    lead_features = extract_opponent_lead_features(
        battle_json['p2_lead_details'],
        battle_json['p1_team_details']
    )
    features.update(lead_features)
    
    # 3. Battle timeline features
    timeline_features = extract_battle_timeline_features(
        battle_json.get('battle_timeline', []),
        battle_json['p1_team_details']
    )
    features.update(timeline_features)
    
    # 4. Derived ratios
    derived_features = extract_derived_ratios(features)
    features.update(derived_features)
    
    return features


def get_feature_names() -> List[str]:
    """
    Return the canonical list of feature names in order.
    
    This ensures consistent feature ordering across training and inference.
    
    Returns:
        List of feature names
    """
    # Generate a dummy example to extract feature names
    dummy_pokemon = {
        'name': 'dummy',
        'level': 100,
        'types': ['normal', 'notype'],
        'base_hp': 100,
        'base_atk': 100,
        'base_def': 100,
        'base_spa': 100,
        'base_spd': 100,
        'base_spe': 100,
    }
    
    dummy_battle = {
        'player_won': True,
        'p1_team_details': [dummy_pokemon] * 6,
        'p2_lead_details': dummy_pokemon,
        'battle_timeline': [],
        'battle_id': 0,
    }
    
    features = extract_all_features(dummy_battle)
    return sorted(features.keys())


if __name__ == '__main__':
    # Test feature extraction on random sample from actual data
    import json
    import random
    from pathlib import Path
    
    # Load sample battles from training data
    train_file = Path('../data/train.jsonl')
    
    if train_file.exists():
        print("Loading random sample from training data...")
        battles = []
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    battles.append(json.loads(line))
        
        # Sample 3 random battles
        num_samples = min(3, len(battles))
        sample_battles = random.sample(battles, num_samples)
        
        print(f"\nTesting feature extraction on {num_samples} random battles:\n")
        print("=" * 80)
        
        for i, battle in enumerate(sample_battles, 1):
            print(f"\nBATTLE {i} (ID: {battle.get('battle_id', 'unknown')})")
            print(f"Actual outcome: {'WIN' if battle.get('player_won') else 'LOSS'}")
            print(f"Number of turns: {len(battle.get('battle_timeline', []))}")
            print("-" * 80)
            
            features = extract_all_features(battle)
            
            print(f"✓ Extracted {len(features)} features")
            print("\nKey features:")
            
            # Display important features
            key_features = [
                'team_avg_total_stats',
                'opp_lead_total_stats',
                'num_turns',
                'ko_differential',
                'momentum_index',
                'offensive_efficiency',
                'battle_control_score',
                'team_vs_lead_net_advantage',
            ]
            
            for feat in key_features:
                if feat in features:
                    print(f"  {feat:<30} {features[feat]:>10.3f}")
            
            print("-" * 80)
        
        print("\n" + "=" * 80)
        print("Feature extraction test completed successfully!")
        print("=" * 80)
        
    else:
        print(f"Training data not found at {train_file}")
        print("Using dummy test battle instead...\n")
        
        # Fallback to dummy data
        test_battle = {
            'player_won': True,
            'p1_team_details': [
                {
                    'name': 'starmie',
                    'level': 100,
                    'types': ['psychic', 'water'],
                    'base_hp': 60,
                    'base_atk': 75,
                    'base_def': 85,
                    'base_spa': 100,
                    'base_spd': 100,
                    'base_spe': 115
                }
            ] * 6,
            'p2_lead_details': {
                'name': 'exeggutor',
                'level': 100,
                'types': ['grass', 'psychic'],
                'base_hp': 95,
                'base_atk': 95,
                'base_def': 85,
                'base_spa': 125,
                'base_spd': 125,
                'base_spe': 55
            },
            'battle_timeline': [],
            'battle_id': 0
        }
        
        features = extract_all_features(test_battle)
        print(f"Extracted {len(features)} features:")
        for name, value in sorted(features.items()):
            print(f"  {name}: {value}")
