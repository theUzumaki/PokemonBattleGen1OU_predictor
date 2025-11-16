from dataclasses import dataclass
from typing import Any

@dataclass
class stats:
	atk: int
	def_: int
	spa: int
	spd: int
	spe: int
	hp: int

@dataclass
class move:
	name: str
	cat: int  # 1 or 0
	type: str
	base_pwr: int
	accuracy: int
	priority: int # -7 to +7

@dataclass
class pkmn:
	id: str
	hps: float # between 0 and 1
	#level: int
	type1: str
	type2: str
	base_stats: stats
	moves: list[move]
	status: str
	effects: list[str]
	boosts: stats

@dataclass
class team:
	pkmns: list[pkmn] # length 6
	revealed: int

@dataclass
class adv_team:
	pkmn_alive: int
	pkmn_dscvrd_alive: int
	types: list[str]
	statuses: list[str]
	hp_leader: float

@dataclass
class battle:
	team1: team
	team2: adv_team
	win: int

@dataclass
class battleline:
	battles: dict[int, battle]

__all__ = [
	'stats','move','pkmn','team','adv_team','battle','battleline'
]
