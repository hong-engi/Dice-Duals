from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cards import CardState


class Team(Enum):
    PLAYER = "player"
    COMRADE = "comrade"
    ENEMY = "enemy"


class GameResult(Enum):
    ONGOING = "ongoing"
    VICTORY = "victory"
    DEFEAT = "defeat"


@dataclass
class AttackProfile:
    power: float = 10.0
    crit_rate: float = 0.0
    crit_multiplier: float = 1.5
    variance: float = 0.0

    def roll_damage(self, rng: Optional[random.Random] = None) -> float:
        r = rng or random
        dmg = self.power
        if self.variance > 0:
            dmg += r.uniform(-self.variance, self.variance)
        if self.crit_rate > 0 and r.random() < self.crit_rate:
            dmg *= self.crit_multiplier
        return max(0.0, dmg)


@dataclass
class DefenseProfile:
    armor: float = 0.0
    defense_power: float = 0.0
    reduction_ratio: float = 0.0

    def mitigate(self, incoming_damage: float) -> float:
        dmg = max(0.0, incoming_damage)
        dmg *= 1.0 - max(0.0, min(0.95, self.reduction_ratio))
        dmg -= max(0.0, self.armor)
        return max(0.0, dmg)


@dataclass
class UnitState:
    alive: bool = True
    stunned: bool = False
    frozen_turns: int = 0
    shield: float = 0.0
    tags: Dict[str, float] = field(default_factory=dict)


class Unit:
    def __init__(
        self,
        unit_id: str,
        name: str,
        max_hp: float,
        hp: Optional[float] = None,
        team: Team = Team.ENEMY,
        attack: Optional[AttackProfile] = None,
        defense: Optional[DefenseProfile] = None,
        state: Optional[UnitState] = None,
    ) -> None:
        self.unit_id = unit_id
        self.name = name
        self.team = team
        self.max_hp = max(1.0, float(max_hp))
        self.hp = self.max_hp if hp is None else max(0.0, min(float(hp), self.max_hp))
        self.attack = attack or AttackProfile()
        self.defense = defense or DefenseProfile()
        self.state = state or UnitState()
        if self.hp <= 0:
            self.on_death()

    @property
    def is_alive(self) -> bool:
        return self.state.alive and self.hp > 0

    def can_act(self) -> bool:
        return self.is_alive and (not self.state.stunned) and self.state.frozen_turns <= 0

    def on_death(self) -> None:
        self.state.alive = False
        self.hp = 0.0

    def take_damage(self, incoming_damage: float) -> float:
        if not self.is_alive:
            return 0.0
        damage = self.defense.mitigate(incoming_damage)
        if self.state.shield > 0:
            absorbed = min(self.state.shield, damage)
            self.state.shield -= absorbed
            damage -= absorbed
        damage = max(0.0, damage)
        if damage <= 0:
            return 0.0
        self.hp = max(0.0, self.hp - damage)
        if self.hp <= 0:
            self.on_death()
        return damage

    def take_true_damage(self, incoming_damage: float) -> float:
        if not self.is_alive:
            return 0.0
        damage = max(0.0, incoming_damage)
        self.hp = max(0.0, self.hp - damage)
        if self.hp <= 0:
            self.on_death()
        return damage

    def heal(self, amount: float) -> float:
        if not self.is_alive:
            return 0.0
        before = self.hp
        self.hp = min(self.max_hp, self.hp + max(0.0, amount))
        return self.hp - before

    def add_shield(self, amount: float) -> float:
        gain = max(0.0, amount)
        self.state.shield += gain
        return gain

    def basic_attack(self, target: "Unit", rng: Optional[random.Random] = None) -> float:
        if not self.can_act() or not target.is_alive:
            return 0.0
        raw_damage = self.attack.roll_damage(rng=rng)
        return target.take_damage(raw_damage)

    def end_turn(self) -> None:
        if self.state.frozen_turns > 0:
            self.state.frozen_turns -= 1
        self.state.stunned = False


class Player(Unit):
    def __init__(
        self,
        unit_id: str,
        name: str,
        max_hp: float,
        hp: Optional[float] = None,
        cards: Optional[List["CardState"]] = None,
        attack: Optional[AttackProfile] = None,
        defense: Optional[DefenseProfile] = None,
        state: Optional[UnitState] = None,
    ) -> None:
        super().__init__(
            unit_id=unit_id,
            name=name,
            max_hp=max_hp,
            hp=hp,
            team=Team.PLAYER,
            attack=attack,
            defense=defense,
            state=state,
        )
        self.cards: List["CardState"] = list(cards or [])
        self.last_result: GameResult = GameResult.ONGOING

    def on_death(self) -> None:
        super().on_death()
        self.last_result = GameResult.DEFEAT

    def add_card(self, card: "CardState") -> None:
        self.cards.append(card)


class Comrade(Unit):
    def __init__(
        self,
        unit_id: str,
        name: str,
        max_hp: float,
        hp: Optional[float] = None,
        attack: Optional[AttackProfile] = None,
        defense: Optional[DefenseProfile] = None,
        state: Optional[UnitState] = None,
    ) -> None:
        super().__init__(
            unit_id=unit_id,
            name=name,
            max_hp=max_hp,
            hp=hp,
            team=Team.COMRADE,
            attack=attack,
            defense=defense,
            state=state,
        )


class Enemy(Unit):
    def __init__(
        self,
        unit_id: str,
        name: str,
        max_hp: float,
        hp: Optional[float] = None,
        attack: Optional[AttackProfile] = None,
        defense: Optional[DefenseProfile] = None,
        state: Optional[UnitState] = None,
    ) -> None:
        super().__init__(
            unit_id=unit_id,
            name=name,
            max_hp=max_hp,
            hp=hp,
            team=Team.ENEMY,
            attack=attack,
            defense=defense,
            state=state,
        )

    @staticmethod
    def all_dead(enemies: List["Enemy"]) -> bool:
        return bool(enemies) and all(not e.is_alive for e in enemies)


@dataclass
class BattleState:
    player: Player
    comrades: List[Comrade] = field(default_factory=list)
    enemies: List[Enemy] = field(default_factory=list)
    result: GameResult = GameResult.ONGOING
    memory: List[str] = field(default_factory=list)

    @property
    def is_game_over(self) -> bool:
        return self.result != GameResult.ONGOING

    def update_result(self) -> GameResult:
        if not self.player.is_alive:
            self.result = GameResult.DEFEAT
            self.player.last_result = GameResult.DEFEAT
            self.memory.append("플레이어가 사망하여 패배했습니다.")
            return self.result

        if Enemy.all_dead(self.enemies):
            self.result = GameResult.VICTORY
            self.player.last_result = GameResult.VICTORY
            self.memory.append("모든 enemy를 처치하여 승리했습니다.")
            return self.result

        self.result = GameResult.ONGOING
        return self.result
