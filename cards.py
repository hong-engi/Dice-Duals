# enhancement_engine.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


@dataclass
class CardState:
    id: str
    name: str
    type: str
    stats: Dict[str, Any]
    effects: Dict[str, Any]

    enhancement_counts: Dict[str, int] = field(default_factory=dict)
    enhancement_log: List[Dict[str, Any]] = field(default_factory=list)

    def ensure_counts(self, tier_names: List[str]) -> None:
        for t in tier_names:
            self.enhancement_counts.setdefault(t, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "stats": self.stats,
            "effects": self.effects,
            "enhancement_counts": self.enhancement_counts,
            "enhancement_log": self.enhancement_log,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CardState":
        return CardState(
            id=d["id"],
            name=d["name"],
            type=d["type"],
            stats=dict(d.get("stats", {})),
            effects=dict(d.get("effects", {})),
            enhancement_counts=dict(d.get("enhancement_counts", {})),
            enhancement_log=list(d.get("enhancement_log", [])),
        )


class EnhancementEngine:
    def __init__(self, enhancements_config: Dict[str, Any]):
        self.cfg = enhancements_config
        self.tiers_cfg: Dict[str, Any] = self.cfg["tiers"]
        self.tier_names = list(self.tiers_cfg.keys())

    @staticmethod
    def load(path: str) -> "EnhancementEngine":
        with open(path, "r", encoding="utf-8") as f:
            return EnhancementEngine(json.load(f))

    def _roll_value(self, spec: Any, rng: random.Random) -> Any:
        # number literal
        if isinstance(spec, (int, float, str, bool)) or spec is None:
            return spec

        # distribution object
        if isinstance(spec, dict) and spec.get("dist") == "uniform":
            return rng.uniform(float(spec["min"]), float(spec["max"]))

        raise ValueError(f"Unknown value spec: {spec}")

    def _apply_effect(self, card: CardState, eff: Dict[str, Any], rng: random.Random) -> None:
        et = eff["type"]

        if et == "stat.add":
            stat = eff["stat"]
            v = eff["value"]
            card.stats[stat] = card.stats.get(stat, 0) + v
            return

        if et == "stat.mul":
            stat = eff["stat"]
            ratio = float(self._roll_value(eff["value"], rng))  # e.g. 0.08~0.12
            base = float(card.stats.get(stat, 0))
            newv = base * (1.0 + ratio)
            if "round" in eff:
                newv = round(newv, int(eff["round"]))
            card.stats[stat] = newv
            return

        if et == "stat.mul_fixed":
            stat = eff["stat"]
            ratio = float(eff["value"])  # e.g. 5.0 for +500%
            base = float(card.stats.get(stat, 0))
            newv = base * (1.0 + ratio)
            if "round" in eff:
                newv = round(newv, int(eff["round"]))
            card.stats[stat] = newv
            return

        if et == "flag.set":
            flag = eff["flag"]
            card.effects[flag] = bool(eff["value"])
            return

        if et == "effect.additive":
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            v = float(eff["value"])
            card.effects[key] = float(card.effects.get(key, 0.0)) + v
            return

        if et == "effect.max":
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            v = eff["value"]
            cur = card.effects.get(key, 0)
            card.effects[key] = max(cur, v)
            return

        if et == "effect.mulstack_reduction":
            # multiplicative stacking reduction:
            # new = 1 - (1-old)*(1-r)
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            r = float(eff["value"])
            old = float(card.effects.get(key, 0.0))
            card.effects[key] = 1.0 - (1.0 - old) * (1.0 - r)
            return

        if et == "effect.set_if_none":
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            if card.effects.get(key, None) is None:
                card.effects[key] = eff["value"]
            return

        raise ValueError(f"Unknown effect type: {et}")

    def apply_tier(self, card: CardState, tier: str, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """
        tier 강화 1회 적용. (유일 2번째부터 신화로 대체 포함)
        return: applied info {tier_used, option_id, display}
        """
        if rng is None:
            rng = random.Random()

        card.ensure_counts(self.tier_names)

        tier_cfg = self.tiers_cfg[tier]

        # ✅ UNIQUE once rule
        if tier_cfg.get("unique_once", False):
            if card.enhancement_counts.get(tier, 0) >= 1:
                overflow = tier_cfg.get("overflow_to", "MYTHICAL")
                tier = overflow
                tier_cfg = self.tiers_cfg[tier]

        # roll option uniformly (can add weights later)
        options = tier_cfg["options"]
        opt = rng.choice(options)

        # apply effects
        for eff in opt.get("effects", []):
            self._apply_effect(card, eff, rng)

        # bookkeeping
        card.enhancement_counts[tier] = card.enhancement_counts.get(tier, 0) + 1
        card.enhancement_log.append({"tier": tier, "option_id": opt["id"]})

        return {
            "tier_used": tier,
            "option_id": opt["id"],
            "display": opt.get("display", opt["id"]),
        }