# cards.py - Card state and EnhancementEngine (richer implementation)
from __future__ import annotations

import json
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple


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
            type=d.get("type", "attack"),
            stats=dict(d.get("stats", {})),
            effects=dict(d.get("effects", {})),
            enhancement_counts=dict(d.get("enhancement_counts", {})),
            enhancement_log=list(d.get("enhancement_log", [])),
        )


# small helpers used by engine (kept local to avoid depending on main.py)
def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def fmt_ratio_0(x: float) -> str:
    return f"{x*100:.0f}%"


def fmt_ratio_1(x: float) -> str:
    return f"{x*100:.1f}%"


def fmt_ratio_2(x: float) -> str:
    return f"{x*100:.2f}%"


class EnhancementEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.tiers_cfg: Dict[str, Any] = self.cfg["tiers"]
        self.tier_names = list(self.tiers_cfg.keys())
        proc_cfg = self.cfg.get("proc_rates", {})
        self.proc_efficiency_base = float(proc_cfg.get("efficiency_base", 0.25))
        self.proc_efficiency_with_higher = float(proc_cfg.get("efficiency_with_higher", 0.50))
        self.proc_lower_base = float(proc_cfg.get("lower_base", 0.20))
        self.proc_lower_with_higher = float(proc_cfg.get("lower_with_higher", 0.40))
        # stat/effect caps: mapping name -> (a, b, is_int)
        # a: soft cap, b: breakthrough allowance (max extra raw amount),
        # is_int: whether value should be integer
        # Values are in the same units as stored in card.stats / card.effects.
        # If stored values are small (<=1) but caps are provided >1, caps are
        # interpreted as percentages and will be divided by 100 at application time.
        self.value_caps: Dict[str, Tuple[float, float, bool]] = {
            # stats
            "attack": (math.inf, 0.0, False),
            "range": (3.0, 0.0, True),
            # effects (keys use the same keys as card.effects) -- values in same units as stored
            "freeze_turns": (1.0, 1.0, False),
            "shield_ratio": (0.40, 0.60, False),
            "bleed_ratio": (1.50, 1.50, False),
            "armor_down_ratio": (0.40, 0.35, False),
            "draw_cards": (1.0, 1.0, False),
        }

    def _apply_with_caps(self, key: str, cur: float, delta: float) -> float:
        a, b, cap_is_int = self.value_caps.get(key, (math.inf, 0.0, False))

        # no cap
        if math.isinf(a) or (a is None):
            out = cur + delta
            return int(round(out)) if cap_is_int else out

        # apply toward cap
        remaining_to_cap = max(0.0, a - cur)
        apply_full = min(delta, remaining_to_cap)
        effective = apply_full
        leftover = max(0.0, delta - apply_full)

        # diminishing overcap up to b
        remaining_overcap = max(0.0, b)
        seg_size = remaining_overcap * 0.5
        seg_multiplier = 0.5
        # iterate until leftover consumed or no overcap left
        while leftover > 1e-12 and remaining_overcap > 1e-12 and seg_size > 1e-12:
            raw = min(seg_size, leftover, remaining_overcap)
            effective += raw * seg_multiplier
            leftover -= raw
            remaining_overcap -= raw
            seg_size *= 0.5
            seg_multiplier *= 0.5

        out = cur + effective
        if cap_is_int:
            return int(round(out))
        return out

    @staticmethod
    def load(path: str) -> "EnhancementEngine":
        with open(path, "r", encoding="utf-8") as f:
            return EnhancementEngine(json.load(f))

    def _roll_value(self, spec: Any, rng: Any) -> Any:
        if isinstance(spec, (int, float, str, bool)) or spec is None:
            return spec
        if isinstance(spec, dict) and spec.get("dist") == "uniform":
            return rng.uniform(float(spec["min"]), float(spec["max"]))
        raise ValueError(f"Unknown value spec: {spec}")

    def _scale_value_spec(self, spec: Any, ratio: float) -> Any:
        if isinstance(spec, (int, float)):
            return float(spec) * ratio
        if isinstance(spec, dict) and spec.get("dist") == "uniform":
            out = dict(spec)
            out["min"] = float(out.get("min", 0.0)) * ratio
            out["max"] = float(out.get("max", 0.0)) * ratio
            return out
        return spec

    def _is_efficiency_eligible(self, eff: Dict[str, Any]) -> bool:
        et = eff.get("type")
        if et in ("flag.set", "effect.set_if_none"):
            return False
        if et == "stat.add" and eff.get("stat") == "range":
            return False
        return "value" in eff

    def _boost_effect(self, eff: Dict[str, Any], ratio: float) -> Dict[str, Any]:
        if not self._is_efficiency_eligible(eff):
            return eff
        out = dict(eff)
        out["value"] = self._scale_value_spec(out.get("value"), ratio)
        return out

    def _apply_effect(self, card: CardState, eff: Dict[str, Any], rng: Any) -> None:
        et = eff["type"]

        if et == "stat.add":
            stat = eff["stat"]
            raw_val = eff.get("value")
            v = self._roll_value(raw_val, rng)
            cur = card.stats.get(stat, 0)
            # apply caps if configured
            nxt = self._apply_with_caps(stat, float(cur), float(v))
            if stat == "range":
                nxt = min(3, int(nxt))
            card.stats[stat] = nxt
            # return a readable applied value for UI only if value was sampled (dict spec)
            if isinstance(raw_val, dict):
                if stat == "range":
                    return f"거리={int(nxt)}"
                return f"{nxt:.2f}"
            return None

        if et == "stat.mul":
            stat = eff["stat"]
            ratio = float(self._roll_value(eff["value"], rng))
            if stat == "attack":
                card.stats["attack_bonus_ratio"] = float(card.stats.get("attack_bonus_ratio", 0.0)) + ratio
                return f"{ratio*100:.1f}%"
            else:
                base = float(card.stats.get(stat, 0))
                new_val = base * (1.0 + ratio)
                # compute delta and apply caps
                delta = new_val - base
                nxt = self._apply_with_caps(stat, base, delta)
                card.stats[stat] = nxt
                return None

        if et == "stat.mul_fixed":
            stat = eff["stat"]
            ratio = float(self._roll_value(eff.get("value"), rng))
            if stat == "attack":
                card.stats["attack_bonus_ratio"] = float(card.stats.get("attack_bonus_ratio", 0.0)) + ratio
            else:
                base = float(card.stats.get(stat, 0))
                new_val = base * (1.0 + ratio)
                delta = new_val - base
                nxt = self._apply_with_caps(stat, base, delta)
                card.stats[stat] = nxt
            return

        if et == "flag.set":
            flag = eff["flag"]
            card.effects[flag] = bool(eff["value"])
            return

        if et == "effect.additive":
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            raw_val = eff.get("value")
            v = float(self._roll_value(raw_val, rng))
            cur = float(card.effects.get(key, 0.0))
            # apply caps if configured
            nxt = self._apply_with_caps(key, cur, v)
            card.effects[key] = nxt
            if effect == "draw" and field == "cards":
                return f"{v:.2f}장" if isinstance(raw_val, dict) else None
            if effect == "freeze" and field == "turns":
                return f"{v:.2f}턴" if isinstance(raw_val, dict) else None
            return fmt_ratio_0(v) if isinstance(raw_val, dict) else None

        if et == "effect.max":
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            raw_val = eff.get("value")
            v = self._roll_value(raw_val, rng)
            cur = float(card.effects.get(key, 0.0))
            desired = float(v) if v is not None else cur
            desired = max(cur, desired)
            delta = desired - cur
            nxt = self._apply_with_caps(key, cur, delta)
            card.effects[key] = nxt
            if isinstance(raw_val, dict):
                if field == "turns":
                    return f"{int(desired)}턴"
                return fmt_ratio_0(desired)
            return None

        if et == "effect.mulstack_reduction":
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            raw_val = eff.get("value")
            r = float(self._roll_value(raw_val, rng))
            old = float(card.effects.get(key, 0.0))
            card.effects[key] = 1.0 - (1.0 - old) * (1.0 - r)
            return fmt_ratio_0(r) if isinstance(raw_val, dict) else None

        if et == "effect.set_if_none":
            effect = eff["effect"]
            field = eff["field"]
            key = f"{effect}_{field}"
            if card.effects.get(key, None) is None:
                card.effects[key] = eff["value"]
            return

        raise ValueError(f"Unknown effect type: {et}")

    def apply_tier(
        self,
        card: CardState,
        tier: str,
        rng: Any = None,
        selected_option: Optional[Dict[str, Any]] = None,
        higher_tier_exists: bool = False,
        enable_bonus_procs: bool = True,
    ) -> Dict[str, Any]:
        if rng is None:
            rng = random
        card.ensure_counts(self.tier_names)

        rolled_tier = tier
        tier_cfg = self.tiers_cfg[tier]

        if tier_cfg.get("unique_once", False):
            if card.enhancement_counts.get(tier, 0) >= 1:
                overflow = tier_cfg.get("overflow_to", "MYTHICAL")
                tier = overflow
                tier_cfg = self.tiers_cfg[tier]

        options = tier_cfg["options"]

        def option_valid(opt: Dict[str, Any]) -> bool:
            for eff in opt.get("effects", []):
                et = eff.get("type")

                if et == "flag.set":
                    flag = eff.get("flag")
                    val = bool(eff.get("value"))
                    if val and bool(card.effects.get(flag, False)):
                        return False

                if et == "effect.set_if_none":
                    effect = eff.get("effect")
                    field = eff.get("field")
                    key = f"{effect}_{field}"
                    if card.effects.get(key, None) is not None:
                        return False

                if et == "stat.add":
                    stat = eff.get("stat")
                    if stat == "range":
                        cur = safe_int(card.stats.get("range", 0))
                        if cur >= 3:
                            return False
                
                if et == "effect.additive":
                    effect = eff.get("effect")
                    field = eff.get("field")
                    key = f"{effect}_{field}"
                    if effect == "draw" and field == "cards":
                        cur = safe_float(card.effects.get(key, 0.0))
                        if cur >= 1.0:
                            return False
                    if effect == "freeze" and field == "turns":
                        cur = safe_float(card.effects.get(key, 0.0))
                        if cur >= 3.0:
                            return False
            return True

        valid = [o for o in options if option_valid(o)]

        # If a specific option was provided (from preview), try to use it
        if selected_option is not None:
            # try to match by id first
            sid = selected_option.get("id") if isinstance(selected_option, dict) else None
            matched = None
            if sid:
                for o in options:
                    if o.get("id") == sid:
                        matched = o
                        break
            # if no id match, and selected_option looks like an option dict, use it directly
            if matched is None and isinstance(selected_option, dict) and selected_option.get("id") is None:
                opt = selected_option
            else:
                opt = matched if matched is not None else (rng.choice(valid) if valid else rng.choice(options))
        else:
            opt = rng.choice(valid if valid else options)

        roll_parts: List[str] = []

        eff_ratio = 1.5
        eff_rate = self.proc_efficiency_with_higher if higher_tier_exists else self.proc_efficiency_base
        lower_rate = self.proc_lower_with_higher if higher_tier_exists else self.proc_lower_base

        efficiency_proc = False
        if enable_bonus_procs:
            has_eligible = any(self._is_efficiency_eligible(eff) for eff in opt.get("effects", []))
            efficiency_proc = has_eligible and (rng.random() < eff_rate)

        main_effects = []
        for eff in opt.get("effects", []):
            main_effects.append(self._boost_effect(eff, eff_ratio) if efficiency_proc else eff)

        for eff in main_effects:
            rolled = self._apply_effect(card, eff, rng)
            if rolled is not None:
                roll_parts.append(rolled)

        # If option forces max range, set range to 2 (overrides other changes)
        if opt.get("max_range", False):
            card.stats["range"] = 2

        card.enhancement_counts[tier] = card.enhancement_counts.get(tier, 0) + 1
        card.enhancement_log.append({"rolled_tier": rolled_tier, "tier": tier, "option_id": opt.get("id", ""), "display": opt.get("display", "")})

        lower_bonus_result: Optional[Dict[str, Any]] = None
        if enable_bonus_procs:
            try:
                tier_idx = self.tier_names.index(tier)
            except ValueError:
                tier_idx = -1
            if tier_idx > 0 and (rng.random() < lower_rate):
                lower_tier = self.tier_names[tier_idx - 1]
                lower_bonus_result = self.apply_tier(
                    card,
                    lower_tier,
                    rng=rng,
                    selected_option=None,
                    higher_tier_exists=False,
                    enable_bonus_procs=False,
                )

        roll_text = ""
        extra_parts = []
        if efficiency_proc:
            extra_parts.append("효율증폭 x1.5")
        if lower_bonus_result is not None:
            extra_parts.append(
                f"하위 보너스 {lower_bonus_result['tier_used']}:{lower_bonus_result['display']}{lower_bonus_result.get('roll_text', '')}"
            )
        all_parts = roll_parts + extra_parts
        if all_parts:
            roll_text = " [" + ", ".join(all_parts) + "]"

        return {
            "rolled_tier": rolled_tier,
            "tier_used": tier,
            "option_id": opt.get("id", ""),
            "display": opt.get("display", opt.get("id", "")),
            "roll_text": roll_text,
            "efficiency_proc": efficiency_proc,
            "lower_bonus_proc": lower_bonus_result is not None,
            "lower_bonus_result": lower_bonus_result,
        }

    def valid_options_for_tier(self, card: CardState, tier: str) -> List[Dict[str, Any]]:
        """주어진 카드 상태에서 해당 `tier`에 대해 적용 가능한 옵션 리스트 반환(상태 변경 없음)."""
        tier_cfg = self.tiers_cfg[tier]
        options = tier_cfg.get("options", [])

        def option_valid(opt: Dict[str, Any]) -> bool:
            for eff in opt.get("effects", []):
                et = eff.get("type")

                if et == "flag.set":
                    flag = eff.get("flag")
                    val = bool(eff.get("value"))
                    if val and bool(card.effects.get(flag, False)):
                        return False

                if et == "effect.set_if_none":
                    effect = eff.get("effect")
                    field = eff.get("field")
                    key = f"{effect}_{field}"
                    if card.effects.get(key, None) is not None:
                        return False

                if et == "stat.add":
                    stat = eff.get("stat")
                    if stat == "range":
                        cur = safe_int(card.stats.get("range", 0))
                        if cur >= 3:
                            return False

                if et == "effect.additive":
                    effect = eff.get("effect")
                    field = eff.get("field")
                    key = f"{effect}_{field}"
                    if effect == "draw" and field == "cards":
                        cur = safe_float(card.effects.get(key, 0.0))
                        if cur >= 1.0:
                            return False
                    if effect == "freeze" and field == "turns":
                        cur = safe_float(card.effects.get(key, 0.0))
                        if cur >= 3.0:
                            return False
            return True

        valid = [o for o in options if option_valid(o)]
        return valid if valid else options

    def preview_option(self, card: CardState, tier: str, rng: Any = None) -> Optional[Dict[str, Any]]:
        """해당 `tier`에서 카드 상태에 대해 실제 적용될(또는 적용 가능한) 옵션 하나를 샘플링하여 반환합니다.
        상태는 변경하지 않습니다. 실패 시 None 반환.
        """
        if rng is None:
            rng = random
        opts = self.valid_options_for_tier(card, tier)
        if not opts:
            return None
        return rng.choice(opts)
