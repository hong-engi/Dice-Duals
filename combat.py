from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cards import CardState, EnhancementEngine, Tier, TIERS_NAME, describe_card, load_cards
from upgrade_main import EnhanceGacha, format_extra_effects
from unit import AttackProfile, Comrade, DefenseProfile, Enemy, Player


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clone_card(base: CardState, new_id: str, new_name: Optional[str] = None) -> CardState:
    c = CardState.from_dict(base.to_dict())
    c.id = new_id
    if new_name:
        c.name = new_name
    return c


def _fallback_attack_card(card_no: int, card_name: str = "기본 공격") -> CardState:
    return CardState(
        id=f"atk_fallback_{card_no}",
        name=card_name,
        type="attack",
        stats={
            "attack_ratio": 1.0,
            "attack_bonus_ratio": 0.0,
            "range": 1,
            "resource": 0,
            "defense_ratio": 0.0,
            "shield_bonus_ratio": 0.0,
        },
        effects={
            "ignore_defense": False,
            "double_hit": False,
            "aoe_all_enemies": False,
            "triple_attack": False,
            "splash_other_ratio": 0.0,
        },
    )


def _fallback_defense_card(card_no: int, card_name: str = "기본 방어") -> CardState:
    return CardState(
        id=f"def_fallback_{card_no}",
        name=card_name,
        type="defense",
        stats={
            "defense_ratio": 1.0,
            "shield_bonus_ratio": 0.0,
            "resource": 0,
            "attack_ratio": 0.0,
            "attack_bonus_ratio": 0.0,
            "range": 0,
        },
        effects={
            "ignore_defense": False,
            "double_hit": False,
            "aoe_all_enemies": False,
            "triple_attack": False,
            "splash_other_ratio": 0.0,
        },
    )


def _build_starting_card_templates(engine: EnhancementEngine) -> List[CardState]:
    cards = load_cards("cards.json")
    attack_cards = [c for c in cards if c.type == "attack"]
    defense_cards = [c for c in cards if c.type == "defense"]
    atk_name = engine.get_card_name("attack") or "기본 공격"
    def_name = engine.get_card_name("defense") or "기본 방어"

    atk1_src = attack_cards[0] if attack_cards else None
    atk2_src = attack_cards[1] if len(attack_cards) >= 2 else atk1_src
    def1_src = defense_cards[0] if defense_cards else None
    def2_src = defense_cards[1] if len(defense_cards) >= 2 else def1_src

    atk1 = _clone_card(atk1_src, "start_atk_1", atk_name) if atk1_src else _fallback_attack_card(1, atk_name)
    atk2 = _clone_card(atk2_src, "start_atk_2", atk_name) if atk2_src else _fallback_attack_card(2, atk_name)
    dfn1 = _clone_card(def1_src, "start_def_1", def_name) if def1_src else _fallback_defense_card(1, def_name)
    dfn2 = _clone_card(def2_src, "start_def_2", def_name) if def2_src else _fallback_defense_card(2, def_name)

    return [atk1, atk2, dfn1, dfn2]


def _tier_name_ko(tier_name: str) -> str:
    try:
        return TIERS_NAME[Tier[tier_name]]
    except Exception:
        return tier_name


def _format_applied_option_for_combat(result: Dict[str, Any]) -> str:
    parts: List[str] = []
    base = str(result.get("display_applied", result.get("display", "")) or "").strip()
    if base:
        parts.append(base)

    lower = result.get("lower_bonus_result")
    if isinstance(lower, dict):
        lower_text = str(lower.get("display_applied", lower.get("display", "")) or "").strip()
        if lower_text:
            parts.append(lower_text)

    same = result.get("same_tier_bonus_result")
    if isinstance(same, dict):
        same_text = str(same.get("display_applied", same.get("display", "")) or "").strip()
        if same_text:
            parts.append(same_text)

    return " + ".join(parts) if parts else "-"


def _card_label(card: CardState) -> str:
    return (card.name or "").strip() or ("공격" if card.type == "attack" else "방어")


def _forced_tier_text(g: EnhanceGacha) -> str:
    forced_idx = g._forced_min_tier()
    if forced_idx < 0:
        return ""
    return f" - {_tier_name_ko(Tier(forced_idx).name)} 확정"


def _print_pity_status(g: EnhanceGacha) -> None:
    print("\n[티어별 남은 천장]")
    for idx in range(len(Tier) - 1, -1, -1):
        tier = Tier(idx)
        cap = int(g.caps[idx]) if idx < len(g.caps) else 0
        fail = int(g.fail_counts[idx]) if idx < len(g.fail_counts) else 0
        if cap <= 0:
            remain_text = "천장 없음"
        else:
            remain = max(0, cap - fail)
            if remain == 0:
                remain_text = "확정 상태"
            else:
                remain_text = f"{remain}회 남음"
        print(f"- {TIERS_NAME[tier]}: {remain_text}")


def _choice_plan_signature(plan: Dict[str, Any], card_index: int) -> tuple:
    opt = plan.get("opt") if isinstance(plan, dict) else None
    if isinstance(opt, dict):
        opt_key = (
            str(opt.get("id", "") or ""),
            str(opt.get("display", "") or ""),
            bool(opt.get("max_range", False)),
        )
    else:
        opt_key = ("", "", False)
    return (
        int(card_index),
        int(plan.get("idx", 0)),
        opt_key,
        bool(plan.get("efficiency_proc", False)),
        bool(plan.get("efficiency_double_proc", False)),
        bool(plan.get("lower_bonus_proc", False)),
        bool(plan.get("same_tier_bonus_proc", False)),
    )


def _roll_center_weighted_count(rng: random.Random, min_count: int, max_count: int, center: int) -> int:
    lo = int(min_count)
    hi = int(max_count)
    if lo > hi:
        lo, hi = hi, lo
    c = max(lo, min(int(center), hi))
    values = list(range(lo, hi + 1))
    max_dist = max(abs(c - lo), abs(hi - c))
    if max_dist == 0:
        return c
    weights: List[float] = []
    for v in values:
        dist = abs(v - c)
        # 양끝 확률이 중심 확률의 1/3이 되도록 선형 가중치 설정.
        # center weight = 3 * max_dist, edge weight = 1 * max_dist
        w = float(max_dist + 2 * (max_dist - dist))
        weights.append(max(1e-9, w))
    return rng.choices(values, weights=weights, k=1)[0]


def _manual_enhance_starting_cards(
    engine: EnhancementEngine,
    cards: List[CardState],
    preview_player: Player,
    rng: random.Random,
    min_count: int = 40,
    max_count: int = 60,
    center_count: int = 50,
    n_choices: int = 3,
) -> None:
    if not cards:
        return

    # upgrade_main과 동일한 난수 소스를 사용하기 위해 전역 random 상태를 잠시 대체한다.
    global_state = random.getstate()
    random.seed(rng.randrange(1 << 30))
    try:
        g = EnhanceGacha()
        total = _roll_center_weighted_count(rng, min_count=min_count, max_count=max_count, center=center_count)
        choice_count = max(1, int(n_choices))
        auto_remaining = False

        print("\n[시작 카드 수동 강화]")
        print(
            f"- 강화 횟수: {total}회 "
            f"(범위 {min_count}~{max_count}, 중심 {center_count})"
        )

        for roll_no in range(1, total + 1):
            slot_count = choice_count
            picks = g.preview_choice_tiers(n_choices=slot_count)
            choice_plans: List[Dict[str, Any]] = []
            used_signatures: set[tuple] = set()

            for idx in picks:
                selected_plan: Optional[Dict[str, Any]] = None
                selected_sig: Optional[tuple] = None
                # 동일 라운드의 완전 중복 선택지는 재추첨한다.
                for _ in range(24):
                    cidx = random.randrange(len(cards))
                    card = cards[cidx]
                    planned_opt = engine.preview_option(card, Tier(idx).name, rng=random)
                    if planned_opt is None:
                        disp = "(옵션 없음)"
                        planned_effects = []
                    else:
                        disp = planned_opt.get("display", planned_opt.get("id", ""))
                        if planned_opt.get("max_range", False):
                            disp = f"{disp} (max range)"
                        planned_effects = planned_opt.get("effects", [])

                    higher_gap_for_choice = max(0, max(picks) - idx) if picks else 0
                    eff_rate, lower_rate, eff2_rate, same_tier_rate = engine.get_proc_rates(
                        Tier(idx).name, higher_tier_gap=higher_gap_for_choice
                    )
                    eff_eligible = bool(planned_opt) and any(
                        engine._is_efficiency_eligible(eff) for eff in planned_effects
                    )
                    efficiency_proc = eff_eligible and (random.random() < eff_rate)
                    efficiency_double_proc = eff_eligible and (random.random() < eff2_rate)
                    lower_bonus_proc = idx > 0 and (random.random() < lower_rate)
                    same_tier_bonus_proc = random.random() < same_tier_rate
                    if efficiency_double_proc:
                        efficiency_proc = False
                    if same_tier_bonus_proc:
                        lower_bonus_proc = False

                    preview_parts: List[str] = []
                    if efficiency_proc:
                        preview_parts.append("x1.5")
                    if efficiency_double_proc:
                        preview_parts.append("x2")
                    if lower_bonus_proc:
                        preview_parts.append("하위 티어 보너스!")
                    if same_tier_bonus_proc:
                        preview_parts.append("동일 티어 보너스!")
                    preview_roll_text = f" [{' ] ['.join(preview_parts)}]" if preview_parts else ""

                    plan = {
                        "idx": idx,
                        "card": card,
                        "card_index": cidx,
                        "opt": planned_opt,
                        "higher_gap": higher_gap_for_choice,
                        "display": disp,
                        "efficiency_proc": efficiency_proc,
                        "efficiency_double_proc": efficiency_double_proc,
                        "lower_bonus_proc": lower_bonus_proc,
                        "same_tier_bonus_proc": same_tier_bonus_proc,
                        "preview_roll_text": preview_roll_text,
                    }
                    sig = _choice_plan_signature(plan, cidx)
                    selected_plan = plan
                    selected_sig = sig
                    if sig not in used_signatures:
                        break

                if selected_plan is not None and selected_sig is not None:
                    used_signatures.add(selected_sig)
                    choice_plans.append(selected_plan)

            print(f"\n[{roll_no}/{total}] 강화 선택{_forced_tier_text(g)}")
            for i, plan in enumerate(choice_plans, start=1):
                idx = plan["idx"]
                card = plan["card"]
                print(f"{i}) {_card_label(card)}")
                print(f"   현재: {describe_card(card, preview_player)}")
                print(
                    f"   제안: {_tier_name_ko(Tier(idx).name)} - "
                    f"{plan.get('display', '')}{plan.get('preview_roll_text', '')}"
                )

            if auto_remaining:
                best_tier = max(p["idx"] for p in choice_plans)
                candidates = [p for p in choice_plans if p["idx"] == best_tier]
                chosen_plan = random.choice(candidates)
                print(
                    f"- 자동 선택: {_tier_name_ko(Tier(best_tier).name)} "
                    f"(남은 {total - roll_no + 1}회 자동)"
                )
            else:
                valid_inputs = {str(i) for i in range(1, slot_count + 1)}
                choices_str = "/".join(str(i) for i in range(1, slot_count + 1))
                while True:
                    sel = input(f"선택({choices_str}, 엔터=1, t=천장보기, a=남은횟수 자동, q=중단): ").strip().lower()
                    if sel in ("q", "quit", "exit"):
                        raise KeyboardInterrupt
                    if sel in ("t", "pity"):
                        _print_pity_status(g)
                        continue
                    if sel in ("a", "auto"):
                        auto_remaining = True
                        best_tier = max(p["idx"] for p in choice_plans)
                        candidates = [p for p in choice_plans if p["idx"] == best_tier]
                        chosen_plan = random.choice(candidates)
                        print(
                            f"- 자동 모드 시작: 남은 {total - roll_no + 1}회 "
                            f"{_tier_name_ko(Tier(best_tier).name)} 우선 선택"
                        )
                        break
                    if sel == "":
                        sel = "1"
                    if sel in valid_inputs:
                        chosen_plan = choice_plans[int(sel) - 1]
                        break
                    print("잘못된 입력입니다.")

            chosen_idx = chosen_plan["idx"]
            chosen_card = chosen_plan["card"]

            g.commit_choice(chosen_idx)
            tier_str = Tier(chosen_idx).name
            applied = engine.apply_tier(
                chosen_card,
                tier_str,
                rng=random,
                selected_option=chosen_plan["opt"],
                higher_tier_gap=chosen_plan["higher_gap"],
                forced_efficiency_proc=chosen_plan["efficiency_proc"],
                forced_efficiency_double_proc=chosen_plan["efficiency_double_proc"],
                forced_lower_bonus_proc=chosen_plan["lower_bonus_proc"],
                forced_same_tier_bonus_proc=chosen_plan["same_tier_bonus_proc"],
            )

            rolled_tier = applied.get("rolled_tier", tier_str)
            used_tier = applied.get("tier_used", tier_str)
            if rolled_tier == used_tier:
                print(
                    f"- 적용 완료: {_card_label(chosen_card)} / "
                    f"{_tier_name_ko(used_tier)}"
                )
            else:
                print(
                    f"- 적용 완료: {_card_label(chosen_card)} / "
                    f"{_tier_name_ko(rolled_tier)} -> {_tier_name_ko(used_tier)}"
                )
            extra = format_extra_effects(applied)
            if extra:
                print(f"  보너스: {extra}")
            print(f"  옵션: {_format_applied_option_for_combat(applied)}")
    finally:
        random.setstate(global_state)


def _build_deck_from_templates(templates: List[CardState], copies_each: int, rng: random.Random) -> List[CardState]:
    deck: List[CardState] = []
    for t in templates:
        for i in range(copies_each):
            c = CardState.from_dict(t.to_dict())
            c.id = f"{t.id}_copy{i+1}"
            deck.append(c)
    rng.shuffle(deck)
    return deck


def _alive_enemies(enemy_rows: List[List[Enemy]]) -> List[Enemy]:
    return [e for row in enemy_rows for e in row if e.is_alive]


def _enemy_distances(enemy_rows: List[List[Enemy]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    d = 1
    for row in enemy_rows:
        alive_row = [e for e in row if e.is_alive]
        if not alive_row:
            continue
        for e in alive_row:
            out[e.unit_id] = d
        d += 1
    return out


def _chance_count(v: float, rng: random.Random) -> int:
    x = max(0.0, _f(v, 0.0))
    whole = int(x)
    frac = x - whole
    return whole + (1 if frac > 0 and rng.random() < frac else 0)


@dataclass
class CombatConfig:
    min_start_enhance: int = 40
    max_start_enhance: int = 60
    center_start_enhance: int = 50
    start_enhance_choices: int = 3
    initial_draw: float = 4.0
    draw_per_turn: float = 2.0
    cards_per_turn: int = 2
    max_hand_size: int = 6
    enemy_row_counts: List[int] = field(default_factory=lambda: [2, 2, 1])
    seed: Optional[int] = None


@dataclass
class CombatRuntime:
    deck: List[CardState]
    discard: List[CardState] = field(default_factory=list)
    hand: List[CardState] = field(default_factory=list)
    comrades: List[Comrade] = field(default_factory=list)
    resource: int = 0
    next_attack_bonus_ratio: float = 0.0
    frenzy_ratio: float = 0.0
    temp_armor_bonus: float = 0.0
    on_hit_freeze_turns: float = 0.0
    on_hit_bleed_ratio: float = 0.0
    on_hit_bleed_turns: float = 0.0
    on_hit_draw_cards: float = 0.0
    on_hit_frenzy_ratio: float = 0.0

    def clear_round_buffs(self) -> None:
        self.temp_armor_bonus = 0.0
        self.on_hit_freeze_turns = 0.0
        self.on_hit_bleed_ratio = 0.0
        self.on_hit_bleed_turns = 0.0
        self.on_hit_draw_cards = 0.0
        self.on_hit_frenzy_ratio = 0.0


def _draw_one(rt: CombatRuntime, cfg: CombatConfig, rng: random.Random, reason: str = "") -> bool:
    if len(rt.hand) >= cfg.max_hand_size:
        if reason:
            print(f"- 드로우 실패({reason}): 손패가 가득 찼습니다. ({len(rt.hand)}/{cfg.max_hand_size})")
        return False
    if not rt.deck:
        if not rt.discard:
            if reason:
                print(f"- 드로우 실패({reason}): 덱/버림 더미가 모두 비어 있습니다.")
            return False
        rt.deck = list(rt.discard)
        rt.discard.clear()
        rng.shuffle(rt.deck)
        print("- 버림 더미를 섞어 덱을 재구성했습니다.")
    rt.hand.append(rt.deck.pop())
    return True


def _draw_n(rt: CombatRuntime, cfg: CombatConfig, rng: random.Random, draw_value: float, reason: str = "") -> int:
    n = _chance_count(draw_value, rng)
    drew = 0
    for _ in range(n):
        if _draw_one(rt, cfg, rng, reason=reason):
            drew += 1
    if reason and n > 0:
        print(f"- 드로우({reason}): {drew}/{n}장")
    return drew


def _gain_resource_from_card(card: CardState, rt: CombatRuntime, rng: random.Random) -> int:
    gain = _chance_count(_f(card.stats.get("resource", 0.0), 0.0), rng)
    if gain > 0:
        rt.resource += gain
        print(f"- 자원 +{gain} (현재 {rt.resource})")
    return gain


def _deal_damage(target: Enemy | Player | Comrade, raw_damage: float, ignore_defense: bool = False) -> float:
    if not target.is_alive:
        return 0.0
    if not ignore_defense:
        return target.take_damage(raw_damage)

    damage = max(0.0, raw_damage)
    if target.state.shield > 0:
        absorbed = min(target.state.shield, damage)
        target.state.shield -= absorbed
        damage -= absorbed
    damage = max(0.0, damage)
    if damage <= 0:
        return 0.0
    target.hp = max(0.0, target.hp - damage)
    if target.hp <= 0:
        target.on_death()
    return damage


def _damage_breakdown(target: Enemy | Player | Comrade, raw_damage: float, ignore_defense: bool = False) -> Dict[str, float]:
    raw = max(0.0, float(raw_damage))
    shield_before = max(0.0, float(target.state.shield))
    if ignore_defense:
        shield_block = min(shield_before, raw)
        final_damage = max(0.0, raw - shield_block)
        return {
            "raw": raw,
            "reduction_amount": 0.0,
            "armor_block": 0.0,
            "shield_block": shield_block,
            "final_damage": final_damage,
        }

    reduction = max(0.0, min(0.95, float(target.defense.reduction_ratio)))
    after_reduction = raw * (1.0 - reduction)
    reduction_amount = raw - after_reduction
    armor = max(0.0, float(target.defense.armor))
    armor_block = min(after_reduction, armor)
    after_armor = max(0.0, after_reduction - armor)
    shield_block = min(shield_before, after_armor)
    final_damage = max(0.0, after_armor - shield_block)
    return {
        "raw": raw,
        "reduction_amount": reduction_amount,
        "armor_block": armor_block,
        "shield_block": shield_block,
        "final_damage": final_damage,
    }


def _format_damage_line(
    card_name: str,
    target_name: str,
    dealt: float,
    breakdown: Dict[str, float],
    hit_idx: int,
    hit_count: int,
) -> str:
    head = (
        f"{card_name} (연타 {hit_idx}/{hit_count}) -> {target_name}에게 "
        if hit_count > 1
        else f"{card_name} -> {target_name}에게 "
    )
    terms = [f"{breakdown.get('raw', 0.0):.1f}"]
    red = breakdown.get("reduction_amount", 0.0)
    arm = breakdown.get("armor_block", 0.0)
    shd = breakdown.get("shield_block", 0.0)
    if red > 0:
        terms.append(f"피해감소 {red:.1f}")
    if arm > 0:
        terms.append(f"갑옷 {arm:.1f}")
    if shd > 0:
        terms.append(f"보호막 {shd:.1f}")
    return head + " - ".join(terms) + f" = {dealt:.1f} 피해"


def _apply_bleed(target: Enemy | Player | Comrade, damage_per_turn: float, turns: int) -> bool:
    if (not target.is_alive) or damage_per_turn <= 0 or turns <= 0:
        return False
    tags = target.state.tags
    tags["bleed_damage"] = float(tags.get("bleed_damage", 0.0)) + float(damage_per_turn)
    tags["bleed_turns"] = max(int(tags.get("bleed_turns", 0)), int(turns))
    return True


def _tick_bleed(unit: Enemy | Player | Comrade) -> float:
    if not unit.is_alive:
        return 0.0
    tags = unit.state.tags
    turns = int(tags.get("bleed_turns", 0))
    dmg = float(tags.get("bleed_damage", 0.0))
    if turns <= 0 or dmg <= 0:
        return 0.0
    dealt = unit.take_true_damage(dmg)
    turns -= 1
    if turns <= 0:
        tags.pop("bleed_turns", None)
        tags.pop("bleed_damage", None)
    else:
        tags["bleed_turns"] = turns
    return dealt


def _apply_freeze(target: Enemy | Player | Comrade, turns: int) -> bool:
    if turns <= 0 or not target.is_alive:
        return False
    target.state.frozen_turns = max(target.state.frozen_turns, turns)
    return True


def _apply_armor_down(target: Enemy | Player | Comrade, ratio: float) -> None:
    r = max(0.0, min(1.0, ratio))
    keep = 1.0 - r
    target.defense.armor *= keep
    target.defense.defense_power *= keep
    target.defense.reduction_ratio *= keep


def _summon_comrade(unit_key: str, rt: CombatRuntime) -> None:
    key = (unit_key or "").strip()
    if not key:
        return
    if key == "wolf":
        if any(c.is_alive and c.unit_id.startswith("wolf_") for c in rt.comrades):
            print("- 소환 실패: 늑대가 이미 전장에 있습니다.")
            return
        idx = sum(1 for c in rt.comrades if c.unit_id.startswith("wolf_")) + 1
        rt.comrades.append(
            Comrade(
                unit_id=f"wolf_{idx}",
                name="늑대",
                max_hp=120.0,
                hp=120.0,
                attack=AttackProfile(power=40.0),
                defense=DefenseProfile(armor=5.0, defense_power=0.0),
            )
        )
        print("- 늑대를 소환했습니다.")
        return
    if key == "barricade":
        if any(c.is_alive and c.unit_id.startswith("barricade_") for c in rt.comrades):
            print("- 소환 실패: 바리케이드가 이미 전장에 있습니다.")
            return
        idx = sum(1 for c in rt.comrades if c.unit_id.startswith("barricade_")) + 1
        rt.comrades.append(
            Comrade(
                unit_id=f"barricade_{idx}",
                name="바리케이드",
                max_hp=240.0,
                hp=240.0,
                attack=AttackProfile(power=0.0),
                defense=DefenseProfile(armor=20.0, defense_power=0.0),
            )
        )
        print("- 바리케이드를 소환했습니다.")
        return
    print(f"- 소환 키 '{key}'는 아직 정의되지 않았습니다.")


def _render_status(
    player: Player,
    enemy_rows: List[List[Enemy]],
    rt: CombatRuntime,
    turn: int,
    cfg: CombatConfig,
) -> None:
    dist = _enemy_distances(enemy_rows)
    print("\n" + "=" * 82)
    print(f"턴 {turn}")
    player_parts = [f"플레이어 HP {player.hp:.1f}/{player.max_hp:.1f}"]
    if player.attack.power > 0:
        player_parts.append(f"기본 피해 {player.attack.power:.1f}")
    if player.defense.defense_power > 0:
        player_parts.append(f"방어력 {player.defense.defense_power:.1f}")
    if player.defense.armor > 0:
        player_parts.append(f"갑옷 {player.defense.armor:.1f}")
    if player.state.shield > 0:
        player_parts.append(f"보호막 {player.state.shield:.1f}")
    if rt.resource > 0:
        player_parts.append(f"자원 {rt.resource}")
    print(" | ".join(player_parts))
    print(
        f"덱 {len(rt.deck)}장 | 버림 {len(rt.discard)}장 | 손패 {len(rt.hand)}장 "
        f"(최대 {cfg.max_hand_size})"
    )
    if rt.comrades:
        print("[아군 소환체]")
        for c in rt.comrades:
            state = "생존" if c.is_alive else "사망"
            parts = [f"- {c.name}({c.unit_id}) {state}", f"HP {c.hp:.1f}/{c.max_hp:.1f}"]
            if c.attack.power > 0:
                parts.append(f"ATK {c.attack.power:.1f}")
            if c.defense.armor > 0:
                parts.append(f"갑옷 {c.defense.armor:.1f}")
            if c.state.shield > 0:
                parts.append(f"보호막 {c.state.shield:.1f}")
            print(" | ".join(parts))

    print("\n[적 상태]")
    alive = _alive_enemies(enemy_rows)
    if not alive:
        print("- 생존한 적이 없습니다.")
    else:
        for row_index, row in enumerate(enemy_rows, start=1):
            alive_row = [e for e in row if e.is_alive]
            if not alive_row:
                continue
            current_dist = dist.get(alive_row[0].unit_id, "?")
            print(f"- 열 {current_dist}")
            for e in alive_row:
                parts = [f"  · {e.name}: HP {e.hp:.1f}/{e.max_hp:.1f}, 거리 {dist.get(e.unit_id, '?')}"]
                if e.attack.power > 0:
                    parts.append(f"ATK {e.attack.power:.1f}")
                if e.defense.defense_power > 0:
                    parts.append(f"방어력 {e.defense.defense_power:.1f}")
                if e.defense.armor > 0:
                    parts.append(f"갑옷 {e.defense.armor:.1f}")
                if e.state.shield > 0:
                    parts.append(f"보호막 {e.state.shield:.1f}")
                print(" | ".join(parts))

    print("\n[손패]")
    if not rt.hand:
        print("- 손패가 없습니다.")
    else:
        for i, c in enumerate(rt.hand, start=1):
            print(f"{i}) {_card_label(c)}")
            print(f"   {describe_card(c, player)}")
    print("=" * 82)


def _front_target(enemy_rows: List[List[Enemy]]) -> Optional[Enemy]:
    for row in enemy_rows:
        for e in row:
            if e.is_alive:
                return e
    return None


def _choose_card_index(rt: CombatRuntime, action_index: int, cfg: CombatConfig) -> Optional[int]:
    if not rt.hand:
        return None
    while True:
        s = input(
            f"{action_index}/{cfg.cards_per_turn}번째 행동 - 카드 번호 선택 "
            f"(1-{len(rt.hand)}, s=턴 종료, q=종료): "
        ).strip().lower()
        if s in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if s in ("s", "skip", ""):
            return None
        if s.isdigit():
            idx = int(s) - 1
            if 0 <= idx < len(rt.hand):
                return idx
        print("잘못된 입력입니다.")


def _choose_enemy_target(
    in_range: List[Enemy],
    distances: Dict[str, int],
    card: CardState,
    hit_idx: int,
    hit_count: int,
) -> Enemy:
    while True:
        print(f"\n[공격 대상 선택] {card.name} ({hit_idx}/{hit_count})")
        for i, e in enumerate(in_range, start=1):
            print(
                f"{i}) {e.name} | 거리 {distances.get(e.unit_id, '?')} | "
                f"HP {e.hp:.1f}/{e.max_hp:.1f} | 갑옷 {e.defense.armor:.1f} | 보호막 {e.state.shield:.1f}"
            )
        s = input(f"대상 번호 선택 (1-{len(in_range)}, q=종료): ").strip().lower()
        if s in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if s.isdigit():
            idx = int(s) - 1
            if 0 <= idx < len(in_range):
                return in_range[idx]
        print("잘못된 입력입니다.")


def _apply_attack_side_effects(
    player: Player,
    target: Enemy,
    card: CardState,
    dealt: float,
    rt: CombatRuntime,
    rng: random.Random,
    cfg: CombatConfig,
) -> None:
    eff = card.effects

    shield_ratio = max(0.0, _f(eff.get("shield_ratio", 0.0)))
    if dealt > 0 and shield_ratio > 0:
        gain = dealt * shield_ratio
        player.add_shield(gain)
        print(f"  -> 피해 비례 보호막 +{gain:.1f}")

    freeze_turns = _chance_count(_f(eff.get("freeze_turns", 0.0)), rng)
    if _apply_freeze(target, freeze_turns):
        print(f"  -> {target.name} 빙결 {freeze_turns}턴")

    bleed_ratio = max(0.0, _f(eff.get("bleed_ratio", 0.0)))
    bleed_turns = _chance_count(_f(eff.get("bleed_turns", 0.0)), rng)
    if bleed_ratio > 0 and bleed_turns > 0:
        bleed_damage = max(0.0, player.attack.power * (1.0 + rt.frenzy_ratio)) * bleed_ratio
        if _apply_bleed(target, bleed_damage, bleed_turns):
            print(f"  -> {target.name} 출혈 부여 ({bleed_damage:.1f}/턴, {bleed_turns}턴)")

    armor_down = max(0.0, _f(eff.get("armor_down_ratio", 0.0)))
    if armor_down > 0:
        _apply_armor_down(target, armor_down)
        print(f"  -> {target.name} 방어 감소 {armor_down*100:.1f}%")

    draw_cards = _f(eff.get("draw_cards", 0.0), 0.0)
    if draw_cards > 0:
        _draw_n(rt, cfg, rng, draw_cards, reason=f"{card.name} 효과")


def _resolve_player_attack(
    player: Player,
    card: CardState,
    enemy_rows: List[List[Enemy]],
    rt: CombatRuntime,
    rng: random.Random,
    cfg: CombatConfig,
) -> None:
    eff = card.effects
    attack_range = int(card.stats.get("range", 1))
    if bool(eff.get("max_range", False)):
        attack_range = max(attack_range, 99)

    aoe = bool(eff.get("aoe_all_enemies", False))
    double_hit = bool(eff.get("double_hit", False))
    ignore_defense = bool(eff.get("ignore_defense", False))
    splash_other_ratio = max(0.0, _f(eff.get("splash_other_ratio", 0.0)))
    hit_count = 2 if double_hit else 1

    attack_power = player.attack.power * (1.0 + rt.frenzy_ratio)
    if rt.next_attack_bonus_ratio > 0:
        before_damage = card.compute_attack_damage(attack_power)
        attack_power *= 1.0 + rt.next_attack_bonus_ratio
        after_damage = card.compute_attack_damage(attack_power)
        print(
            f"- 다음 공격 보너스 적용: +{rt.next_attack_bonus_ratio*100:.1f}% "
            f"(기본 피해 {before_damage:.1f} -> {after_damage:.1f})"
        )
        rt.next_attack_bonus_ratio = 0.0

    per_hit = card.compute_attack_damage(attack_power)
    if per_hit <= 0:
        print("- 공격 카드지만 피해량이 0입니다.")
        return

    any_hit = False
    for hit_idx in range(1, hit_count + 1):
        distances = _enemy_distances(enemy_rows)
        in_range = [
            e
            for row in enemy_rows
            for e in row
            if e.is_alive and distances.get(e.unit_id, 999) <= attack_range
        ]
        if not in_range:
            if not any_hit:
                print(f"- 사정거리({attack_range}) 안에 적이 없습니다.")
            break

        if aoe:
            targets = list(in_range)
        else:
            target = _choose_enemy_target(in_range, distances, card, hit_idx, hit_count)
            targets = [target]

        for t in targets:
            breakdown = _damage_breakdown(t, per_hit, ignore_defense=ignore_defense)
            dealt = _deal_damage(t, per_hit, ignore_defense=ignore_defense)
            any_hit = any_hit or dealt > 0
            print(_format_damage_line(card.name, t.name, dealt, breakdown, hit_idx, hit_count))
            if dealt > 0:
                _apply_attack_side_effects(player, t, card, dealt, rt, rng, cfg)
            if not t.is_alive:
                print(f"{t.name} 처치!")

            if (not aoe) and splash_other_ratio > 0:
                splash_damage = per_hit * splash_other_ratio
                others = [e for e in _alive_enemies(enemy_rows) if e.unit_id != t.unit_id]
                for other in others:
                    splash_breakdown = _damage_breakdown(other, splash_damage, ignore_defense=ignore_defense)
                    splash_dealt = _deal_damage(other, splash_damage, ignore_defense=ignore_defense)
                    print(
                        f"  -> 파급 피해({splash_other_ratio*100:.0f}%): "
                        f"{splash_breakdown.get('raw', 0.0):.1f}"
                        + (f" - 피해감소 {splash_breakdown.get('reduction_amount', 0.0):.1f}" if splash_breakdown.get("reduction_amount", 0.0) > 0 else "")
                        + (f" - 갑옷 {splash_breakdown.get('armor_block', 0.0):.1f}" if splash_breakdown.get("armor_block", 0.0) > 0 else "")
                        + (f" - 보호막 {splash_breakdown.get('shield_block', 0.0):.1f}" if splash_breakdown.get("shield_block", 0.0) > 0 else "")
                        + f" = {splash_dealt:.1f} ({other.name})"
                    )
                    if not other.is_alive:
                        print(f"  {other.name} 처치!")

    summon_key = str(eff.get("summon_unit", "") or "")
    if summon_key:
        _summon_comrade(summon_key, rt)


def _resolve_player_defense(
    player: Player,
    card: CardState,
    rt: CombatRuntime,
    rng: random.Random,
    cfg: CombatConfig,
) -> None:
    eff = card.effects
    gain = card.compute_shield_amount(player.defense.defense_power)
    player.add_shield(gain)
    print(f"{card.name} 사용: 보호막 {gain:.1f} 획득")

    temp_armor_ratio = max(0.0, _f(eff.get("temp_armor_ratio", 0.0)) + _f(eff.get("temp_defense_ratio", 0.0)))
    if temp_armor_ratio > 0:
        bonus_armor = max(0.0, player.defense.armor * temp_armor_ratio)
        rt.temp_armor_bonus += bonus_armor
        print(f"  -> 이번 라운드 갑옷 +{bonus_armor:.1f} ({temp_armor_ratio*100:.1f}%)")

    next_attack = max(0.0, _f(eff.get("next_attack_bonus_ratio", 0.0)))
    if next_attack > 0:
        rt.next_attack_bonus_ratio += next_attack
        print(f"  -> 다음 공격 보너스 +{next_attack*100:.1f}%")

    oh_freeze = max(0.0, _f(eff.get("on_hit_freeze_turns", 0.0)))
    if oh_freeze > 0:
        rt.on_hit_freeze_turns += oh_freeze
        print(f"  -> 피격 시 빙결 반격 준비 ({rt.on_hit_freeze_turns:.2f}턴)")

    oh_bleed_ratio = max(0.0, _f(eff.get("on_hit_bleed_ratio", 0.0)))
    oh_bleed_turns = max(0.0, _f(eff.get("on_hit_bleed_turns", 0.0)))
    if oh_bleed_ratio > 0 and oh_bleed_turns > 0:
        rt.on_hit_bleed_ratio += oh_bleed_ratio
        rt.on_hit_bleed_turns += oh_bleed_turns
        print(
            f"  -> 피격 시 출혈 반격 준비 "
            f"({rt.on_hit_bleed_ratio*100:.1f}%, {rt.on_hit_bleed_turns:.2f}턴)"
        )

    oh_draw = max(0.0, _f(eff.get("on_hit_draw_cards", 0.0)))
    if oh_draw > 0:
        rt.on_hit_draw_cards += oh_draw
        print(f"  -> 피격 시 드로우 준비 ({rt.on_hit_draw_cards:.2f}장)")

    oh_frenzy = max(0.0, _f(eff.get("on_hit_frenzy_ratio", 0.0)))
    if oh_frenzy > 0:
        rt.on_hit_frenzy_ratio += oh_frenzy
        print(f"  -> 피격 시 광란 준비 ({rt.on_hit_frenzy_ratio*100:.1f}%)")

    draw_cards = _f(eff.get("draw_cards", 0.0), 0.0)
    if draw_cards > 0:
        _draw_n(rt, cfg, rng, draw_cards, reason=f"{card.name} 효과")

    summon_key = str(eff.get("summon_unit", "") or "")
    if summon_key:
        _summon_comrade(summon_key, rt)


def _player_action_phase(
    player: Player,
    enemy_rows: List[List[Enemy]],
    rt: CombatRuntime,
    turn: int,
    cfg: CombatConfig,
    rng: random.Random,
) -> None:
    if not player.can_act():
        print("- 플레이어는 빙결 상태라 이번 턴에 행동할 수 없습니다.")
        return

    for action_idx in range(1, cfg.cards_per_turn + 1):
        if not rt.hand:
            print("- 손패가 없어 행동을 종료합니다.")
            return
        _render_status(player, enemy_rows, rt, turn, cfg)
        idx = _choose_card_index(rt, action_idx, cfg)
        if idx is None:
            print("- 남은 행동을 종료합니다.")
            return

        card = rt.hand.pop(idx)
        print(f"\n[{action_idx}번째 행동] {_card_label(card)} 사용")
        _gain_resource_from_card(card, rt, rng)

        if card.type == "attack":
            _resolve_player_attack(player, card, enemy_rows, rt, rng, cfg)
        elif card.type == "defense":
            _resolve_player_defense(player, card, rt, rng, cfg)
        else:
            print(f"- {_card_label(card)}: 미지원 카드 타입({card.type})")

        rt.discard.append(card)

        if not _alive_enemies(enemy_rows):
            return


def _comrade_phase(rt: CombatRuntime, enemy_rows: List[List[Enemy]]) -> None:
    alive_comrades = [c for c in rt.comrades if c.is_alive]
    if not alive_comrades:
        return
    print("\n[아군 소환체 행동]")
    for c in alive_comrades:
        target = _front_target(enemy_rows)
        if target is None:
            return
        if not c.can_act():
            print(f"- {c.name}({c.unit_id})은 빙결 상태로 행동 불가")
        else:
            dealt = c.basic_attack(target)
            print(f"- {c.name} -> {target.name} {dealt:.1f} 피해")
            if not target.is_alive:
                print(f"  {target.name} 처치!")

        bleed = _tick_bleed(c)
        if bleed > 0:
            print(f"  {c.name} 출혈 피해 {bleed:.1f}")
        c.end_turn()


def _trigger_on_hit_reactions(
    player: Player,
    attacker: Enemy,
    rt: CombatRuntime,
    cfg: CombatConfig,
    rng: random.Random,
) -> None:
    freeze_turns = _chance_count(rt.on_hit_freeze_turns, rng)
    if _apply_freeze(attacker, freeze_turns):
        print(f"  -> 반격 빙결: {attacker.name} {freeze_turns}턴")

    bleed_turns = _chance_count(rt.on_hit_bleed_turns, rng)
    if rt.on_hit_bleed_ratio > 0 and bleed_turns > 0:
        bleed_damage = max(0.0, player.attack.power * (1.0 + rt.frenzy_ratio)) * rt.on_hit_bleed_ratio
        if _apply_bleed(attacker, bleed_damage, bleed_turns):
            print(f"  -> 반격 출혈: {attacker.name} ({bleed_damage:.1f}/턴, {bleed_turns}턴)")

    if rt.on_hit_draw_cards > 0:
        _draw_n(rt, cfg, rng, rt.on_hit_draw_cards, reason="피격 반응")

    if rt.on_hit_frenzy_ratio > 0:
        rt.frenzy_ratio += rt.on_hit_frenzy_ratio
        print(f"  -> 광란 증가: +{rt.on_hit_frenzy_ratio*100:.1f}% (총 {rt.frenzy_ratio*100:.1f}%)")


def _enemy_phase(
    player: Player,
    enemy_rows: List[List[Enemy]],
    rt: CombatRuntime,
    cfg: CombatConfig,
    rng: random.Random,
) -> None:
    alive = _alive_enemies(enemy_rows)
    if not alive:
        return

    print("\n[적 턴]")
    base_armor = player.defense.armor
    player.defense.armor = base_armor + rt.temp_armor_bonus
    try:
        for e in alive:
            if not e.can_act():
                print(f"- {e.name}은 빙결 상태로 행동 불가")
            else:
                dealt = e.basic_attack(player)
                print(f"- {e.name}의 공격: 플레이어 {dealt:.1f} 피해")
                if player.is_alive:
                    _trigger_on_hit_reactions(player, e, rt, cfg, rng)

            bleed = _tick_bleed(e)
            if bleed > 0:
                print(f"  {e.name} 출혈 피해 {bleed:.1f}")
            e.end_turn()

            if not player.is_alive:
                break
    finally:
        player.defense.armor = base_armor


def _build_enemy_rows(row_counts: List[int]) -> List[List[Enemy]]:
    enemy_rows: List[List[Enemy]] = []
    
    # for row_idx, count in enumerate(row_counts, start=1):
    #     row: List[Enemy] = []
    #     for col_idx in range(1, max(0, count) + 1):
    #         row.append(
    #             Enemy(
    #                 unit_id=f"enemy_r{row_idx}c{col_idx}",
    #                 name=f"몬스터 {row_idx}-{col_idx}",
    #                 max_hp=300.0,
    #                 hp=300.0,
    #                 attack=AttackProfile(power=30.0),
    #                 defense=DefenseProfile(armor=0.0, defense_power=0.0),
    #             )
    #         )
    #     if row:
    #         enemy_rows.append(row)
    row1: List[Enemy] = []
    row1.extend(
        [
            Enemy(
                unit_id=f"wolf",
                name="늑대1",
                max_hp=300.0,
                hp=300.0,
                attack=AttackProfile(power=30.0),
                defense=DefenseProfile(armor=20.0, defense_power=0.0),
            ),
            Enemy(
                unit_id=f"wolf",
                name="늑대2",
                max_hp=300.0,
                hp=300.0,
                attack=AttackProfile(power=30.0),
                defense=DefenseProfile(armor=20.0, defense_power=0.0),
            )
        ]
    )
    row2: List[Enemy] = []
    row2.extend(
        [
            Enemy(
                unit_id=f"archer",
                name="궁수1",
                max_hp=200.0,
                hp=200.0,
                attack=AttackProfile(power=60.0),
                defense=DefenseProfile(armor=0.0, defense_power=0.0),
            ),
            Enemy(
                unit_id=f"archer",
                name="궁수2",
                max_hp=200.0,
                hp=200.0,
                attack=AttackProfile(power=60.0),
                defense=DefenseProfile(armor=0.0, defense_power=0.0),
            )
        ]
    )
    row3: List[Enemy] = []
    row3.extend(
        [
            Enemy(
                unit_id=f"mage",
                name="마법사1",
                max_hp=50.0,
                hp=50.0,
                attack=AttackProfile(power=150.0),
                defense=DefenseProfile(armor=0.0, defense_power=0.0),
            ),
        ]
    )
    
    enemy_rows.append(row1)
    enemy_rows.append(row2)
    enemy_rows.append(row3)
    return enemy_rows


def run_basic_combat(cfg: Optional[CombatConfig] = None) -> None:
    cfg = cfg or CombatConfig()
    rng = random.Random(cfg.seed)
    if cfg.seed is not None:
        random.seed(cfg.seed)

    engine = EnhancementEngine.load("card_definitions.json")
    player = Player(
        unit_id="player",
        name="플레이어",
        max_hp=500.0,
        hp=500.0,
        cards=[],
        attack=AttackProfile(power=50.0),
        defense=DefenseProfile(armor=15.0, defense_power=50.0),
    )
    templates = _build_starting_card_templates(engine)
    try:
        _manual_enhance_starting_cards(
            engine=engine,
            cards=templates,
            preview_player=player,
            rng=rng,
            min_count=cfg.min_start_enhance,
            max_count=cfg.max_start_enhance,
            center_count=cfg.center_start_enhance,
            n_choices=cfg.start_enhance_choices,
        )
    except KeyboardInterrupt:
        print("\n강화를 종료하여 전투를 시작하지 않았습니다.")
        return

    deck = _build_deck_from_templates(templates, copies_each=5, rng=rng)
    rt = CombatRuntime(deck=deck)
    enemy_rows = _build_enemy_rows(cfg.enemy_row_counts)

    _draw_n(rt, cfg, rng, cfg.initial_draw, reason="초기 손패")

    turn = 1
    print(
        f"\n전투 시작: 적 배치 {cfg.enemy_row_counts} "
        f"(각 HP 300 / ATK 30), 덱 {len(rt.deck)}장+손패 {len(rt.hand)}장"
    )
    print("규칙: 턴당 카드 2장 사용, 턴 시작 시 2장 드로우(손패 최대 6), 카드 사용 시 버림.")

    try:
        while True:
            if not player.is_alive:
                print("\n패배: 플레이어가 사망했습니다.")
                break
            if not _alive_enemies(enemy_rows):
                print("\n승리: 모든 몬스터를 처치했습니다.")
                break

            print(f"\n--- 턴 {turn} 시작 ---")
            if turn > 1:
                _draw_n(rt, cfg, rng, cfg.draw_per_turn, reason="턴 시작")

            _comrade_phase(rt, enemy_rows)
            if not _alive_enemies(enemy_rows):
                print("\n승리: 모든 몬스터를 처치했습니다.")
                break

            _player_action_phase(player, enemy_rows, rt, turn, cfg, rng)
            if not _alive_enemies(enemy_rows):
                print("\n승리: 모든 몬스터를 처치했습니다.")
                break

            p_bleed = _tick_bleed(player)
            if p_bleed > 0:
                print(f"- 플레이어 출혈 피해 {p_bleed:.1f}")
            player.end_turn()
            if not player.is_alive:
                print("\n패배: 플레이어가 사망했습니다.")
                break

            _enemy_phase(player, enemy_rows, rt, cfg, rng)
            if not player.is_alive:
                print("\n패배: 플레이어가 사망했습니다.")
                break

            rt.clear_round_buffs()
            turn += 1
    except KeyboardInterrupt:
        print("\n전투를 종료했습니다.")


if __name__ == "__main__":
    run_basic_combat()
