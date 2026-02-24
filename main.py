import os
import json
import copy
import random
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
from cards import EnhancementEngine


class Tier(IntEnum):
    COMMON = 0
    RARE = 1
    SPECIAL = 2
    HEROIC = 3
    LEGENDARY = 4
    MYTHICAL = 5
    UNIQUE = 6


TIERS_NAME = {
    Tier.COMMON: "일반적인 강화",
    Tier.RARE: "희귀한 강화",
    Tier.SPECIAL: "특별한 강화",
    Tier.HEROIC: "영웅적인 강화",
    Tier.LEGENDARY: "전설적인 강화",
    Tier.MYTHICAL: "신화적인 강화",
    Tier.UNIQUE: "유일한 강화",
}


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def ensure_json_ext(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    if not name.lower().endswith(".json"):
        name += ".json"
    return name


def pct(x: float) -> str:
    return f"{x*100:.3f}%"


def fmt_ratio_0(x: float) -> str:
    return f"{x*100:.0f}%"


def fmt_ratio_2(x: float) -> str:
    return f"{x*100:.2f}%"


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


@dataclass
class EnhanceGacha:
    base_probs: List[float] = None
    caps: List[int] = None
    pity_strengths: List[float] = None
    soft_k: float = 8.0
    soft_power: float = 4.0

    def __post_init__(self):
        if self.base_probs is None:
            base_number = [1000, 300, 100, 70, 30, 10, 3]
            self.base_probs = [x / sum(base_number) for x in base_number]
        if len(self.base_probs) != len(Tier):
            raise ValueError("base_probs != len(Tier)")
        s = sum(self.base_probs)
        if abs(s - 1.0) > 1e-9:
            self.base_probs = [p / s for p in self.base_probs]
        if self.caps is None:
            self.caps = [1, 3, 7, 15, 30, 50, 100]
        if len(self.caps) != len(Tier):
            raise ValueError("caps 길이는 Tier와 같아야 합니다.")
        if self.pity_strengths is None:
            self.pity_strengths = [0.5, 0.7, 2, 3, 10, 10, 10]
        if len(self.pity_strengths) != len(Tier):
            raise ValueError("pity_strengths 길이는 Tier와 같아야 합니다.")
        self.fail_counts = [0 for _ in range(len(Tier))]

    def reset_state(self) -> None:
        self.fail_counts = [0 for _ in range(len(Tier))]

    def _forced_min_tier(self) -> int:
        forced = -1
        for i, (cnt, cap) in enumerate(zip(self.fail_counts, self.caps)):
            if cap > 0 and cnt >= cap:
                forced = max(forced, i)
        return forced

    def _adjusted_probs(self) -> List[float]:
        n = len(self.base_probs)
        boosted = []
        for i, base_p in enumerate(self.base_probs):
            cap = self.caps[i]
            if cap <= 0:
                boosted.append(base_p)
                continue
            progress = min(1.0, self.fail_counts[i] / cap)
            rank = i / (n - 1)
            expo = self.soft_k * self.pity_strengths[i] * (progress ** self.soft_power) * (rank ** 2)
            if expo > 60:
                expo = 60
            boosted.append(base_p * math.exp(expo))

        min_tier = self._forced_min_tier()
        if min_tier >= 0:
            for i in range(min_tier):
                boosted[i] = 0.0

        total = sum(boosted)
        if total <= 0:
            out = [0.0] * n
            out[min_tier if min_tier >= 0 else 0] = 1.0
            return out
        return [x / total for x in boosted]

    def _sample_index(self, probs: List[float]) -> int:
        r = random.random()
        acc = 0.0
        idx = 0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                idx = i
                break
        return idx

    def roll_with_probs(self, probs: List[float]) -> Tuple[int, str]:
        idx = self._sample_index(probs)
        n = len(probs)
        for i in range(n):
            if idx >= i:
                self.fail_counts[i] = 0
            else:
                self.fail_counts[i] += 1
        return idx, TIERS_NAME[Tier(idx)]

    def simulate_many(self, trials: int = 1000, seed: int = 0, reset: bool = True, show_samples: int = 0) -> None:
        random.seed(seed)
        if reset:
            self.reset_state()
        tier_counts = Counter()
        forced_counts = Counter()
        unique_gaps = []
        since_unique = 0
        sample_lines = []
        for t in range(1, trials + 1):
            probs = self._adjusted_probs()
            forced = self._forced_min_tier()
            idx, name = self.roll_with_probs(probs)
            tier_counts[idx] += 1
            if forced >= 0:
                forced_counts[forced] += 1
            since_unique += 1
            if idx == Tier.UNIQUE:
                unique_gaps.append(since_unique)
                since_unique = 0
            if show_samples > 0 and t <= show_samples:
                forced_name = TIERS_NAME[Tier(forced)] if forced >= 0 else "없음"
                sample_lines.append(
                    f"{t:04d}회차: {name:10s} | 천장강제={forced_name:10s} | "
                    f"유일={probs[Tier.UNIQUE]*100:.3f}% 신화={probs[Tier.MYTHICAL]*100:.3f}% 전설={probs[Tier.LEGENDARY]*100:.3f}%"
                )
        print(f"=== 시뮬레이션 결과: trials={trials}, seed={seed}, reset={reset} ===")
        if sample_lines:
            print("\n--- 샘플(초반) ---")
            for line in sample_lines:
                print(line)
        print("\n--- 티어 분포 ---")
        for i in range(len(Tier)):
            c = tier_counts[i]
            pctv = c / trials * 100.0
            print(f"{TIERS_NAME[Tier(i)]:10s}: {c:4d}회 ({pctv:6.2f}%)")
        print("\n--- 천장(강제 최소 티어) 발동 ---")
        total_forced = sum(forced_counts.values())
        print(f"천장 발동 총합: {total_forced}회 ({total_forced / trials * 100:.2f}%)")
        if total_forced > 0:
            for i in range(len(Tier)):
                if i in forced_counts:
                    c = forced_counts[i]
                    pctv = c / trials * 100.0
                    print(f"  {TIERS_NAME[Tier(i)]:10s} 강제: {c:4d}회 ({pctv:6.2f}%)")
        print("\n--- UNIQUE 통계 ---")
        unique_count = tier_counts[Tier.UNIQUE]
        print(f"UNIQUE 횟수: {unique_count}회 ({unique_count / trials * 100:.2f}%)")
        if unique_gaps:
            avg_gap = sum(unique_gaps) / len(unique_gaps)
            max_gap = max(unique_gaps)
            print(f"UNIQUE 등장 간격: 평균 {avg_gap:.2f}회, 최대 {max_gap}회")
            if since_unique > 0:
                print(f"마지막 UNIQUE 이후 누적: {since_unique}회")
        else:
            print("UNIQUE가 한 번도 나오지 않았습니다.")

    def batch_apply_enhancements(self, engine: "EnhancementEngine", card_template: "CardState", trials: int = 1000, seed: int = 0, reset: bool = True, show_samples: int = 0) -> None:
        """티켓을 n번 반복하여 실제로 `engine.apply_tier`를 적용하고 결과를 집계해 출력합니다.

        - `engine`: `EnhancementEngine` 인스턴스
        - `card_template`: 각 시도에 복사해 사용할 `CardState`
        - `trials`: 시도 횟수
        - `seed`: 랜덤 시드
        - `reset`: 시작 전 `fail_counts` 리셋 여부
        - `show_samples`: 초반 샘플 출력 수
        """
        random.seed(seed)
        if reset:
            self.reset_state()

        tier_counts = Counter()
        forced_counts = Counter()
        applied_counts = Counter()
        unique_gaps = []
        since_unique = 0
        sample_lines = []

        for t in range(1, trials + 1):
            probs = self._adjusted_probs()
            forced = self._forced_min_tier()
            idx, _ = self.roll_with_probs(probs)

            # 각 시도마다 카드 템플릿을 복사해서 실제로 효과를 적용
            card = CardState.from_dict(card_template.to_dict())
            before = CardState.from_dict(card.to_dict())
            tier_str = Tier(idx).name
            res = engine.apply_tier(card, tier_str, rng=random)
            used = res.get("tier_used", tier_str)

            tier_counts[idx] += 1
            applied_counts[used] += 1
            if forced >= 0:
                forced_counts[forced] += 1

            since_unique += 1
            if used == Tier.UNIQUE.name:
                unique_gaps.append(since_unique)
                since_unique = 0

            if show_samples > 0 and t <= show_samples:
                forced_name = TIERS_NAME[Tier(forced)] if forced >= 0 else "없음"
                sample_lines.append(
                    f"{t:04d}회차: 뽑힘={TIERS_NAME[Tier(idx)]:10s} | 적용={TIERS_NAME[Tier[used]]:10s} | 천장강제={forced_name}"
                )

        print(f"=== 배치 강화 결과: trials={trials}, seed={seed}, reset={reset} ===")
        if sample_lines:
            print("\n--- 샘플(초반) ---")
            for line in sample_lines:
                print(line)

        print("\n--- 뽑힌 티어 분포(뽑힘 기준) ---")
        for i in range(len(Tier)):
            c = tier_counts[i]
            pctv = c / trials * 100.0
            print(f"{TIERS_NAME[Tier(i)]:10s}: {c:4d}회 ({pctv:6.2f}%)")

        print("\n--- 적용된 티어 분포(실제 적용된 기준) ---")
        total_applied = sum(applied_counts.values())
        for name, c in sorted(applied_counts.items(), key=lambda x: Tier[x[0]].value if x[0] in Tier.__members__ else -1):
            pctv = c / trials * 100.0
            try:
                enum_val = Tier[name]
                label = TIERS_NAME[enum_val]
            except Exception:
                label = name
            print(f"{label:10s}: {c:4d}회 ({pctv:6.2f}%)")

        print("\n--- 천장(강제 최소 티어) 발동 ---")
        total_forced = sum(forced_counts.values())
        print(f"천장 발동 총합: {total_forced}회 ({total_forced / trials * 100:.2f}%)")
        if total_forced > 0:
            for i in range(len(Tier)):
                if i in forced_counts:
                    c = forced_counts[i]
                    pctv = c / trials * 100.0
                    print(f"  {TIERS_NAME[Tier(i)]:10s} 강제: {c:4d}회 ({pctv:6.2f}%)")

        print("\n--- UNIQUE 통계 ---")
        unique_count = applied_counts.get(Tier.UNIQUE.name, 0)
        print(f"UNIQUE 횟수: {unique_count}회 ({unique_count / trials * 100:.2f}%)")
        if unique_gaps:
            avg_gap = sum(unique_gaps) / len(unique_gaps)
            max_gap = max(unique_gaps)
            print(f"UNIQUE 등장 간격: 평균 {avg_gap:.2f}회, 최대 {max_gap}회")
        else:
            print("UNIQUE가 한 번도 나오지 않았습니다.")

    def batch_apply_enhancements_choice_max(
        self,
        engine: "EnhancementEngine",
        card_template: "CardState",
        trials: int = 1000,
        seed: int = 0,
        reset: bool = True,
        show_samples: int = 0,
        n_choices: int = 3,
    ) -> None:
        random.seed(seed)
        if reset:
            self.reset_state()

        tier_counts = Counter()       # '제시 후보 중 선택된(=최종)' 티어 인덱스 분포
        forced_counts = Counter()
        applied_counts = Counter()
        sample_lines = []

        for t in range(1, trials + 1):
            forced = self._forced_min_tier()

            # 1) 후보 n개 미리보기(상태 변화 없음)
            picks, option_infos = self.preview_choices(engine, card_template, n_choices)

            # 2) 최댓값 선택
            chosen_idx = max(picks)

            # 3) 선택 확정(여기서만 pity/천장 상태 갱신)
            self.commit_choice(chosen_idx)

            # 4) 카드에 적용
            card = CardState.from_dict(card_template.to_dict())
            tier_str = Tier(chosen_idx).name
            res = engine.apply_tier(card, tier_str, rng=random)
            used = res.get("tier_used", tier_str)

            tier_counts[chosen_idx] += 1
            applied_counts[used] += 1
            if forced >= 0:
                forced_counts[forced] += 1

            if show_samples > 0 and t <= show_samples:
                forced_name = TIERS_NAME[Tier(forced)] if forced >= 0 else "없음"
                candidate_strs = [f"{TIERS_NAME[Tier(info['tier'])]}:{info['display']}" for info in option_infos]
                sample_lines.append(
                    f"{t:04d}회차: 후보={candidate_strs} | 선택={TIERS_NAME[Tier(chosen_idx)]} | 적용={TIERS_NAME[Tier[used]]} | 천장강제={forced_name}"
                )

        print(f"=== 배치(후보 {n_choices}개 중 최고 선택) 결과: trials={trials}, seed={seed}, reset={reset} ===")
        if sample_lines:
            print("\n--- 샘플(초반) ---")
            for line in sample_lines:
                print(line)

        print("\n--- 최종 선택 티어 분포(선택된 idx 기준) ---")
        for i in range(len(Tier)):
            c = tier_counts[i]
            pctv = c / trials * 100.0
            print(f"{TIERS_NAME[Tier(i)]:10s}: {c:4d}회 ({pctv:6.2f}%)")

        print("\n--- 적용된 티어 분포(실제 적용 기준) ---")
        for name, c in sorted(applied_counts.items(), key=lambda x: Tier[x[0]].value if x[0] in Tier.__members__ else -1):
            pctv = c / trials * 100.0
            label = TIERS_NAME[Tier[name]] if name in Tier.__members__ else name
            print(f"{label:10s}: {c:4d}회 ({pctv:6.2f}%)")

        print("\n--- 천장(강제 최소 티어) 발동 ---")
        total_forced = sum(forced_counts.values())
        print(f"천장 발동 총합: {total_forced}회 ({total_forced / trials * 100:.2f}%)")
        
    def _apply_roll_to_state(self, idx: int) -> None:
        """roll_with_probs에서 하던 fail_counts 갱신만 분리"""
        n = len(self.base_probs)
        for i in range(n):
            if idx >= i:
                self.fail_counts[i] = 0
            else:
                self.fail_counts[i] += 1

    @staticmethod
    def transform_probs_for_choice(probs: List[float], n_choices: int = 3) -> List[float]:
        """
        'n_choices개 중 하나 선택'에서 최종 선택 분포가
        기존 단일 뽑기와 같아지도록 각 슬롯 확률을 낮추는 변환.
        """
        n = len(probs)
        # 생존함수 S[i] = P(tier >= i)
        S = [0.0] * (n + 1)
        S[n] = 0.0
        acc = 0.0
        for i in range(n - 1, -1, -1):
            acc += probs[i]
            S[i] = acc

        # 변환된 생존함수 S'
        Sp = [0.0] * (n + 1)
        Sp[n] = 0.0
        inv = 1.0 / float(n_choices)
        for i in range(n):
            # S'_i = 1 - (1 - S_i)^(1/n)
            x = max(0.0, min(1.0, S[i]))
            Sp[i] = 1.0 - ((1.0 - x) ** inv)

        # 티어별 확률 q[i] = S'[i] - S'[i+1]
        q = [0.0] * n
        for i in range(n):
            q[i] = max(0.0, Sp[i] - Sp[i + 1])

        # 정규화(부동소수 오차 방지)
        total = sum(q)
        if total <= 0:
            q[0] = 1.0
            return q
        return [x / total for x in q]

    def preview_choices(self, engine: "EnhancementEngine", card: "CardState", n_choices: int = 3) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        현재 상태에서 (천장/피티 반영된) 확률을 기반으로
        '보기용' 강화 티어 n개를 뽑아주고, 각 후보에 대해
        `engine`을 사용해 그 티어에서 선택될 "옵션"(display 및 effects)을
        미리 샘플링해 반환합니다. 상태 업데이트는 발생하지 않습니다.

        반환값: (picks, option_infos)
          - picks: 티어 인덱스 리스트
          - option_infos: 각 후보에 대응하는 옵션 요약(dict)
        """
        base = self._adjusted_probs()
        offer_probs = self.transform_probs_for_choice(base, n_choices=n_choices)
        picks = [self._sample_index(offer_probs) for _ in range(n_choices)]

        option_infos: List[Dict[str, Any]] = []
        for idx in picks:
            opt = engine.preview_option(card, Tier(idx).name, rng=random)
            if opt is None:
                option_infos.append({"tier": idx, "display": "(옵션 없음)", "effects": [], "opt": None})
            else:
                disp = opt.get("display", opt.get("id", ""))
                if opt.get("max_range", False):
                    disp = f"{disp} (max range)"
                option_infos.append({"tier": idx, "display": disp, "effects": opt.get("effects", []), "opt": opt})

        return picks, option_infos

    def commit_choice(self, idx: int) -> Tuple[int, str]:
        """선택된 강화 1개를 확정(상태 업데이트)"""
        self._apply_roll_to_state(idx)
        return idx, TIERS_NAME[Tier(idx)]

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

def load_first_card(path: str = "cards.json") -> CardState:
    if not os.path.exists(path):
        data = {
            "cards": [
                {
                    "id": "atk_001",
                    "name": "기본 공격",
                    "type": "attack",
                    "stats": {"attack": 100, "range": 1},
                    "effects": {
                        "ignore_defense": False,
                        "double_hit": False,
                        "aoe_all_enemies": False,
                        "triple_attack": False,
                        "freeze_turns": 0.0,
                        "shield_ratio": 0.0,
                        "bleed_ratio": 0.0,
                        "bleed_turns": 0,
                        "armor_down_ratio": 0.0,
                        "draw_cards": 0.0,
                        "summon_unit": None
                    }
                }
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cards = data.get("cards", [])
    if not cards:
        raise ValueError("cards.json에 cards가 없습니다.")

    cd = cards[0]
    cid = cd["id"]
    name = cd.get("name", cid)
    ctype = cd.get("type", "attack")

    stats = cd.get("stats")
    if stats is None:
        stats = {"attack": cd.get("attack", 0), "range": cd.get("range", 0)}
    stats = dict(stats)

    effects = dict(cd.get("effects", {}) or {})

    base_atk = float(stats.get("attack", 0.0))
    stats.setdefault("base_attack", base_atk)
    stats.setdefault("attack_bonus_ratio", 0.0)
    stats.setdefault("resource", 0)
    effects.setdefault("ignore_defense", False)
    effects.setdefault("double_hit", False)
    effects.setdefault("aoe_all_enemies", False)
    effects.setdefault("freeze_turns", 0.0)
    effects.setdefault("shield_ratio", 0.0)
    effects.setdefault("bleed_ratio", 0.0)
    effects.setdefault("bleed_turns", 0)
    effects.setdefault("armor_down_ratio", 0.0)
    effects.setdefault("draw_cards", 0.0)
    effects.setdefault("summon_unit", None)
    effects.setdefault("triple_attack", False)

    return CardState(id=cid, name=name, type=ctype, stats=stats, effects=effects)

def describe_card(card: CardState) -> str:
    parts: List[str] = []
    eff = card.effects
    resource = safe_int(card.stats.get("resource", 0))
    if card.type == "attack":
        base_atk = safe_float(card.stats.get("base_attack", card.stats.get("attack", 0.0)))
        bonus = safe_float(card.stats.get("attack_bonus_ratio", 0.0))
        atk = base_atk * (1.0 + bonus)
        rngv = safe_int(card.stats.get("range", 0))
        if bool(eff.get("triple_attack", False)):
            parts.append(f"피해를 {atk:.1f} * 3 = {atk * 3:.1f} 줍니다.")
        else:
            parts.append(f"피해를 {atk:.1f} 줍니다.")
        parts.append(f"거리({rngv})")
    else:
        parts.append(f"{card.name}")

    eff_parts: List[str] = []

    if bool(eff.get("ignore_defense", False)):
        eff_parts.append("방어를 무시한다")
    if bool(eff.get("double_hit", False)):
        eff_parts.append("공격이 두 번 적용된다")
    if bool(eff.get("aoe_all_enemies", False)):
        eff_parts.append("모든 적에게 피해를 입힌다")
    if bool(eff.get("triple_attack", False)):
        eff_parts.append("전체 공격력이 300% 강화된다")

    ft = safe_float(eff.get("freeze_turns", 0.0))
    if ft > 0:
        eff_parts.append(f"빙결 {ft:.2f}턴")

    dr = safe_float(eff.get("draw_cards", 0.0))
    if dr > 0:
        eff_parts.append(f"카드를 {dr:.2f}장 뽑는다")

    sr = safe_float(eff.get("shield_ratio", 0.0))
    if sr > 0:
        eff_parts.append(f"피해량의 {fmt_ratio_0(sr)}만큼 보호막을 얻는다")

    br = safe_float(eff.get("bleed_ratio", 0.0))
    bt = safe_int(eff.get("bleed_turns", 0))
    if br > 0 and bt > 0:
        eff_parts.append(f"출혈(공격력의 {fmt_ratio_0(br)}) {bt}턴")

    ar = safe_float(eff.get("armor_down_ratio", 0.0))
    if ar > 0:
        eff_parts.append(f"상대의 방어력을 {fmt_ratio_0(ar)} 감소시킨다")
        
    if resource > 0:
        eff_parts.append(f"자원 +{resource}")

    su = eff.get("summon_unit", None)
    if su:
        if su == "wolf":
            eff_parts.append("늑대를 소환한다")
        else:
            eff_parts.append(f"{su}을(를) 소환한다")

    if eff_parts:
        parts.append("/ " + " / ".join(eff_parts))

    return " ".join(parts).replace("  ", " ").strip()


STAT_LABEL = {"attack": "공격력", "range": "거리", "resource": "자원"}
EFFECT_LABEL = {
    "ignore_defense": "방어 무시",
    "double_hit": "공격 2회",
    "aoe_all_enemies": "전체 공격",
    "freeze_turns": "빙결",
    "shield_ratio": "보호막 비율",
    "bleed_ratio": "출혈 비율",
    "bleed_turns": "출혈 턴",
    "armor_down_ratio": "방어 감소",
    "draw_cards": "드로우",
    "summon_unit": "소환",
    "triple_attack": "3배 공격",
}


def format_field(key: str, v: Any) -> str:
    if key in ("shield_ratio", "bleed_ratio", "armor_down_ratio"):
        return fmt_ratio_2(safe_float(v))
    if key == "freeze_turns":
        return f"{safe_float(v):.2f}턴"
    if key == "draw_cards":
        return f"{safe_float(v):.2f}장"
    if key == "bleed_turns":
        return f"{safe_int(v)}턴"
    return str(v)


def diff_card(before: CardState, after: CardState) -> List[str]:
    def calc_attack(c: CardState) -> float:
        base_atk = safe_float(c.stats.get("base_attack", c.stats.get("attack", 0.0)))
        bonus = safe_float(c.stats.get("attack_bonus_ratio", 0.0))
        return base_atk * (1.0 + bonus)
    lines: List[str] = []

    s_keys = sorted(set(before.stats.keys()) | set(after.stats.keys()))
    for k in s_keys:
        b = before.stats.get(k, None)
        a = after.stats.get(k, None)
        if b == a:
            continue
        label = STAT_LABEL.get(k, k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
            if k == "attack_bonus_ratio":
                b_atk = calc_attack(before)
                a_atk = calc_attack(after)
                if b_atk == a_atk:
                    continue
                delta = a_atk - b_atk
                lines.append(f"- 공격력: {b_atk:.1f} -> {a_atk:.1f} ({delta:+.1f})")
                continue
            else:
                lines.append(f"- {label}: {b} -> {a} ({delta:+})")
        else:
            lines.append(f"- {label}: {b} -> {a}")

    e_keys = sorted(set(before.effects.keys()) | set(after.effects.keys()))
    s_keys = sorted(set(before.stats.keys()) | set(after.stats.keys()) | {"attack"})
    for k in e_keys:
        b = before.effects.get(k, None)
        a = after.effects.get(k, None)
        if b == a:
            continue
        label = EFFECT_LABEL.get(k, k)
        lines.append(f"- {label}: {format_field(k, b)} -> {format_field(k, a)}")

    return lines


def probs_block(probs: List[float]) -> str:
    rows = []
    for i in range(len(Tier)):
        t = Tier(i)
        rows.append(f"{TIERS_NAME[t]} {probs[i]*100:6.3f}%")
    return "\n".join(rows)


def enhancement_counts_line(card: CardState) -> str:
    items = []
    for t in Tier:
        name = t.name
        c = card.enhancement_counts.get(name, 0)
        if c:
            items.append(f"{TIERS_NAME[t]}×{c}")
    return ", ".join(items) if items else "없음"


def save_card(card: CardState, filename: str) -> str:
    filename = ensure_json_ext(filename)
    if not filename:
        return ""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(card.to_dict(), f, ensure_ascii=False, indent=2)
    return filename


def interactive_session(g: EnhanceGacha, engine: EnhancementEngine, card: CardState, seed: Optional[int] = None, n_choices: int = 3) -> None:
    if seed is not None:
        random.seed(seed)

    last_result: Optional[Dict[str, Any]] = None
    last_before: Optional[CardState] = None
    last_after: Optional[CardState] = None
    count = 0
    status = ""

    while True:
        clear_screen()

        print(f"카드: {card.name} ({card.id})")
        print(describe_card(card))
        print(f"티어 강화 횟수: {enhancement_counts_line(card)}")
        if status:
            print(f"\n{status}")
        status = ""

        if last_result is not None and last_before is not None and last_after is not None:
            rolled = last_result["rolled_tier"]
            used = last_result["tier_used"]
            rolled_name = TIERS_NAME[Tier[rolled]]
            used_name = TIERS_NAME[Tier[used]]
            print("\n" + "=" * 60)
            print(f"{count:02d}회차 결과")
            if rolled == used:
                print(f"뽑힌 강화: {rolled_name}")
            else:
                print(f"뽑힌 강화: {rolled_name} -> 적용: {used_name}")
            print(f"적용 옵션: {last_result['display']}{last_result.get('roll_text','')}")
            print("\n[이전 카드]")
            print(describe_card(last_before))
            print("\n[이후 카드]")
            print(describe_card(last_after))
            changes = diff_card(last_before, last_after)
            print("\n[변화]")
            if changes:
                for line in changes:
                    print(line)
            else:
                print("변화 없음")
            print("=" * 60)

        probs = g._adjusted_probs()
        forced = g._forced_min_tier()
        forced_name = TIERS_NAME[Tier(forced)] if forced >= 0 else "없음"

        print("\n천장 강제 최소 티어:", forced_name)
        print("\n[현재 강화 확률]")
        print(probs_block(probs))
        
        print("\n엔터: 강화 1회 | tN: N티어 이상 나올 때까지 연속 강화 (예: t4) | r: 리셋| s: 저장 | q: 종료")
        cmd = input("> ").strip().lower()
        if cmd in ("q", "quit"):
            clear_screen()
            print("종료합니다.")
            break

        if cmd in ("r", "quit"):
            card = load_first_card("cards.json")
            g.reset_state()
            count = 0
            last_result = None
            last_before = None
            last_after = None
            status = ""
            continue

        if cmd == "s":
            name = input("저장할 파일 이름(.json 생략 가능): ").strip()
            out = save_card(card, name)
            status = f"저장 완료: {out}" if out else "저장 취소"
            continue

        target_tier: Optional[int] = None
        if cmd.startswith("t") and len(cmd) >= 2 and cmd[1:].isdigit():
            target_tier = int(cmd[1:])
            if target_tier < 0:
                target_tier = 0
            if target_tier > 6:
                target_tier = 6

        if cmd != "" and target_tier is None:
            status = "알 수 없는 입력"
            continue

        def do_one_roll(auto_pick_best: bool = False) -> None:
            nonlocal last_result, last_before, last_after, count

            # 1) 보기용 3개 뽑기(상태 변화 없음)
            picks, option_infos = g.preview_choices(engine, card, n_choices=n_choices)

            # 2) 선택
            if auto_pick_best:
                # 가장 높은 티어(숫자 큰 것) 자동 선택
                best_idx = max(picks)
                chosen_idx = best_idx
                try:
                    sel_idx = picks.index(chosen_idx)
                    chosen_opt = option_infos[sel_idx].get("opt")
                except Exception:
                    chosen_opt = None
            else:
                # 사용자에게 3개 보여주고 선택 받기
                print(f"\n[강화 선택지 {n_choices}개]")
                for i, (idx, info) in enumerate(zip(picks, option_infos), start=1):
                    print(f"{i}) {TIERS_NAME[Tier(idx)]} - {info.get('display','')}")

                choices_str = "/".join(map(str, range(1, n_choices+1)))
                sel = input(f"선택({choices_str}, 엔터=1): ").strip()
                if sel == "" or sel not in [str(i) for i in range(1, n_choices+1)]:
                    sel = "1"
                sel_idx = int(sel) - 1
                chosen_idx = picks[sel_idx]
                chosen_opt = option_infos[sel_idx].get("opt")

            # 3) 선택 확정(여기서만 pity/천장 상태 갱신)
            g.commit_choice(chosen_idx)
            tier_str = Tier(chosen_idx).name

            # 4) 카드에 강화 적용 (선택된 옵션을 전달하여 동일 옵션 적용)
            before = CardState.from_dict(card.to_dict())
            applied = engine.apply_tier(card, tier_str, rng=random, selected_option=chosen_opt)

            count += 1
            last_result = applied
            last_before = before
            last_after = CardState.from_dict(card.to_dict())
            
        if target_tier is None:
            do_one_roll()
            continue

        loops = 0
        max_loops = 10000
        while True:
            do_one_roll(auto_pick_best=True)
            loops += 1
            used = last_result["tier_used"]
            if Tier[used].value >= target_tier:
                status = f"t{target_tier}: {loops}번 시도 끝에 {TIERS_NAME[Tier[used]]} 획득"
                break
            if loops >= max_loops:
                status = f"t{target_tier}: {max_loops}번을 초과하여 중단"
                break
    
    


if __name__ == "__main__":
    engine = EnhancementEngine.load("enhancements.json")
    card = load_first_card("cards.json")
    g = EnhanceGacha()
    interactive_session(g, engine, card, n_choices=3)
    # g.batch_apply_enhancements(engine, card, trials=100000, reset=True)
    # g.batch_apply_enhancements_choice_max(engine, card, trials=100000, reset=True, n_choices=3)
