import os
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
from cards import (
    CardState,
    EnhancementEngine,
    Tier,
    TIER_LABELS_BY_NAME,
    TIERS_NAME,
    describe_card,
    diff_card,
    enhancement_counts_line,
    load_cards,
    save_cards,
)
from unit import AttackProfile, DefenseProfile, Player


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pct(x: float) -> str:
    return f"{x*100:.3f}%"


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

    def _adjusted_probs(self, apply_forced_floor: bool = True) -> List[float]:
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

        min_tier = self._forced_min_tier() if apply_forced_floor else -1
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
        efficiency_proc_count = 0
        efficiency_double_proc_count = 0
        lower_bonus_proc_count = 0
        same_tier_bonus_proc_count = 0
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
            if res.get("efficiency_proc", False):
                efficiency_proc_count += 1
            if res.get("efficiency_double_proc", False):
                efficiency_double_proc_count += 1
            if res.get("lower_bonus_proc", False):
                lower_bonus_proc_count += 1
            if res.get("same_tier_bonus_proc", False):
                same_tier_bonus_proc_count += 1

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
        print("\n--- 추가 발동 통계 ---")
        print(f"강화 효율 50% 증가 발동: {efficiency_proc_count}회 ({efficiency_proc_count / trials * 100:.2f}%)")
        print(f"강화 효율 두 배 발동: {efficiency_double_proc_count}회 ({efficiency_double_proc_count / trials * 100:.2f}%)")
        print(f"하위 티어 무작위 업그레이드 발동: {lower_bonus_proc_count}회 ({lower_bonus_proc_count / trials * 100:.2f}%)")
        print(f"동일 티어 무작위 업그레이드 발동: {same_tier_bonus_proc_count}회 ({same_tier_bonus_proc_count / trials * 100:.2f}%)")

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
        efficiency_proc_count = 0
        efficiency_double_proc_count = 0
        lower_bonus_proc_count = 0
        same_tier_bonus_proc_count = 0
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
            higher_gap = max(0, max(picks) - chosen_idx) if picks else 0
            res = engine.apply_tier(card, tier_str, rng=random, higher_tier_gap=higher_gap)
            used = res.get("tier_used", tier_str)
            if res.get("efficiency_proc", False):
                efficiency_proc_count += 1
            if res.get("efficiency_double_proc", False):
                efficiency_double_proc_count += 1
            if res.get("lower_bonus_proc", False):
                lower_bonus_proc_count += 1
            if res.get("same_tier_bonus_proc", False):
                same_tier_bonus_proc_count += 1

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
        print("\n--- 추가 발동 통계 ---")
        print(f"강화 효율 50% 증가 발동: {efficiency_proc_count}회 ({efficiency_proc_count / trials * 100:.2f}%)")
        print(f"강화 효율 두 배 발동: {efficiency_double_proc_count}회 ({efficiency_double_proc_count / trials * 100:.2f}%)")
        print(f"하위 티어 무작위 업그레이드 발동: {lower_bonus_proc_count}회 ({lower_bonus_proc_count / trials * 100:.2f}%)")
        print(f"동일 티어 무작위 업그레이드 발동: {same_tier_bonus_proc_count}회 ({same_tier_bonus_proc_count / trials * 100:.2f}%)")
        
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

    def preview_choice_tiers(self, n_choices: int = 3) -> List[int]:
        """현재 피티/천장 상태에서 선택지용 티어 인덱스 n개를 샘플링(상태 변화 없음)."""
        forced = self._forced_min_tier()
        base_normal = self._adjusted_probs(apply_forced_floor=False)
        offer_probs_normal = self.transform_probs_for_choice(base_normal, n_choices=n_choices)
        picks = [self._sample_index(offer_probs_normal) for _ in range(n_choices)]
        if forced >= 0 and n_choices > 0:
            base_forced = self._adjusted_probs(apply_forced_floor=True)
            offer_probs_forced = self.transform_probs_for_choice(base_forced, n_choices=n_choices)
            forced_slot = random.randrange(n_choices)
            picks[forced_slot] = self._sample_index(offer_probs_forced)
        return picks

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
        picks = self.preview_choice_tiers(n_choices=n_choices)

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


def compact_option_text(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("공격력이 "):
        s = "공격력 " + s[len("공격력이 "):]
    if " 강화된다" in s:
        s = s.replace(" 강화된다", " 강화")
    return s


def format_extra_effects(result: Dict[str, Any]) -> str:
    parts: List[str] = []
    if result.get("efficiency_proc", False):
        parts.append("효율 x1.5")
    if result.get("efficiency_double_proc", False):
        parts.append("효율 x2")
    if result.get("lower_bonus_proc", False):
        parts.append("하위 티어 보너스")
    if result.get("same_tier_bonus_proc", False):
        parts.append("동일 티어 보너스")
    return " + ".join(parts)


def format_applied_option(result: Dict[str, Any]) -> str:
    base = compact_option_text(result.get("display_applied", result.get("display", "")))
    parts = [base] if base else []

    lower = result.get("lower_bonus_result")
    if isinstance(lower, dict):
        lower_text = compact_option_text(lower.get("display_applied", lower.get("display", "")))
        if lower_text:
            parts.append(f"하위 티어 {lower_text}")

    same = result.get("same_tier_bonus_result")
    if isinstance(same, dict):
        same_text = compact_option_text(same.get("display_applied", same.get("display", "")))
        if same_text:
            parts.append(same_text)

    return " + ".join(parts)


def probs_block(probs: List[float]) -> str:
    rows = []
    for i in range(len(Tier)):
        t = Tier(i)
        rows.append(f"{TIERS_NAME[t]} {probs[i]*100:6.3f}%")
    return "\n".join(rows)


def interactive_session(
    g: EnhanceGacha,
    engine: EnhancementEngine,
    cards: List[CardState],
    seed: Optional[int] = None,
    n_choices: int = 3,
) -> None:
    if seed is not None:
        random.seed(seed)

    dummy_player = Player(
        unit_id="dummy_player",
        name="Dummy Player",
        max_hp=1.0,
        hp=1.0,
        attack=AttackProfile(power=100.0),
        defense=DefenseProfile(armor=0.0, shield_power=100.0),
    )
    last_result: Optional[Dict[str, Any]] = None
    last_before: Optional[CardState] = None
    last_after: Optional[CardState] = None
    last_card_label: str = ""
    count = 0
    status = ""

    while True:
        clear_screen()

        print(f"보유 카드 수: {len(cards)}")
        for i, c in enumerate(cards, start=1):
            print(f"[{i}] {c.name} ({c.id})")
            print(f"  {describe_card(c, dummy_player)}")
            print(
                f"  티어 강화 횟수: "
                f"{enhancement_counts_line(c, [t.name for t in Tier], TIER_LABELS_BY_NAME)}"
            )
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
            if last_card_label:
                print(f"대상 카드: {last_card_label}")
            if rolled == used:
                print(f"뽑힌 강화: {rolled_name}")
            else:
                print(f"뽑힌 강화: {rolled_name} -> 적용: {used_name}")
            extra_effects = format_extra_effects(last_result)
            if extra_effects:
                print(f"추가 효과: {extra_effects}")
            applied_text = format_applied_option(last_result)
            print(f"적용 옵션: {applied_text}")
            print("\n[이전 카드]")
            print(describe_card(last_before, dummy_player))
            print("\n[이후 카드]")
            print(describe_card(last_after, dummy_player))
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
        forced_prob_text = pct(probs[forced]) if forced >= 0 else "-"
        print(f"\n천장 티어: {forced_name} | 확률 {forced_prob_text}")
        print("\n[현재 강화 확률]")
        print(probs_block(probs))
        
        print("\n엔터: 강화 1회 | c: 카드 전체 보기 | tN: N티어 이상 나올 때까지 연속 강화 (예: t4) | r: 리셋| s: 저장 | q: 종료")
        cmd = input("> ").strip().lower()
        if cmd in ("q", "quit"):
            clear_screen()
            print("종료합니다.")
            break

        if cmd in ("r", "quit"):
            cards = load_cards("cards.json")
            g.reset_state()
            count = 0
            last_result = None
            last_before = None
            last_after = None
            last_card_label = ""
            status = ""
            continue

        if cmd == "s":
            name = input("저장할 파일 이름(.json 생략 가능): ").strip()
            out = save_cards(cards, name)
            status = f"저장 완료: {out}" if out else "저장 취소"
            continue

        if cmd == "c":
            clear_screen()
            print("[보유 카드 전체]\n")
            for i, c in enumerate(cards, start=1):
                print(f"{i}. {c.name} ({c.id})")
                print(f"   {describe_card(c, dummy_player)}")
                print(
                    f"   티어 강화 횟수: "
                    f"{enhancement_counts_line(c, [t.name for t in Tier], TIER_LABELS_BY_NAME)}"
                )
                print()
            input("엔터를 누르면 돌아갑니다...")
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
            nonlocal last_result, last_before, last_after, last_card_label, status, count

            if not cards:
                status = "보유 카드가 없습니다."
                return
            slot_count = n_choices
            if slot_count <= 0:
                status = "보유 카드가 없습니다."
                return

            # 1) 보기용 n개 뽑기(상태 변화 없음)
            picks = g.preview_choice_tiers(n_choices=slot_count)
            candidate_card_indices = random.choices(range(len(cards)), k=slot_count)
            choice_plans: List[Dict[str, Any]] = []
            for idx, cidx in zip(picks, candidate_card_indices):
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
                if preview_parts:
                    preview_roll_text = " [" + "] [".join(preview_parts) + "]"
                else:
                    preview_roll_text = ""

                choice_plans.append(
                    {
                        "idx": idx,
                        "card_idx": cidx,
                        "card": card,
                        "opt": planned_opt,
                        "higher_gap": higher_gap_for_choice,
                        "display": disp,
                        "efficiency_proc": efficiency_proc,
                        "efficiency_double_proc": efficiency_double_proc,
                        "lower_bonus_proc": lower_bonus_proc,
                        "same_tier_bonus_proc": same_tier_bonus_proc,
                        "preview_roll_text": preview_roll_text,
                    }
                )

            # 2) 선택
            if auto_pick_best:
                # 가장 높은 티어(숫자 큰 것) 자동 선택
                best_idx = max(picks)
                candidate_sel = [i for i, v in enumerate(picks) if v == best_idx]
                sel_idx = random.choice(candidate_sel)
                chosen_idx = choice_plans[sel_idx]["idx"]
                try:
                    chosen_plan = choice_plans[sel_idx]
                except Exception:
                    chosen_plan = None
            else:
                # 사용자에게 3개 보여주고 선택 받기
                print(f"\n[강화 선택지 {slot_count}개]")
                for i, plan in enumerate(choice_plans, start=1):
                    idx = plan["idx"]
                    c = plan["card"]
                    print(f"{i}) {c.id}: {describe_card(c, dummy_player)}")
                    print(
                        f"   {TIERS_NAME[Tier(idx)]} - "
                        f"{plan.get('display','')}{plan.get('preview_roll_text','')}"
                    )

                choices_str = "/".join(map(str, range(1, slot_count + 1)))
                sel = input(f"선택({choices_str}, 엔터=1): ").strip()
                if sel == "" or sel not in [str(i) for i in range(1, slot_count + 1)]:
                    sel = "1"
                sel_idx = int(sel) - 1
                chosen_plan = choice_plans[sel_idx]
                chosen_idx = chosen_plan["idx"]

            if chosen_plan is None:
                status = "선택된 강화가 없습니다."
                return

            card = chosen_plan["card"]
            last_card_label = f"{card.name} ({card.id})"

            # 3) 선택 확정(여기서만 pity/천장 상태 갱신)
            g.commit_choice(chosen_idx)
            tier_str = Tier(chosen_idx).name

            # 4) 카드에 강화 적용 (선택된 옵션을 전달하여 동일 옵션 적용)
            before = CardState.from_dict(card.to_dict())
            chosen_opt = chosen_plan["opt"] if chosen_plan is not None else None
            higher_gap = chosen_plan["higher_gap"] if chosen_plan is not None else (max(0, max(picks) - chosen_idx) if picks else 0)
            forced_efficiency_proc = chosen_plan["efficiency_proc"] if chosen_plan is not None else None
            forced_efficiency_double_proc = chosen_plan["efficiency_double_proc"] if chosen_plan is not None else None
            forced_lower_bonus_proc = chosen_plan["lower_bonus_proc"] if chosen_plan is not None else None
            forced_same_tier_bonus_proc = chosen_plan["same_tier_bonus_proc"] if chosen_plan is not None else None
            applied = engine.apply_tier(
                card,
                tier_str,
                rng=random,
                selected_option=chosen_opt,
                higher_tier_gap=higher_gap,
                forced_efficiency_proc=forced_efficiency_proc,
                forced_efficiency_double_proc=forced_efficiency_double_proc,
                forced_lower_bonus_proc=forced_lower_bonus_proc,
                forced_same_tier_bonus_proc=forced_same_tier_bonus_proc,
            )

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
    engine = EnhancementEngine.load("card_definitions.json")
    cards = load_cards("cards.json")
    g = EnhanceGacha()
    interactive_session(g, engine, cards, n_choices=3)
    # g.batch_apply_enhancements(engine, card, trials=100000, reset=True)
    # g.batch_apply_enhancements_choice_max(engine, card, trials=100000, reset=True, n_choices=3)
