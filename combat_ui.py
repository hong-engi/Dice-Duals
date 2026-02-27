from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

from cards import CardState, EnhancementEngine, Tier, TIERS_NAME, describe_card
from combat import (
    CombatConfig,
    CombatRuntime,
    _alive_enemies,
    _apply_armor_down,
    _apply_bleed,
    _apply_freeze,
    _build_deck_from_templates,
    _build_starting_card_templates,
    _chance_count,
    _damage_breakdown,
    _deal_damage,
    _draw_n,
    _tick_bleed,
)
from upgrade_main import EnhanceGacha, format_extra_effects
from unit import AttackProfile, DefenseProfile, Enemy, Player


BG = (18, 22, 31)
PANEL = (28, 34, 46)
BORDER = (70, 88, 115)
TEXT = (234, 238, 246)
MUTED = (164, 177, 204)
PLAYER_COLOR = (90, 170, 255)
ENEMY_COLOR = (255, 110, 110)
ENEMY_HL = (255, 220, 100)
CARD_ATTACK = (145, 76, 76)
CARD_DEFENSE = (72, 103, 150)
GREEN = (100, 220, 140)
RED = (235, 95, 95)
TEMP_BUFF = (132, 204, 255)
TIER_COLOR_BY_INDEX: Dict[int, Tuple[int, int, int]] = {
    0: (245, 245, 245),   # t0: white
    1: (255, 242, 170),   # t1: light yellow
    2: (184, 238, 170),   # t2: light green
    3: (112, 170, 255),   # t3: blue
    4: (186, 130, 255),   # t4: purple
    5: (255, 104, 104),   # t5: red
    6: (255, 104, 104),   # t6: red
}
UNIT_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "wolf": {"name": "늑대", "max_hp": 300.0, "attack": 30.0, "armor": 20.0},
    "archer": {"name": "궁수", "max_hp": 200.0, "attack": 60.0, "armor": 0.0},
    "mage": {"name": "마법사", "max_hp": 50.0, "attack": 150.0, "armor": 0.0},
}
ENEMY_UNIT_KEYS: Tuple[str, ...] = tuple(UNIT_ARCHETYPES.keys())
ENEMY_REMOVE_EFFECT_MS = 650.0


def _tier_color(tier_idx: int) -> Tuple[int, int, int]:
    return TIER_COLOR_BY_INDEX.get(int(tier_idx), TEXT)


def _card_tier_index(card: CardState) -> int:
    max_idx = 0
    for tier_name, cnt in (card.enhancement_counts or {}).items():
        if cnt and tier_name in Tier.__members__:
            max_idx = max(max_idx, int(Tier[tier_name].value))
    for rec in (card.enhancement_log or []):
        if not isinstance(rec, dict):
            continue
        tier_name = rec.get("tier") or rec.get("tier_used")
        if isinstance(tier_name, str) and tier_name in Tier.__members__:
            max_idx = max(max_idx, int(Tier[tier_name].value))
    return max_idx


@dataclass
class PendingAttack:
    card: CardState
    hit_count: int
    hit_index: int
    per_hit: float
    attack_range: int
    ignore_defense: bool
    splash_other_ratio: float


def _roll_center_weighted_count(min_count: int, max_count: int, center: int, rng: random.Random) -> int:
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
        w = float(max_dist + 2 * (max_dist - dist))
        weights.append(max(1e-9, w))
    return rng.choices(values, weights=weights, k=1)[0]


def _format_applied_option(result: Dict[str, Any]) -> str:
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


class CombatUI:
    def __init__(
        self,
        width: int = 1280,
        height: int = 800,
        seed: Optional[int] = None,
        encounter_enemy_count: Optional[int] = None,
        encounter_enemy_types: Optional[List[str]] = None,
        forced_start_enhance_rolls: Optional[int] = None,
        forced_initial_draw: Optional[float] = None,
        forced_min_enhance_tier: Optional[int] = None,
        forced_min_enhance_tier_rolls: Optional[int] = None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Dice Duals - Combat UI")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.font = self._load_font(20, preferred="katuri")
        self.small = self._load_font(16, preferred="katuri")
        self.tiny = self._load_font(14, preferred="katuri")
        self.enh_choice_big = self._load_font(18, preferred="katuri")
        self.enh_choice_mid = self._load_font(17, preferred="katuri")
        self.enh_heading_font = self._load_font(20, preferred="keriskedu_bold")
        self.enh_heading_small = self._load_font(16, preferred="keriskedu_bold")
        self.attack_card_image = self._load_image(os.path.join("images", "card_images", "card_attack.png"))
        self.defense_card_image = self._load_image(os.path.join("images", "card_images", "card_defense.png"))
        self.unit_image_raw: Dict[str, Optional[pygame.Surface]] = {
            "player": self._load_image(os.path.join("images", "units", "player.png")),
            "wolf": self._load_image(os.path.join("images", "units", "wolf.png")),
            "archer": self._load_image(os.path.join("images", "units", "archer.png")),
            "mage": self._load_image(os.path.join("images", "units", "mage.png")),
        }
        self.effect_image_raw: Dict[str, Optional[pygame.Surface]] = {
            "freeze_back": self._load_image(os.path.join("images", "effects", "freeze_back.png")),
            "freeze_front": self._load_image(os.path.join("images", "effects", "freeze_front.png")),
        }
        self.card_surface_cache: Dict[str, pygame.Surface] = {}
        self.card_scaled_cache: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self.unit_scaled_cache: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self.effect_scaled_cache: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self.tier_icon_cache: Dict[int, Optional[pygame.Surface]] = {}

        self.cfg = CombatConfig(seed=seed)
        self.encounter_enemy_count = (
            max(1, int(encounter_enemy_count))
            if encounter_enemy_count is not None
            else None
        )
        self.encounter_enemy_types = [
            k for k in (encounter_enemy_types or []) if k in UNIT_ARCHETYPES
        ] or None
        self.forced_start_enhance_rolls = (
            max(0, int(forced_start_enhance_rolls))
            if forced_start_enhance_rolls is not None
            else None
        )
        self.forced_initial_draw = (
            max(0.0, float(forced_initial_draw))
            if forced_initial_draw is not None
            else None
        )
        if self.forced_initial_draw is not None:
            self.cfg.initial_draw = self.forced_initial_draw
        self.forced_min_enhance_tier = (
            max(0, min(int(forced_min_enhance_tier), len(Tier) - 1))
            if forced_min_enhance_tier is not None
            else None
        )
        self.forced_min_enhance_tier_rolls = (
            max(0, int(forced_min_enhance_tier_rolls))
            if forced_min_enhance_tier_rolls is not None
            else 0
        )
        self.rng = random.Random(seed)

        self.running = True
        self.logs: List[str] = []
        self.mode = "enhance"

        self.player: Player
        self.enemy_rows: List[List[Enemy]]
        self.rt: CombatRuntime

        self.turn = 1
        self.actions_left = 0
        self.phase = "player"
        self.game_over = False
        self.win = False
        self.selected_card_index: Optional[int] = None
        self.selected_unit_key: Optional[Tuple[str, str]] = None
        self.pending_attack: Optional[PendingAttack] = None
        self.end_button_rect = pygame.Rect(self.width - 190, 18, 160, 48)

        self.enh_engine: Optional[EnhancementEngine] = None
        self.enh_templates: List[CardState] = []
        self.enh_gacha: Optional[EnhanceGacha] = None
        self.enh_total_rolls = 0
        self.enh_roll_index = 1
        self.enh_choices: List[Dict[str, Any]] = []
        self.enh_auto = False
        self.enh_show_pity = False
        self.enh_logs: List[str] = []
        self.enh_log_expanded = False
        self.enh_log_scroll = 0
        self.enh_log_lines_per_page = 1
        self.battle_log_expanded = False
        self.battle_log_scroll = 0
        self.battle_log_lines_per_page = 1
        self.card_hover_anim: List[float] = []
        self.card_use_anims: List[Dict[str, Any]] = []
        self.enemy_remove_effects: List[Dict[str, Any]] = []
        self._enh_rng_backup: Optional[object] = None

        self.new_game()

    def _load_font(self, size: int, preferred: str = "katuri") -> pygame.font.Font:
        katuri_path = os.path.join("fonts", "Katuri.ttf")
        regular_path = os.path.join("fonts", "KERISKEDU_R.ttf")
        bold_path = os.path.join("fonts", "KERISKEDU_B.ttf")
        if preferred == "keriskedu_bold":
            if os.path.exists(bold_path):
                return pygame.font.Font(bold_path, size)
            if os.path.exists(regular_path):
                return pygame.font.Font(regular_path, size)
            if os.path.exists(katuri_path):
                return pygame.font.Font(katuri_path, size)
            return pygame.font.SysFont("malgungothic", size)
        if os.path.exists(katuri_path):
            return pygame.font.Font(katuri_path, size)
        if os.path.exists(regular_path):
            return pygame.font.Font(regular_path, size)
        if os.path.exists(bold_path):
            return pygame.font.Font(bold_path, size)
        return pygame.font.SysFont("malgungothic", size)

    def _load_image(self, path: str) -> Optional[pygame.Surface]:
        if not os.path.exists(path):
            return None
        try:
            return pygame.image.load(path).convert_alpha()
        except Exception:
            return None

    def _scaled_unit_image(self, unit_key: str, size: Tuple[int, int]) -> Optional[pygame.Surface]:
        raw = self.unit_image_raw.get(unit_key)
        if raw is None:
            return None
        w = max(1, int(size[0]))
        h = max(1, int(size[1]))
        cache_key = (unit_key, w, h)
        cached = self.unit_scaled_cache.get(cache_key)
        if cached is not None:
            return cached
        scaled = pygame.transform.smoothscale(raw, (w, h))
        self.unit_scaled_cache[cache_key] = scaled
        return scaled

    def _scaled_effect_image(self, effect_key: str, size: Tuple[int, int]) -> Optional[pygame.Surface]:
        raw = self.effect_image_raw.get(effect_key)
        if raw is None:
            return None
        w = max(1, int(size[0]))
        h = max(1, int(size[1]))
        cache_key = (effect_key, w, h)
        cached = self.effect_scaled_cache.get(cache_key)
        if cached is not None:
            return cached
        scaled = pygame.transform.smoothscale(raw, (w, h)).convert_alpha()
        # 현재 freeze 에셋은 흰 배경이 포함되어 있어 컬러키로 제거한다.
        scaled.set_colorkey((255, 255, 255))
        self.effect_scaled_cache[cache_key] = scaled
        return scaled

    def _draw_freeze_effect_back(self, rect: pygame.Rect) -> None:
        back_w = max(20, int(rect.w * 2.10))
        back_h = max(20, int(rect.h * 1.45))
        back = self._scaled_effect_image("freeze_back", (back_w, back_h))
        if back is None:
            return
        back_draw = back.copy()
        back_draw.set_alpha(210)
        back_rect = back_draw.get_rect(
            midbottom=(rect.centerx, rect.bottom + int(rect.h * 0.18))
        )
        self.screen.blit(back_draw, back_rect.topleft)

    def _draw_freeze_effect_front(self, rect: pygame.Rect) -> None:
        front_w = max(20, int(rect.w * 1.60))
        front_h = max(20, int(rect.h * 1.55))
        front = self._scaled_effect_image("freeze_front", (front_w, front_h))
        if front is None:
            pygame.draw.circle(
                self.screen,
                (98, 188, 255),
                rect.center,
                max(12, int(max(rect.w, rect.h) * 0.52)),
                width=2,
            )
            return
        front_draw = front.copy()
        front_draw.set_alpha(230)
        front_rect = front_draw.get_rect(
            midbottom=(rect.centerx, rect.bottom + int(rect.h * 0.14))
        )
        self.screen.blit(front_draw, front_rect.topleft)

    @staticmethod
    def _enemy_kind(enemy: Enemy) -> str:
        prefix = enemy.unit_id.split("_", 1)[0].lower()
        return prefix if prefix in UNIT_ARCHETYPES else "wolf"

    def new_game(self) -> None:
        if self._enh_rng_backup is not None:
            random.setstate(self._enh_rng_backup)
            self._enh_rng_backup = None

        self.card_surface_cache.clear()
        self.card_scaled_cache.clear()
        self.unit_scaled_cache.clear()
        self.effect_scaled_cache.clear()
        self.rt = CombatRuntime(deck=[])

        self.player = Player(
            unit_id="player",
            name="플레이어",
            max_hp=500.0,
            hp=500.0,
            cards=[],
            attack=AttackProfile(power=50.0),
            defense=DefenseProfile(armor=20.0, defense_power=50.0),
        )
        self.enemy_rows = self._build_enemy_rows_unique(
            enemy_count=self.encounter_enemy_count,
            enemy_types=self.encounter_enemy_types,
        )

        self.logs.clear()
        self.battle_log_expanded = False
        self.battle_log_scroll = 0
        self.battle_log_lines_per_page = 1
        self.card_hover_anim = []
        self.card_use_anims = []
        self.enemy_remove_effects = []
        self.selected_card_index = None
        self.selected_unit_key = None
        self.pending_attack = None
        self.phase = "player"
        self.game_over = False
        self.win = False
        self.turn = 1
        self.actions_left = self.cfg.cards_per_turn

        self.enh_engine = EnhancementEngine.load("card_definitions.json")
        self.enh_templates = _build_starting_card_templates(self.enh_engine)
        self._begin_enhancement_phase()

    def _build_enemy_rows_unique(
        self,
        enemy_count: Optional[int] = None,
        enemy_types: Optional[List[str]] = None,
    ) -> List[List[Enemy]]:
        # 지정 타입이 없으면 wolf/archer/mage를 랜덤 소환한다.
        planned_types = [k for k in (enemy_types or []) if k in UNIT_ARCHETYPES]
        default_count = 5
        target_count = (
            max(1, int(enemy_count))
            if enemy_count is not None
            else (len(planned_types) if planned_types else default_count)
        )
        if len(planned_types) < target_count:
            while len(planned_types) < target_count:
                planned_types.append(self.rng.choice(ENEMY_UNIT_KEYS))
        else:
            planned_types = planned_types[:target_count]

        all_enemies: List[Enemy] = []
        for idx, unit_key in enumerate(planned_types, start=1):
            bp = UNIT_ARCHETYPES[unit_key]
            all_enemies.append(
                Enemy(
                    unit_id=f"{unit_key}_{idx}",
                    name=f"{bp['name']}",
                    max_hp=float(bp["max_hp"]),
                    hp=float(bp["max_hp"]),
                    attack=AttackProfile(power=float(bp["attack"])),
                    defense=DefenseProfile(armor=float(bp["armor"]), defense_power=0.0),
                )
            )

        row_pattern = [max(1, int(x)) for x in (self.cfg.enemy_row_counts or [target_count])]
        rows: List[List[Enemy]] = []
        cursor = 0
        row_idx = 0
        while cursor < len(all_enemies):
            cap = row_pattern[row_idx % len(row_pattern)]
            rows.append(all_enemies[cursor:cursor + cap])
            cursor += cap
            row_idx += 1
        return rows

    def _begin_enhancement_phase(self) -> None:
        self.mode = "enhance"
        self.enh_gacha = EnhanceGacha()
        self.enh_auto = False
        self.enh_show_pity = False
        self.enh_logs = []
        self.enh_log_expanded = False
        self.enh_log_scroll = 0
        self.enh_log_lines_per_page = 1
        self.enh_roll_index = 1
        if self.forced_start_enhance_rolls is not None:
            self.enh_total_rolls = self.forced_start_enhance_rolls
        else:
            self.enh_total_rolls = _roll_center_weighted_count(
                self.cfg.min_start_enhance,
                self.cfg.max_start_enhance,
                self.cfg.center_start_enhance,
                self.rng,
            )

        self._enh_rng_backup = random.getstate()
        random.seed(self.rng.randrange(1 << 30))

        if self.forced_start_enhance_rolls is not None:
            self._enh_log(f"강화 시작: {self.enh_total_rolls}회 (고정)")
        else:
            self._enh_log(
                f"강화 시작: {self.enh_total_rolls}회 "
                f"(범위 {self.cfg.min_start_enhance}~{self.cfg.max_start_enhance}, 중심 {self.cfg.center_start_enhance})"
            )
        if (
            self.forced_min_enhance_tier is not None
            and self.forced_min_enhance_tier > 0
            and self.forced_min_enhance_tier_rolls > 0
        ):
            self._enh_log(
                f"티어 하한 적용: {TIERS_NAME[Tier(self.forced_min_enhance_tier)]} 이상 "
                f"({self.forced_min_enhance_tier_rolls}회)"
            )
        self._enh_log("선택지 카드를 클릭하세요. A: 남은 횟수 자동, T: 천장 보기")
        if self.enh_total_rolls <= 0:
            self._end_enhancement_phase_and_start_battle()
            return
        self._build_enhancement_choices()

    def _end_enhancement_phase_and_start_battle(self) -> None:
        if self._enh_rng_backup is not None:
            random.setstate(self._enh_rng_backup)
            self._enh_rng_backup = None

        deck = _build_deck_from_templates(self.enh_templates, copies_each=5, rng=self.rng)
        self.rt = CombatRuntime(deck=deck)
        _draw_n(self.rt, self.cfg, self.rng, self.cfg.initial_draw)

        self.mode = "battle"
        self.turn = 1
        self.actions_left = self.cfg.cards_per_turn
        self.phase = "player"
        self.game_over = False
        self.win = False
        self.pending_attack = None
        self.logs.clear()
        self.log("전투 시작")
        self.log("카드를 클릭해서 사용하세요. 스페이스: 턴 종료, R: 재시작")

    def _enh_log(self, text: str) -> None:
        self.enh_logs.append(text)
        if self.enh_log_expanded:
            self._clamp_enh_log_scroll()

    def _enh_forced_tier_text(self) -> str:
        forced_idx = self._enh_forced_tier_index()
        if forced_idx is None:
            return ""
        return f" - {TIERS_NAME[Tier(forced_idx)]} 확정"

    def _active_forced_min_tier(self) -> Optional[int]:
        if self.forced_min_enhance_tier is None:
            return None
        if self.forced_min_enhance_tier_rolls <= 0:
            return None
        if self.enh_roll_index > self.forced_min_enhance_tier_rolls:
            return None
        return int(self.forced_min_enhance_tier)

    def _enh_forced_tier_index(self) -> Optional[int]:
        forced_idxs: List[int] = []
        if self.enh_gacha is not None:
            gacha_forced = self.enh_gacha._forced_min_tier()
            if gacha_forced >= 0:
                forced_idxs.append(int(gacha_forced))
        active_floor = self._active_forced_min_tier()
        if active_floor is not None:
            forced_idxs.append(active_floor)
        if not forced_idxs:
            return None
        return max(forced_idxs)

    def _enh_choice_signature(self, plan: Dict[str, Any]) -> Tuple[Any, ...]:
        opt = plan.get("opt")
        if isinstance(opt, dict):
            opt_key = (
                str(opt.get("id", "") or ""),
                str(opt.get("display", "") or ""),
                bool(opt.get("max_range", False)),
            )
        else:
            opt_key = ("", "", False)
        return (
            int(plan.get("card_index", -1)),
            int(plan.get("idx", 0)),
            opt_key,
            bool(plan.get("efficiency_proc", False)),
            bool(plan.get("efficiency_double_proc", False)),
            bool(plan.get("lower_bonus_proc", False)),
            bool(plan.get("same_tier_bonus_proc", False)),
        )

    def _build_enhancement_choices(self) -> None:
        if self.enh_gacha is None or self.enh_engine is None or not self.enh_templates:
            return

        slot_count = max(1, int(self.cfg.start_enhance_choices))
        picks = self.enh_gacha.preview_choice_tiers(n_choices=slot_count)
        floor_tier = self._active_forced_min_tier()
        if floor_tier is not None:
            picks = [max(floor_tier, int(idx)) for idx in picks]
        self.enh_choices = []
        used_sigs: set[Tuple[Any, ...]] = set()

        for idx in picks:
            selected_plan: Optional[Dict[str, Any]] = None
            selected_sig: Optional[Tuple[Any, ...]] = None
            for _ in range(24):
                card_index = random.randrange(len(self.enh_templates))
                card = self.enh_templates[card_index]
                planned_opt = self.enh_engine.preview_option(card, Tier(idx).name, rng=random)
                if planned_opt is None:
                    display = "(옵션 없음)"
                    planned_effects: List[Dict[str, Any]] = []
                else:
                    display = str(planned_opt.get("display", planned_opt.get("id", "")))
                    planned_effects = list(planned_opt.get("effects", []))

                higher_gap_for_choice = max(0, max(picks) - idx) if picks else 0
                eff_rate, lower_rate, eff2_rate, same_tier_rate = self.enh_engine.get_proc_rates(
                    Tier(idx).name,
                    higher_tier_gap=higher_gap_for_choice,
                )
                eff_eligible = bool(planned_opt) and any(
                    self.enh_engine._is_efficiency_eligible(eff) for eff in planned_effects
                )

                efficiency_proc = eff_eligible and (random.random() < eff_rate)
                efficiency_double_proc = eff_eligible and (random.random() < eff2_rate)
                lower_bonus_proc = idx > 0 and (random.random() < lower_rate)
                same_tier_bonus_proc = random.random() < same_tier_rate
                if efficiency_double_proc:
                    efficiency_proc = False
                if same_tier_bonus_proc:
                    lower_bonus_proc = False

                bonus_tags: List[str] = []
                if efficiency_proc:
                    bonus_tags.append("x1.5")
                if efficiency_double_proc:
                    bonus_tags.append("x2")
                if lower_bonus_proc:
                    bonus_tags.append("하위 티어")
                if same_tier_bonus_proc:
                    bonus_tags.append("동일 티어")

                plan = {
                    "idx": idx,
                    "card": card,
                    "card_index": card_index,
                    "opt": planned_opt,
                    "higher_gap": higher_gap_for_choice,
                    "display": display,
                    "efficiency_proc": efficiency_proc,
                    "efficiency_double_proc": efficiency_double_proc,
                    "lower_bonus_proc": lower_bonus_proc,
                    "same_tier_bonus_proc": same_tier_bonus_proc,
                    "bonus_text": " / ".join(bonus_tags),
                }
                sig = self._enh_choice_signature(plan)
                selected_plan = plan
                selected_sig = sig
                if sig not in used_sigs:
                    break

            if selected_plan is not None and selected_sig is not None:
                used_sigs.add(selected_sig)
                self.enh_choices.append(selected_plan)

    def _best_enhance_choice_index(self) -> int:
        if not self.enh_choices:
            return 0
        best_tier = max(int(p.get("idx", 0)) for p in self.enh_choices)
        best_indices = [i for i, p in enumerate(self.enh_choices) if int(p.get("idx", 0)) == best_tier]
        return self.rng.choice(best_indices)

    def _commit_enhance_choice(self, choice_index: int) -> None:
        if self.enh_gacha is None or self.enh_engine is None:
            return
        if not (0 <= choice_index < len(self.enh_choices)):
            return

        plan = self.enh_choices[choice_index]
        tier_idx = int(plan["idx"])
        chosen_card = plan["card"]

        self.enh_gacha.commit_choice(tier_idx)
        tier_str = Tier(tier_idx).name
        applied = self.enh_engine.apply_tier(
            chosen_card,
            tier_str,
            rng=random,
            selected_option=plan.get("opt"),
            higher_tier_gap=int(plan.get("higher_gap", 0)),
            forced_efficiency_proc=bool(plan.get("efficiency_proc", False)),
            forced_efficiency_double_proc=bool(plan.get("efficiency_double_proc", False)),
            forced_lower_bonus_proc=bool(plan.get("lower_bonus_proc", False)),
            forced_same_tier_bonus_proc=bool(plan.get("same_tier_bonus_proc", False)),
        )
        rolled_tier = str(applied.get("rolled_tier", tier_str))
        used_tier = str(applied.get("tier_used", tier_str))
        if rolled_tier == used_tier:
            self._enh_log(f"적용: {chosen_card.name} / {TIERS_NAME[Tier[used_tier]]}")
        else:
            self._enh_log(
                f"적용: {chosen_card.name} / {TIERS_NAME[Tier[rolled_tier]]} -> {TIERS_NAME[Tier[used_tier]]}"
            )

        extra = format_extra_effects(applied)
        if extra:
            self._enh_log(f"보너스: {extra}")
        self._enh_log(f"옵션: {_format_applied_option(applied)}")

        self.enh_roll_index += 1
        if self.enh_roll_index > self.enh_total_rolls:
            self._end_enhancement_phase_and_start_battle()
            return
        self._build_enhancement_choices()

    def _run_auto_enhance(self) -> None:
        while self.mode == "enhance" and self.enh_auto and self.enh_roll_index <= self.enh_total_rolls:
            idx = self._best_enhance_choice_index()
            self._commit_enhance_choice(idx)
            if self.mode != "enhance":
                break

    def _enh_log_collapsed_rect(self) -> pygame.Rect:
        return pygame.Rect(18, self.height - 46, 118, 30)

    def _enh_log_expanded_rect(self) -> pygame.Rect:
        return pygame.Rect(36, 120, self.width - 72, self.height - 170)

    def _clamp_enh_log_scroll(self) -> None:
        max_start = max(0, len(self.enh_logs) - max(1, self.enh_log_lines_per_page))
        self.enh_log_scroll = max(0, min(self.enh_log_scroll, max_start))

    def _open_enh_log_panel(self) -> None:
        self.enh_log_expanded = True
        self._clamp_enh_log_scroll()
        self.enh_log_scroll = max(0, len(self.enh_logs) - max(1, self.enh_log_lines_per_page))

    def _toggle_enh_log_panel(self) -> None:
        if self.enh_log_expanded:
            self.enh_log_expanded = False
            return
        self._open_enh_log_panel()

    def _scroll_enh_log(self, delta: int) -> None:
        if not self.enh_log_expanded:
            return
        self.enh_log_scroll -= int(delta)
        self._clamp_enh_log_scroll()

    def log(self, text: str) -> None:
        self.logs.append(text)
        if self.battle_log_expanded:
            self._clamp_battle_log_scroll()

    def _battle_log_button_rect(self) -> pygame.Rect:
        return pygame.Rect(18, 96, 110, 34)

    def _battle_log_expanded_rect(self) -> pygame.Rect:
        return pygame.Rect(36, 120, self.width - 72, self.height - 170)

    def _clamp_battle_log_scroll(self) -> None:
        max_start = max(0, len(self.logs) - max(1, self.battle_log_lines_per_page))
        self.battle_log_scroll = max(0, min(self.battle_log_scroll, max_start))

    def _open_battle_log_panel(self) -> None:
        self.battle_log_expanded = True
        self._clamp_battle_log_scroll()
        self.battle_log_scroll = max(0, len(self.logs) - max(1, self.battle_log_lines_per_page))

    def _scroll_battle_log(self, delta: int) -> None:
        if not self.battle_log_expanded:
            return
        self.battle_log_scroll -= int(delta)
        self._clamp_battle_log_scroll()

    def run(
        self,
        quit_on_exit: bool = True,
        stop_on_game_over: bool = False,
        stop_on_enhance_complete: bool = False,
    ) -> Optional[bool]:
        encounter_result: Optional[bool] = None
        while self.running:
            self._handle_events()
            self._render()
            pygame.display.flip()
            self.clock.tick(60)
            if stop_on_enhance_complete and self.mode == "battle":
                self.running = False
                break
            if stop_on_game_over and self.mode == "battle" and self.game_over:
                encounter_result = bool(self.win)
                self.running = False
                break
        if quit_on_exit:
            pygame.quit()
        return encounter_result

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.mode == "enhance":
                        if self.enh_show_pity:
                            self.enh_show_pity = False
                            return
                        if self.enh_log_expanded:
                            self.enh_log_expanded = False
                        return
                    if self.mode == "battle":
                        if self.battle_log_expanded:
                            self.battle_log_expanded = False
                            return
                        if self.selected_card_index is not None:
                            self.selected_card_index = None
                            self.log("카드 선택 취소")
                            return
                        if self.selected_unit_key is not None:
                            self.selected_unit_key = None
                            self.log("유닛 선택 취소")
                        return
                    return
                if event.key == pygame.K_r:
                    self.new_game()
                    return
                if self.mode == "enhance":
                    key_to_index = {
                        pygame.K_1: 0,
                        pygame.K_KP1: 0,
                        pygame.K_2: 1,
                        pygame.K_KP2: 1,
                        pygame.K_3: 2,
                        pygame.K_KP3: 2,
                    }
                    if event.key in key_to_index:
                        idx = key_to_index[event.key]
                        if 0 <= idx < len(self.enh_choices):
                            self._commit_enhance_choice(idx)
                            if self.mode == "enhance" and self.enh_auto:
                                self._run_auto_enhance()
                        return
                    if event.key == pygame.K_a:
                        self.enh_auto = True
                        self._enh_log("자동 강화 시작: 남은 횟수를 최고 티어 우선으로 진행")
                        self._run_auto_enhance()
                        return
                    if event.key == pygame.K_t:
                        self.enh_show_pity = not self.enh_show_pity
                        return
                    if event.key == pygame.K_PAGEUP:
                        self._scroll_enh_log(+6)
                        return
                    if event.key == pygame.K_PAGEDOWN:
                        self._scroll_enh_log(-6)
                        return
                    if event.key == pygame.K_HOME:
                        if self.enh_log_expanded:
                            self.enh_log_scroll = 0
                            self._clamp_enh_log_scroll()
                        return
                    if event.key == pygame.K_END:
                        if self.enh_log_expanded:
                            self._clamp_enh_log_scroll()
                            self.enh_log_scroll = max(0, len(self.enh_logs) - max(1, self.enh_log_lines_per_page))
                        return
                if self.mode == "battle":
                    hot_idx = self._battle_card_hotkey_index(event.key)
                    if hot_idx is not None:
                        if self._handle_battle_card_hotkey(hot_idx):
                            return
                    if event.key == pygame.K_PAGEUP:
                        self._scroll_battle_log(+6)
                        return
                    if event.key == pygame.K_PAGEDOWN:
                        self._scroll_battle_log(-6)
                        return
                    if event.key == pygame.K_HOME:
                        if self.battle_log_expanded:
                            self.battle_log_scroll = 0
                            self._clamp_battle_log_scroll()
                        return
                    if event.key == pygame.K_END:
                        if self.battle_log_expanded:
                            self._clamp_battle_log_scroll()
                            self.battle_log_scroll = max(0, len(self.logs) - max(1, self.battle_log_lines_per_page))
                        return
                if event.key == pygame.K_SPACE:
                    if self.mode == "battle" and self.phase == "player" and self.pending_attack is None and not self.game_over:
                        self._end_player_turn()
                    return
            if event.type == pygame.MOUSEWHEEL:
                if self.mode == "enhance" and self.enh_log_expanded:
                    self._scroll_enh_log(event.y)
                if self.mode == "battle" and self.battle_log_expanded:
                    self._scroll_battle_log(event.y)
                continue
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.mode == "enhance":
                    self._handle_enhance_click(event.pos)
                    continue
                self._handle_left_click(event.pos)

    @staticmethod
    def _battle_card_hotkey_index(key: int) -> Optional[int]:
        key_map = {
            pygame.K_1: 0,
            pygame.K_KP1: 0,
            pygame.K_2: 1,
            pygame.K_KP2: 1,
            pygame.K_3: 2,
            pygame.K_KP3: 2,
            pygame.K_4: 3,
            pygame.K_KP4: 3,
            pygame.K_5: 4,
            pygame.K_KP5: 4,
            pygame.K_6: 5,
            pygame.K_KP6: 5,
            pygame.K_7: 6,
            pygame.K_KP7: 6,
            pygame.K_8: 7,
            pygame.K_KP8: 7,
            pygame.K_9: 8,
            pygame.K_KP9: 8,
            pygame.K_0: 9,
            pygame.K_KP0: 9,
        }
        return key_map.get(key)

    def _handle_battle_card_hotkey(self, hot_idx: int) -> bool:
        if self.mode != "battle" or self.game_over or self.phase != "player":
            return False
        if self.pending_attack is not None:
            return False
        if self.actions_left <= 0:
            return True
        if not (0 <= hot_idx < len(self.rt.hand)):
            return False

        if self.selected_card_index == hot_idx:
            card = self.rt.hand[hot_idx]
            if card.type == "attack":
                self.log("공격 카드는 사거리 내 적 클릭으로 사용")
                return True
            self._play_card(hot_idx)
            return True

        self.selected_card_index = hot_idx
        self.log(f"카드 선택: {self.rt.hand[hot_idx].name} (숫자키 재입력 사용 / ESC,X 취소)")
        return True

    def _handle_enhance_click(self, pos: Tuple[int, int]) -> None:
        collapsed = self._enh_log_collapsed_rect()
        expanded = self._enh_log_expanded_rect()
        close_rect = pygame.Rect(expanded.right - 92, expanded.y + 10, 76, 28)

        if self.enh_log_expanded:
            if close_rect.collidepoint(pos):
                self.enh_log_expanded = False
                return
            if expanded.collidepoint(pos):
                return
            self.enh_log_expanded = False
            return

        if collapsed.collidepoint(pos):
            self._toggle_enh_log_panel()
            return

        rects = self._enhance_choice_rects()
        for i, rect in enumerate(rects):
            if rect.collidepoint(pos):
                self._commit_enhance_choice(i)
                if self.mode == "enhance" and self.enh_auto:
                    self._run_auto_enhance()
                return

    def _handle_left_click(self, pos: Tuple[int, int]) -> None:
        if self.game_over:
            return
        if self.mode != "battle":
            return
        if self.phase != "player":
            return

        if self.pending_attack is not None:
            if not self._try_target_click(pos):
                self._cancel_pending_attack()
            return

        selected_attack_idx: Optional[int] = None
        if (
            self.selected_card_index is not None
            and 0 <= self.selected_card_index < len(self.rt.hand)
            and self.rt.hand[self.selected_card_index].type == "attack"
        ):
            selected_attack_idx = self.selected_card_index

        unit_hit = self._unit_at_pos(pos)
        if unit_hit is not None:
            kind, unit = unit_hit
            if kind == "enemy" and selected_attack_idx is not None:
                card = self.rt.hand[selected_attack_idx]
                attack_range = int(card.stats.get("range", 1) or 1)
                if bool(card.effects.get("max_range", False)):
                    attack_range = max(attack_range, 99)
                in_range = set(self._in_range_enemies(attack_range))
                if unit in in_range:
                    # 공격 대상 클릭은 유닛 선택과 분리한다.
                    self.selected_unit_key = None
                    self._play_card(selected_attack_idx)
                    if self.pending_attack is not None and not self._resolve_pending_attack_target(unit):
                        self._cancel_pending_attack()
                    return
                # 공격 카드가 선택된 상태에서 다른 유닛을 터치하면 카드 선택을 해제한다.
                self.selected_card_index = None
                self.log("카드 선택 취소")
            elif self.selected_card_index is not None:
                self.selected_card_index = None
                self.log("카드 선택 취소")

            key = (kind, unit.unit_id)
            if self.selected_unit_key == key:
                self.selected_unit_key = None
            else:
                self.selected_unit_key = key
            return
        else:
            # 유닛 외 영역 클릭 시 선택 해제
            self.selected_unit_key = None

        log_button = self._battle_log_button_rect()
        log_panel = self._battle_log_expanded_rect()
        close_log_rect = pygame.Rect(log_panel.right - 92, log_panel.y + 10, 76, 28)
        if self.battle_log_expanded:
            if close_log_rect.collidepoint(pos):
                self.battle_log_expanded = False
                return
            if log_panel.collidepoint(pos):
                return
            self.battle_log_expanded = False
            return
        if log_button.collidepoint(pos):
            if self.selected_card_index is not None:
                self.selected_card_index = None
            self._open_battle_log_panel()
            return

        if self.end_button_rect.collidepoint(pos):
            if self.selected_card_index is not None:
                self.selected_card_index = None
            self._end_player_turn()
            return

        if self.actions_left <= 0:
            return

        card_rects = self._display_card_rects()
        cancel_rect = self._selected_card_cancel_rect(card_rects)
        if cancel_rect is not None and cancel_rect.collidepoint(pos):
            self.selected_card_index = None
            self.log("카드 선택 취소")
            return

        clicked_card = False
        for i, rect in enumerate(card_rects):
            if rect.collidepoint(pos):
                clicked_card = True
                card = self.rt.hand[i]
                if self.selected_card_index == i:
                    if card.type == "attack":
                        self.selected_card_index = None
                        self.log("카드 선택 취소")
                    else:
                        self._play_card(i)
                else:
                    self.selected_card_index = i
                    self.log(f"카드 선택: {self.rt.hand[i].name} (재클릭 사용 / ESC,X 취소)")
                return

        if not clicked_card and self.selected_card_index is not None:
            self.selected_card_index = None
            self.log("카드 선택 취소")

    def _card_rects(self) -> List[pygame.Rect]:
        n = len(self.rt.hand)
        if n <= 0:
            return []
        desired_w = 420
        desired_h = 240
        gap = 18
        margin = 20
        max_fit_w = int((self.width - (margin * 2) - (n - 1) * gap) / n)
        card_w = max(140, min(desired_w, max_fit_w))
        card_h = int(card_w * (desired_h / desired_w))
        total_w = n * card_w + (n - 1) * gap
        x0 = max(margin, (self.width - total_w) // 2)
        y = self.height - card_h - 18
        return [pygame.Rect(x0 + i * (card_w + gap), y, card_w, card_h) for i in range(n)]

    def _display_card_rects(self) -> List[pygame.Rect]:
        rects = [r.copy() for r in self._card_rects()]
        idx = self.selected_card_index
        if idx is not None and 0 <= idx < len(rects):
            rects[idx].move_ip(0, -28)
        return rects

    def _selected_card_cancel_rect(self, rects: Optional[List[pygame.Rect]] = None) -> Optional[pygame.Rect]:
        idx = self.selected_card_index
        if idx is None:
            return None
        rs = rects if rects is not None else self._display_card_rects()
        if not (0 <= idx < len(rs)):
            return None
        r = rs[idx]
        return pygame.Rect(r.right - 30, r.y + 6, 24, 24)

    def _hovered_card_index(self) -> Optional[int]:
        if self.mode != "battle":
            return None
        mx, my = pygame.mouse.get_pos()
        for i, rect in enumerate(self._display_card_rects()):
            if rect.collidepoint((mx, my)):
                return i
        return None

    def _battlefield_bounds(self) -> Tuple[pygame.Rect, pygame.Rect, int]:
        card_rects = self._card_rects()
        cards_top = card_rects[0].top if card_rects else (self.height - 220)
        field_top = 96
        field_bottom = max(field_top + 220, cards_top - 24)
        field_h = max(220, field_bottom - field_top)
        half_gap = 16
        left = pygame.Rect(24, field_top, max(240, self.width // 2 - 24 - half_gap), field_h)
        right = pygame.Rect(self.width // 2 + half_gap, field_top, max(240, self.width - (self.width // 2 + half_gap) - 24), field_h)
        y_center = field_top + field_h // 2
        return left, right, y_center

    def _player_shape(self) -> Tuple[Tuple[int, int], int]:
        left_zone, _, y_center = self._battlefield_bounds()
        return (left_zone.centerx, y_center), 54

    def _enemy_distance_map(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        d = 1
        for row in self.enemy_rows:
            alive_row = [e for e in row if e.is_alive]
            if not alive_row:
                continue
            for e in alive_row:
                dist[e.unit_id] = d
            d += 1
        return dist

    def _unit_at_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[str, Any]]:
        for enemy, rect, _ in self._enemy_slots():
            if rect.collidepoint(pos):
                return ("enemy", enemy)
        (cx, cy), r = self._player_shape()
        dx = pos[0] - cx
        dy = pos[1] - cy
        if (dx * dx + dy * dy) <= (r * r):
            return ("player", self.player)
        return None

    def _get_selected_unit(self) -> Optional[Tuple[str, Any]]:
        if self.selected_unit_key is None:
            return None
        k, uid = self.selected_unit_key
        if k == "player":
            if self.player.is_alive:
                return ("player", self.player)
            return None
        if k == "enemy":
            for row in self.enemy_rows:
                for e in row:
                    if e.unit_id == uid and e.is_alive:
                        return ("enemy", e)
        return None

    def _unit_info_lines(self, kind: str, unit: Any) -> List[str]:
        lines: List[str] = []
        lines.append(unit.name)
        lines.append(f"HP {unit.hp:.1f}/{unit.max_hp:.1f}")
        if unit.attack.power > 0:
            lines.append(f"ATK {unit.attack.power:.1f}")
        if unit.defense.defense_power > 0:
            lines.append(f"방어력 {unit.defense.defense_power:.1f}")
        if kind == "player":
            base_armor, bonus_armor = self._player_armor_parts()
            if base_armor > 0 or bonus_armor > 0:
                if bonus_armor > 0:
                    lines.append(f"갑옷 {base_armor:.1f} (+{bonus_armor:.1f})")
                else:
                    lines.append(f"갑옷 {base_armor:.1f}")
        elif unit.defense.armor > 0:
            lines.append(f"갑옷 {unit.defense.armor:.1f}")
        if unit.state.shield > 0:
            lines.append(f"보호막 {unit.state.shield:.1f}")
        if kind == "enemy":
            dist = self._enemy_distance_map().get(unit.unit_id)
            if dist is not None:
                lines.append(f"거리 {dist}")
        if kind == "player":
            lines.append(f"자원 {self.rt.resource}")

        bleed_turns = int(max(0, unit.state.tags.get("bleed_turns", 0)))
        bleed_dmg = float(max(0.0, unit.state.tags.get("bleed_damage", 0.0)))
        if bleed_turns > 0 and bleed_dmg > 0:
            lines.append(f"출혈 {bleed_dmg:.1f}/턴 ({bleed_turns}턴)")
        freeze_turns = int(max(0, unit.state.frozen_turns))
        if freeze_turns > 0:
            lines.append(f"빙결 {freeze_turns}")
        return lines

    def _draw_unit_tooltip(self) -> None:
        if self.mode != "battle":
            return
        if self.battle_log_expanded:
            return

        hovered = self._unit_at_pos(pygame.mouse.get_pos())
        selected = self._get_selected_unit()
        selected_panel = pygame.Rect(0, 0, 0, 0)
        if selected is not None:
            sk, su = selected
            selected_panel = self._draw_unit_info_panel(sk, su)

        if hovered is not None:
            hk, hu = hovered
            # 같은 유닛이면 중복 패널을 그리지 않는다.
            if selected is None or not (selected[0] == hk and selected[1] is hu):
                lines = self._unit_info_lines(hk, hu)
                w = 250
                line_h = self.tiny.get_linesize() + 2
                h = 18 + 14 + line_h * len(lines) + 10
                hx, hy = self._selected_tooltip_anchor(hk, hu)
                if hx + w > self.width - 8:
                    hx = self.width - w - 8
                if hy + h > self.height - 8:
                    hy = self.height - h - 8
                if hx < 8:
                    hx = 8
                if hy < 8:
                    hy = 8
                hover_rect = pygame.Rect(hx, hy, w, h)
                if selected_panel.width > 0 and hover_rect.colliderect(selected_panel):
                    # 겹치면 hover 패널만 아래(또는 위)로 재배치
                    self._draw_unit_info_panel_with_anchor(
                        hk,
                        hu,
                        selected_panel.bottom + 8,
                        fallback_top=max(8, selected_panel.y - h - 8),
                    )
                else:
                    self._draw_unit_info_panel(hk, hu)

    def _draw_unit_info_panel_with_anchor(
        self,
        kind: str,
        unit: Any,
        forced_y: int,
        fallback_top: int,
    ) -> pygame.Rect:
        lines = self._unit_info_lines(kind, unit)
        if not lines:
            return pygame.Rect(0, 0, 0, 0)
        w = 250
        line_h = self.tiny.get_linesize() + 2
        h = 14 + line_h * len(lines) + 10
        panel = self._unit_info_panel_rect(kind, unit, w, h, forced_y=forced_y, fallback_top=fallback_top)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, panel, width=2, border_radius=8)
        yy = panel.y + 8
        for i, line in enumerate(lines):
            color = TEXT if i == 0 else MUTED
            self._draw_info_line(panel.x + 10, yy, line, color)
            yy += line_h
        return panel

    def _draw_unit_info_panel(
        self,
        kind: str,
        unit: Any,
    ) -> pygame.Rect:
        lines = self._unit_info_lines(kind, unit)
        if not lines:
            return pygame.Rect(0, 0, 0, 0)
        w = 250
        line_h = self.tiny.get_linesize() + 2
        h = 14 + line_h * len(lines) + 10
        panel = self._unit_info_panel_rect(kind, unit, w, h)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, panel, width=2, border_radius=8)
        yy = panel.y + 8
        for i, line in enumerate(lines):
            color = TEXT if i == 0 else MUTED
            self._draw_info_line(panel.x + 10, yy, line, color)
            yy += line_h
        return panel

    def _unit_rect(self, kind: str, unit: Any) -> pygame.Rect:
        if kind == "player":
            (cx, cy), r = self._player_shape()
            return pygame.Rect(cx - r, cy - r, r * 2, r * 2)
        if kind == "enemy":
            for enemy, rect, _ in self._enemy_slots():
                if enemy is unit:
                    return rect.copy()
        return pygame.Rect(16, 16, 1, 1)

    def _clamp_panel_rect(self, rect: pygame.Rect) -> pygame.Rect:
        x = min(max(8, rect.x), self.width - rect.w - 8)
        y = min(max(8, rect.y), self.height - rect.h - 8)
        return pygame.Rect(x, y, rect.w, rect.h)

    def _unit_info_panel_rect(
        self,
        kind: str,
        unit: Any,
        w: int,
        h: int,
        forced_y: Optional[int] = None,
        fallback_top: Optional[int] = None,
    ) -> pygame.Rect:
        u = self._unit_rect(kind, unit).inflate(10, 10)

        candidates: List[pygame.Rect] = []
        if forced_y is not None:
            candidates.append(pygame.Rect(u.right + 12, forced_y, w, h))
            candidates.append(pygame.Rect(u.left - w - 12, forced_y, w, h))
            if fallback_top is not None:
                candidates.append(pygame.Rect(u.right + 12, fallback_top, w, h))
                candidates.append(pygame.Rect(u.left - w - 12, fallback_top, w, h))
        else:
            candidates.extend(
                [
                    pygame.Rect(u.right + 12, u.y - 4, w, h),
                    pygame.Rect(u.left - w - 12, u.y - 4, w, h),
                    pygame.Rect(u.centerx - (w // 2), u.top - h - 8, w, h),
                    pygame.Rect(u.centerx - (w // 2), u.bottom + 8, w, h),
                ]
            )

        best = None
        best_score = None
        for cand in candidates:
            clamped = self._clamp_panel_rect(cand)
            overlap_area = clamped.clip(u).width * clamped.clip(u).height
            # 상단 패널(헤더)와 겹치지 않도록 약한 페널티
            header_overlap = 1 if clamped.y < 80 else 0
            score = overlap_area * 1000 + header_overlap
            if best is None or score < best_score:
                best = clamped
                best_score = score
            if score == 0:
                break
        return best if best is not None else self._clamp_panel_rect(pygame.Rect(16, 16, w, h))

    def _selected_tooltip_anchor(self, kind: str, unit: Any) -> Tuple[int, int]:
        if kind == "player":
            (cx, cy), r = self._player_shape()
            return cx + r + 14, cy - r
        if kind == "enemy":
            for enemy, rect, _ in self._enemy_slots():
                if enemy is unit:
                    return rect.right + 12, rect.y - 4
        return 16, 16

    def _enhance_choice_rects(self) -> List[pygame.Rect]:
        n = len(self.enh_choices)
        if n <= 0:
            return []
        top = 108
        bottom = self.height - 64
        h = max(360, min(720, bottom - top))
        gap = 20
        margin_x = 16
        avail_w = self.width - margin_x * 2 - (n - 1) * gap
        w = max(280, int(avail_w / n))
        total_w = n * w + (n - 1) * gap
        x0 = max(margin_x, (self.width - total_w) // 2)
        return [pygame.Rect(x0 + i * (w + gap), top, w, h) for i in range(n)]

    def _draw_wrapped(self, text: str, rect: pygame.Rect, font: pygame.font.Font, color: Tuple[int, int, int], max_lines: int) -> None:
        words = (text or "").split()
        if not words:
            return
        lines: List[str] = []
        current = words[0]
        for w in words[1:]:
            candidate = f"{current} {w}"
            if font.size(candidate)[0] <= rect.w:
                current = candidate
            else:
                lines.append(current)
                current = w
                if len(lines) >= max_lines:
                    break
        if len(lines) < max_lines:
            lines.append(current)
        for i, line in enumerate(lines[:max_lines]):
            self.screen.blit(font.render(line, True, color), (rect.x, rect.y + i * (font.get_linesize() + 1)))

    def _draw_inline_segments(
        self,
        x: int,
        y: int,
        segments: List[Tuple[str, Tuple[int, int, int]]],
        font: pygame.font.Font,
    ) -> int:
        cx = x
        for text, color in segments:
            if not text:
                continue
            surf = font.render(text, True, color)
            self.screen.blit(surf, (cx, y))
            cx += surf.get_width()
        return cx

    def _player_armor_parts(self) -> Tuple[float, float]:
        bonus = max(0.0, float(self.rt.temp_armor_bonus))
        if bonus <= 0:
            return max(0.0, float(self.player.defense.armor)), 0.0
        # 적 턴 중에는 armor가 임시 보너스 포함값일 수 있어 기본값을 역산한다.
        if self.phase == "enemy":
            base = max(0.0, float(self.player.defense.armor) - bonus)
        else:
            base = max(0.0, float(self.player.defense.armor))
        return base, bonus

    def _draw_armor_line(
        self,
        x: int,
        y: int,
        base_armor: float,
        bonus_armor: float,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int],
    ) -> None:
        if bonus_armor > 0:
            self._draw_inline_segments(
                x,
                y,
                [
                    (f"갑옷 {base_armor:.1f} ", base_color),
                    (f"(+{bonus_armor:.1f})", TEMP_BUFF),
                ],
                font,
            )
        else:
            self.screen.blit(font.render(f"갑옷 {base_armor:.1f}", True, base_color), (x, y))

    def _draw_info_line(self, x: int, y: int, line: str, color: Tuple[int, int, int]) -> None:
        if line.startswith("갑옷 ") and "(+" in line and line.endswith(")"):
            try:
                pivot = line.index(" (+")
                left = line[:pivot + 1]  # trailing space 유지
                right = line[pivot + 1:]  # "(+12.0)"
                self._draw_inline_segments(x, y, [(left, color), (right, TEMP_BUFF)], self.tiny)
                return
            except Exception:
                pass
        self.screen.blit(self.tiny.render(line, True, color), (x, y))

    def _draw_card_background(self, rect: pygame.Rect, card_type: str) -> None:
        img = self.attack_card_image if card_type == "attack" else self.defense_card_image
        if img is None:
            fill = CARD_ATTACK if card_type == "attack" else CARD_DEFENSE
            pygame.draw.rect(self.screen, fill, rect, border_radius=8)
            return
        scaled = pygame.transform.smoothscale(img, (rect.w, rect.h))
        self.screen.blit(scaled, rect.topleft)

    def _tier_icon_surface(self, tier_idx: int) -> Optional[pygame.Surface]:
        ti = int(tier_idx)
        if ti in self.tier_icon_cache:
            return self.tier_icon_cache[ti]
        path = os.path.join("images", "tier_icons", f"t{ti}.png")
        surf: Optional[pygame.Surface]
        if os.path.exists(path):
            try:
                surf = pygame.image.load(path).convert_alpha()
            except Exception:
                surf = None
        else:
            surf = None
        self.tier_icon_cache[ti] = surf
        return surf

    def _draw_efficiency_badge(self, center: Tuple[int, int], text: str, border_color: Tuple[int, int, int]) -> None:
        radius = 19
        pygame.draw.circle(self.screen, (24, 30, 44), center, radius)
        pygame.draw.circle(self.screen, border_color, center, radius, width=3)
        txt = self.tiny.render(text, True, (255, 255, 255))
        tr = txt.get_rect(center=(center[0] - 2, center[1]))
        self.screen.blit(txt, tr.topleft)

    def _same_tier_bonus_dims(self, tier_idx: int) -> Tuple[int, int]:
        icon = self._tier_icon_surface(tier_idx)
        if icon is not None:
            return 38, 38
        return 20, 20

    def _lower_tier_bonus_dims(self, tier_idx: int) -> Tuple[int, int]:
        lower_idx = max(0, int(tier_idx) - 1)
        icon = self._tier_icon_surface(lower_idx)
        if icon is not None:
            return 38, 38
        return 20, 20

    def _draw_same_tier_bonus(self, x: int, y: int, tier_idx: int) -> Tuple[int, int]:
        icon = self._tier_icon_surface(tier_idx)
        if icon is not None:
            iw, ih = 38, 38
            icon_s = pygame.transform.smoothscale(icon, (iw, ih))
            self.screen.blit(icon_s, (x, y))
            return iw, ih
        pygame.draw.circle(self.screen, _tier_color(tier_idx), (x + 10, y + 10), 10)
        return 20, 20

    def _draw_lower_tier_bonus(self, x: int, y: int, tier_idx: int) -> Tuple[int, int]:
        lower_idx = max(0, int(tier_idx) - 1)
        icon = self._tier_icon_surface(lower_idx)
        if icon is not None:
            iw, ih = 38, 38
            icon_s = pygame.transform.smoothscale(icon, (iw, ih))
            self.screen.blit(icon_s, (x, y))
            return iw, ih
        pygame.draw.circle(self.screen, _tier_color(lower_idx), (x + 10, y + 10), 10)
        return 20, 20

    def _card_cache_key(self, card: CardState) -> str:
        try:
            return json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return f"{card.id}|{card.name}|{card.type}|{id(card)}"

    def _card_surface(self, card: CardState) -> Optional[pygame.Surface]:
        key = self._card_cache_key(card)
        cached = self.card_surface_cache.get(key)
        if cached is not None:
            return cached
        if self.enh_engine is None:
            return None
        try:
            pil_img = card.render_visual(self.enh_engine)
            if pil_img.mode != "RGBA":
                pil_img = pil_img.convert("RGBA")
            surface = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode).convert_alpha()
            self.card_surface_cache[key] = surface
            return surface
        except Exception:
            return None

    def _draw_card_visual(self, card: CardState, rect: pygame.Rect) -> None:
        surf = self._card_surface(card)
        if surf is None:
            self._draw_card_background(rect, card.type)
            return

        key = self._card_cache_key(card)
        skey = (key, rect.w, rect.h)
        scaled = self.card_scaled_cache.get(skey)
        if scaled is None:
            sw, sh = surf.get_size()
            if sw <= 0 or sh <= 0:
                self._draw_card_background(rect, card.type)
                return
            scale = min(rect.w / sw, rect.h / sh)
            tw = max(1, int(sw * scale))
            th = max(1, int(sh * scale))
            scaled = pygame.transform.smoothscale(surf, (tw, th))
            self.card_scaled_cache[skey] = scaled

        bg = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        bg.fill((10, 12, 18, 140))
        self.screen.blit(bg, rect.topleft)
        dst = scaled.get_rect(center=rect.center)
        self.screen.blit(scaled, dst.topleft)

    def _draw_card_visual_alpha(self, card: CardState, rect: pygame.Rect, alpha: int) -> None:
        a = max(0, min(255, int(alpha)))
        if a <= 0:
            return
        surf = self._card_surface(card)
        if surf is None:
            bg = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            bg.fill((80, 90, 110, int(90 * (a / 255.0))))
            self.screen.blit(bg, rect.topleft)
            return

        key = self._card_cache_key(card)
        skey = (key, rect.w, rect.h)
        scaled = self.card_scaled_cache.get(skey)
        if scaled is None:
            sw, sh = surf.get_size()
            if sw <= 0 or sh <= 0:
                return
            scale = min(rect.w / sw, rect.h / sh)
            tw = max(1, int(sw * scale))
            th = max(1, int(sh * scale))
            scaled = pygame.transform.smoothscale(surf, (tw, th))
            self.card_scaled_cache[skey] = scaled

        bg = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        bg.fill((10, 12, 18, int(140 * (a / 255.0))))
        self.screen.blit(bg, rect.topleft)

        draw_surf = scaled.copy()
        draw_surf.set_alpha(a)
        dst = draw_surf.get_rect(center=rect.center)
        self.screen.blit(draw_surf, dst.topleft)

    def _spawn_card_use_animation(self, card: CardState, rect: pygame.Rect) -> None:
        self.card_use_anims.append(
            {
                "card": card,
                "rect": rect.copy(),
                "t": 0.0,
            }
        )

    def _enemy_slots(self) -> List[Tuple[Enemy, pygame.Rect, int]]:
        slots: List[Tuple[Enemy, pygame.Rect, int]] = []
        _, right_zone, y_center = self._battlefield_bounds()
        alive_rows: List[List[Enemy]] = []
        for row in self.enemy_rows:
            alive_row = [e for e in row if e.is_alive]
            if alive_row:
                alive_rows.append(alive_row)

        if not alive_rows:
            return slots

        row_count = len(alive_rows)
        for pos, alive_row in enumerate(alive_rows):
            x = int(right_zone.x + ((pos + 0.5) * right_zone.w / row_count))
            spacing = 160
            if len(alive_row) > 1:
                spacing = min(220, max(160, int((right_zone.h - 120) / len(alive_row))))
            y0 = y_center - spacing * (len(alive_row) - 1) // 2
            for j, e in enumerate(alive_row):
                y = y0 + j * spacing
                rect = pygame.Rect(x - 44, y - 44, 88, 88)
                # 세 번째 값은 "현재 압축된 거리(열)"이다.
                slots.append((e, rect, pos + 1))
        return slots

    def _enemy_rect_for(self, target: Enemy) -> Optional[pygame.Rect]:
        for enemy, rect, _ in self._enemy_slots():
            if enemy is target:
                return rect.copy()
        return None

    def _spawn_enemy_remove_effect(self, target: Enemy, rect: Optional[pygame.Rect]) -> None:
        if rect is None:
            return
        self.enemy_remove_effects.append(
            {
                "start_ms": float(pygame.time.get_ticks()),
                "duration_ms": float(ENEMY_REMOVE_EFFECT_MS),
                "cx": float(rect.centerx),
                "cy": float(rect.centery),
                "size": float(max(rect.w, rect.h)),
                "kind": self._enemy_kind(target),
                "phase": float(self.rng.random() * 6.283185307),
            }
        )

    def _draw_triangle(self, center: Tuple[int, int], size: int, color: Tuple[int, int, int]) -> None:
        cx, cy = center
        pts = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
        pygame.draw.polygon(self.screen, color, pts)

    def _draw_rect_x(self, rect: pygame.Rect, color: Tuple[int, int, int]) -> None:
        cx, cy = rect.center
        bar_len = 12
        bar_th = 3
        base = pygame.Surface((bar_len, bar_th), pygame.SRCALPHA)
        pygame.draw.rect(base, color, (0, 0, bar_len, bar_th), border_radius=1)
        for ang in (45, -45):
            rot = pygame.transform.rotate(base, ang)
            rr = rot.get_rect(center=(cx, cy))
            self.screen.blit(rot, rr.topleft)

    def _play_card(self, hand_index: int) -> None:
        if not (0 <= hand_index < len(self.rt.hand)):
            return
        if self.actions_left <= 0:
            return

        src_rects = self._display_card_rects()
        src_rect = src_rects[hand_index].copy() if 0 <= hand_index < len(src_rects) else None
        card = self.rt.hand.pop(hand_index)
        if src_rect is not None:
            self._spawn_card_use_animation(card, src_rect)
        self.selected_card_index = None
        self._gain_resource(card)

        if card.type == "attack":
            self._prepare_attack(card)
        elif card.type == "defense":
            self._play_defense(card)
            self._finish_card_action(card)
        else:
            self.log(f"{card.name}: 미지원 카드 타입")
            self._finish_card_action(card)

    def _gain_resource(self, card: CardState) -> None:
        gain = _chance_count(float(card.stats.get("resource", 0.0) or 0.0), self.rng)
        if gain > 0:
            self.rt.resource += gain
            self.log(f"자원 +{gain} (현재 {self.rt.resource})")

    def _prepare_attack(self, card: CardState) -> None:
        eff = card.effects
        attack_range = int(card.stats.get("range", 1) or 1)
        if bool(eff.get("max_range", False)):
            attack_range = max(attack_range, 99)
        aoe = bool(eff.get("aoe_all_enemies", False))
        ignore_defense = bool(eff.get("ignore_defense", False))
        splash_other_ratio = max(0.0, float(eff.get("splash_other_ratio", 0.0) or 0.0))
        hit_count = 2 if bool(eff.get("double_hit", False)) else 1

        attack_power = self.player.attack.power * (1.0 + self.rt.frenzy_ratio)
        if self.rt.next_attack_bonus_ratio > 0:
            before = card.compute_attack_damage(attack_power)
            attack_power *= 1.0 + self.rt.next_attack_bonus_ratio
            after = card.compute_attack_damage(attack_power)
            self.log(
                f"다음 공격 보너스 +{self.rt.next_attack_bonus_ratio*100:.1f}% "
                f"({before:.1f} -> {after:.1f})"
            )
            self.rt.next_attack_bonus_ratio = 0.0

        per_hit = card.compute_attack_damage(attack_power)
        if per_hit <= 0:
            self.log(f"{card.name}: 피해량이 0입니다.")
            self._finish_card_action(card)
            return

        if aoe:
            for hit in range(1, hit_count + 1):
                in_range = self._in_range_enemies(attack_range)
                if not in_range:
                    break
                for target in list(in_range):
                    self._apply_attack_hit(
                        card=card,
                        target=target,
                        per_hit=per_hit,
                        hit_idx=hit,
                        hit_count=hit_count,
                        ignore_defense=ignore_defense,
                        splash_other_ratio=0.0,
                    )
                if self._check_end():
                    return
            self._finish_card_action(card)
            return

        in_range = self._in_range_enemies(attack_range)
        if not in_range:
            self.log(f"{card.name}: 사정거리({attack_range}) 안에 적이 없습니다.")
            self._finish_card_action(card)
            return

        self.pending_attack = PendingAttack(
            card=card,
            hit_count=hit_count,
            hit_index=1,
            per_hit=per_hit,
            attack_range=attack_range,
            ignore_defense=ignore_defense,
            splash_other_ratio=splash_other_ratio,
        )
        self.log(f"대상 선택: {card.name} ({self.pending_attack.hit_index}/{self.pending_attack.hit_count})")

    def _in_range_enemies(self, attack_range: int) -> List[Enemy]:
        out: List[Enemy] = []
        current_dist = 1
        for row in self.enemy_rows:
            alive_row = [e for e in row if e.is_alive]
            if not alive_row:
                continue
            if current_dist > attack_range:
                break
            out.extend(alive_row)
            current_dist += 1
        return out

    def _resolve_pending_attack_target(self, target: Enemy) -> bool:
        if self.pending_attack is None:
            return False
        p = self.pending_attack
        self._apply_attack_hit(
            card=p.card,
            target=target,
            per_hit=p.per_hit,
            hit_idx=p.hit_index,
            hit_count=p.hit_count,
            ignore_defense=p.ignore_defense,
            splash_other_ratio=p.splash_other_ratio,
        )
        if self._check_end():
            return True

        p.hit_index += 1
        if p.hit_index > p.hit_count:
            finished_card = p.card
            self.pending_attack = None
            self._finish_card_action(finished_card)
            return True

        if not self._in_range_enemies(p.attack_range):
            self.log("사정거리 안에 대상이 없어 연타가 종료되었습니다.")
            finished_card = p.card
            self.pending_attack = None
            self._finish_card_action(finished_card)
            return True

        self.log(f"대상 선택: {p.card.name} ({p.hit_index}/{p.hit_count})")
        return True

    def _try_target_click(self, pos: Tuple[int, int]) -> bool:
        if self.pending_attack is None:
            return False
        valid = set(self._in_range_enemies(self.pending_attack.attack_range))
        clicked_enemy: Optional[Enemy] = None
        for enemy, rect, _ in self._enemy_slots():
            if enemy in valid and rect.collidepoint(pos):
                clicked_enemy = enemy
                break
        if clicked_enemy is None:
            return False
        return self._resolve_pending_attack_target(clicked_enemy)

    def _cancel_pending_attack(self) -> None:
        if self.pending_attack is None:
            return
        card = self.pending_attack.card
        self.pending_attack = None
        self.rt.hand.append(card)
        self.selected_card_index = None
        self.log("카드 선택 취소")

    def _apply_attack_hit(
        self,
        card: CardState,
        target: Enemy,
        per_hit: float,
        hit_idx: int,
        hit_count: int,
        ignore_defense: bool,
        splash_other_ratio: float,
    ) -> None:
        target_rect_before = self._enemy_rect_for(target)
        breakdown = _damage_breakdown(target, per_hit, ignore_defense=ignore_defense)
        dealt = _deal_damage(target, per_hit, ignore_defense=ignore_defense)
        parts = [f"{breakdown['raw']:.1f}"]
        if breakdown["reduction_amount"] > 0:
            parts.append(f"감소 {breakdown['reduction_amount']:.1f}")
        if breakdown["armor_block"] > 0:
            parts.append(f"갑옷 {breakdown['armor_block']:.1f}")
        if breakdown["shield_block"] > 0:
            parts.append(f"보호막 {breakdown['shield_block']:.1f}")
        self.log(
            f"{card.name}({hit_idx}/{hit_count}) -> {target.name}: "
            + " - ".join(parts)
            + f" = {dealt:.1f}"
        )
        if dealt > 0:
            self._apply_attack_side_effects(target, card, dealt)
        if not target.is_alive:
            self._spawn_enemy_remove_effect(target, target_rect_before)
            self.log(f"{target.name} 처치!")

        if splash_other_ratio > 0:
            splash_damage = per_hit * splash_other_ratio
            for other in [e for e in _alive_enemies(self.enemy_rows) if e is not target]:
                other_rect_before = self._enemy_rect_for(other)
                sp_break = _damage_breakdown(other, splash_damage, ignore_defense=ignore_defense)
                sp_dealt = _deal_damage(other, splash_damage, ignore_defense=ignore_defense)
                self.log(
                    f"파급 {other.name}: {sp_break['raw']:.1f}"
                    + (f" - 갑옷 {sp_break['armor_block']:.1f}" if sp_break["armor_block"] > 0 else "")
                    + (f" - 보호막 {sp_break['shield_block']:.1f}" if sp_break["shield_block"] > 0 else "")
                    + f" = {sp_dealt:.1f}"
                )
                if not other.is_alive:
                    self._spawn_enemy_remove_effect(other, other_rect_before)
                    self.log(f"{other.name} 처치!")

    def _apply_attack_side_effects(self, target: Enemy, card: CardState, dealt: float) -> None:
        eff = card.effects

        shield_ratio = max(0.0, float(eff.get("shield_ratio", 0.0) or 0.0))
        if dealt > 0 and shield_ratio > 0:
            gain = self.player.add_shield(dealt * shield_ratio)
            self.log(f"보호막 +{gain:.1f}")

        freeze_turns = _chance_count(float(eff.get("freeze_turns", 0.0) or 0.0), self.rng)
        if _apply_freeze(target, freeze_turns):
            self.log(f"{target.name} 빙결 {freeze_turns}턴")

        bleed_ratio = max(0.0, float(eff.get("bleed_ratio", 0.0) or 0.0))
        bleed_turns = _chance_count(float(eff.get("bleed_turns", 0.0) or 0.0), self.rng)
        if bleed_ratio > 0 and bleed_turns > 0:
            bleed_damage = max(0.0, self.player.attack.power * (1.0 + self.rt.frenzy_ratio)) * bleed_ratio
            if _apply_bleed(target, bleed_damage, bleed_turns):
                self.log(f"{target.name} 출혈 {bleed_damage:.1f}/턴 ({bleed_turns}턴)")

        armor_down = max(0.0, float(eff.get("armor_down_ratio", 0.0) or 0.0))
        if armor_down > 0:
            _apply_armor_down(target, armor_down)
            self.log(f"{target.name} 방어 감소 {armor_down*100:.1f}%")

        draw_cards = max(0.0, float(eff.get("draw_cards", 0.0) or 0.0))
        if draw_cards > 0:
            drew = _draw_n(self.rt, self.cfg, self.rng, draw_cards)
            self.log(f"드로우 +{drew}장")

    def _play_defense(self, card: CardState) -> None:
        eff = card.effects
        gain = card.compute_shield_amount(self.player.defense.defense_power)
        self.player.add_shield(gain)
        self.log(f"{card.name}: 보호막 +{gain:.1f}")

        temp_armor_ratio = max(
            0.0,
            float(eff.get("temp_armor_ratio", 0.0) or 0.0) + float(eff.get("temp_defense_ratio", 0.0) or 0.0),
        )
        if temp_armor_ratio > 0:
            bonus_armor = max(0.0, self.player.defense.armor * temp_armor_ratio)
            self.rt.temp_armor_bonus += bonus_armor
            self.log(f"이번 적 턴 갑옷 +{bonus_armor:.1f}")

        next_attack = max(0.0, float(eff.get("next_attack_bonus_ratio", 0.0) or 0.0))
        if next_attack > 0:
            self.rt.next_attack_bonus_ratio += next_attack
            self.log(f"다음 공격 +{next_attack*100:.1f}%")

        oh_freeze = max(0.0, float(eff.get("on_hit_freeze_turns", 0.0) or 0.0))
        if oh_freeze > 0:
            self.rt.on_hit_freeze_turns += oh_freeze
            self.log(f"피격 반격 빙결 {self.rt.on_hit_freeze_turns:.2f}턴")

        oh_bleed_ratio = max(0.0, float(eff.get("on_hit_bleed_ratio", 0.0) or 0.0))
        oh_bleed_turns = max(0.0, float(eff.get("on_hit_bleed_turns", 0.0) or 0.0))
        if oh_bleed_ratio > 0 and oh_bleed_turns > 0:
            self.rt.on_hit_bleed_ratio += oh_bleed_ratio
            self.rt.on_hit_bleed_turns += oh_bleed_turns
            self.log(f"피격 반격 출혈 {self.rt.on_hit_bleed_ratio*100:.1f}%/{self.rt.on_hit_bleed_turns:.1f}턴")

        oh_draw = max(0.0, float(eff.get("on_hit_draw_cards", 0.0) or 0.0))
        if oh_draw > 0:
            self.rt.on_hit_draw_cards += oh_draw
            self.log(f"피격 반격 드로우 {self.rt.on_hit_draw_cards:.2f}")

        oh_frenzy = max(0.0, float(eff.get("on_hit_frenzy_ratio", 0.0) or 0.0))
        if oh_frenzy > 0:
            self.rt.on_hit_frenzy_ratio += oh_frenzy
            self.log(f"피격 반격 광란 {self.rt.on_hit_frenzy_ratio*100:.1f}%")

        draw_cards = max(0.0, float(eff.get("draw_cards", 0.0) or 0.0))
        if draw_cards > 0:
            drew = _draw_n(self.rt, self.cfg, self.rng, draw_cards)
            self.log(f"드로우 +{drew}장")

    def _finish_card_action(self, card: CardState) -> None:
        self.rt.discard.append(card)
        self.actions_left -= 1
        if self.actions_left <= 0 and self.phase == "player":
            self._end_player_turn()

    def _end_player_turn(self) -> None:
        if self.game_over:
            return
        if self.phase != "player":
            return
        if self.pending_attack is not None:
            return

        self.selected_card_index = None
        p_bleed = _tick_bleed(self.player)
        if p_bleed > 0:
            self.log(f"플레이어 출혈 피해 {p_bleed:.1f}")
        self.player.end_turn()
        if self._check_end():
            return

        self.phase = "enemy"
        self._run_enemy_phase()
        if self._check_end():
            return

        self.rt.clear_round_buffs()
        self.turn += 1
        self.actions_left = self.cfg.cards_per_turn
        drew = _draw_n(self.rt, self.cfg, self.rng, self.cfg.draw_per_turn)
        self.phase = "player"
        self.log(f"턴 {self.turn} 시작 (드로우 {drew}장)")

    def _run_enemy_phase(self) -> None:
        alive = _alive_enemies(self.enemy_rows)
        if not alive:
            return
        self.log("적 턴")
        base_armor = self.player.defense.armor
        self.player.defense.armor = base_armor + self.rt.temp_armor_bonus
        try:
            for e in alive:
                skipped_by_freeze = False
                if e.state.frozen_turns > 0:
                    skipped_by_freeze = True
                    self.log(f"{e.name}: 빙결로 행동 불가")
                elif e.state.stunned:
                    self.log(f"{e.name}: 기절로 행동 불가")
                else:
                    dealt = e.basic_attack(self.player)
                    self.log(f"{e.name} 공격 -> 플레이어 {dealt:.1f}")
                    if self.player.is_alive:
                        self._trigger_on_hit_reactions(e)

                bleed = _tick_bleed(e)
                if bleed > 0:
                    self.log(f"{e.name} 출혈 피해 {bleed:.1f}")
                if skipped_by_freeze and e.state.frozen_turns > 0:
                    e.state.frozen_turns -= 1
                e.state.stunned = False
                if not self.player.is_alive:
                    break
        finally:
            self.player.defense.armor = base_armor

    def _trigger_on_hit_reactions(self, attacker: Enemy) -> None:
        freeze_turns = _chance_count(self.rt.on_hit_freeze_turns, self.rng)
        if _apply_freeze(attacker, freeze_turns):
            self.log(f"반격 빙결: {attacker.name} {freeze_turns}턴")

        bleed_turns = _chance_count(self.rt.on_hit_bleed_turns, self.rng)
        if self.rt.on_hit_bleed_ratio > 0 and bleed_turns > 0:
            bleed_damage = max(0.0, self.player.attack.power * (1.0 + self.rt.frenzy_ratio)) * self.rt.on_hit_bleed_ratio
            if _apply_bleed(attacker, bleed_damage, bleed_turns):
                self.log(f"반격 출혈: {attacker.name} {bleed_damage:.1f}/턴 ({bleed_turns}턴)")

        if self.rt.on_hit_draw_cards > 0:
            drew = _draw_n(self.rt, self.cfg, self.rng, self.rt.on_hit_draw_cards)
            self.log(f"피격 반응 드로우 {drew}장")

        if self.rt.on_hit_frenzy_ratio > 0:
            self.rt.frenzy_ratio += self.rt.on_hit_frenzy_ratio
            self.log(f"광란 +{self.rt.on_hit_frenzy_ratio*100:.1f}% (총 {self.rt.frenzy_ratio*100:.1f}%)")

    def _check_end(self) -> bool:
        if not self.player.is_alive:
            self.game_over = True
            self.win = False
            self.phase = "done"
            self.log("패배: 플레이어 사망")
            return True
        if not _alive_enemies(self.enemy_rows):
            self.game_over = True
            self.win = True
            self.phase = "done"
            self.log("승리: 모든 적 처치")
            return True
        return False

    def _render(self) -> None:
        self.screen.fill(BG)
        if self.mode == "enhance":
            self._draw_enhancement_screen()
            return
        self._draw_top_panel()
        self._draw_units()
        self._draw_cards()
        self._draw_logs()
        self._draw_overlay()

    def _draw_enhancement_screen(self) -> None:
        pygame.draw.rect(self.screen, PANEL, (0, 0, self.width, 92))
        pygame.draw.line(self.screen, BORDER, (0, 92), (self.width, 92), 2)

        title_base = f"[{self.enh_roll_index}/{self.enh_total_rolls}] 강화 선택"
        title_surface = self.enh_heading_font.render(title_base, True, TEXT)
        self.screen.blit(title_surface, (18, 18))
        forced_idx = self._enh_forced_tier_index()
        if forced_idx is not None:
            forced_text = f" - {TIERS_NAME[Tier(forced_idx)]} 확정"
            self.screen.blit(
                self.enh_heading_font.render(forced_text, True, _tier_color(forced_idx)),
                (18 + title_surface.get_width(), 18),
            )
        self.screen.blit(
            self.enh_heading_small.render("카드를 클릭해서 선택 | A: 남은 횟수 자동 | T: 천장 패널 | 강화 로그 버튼", True, MUTED),
            (18, 52),
        )

        for i, (plan, rect) in enumerate(zip(self.enh_choices, self._enhance_choice_rects()), start=1):
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
            pygame.draw.rect(self.screen, BORDER, rect, width=2, border_radius=10)

            inner_x = rect.x + 12
            inner_w = rect.w - 24
            reserve_for_text = 172
            art_h = max(180, min(int(rect.h * 0.68), rect.h - reserve_for_text))
            art_rect = pygame.Rect(inner_x, rect.y + 12, inner_w, art_h)
            self._draw_card_visual(plan["card"], art_rect)
            pygame.draw.rect(self.screen, BORDER, art_rect, width=2, border_radius=8)

            text_x = inner_x
            text_w = inner_w
            text_y = art_rect.bottom + 10
            has_bonus = any(
                bool(plan.get(k, False))
                for k in ("efficiency_proc", "efficiency_double_proc", "same_tier_bonus_proc", "lower_bonus_proc")
            )
            main_text_w = text_w
            title_text = f"{i}) {plan['card'].name}"
            self.screen.blit(self.enh_choice_big.render(title_text, True, TEXT), (text_x, text_y))

            tier_idx = int(plan["idx"])
            tier_text = TIERS_NAME[Tier(tier_idx)]
            tier_surf = self.enh_choice_big.render(f"티어: {tier_text}", True, _tier_color(tier_idx))
            tier_pos = (text_x, text_y + 26)
            self.screen.blit(tier_surf, tier_pos)
            tier_icon = self._tier_icon_surface(tier_idx)
            if tier_icon is not None:
                icon_size = 24
                icon_surf = pygame.transform.smoothscale(tier_icon, (icon_size, icon_size))
                icon_x = tier_pos[0] + tier_surf.get_width() + 4
                icon_y = tier_pos[1] + max(0, (tier_surf.get_height() - icon_size) // 2)
                self.screen.blit(icon_surf, (icon_x, icon_y))

            current_desc = describe_card(plan["card"], self.player)
            self._draw_wrapped(
                f"현재: {current_desc}",
                pygame.Rect(text_x, text_y + 56, main_text_w, 36),
                self.small,
                MUTED,
                max_lines=2,
            )
            self._draw_wrapped(
                f"강화: {plan.get('display', '-')}",
                pygame.Rect(text_x, text_y + 98, main_text_w, 56),
                self.enh_choice_mid,
                _tier_color(tier_idx),
                max_lines=2,
            )

            if has_bonus:
                items: List[Tuple[str, Any, int, int]] = []
                if bool(plan.get("efficiency_proc", False)):
                    items.append(("eff", ("×1.5", (212, 205, 152)), 38, 38))
                if bool(plan.get("efficiency_double_proc", False)):
                    items.append(("eff", ("×2", (150, 204, 166)), 38, 38))
                if bool(plan.get("same_tier_bonus_proc", False)):
                    iw, ih = self._same_tier_bonus_dims(tier_idx)
                    items.append(("gem", tier_idx, iw, ih))
                if bool(plan.get("lower_bonus_proc", False)):
                    iw, ih = self._lower_tier_bonus_dims(tier_idx)
                    items.append(("lower_gem", tier_idx, iw, ih))

                if items:
                    pad_x = 8
                    pad_y = 8
                    gap = 6
                    content_w = sum(it[2] for it in items) + gap * (len(items) - 1)
                    content_h = max(it[3] for it in items)
                    box_w = content_w + pad_x * 2
                    box_h = content_h + pad_y * 2

                    bonus_box = pygame.Rect(
                        rect.right - 12 - box_w,
                        rect.bottom - 12 - box_h,
                        box_w,
                        box_h,
                    )
                    pygame.draw.rect(self.screen, (24, 30, 44), bonus_box, border_radius=8)
                    pygame.draw.rect(self.screen, BORDER, bonus_box, width=1, border_radius=8)

                    bx = bonus_box.x + pad_x
                    for idx, (kind, payload, iw, ih) in enumerate(items):
                        iy = bonus_box.y + pad_y + (content_h - ih) // 2
                        if kind == "eff":
                            label, color = payload
                            self._draw_efficiency_badge((bx + 19, iy + 19), label, color)
                        elif kind == "gem":
                            self._draw_same_tier_bonus(bx, iy, int(payload))
                        elif kind == "lower_gem":
                            self._draw_lower_tier_bonus(bx, iy, int(payload))
                        else:
                            txt = payload
                            self.screen.blit(txt, (bx, iy))
                        if idx < len(items) - 1:
                            sep_x = bx + iw + (gap // 2)
                            pygame.draw.line(
                                self.screen,
                                BORDER,
                                (sep_x, bonus_box.y + 6),
                                (sep_x, bonus_box.bottom - 6),
                                1,
                            )
                        bx += iw + gap

        log_panel = self._enh_log_collapsed_rect()
        pygame.draw.rect(self.screen, PANEL, log_panel, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, log_panel, width=2, border_radius=10)
        self.screen.blit(self.tiny.render("강화 로그", True, TEXT), (log_panel.x + 20, log_panel.y + 8))

        if self.enh_log_expanded:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((8, 10, 14, 170))
            self.screen.blit(overlay, (0, 0))

            panel = self._enh_log_expanded_rect()
            pygame.draw.rect(self.screen, PANEL, panel, border_radius=12)
            pygame.draw.rect(self.screen, BORDER, panel, width=2, border_radius=12)
            self.screen.blit(self.font.render("강화 로그 전체", True, TEXT), (panel.x + 14, panel.y + 10))

            close_rect = pygame.Rect(panel.right - 92, panel.y + 10, 76, 28)
            pygame.draw.rect(self.screen, (56, 64, 84), close_rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, close_rect, width=1, border_radius=8)
            self.screen.blit(self.tiny.render("닫기", True, TEXT), (close_rect.x + 26, close_rect.y + 7))

            content = pygame.Rect(panel.x + 14, panel.y + 66, panel.w - 28, panel.h - 80)
            pygame.draw.rect(self.screen, (22, 28, 39), content, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, content, width=1, border_radius=8)
            line_h = self.tiny.get_linesize() + 4
            self.enh_log_lines_per_page = max(1, content.h // line_h)
            self._clamp_enh_log_scroll()
            start = self.enh_log_scroll
            end = min(len(self.enh_logs), start + self.enh_log_lines_per_page)

            yy = content.y
            for line in self.enh_logs[start:end]:
                self.screen.blit(self.tiny.render(line, True, MUTED), (content.x, yy))
                yy += line_h

            if len(self.enh_logs) > self.enh_log_lines_per_page:
                pos_txt = f"{start + 1}-{end} / {len(self.enh_logs)}"
                self.screen.blit(self.tiny.render(pos_txt, True, MUTED), (panel.right - 150, panel.bottom - 22))

        if self.enh_show_pity and self.enh_gacha is not None:
            pity = pygame.Rect(self.width - 290, 104, 250, 180)
            pygame.draw.rect(self.screen, PANEL, pity, border_radius=10)
            pygame.draw.rect(self.screen, BORDER, pity, width=2, border_radius=10)
            self.screen.blit(self.small.render("티어별 남은 천장", True, TEXT), (pity.x + 10, pity.y + 8))
            py = pity.y + 36
            for idx in range(len(Tier) - 1, 0, -1):
                cap = int(self.enh_gacha.caps[idx]) if idx < len(self.enh_gacha.caps) else 0
                fail = int(self.enh_gacha.fail_counts[idx]) if idx < len(self.enh_gacha.fail_counts) else 0
                if cap <= 0:
                    remain_text = "천장 없음"
                else:
                    remain = max(0, cap - fail)
                    remain_text = "확정 상태" if remain == 0 else f"{remain}회 남음"
                txt = f"{TIERS_NAME[Tier(idx)]}: {remain_text}"
                self.screen.blit(self.tiny.render(txt, True, _tier_color(idx)), (pity.x + 10, py))
                py += 24

    def _draw_top_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL, (0, 0, self.width, 78))
        pygame.draw.line(self.screen, BORDER, (0, 78), (self.width, 78), 2)

        head = (
            f"턴 {self.turn} | 단계: {'플레이어' if self.phase == 'player' else ('적' if self.phase == 'enemy' else '종료')} "
            f"| 남은 행동 {self.actions_left}"
        )
        self.screen.blit(self.font.render(head, True, TEXT), (18, 18))

        base_armor, bonus_armor = self._player_armor_parts()
        status_segments: List[Tuple[str, Tuple[int, int, int]]] = [
            (f"HP {self.player.hp:.1f}/{self.player.max_hp:.1f}", MUTED),
            (" | ", MUTED),
            (f"갑옷 {base_armor:.1f}", MUTED),
        ]
        if bonus_armor > 0:
            status_segments.append((f" (+{bonus_armor:.1f})", TEMP_BUFF))
        status_segments.extend(
            [
                (" | ", MUTED),
                (f"보호막 {self.player.state.shield:.1f}", MUTED),
                (" | ", MUTED),
                (f"덱 {len(self.rt.deck)} / 버림 {len(self.rt.discard)} / 손패 {len(self.rt.hand)}", MUTED),
            ]
        )
        self._draw_inline_segments(18, 46, status_segments, self.small)

        btn_color = (56, 86, 126) if self.phase == "player" and not self.game_over else (76, 76, 86)
        pygame.draw.rect(self.screen, btn_color, self.end_button_rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, self.end_button_rect, width=2, border_radius=8)
        self.screen.blit(self.small.render("턴 종료 (Space)", True, TEXT), (self.end_button_rect.x + 15, self.end_button_rect.y + 14))

    def _draw_units(self) -> None:
        left_zone, right_zone, y_center = self._battlefield_bounds()
        player_center = (left_zone.centerx, y_center)
        selected = self._get_selected_unit()
        selected_kind = selected[0] if selected is not None else None
        selected_unit = selected[1] if selected is not None else None
        if selected_kind == "player" and selected_unit is self.player and self.player.is_alive:
            pygame.draw.circle(self.screen, (255, 224, 120), player_center, 68, width=5)
        player_rect = pygame.Rect(0, 0, 108, 108)
        player_rect.center = player_center
        player_frozen_turns = int(max(0, self.player.state.frozen_turns))
        if player_frozen_turns > 0:
            self._draw_freeze_effect_back(player_rect)
        player_img = self._scaled_unit_image("player", (player_rect.w, player_rect.h))
        if player_img is not None:
            self.screen.blit(player_img, player_rect.topleft)
        else:
            pygame.draw.circle(self.screen, PLAYER_COLOR, player_center, 54)
        if player_frozen_turns > 0:
            self._draw_freeze_effect_front(player_rect)
        self.screen.blit(self.small.render("플레이어", True, TEXT), (player_center[0] - 34, player_center[1] + 70))

        hp_ratio = max(0.0, min(1.0, self.player.hp / self.player.max_hp))
        hp_w = 220
        hp_rect = pygame.Rect(player_center[0] - hp_w // 2, player_center[1] + 96, hp_w, 18)
        pygame.draw.rect(self.screen, (70, 36, 42), hp_rect, border_radius=6)
        pygame.draw.rect(self.screen, GREEN, (hp_rect.x, hp_rect.y, int(hp_w * hp_ratio), hp_rect.h), border_radius=6)
        hp_text = f"{self.player.hp:.1f}/{self.player.max_hp:.1f}"
        hp_text_surf = self.tiny.render(hp_text, True, (255, 255, 255))
        hp_text_rect = hp_text_surf.get_rect(center=hp_rect.center)
        self.screen.blit(hp_text_surf, hp_text_rect.topleft)

        player_shield = max(0.0, float(self.player.state.shield))
        if player_shield > 0.0:
            shield_ratio = max(0.0, min(1.0, player_shield / max(1.0, self.player.max_hp)))
            shield_rect = pygame.Rect(hp_rect.x, hp_rect.bottom + 8, hp_rect.w, 12)
            pygame.draw.rect(self.screen, (70, 74, 84), shield_rect, border_radius=6)
            pygame.draw.rect(
                self.screen,
                (126, 132, 146),
                (shield_rect.x, shield_rect.y, int(shield_rect.w * shield_ratio), shield_rect.h),
                border_radius=6,
            )
            shield_text = f"보호막 {player_shield:.1f}"
            shield_text_surf = self.tiny.render(shield_text, True, (255, 255, 255))
            shield_text_rect = shield_text_surf.get_rect(center=shield_rect.center)
            self.screen.blit(shield_text_surf, shield_text_rect.topleft)

        pending_targets = set()
        if self.pending_attack is not None:
            pending_targets = set(self._in_range_enemies(self.pending_attack.attack_range))
        elif (
            self.selected_card_index is not None
            and 0 <= self.selected_card_index < len(self.rt.hand)
            and self.rt.hand[self.selected_card_index].type == "attack"
        ):
            selected_card = self.rt.hand[self.selected_card_index]
            attack_range = int(selected_card.stats.get("range", 1) or 1)
            if bool(selected_card.effects.get("max_range", False)):
                attack_range = max(attack_range, 99)
            pending_targets = set(self._in_range_enemies(attack_range))

        enemy_slots = self._enemy_slots()
        row_rects: Dict[int, List[pygame.Rect]] = {}
        for _, rect, row_idx in enemy_slots:
            row_rects.setdefault(row_idx, []).append(rect)

        row_label_y = right_zone.y + 8
        for row_idx, rects in row_rects.items():
            center_x = int(sum(r.centerx for r in rects) / len(rects))
            row_text = f"열 {row_idx}"
            txt = self.tiny.render(row_text, True, MUTED)
            self.screen.blit(txt, (center_x - txt.get_width() // 2, row_label_y))

        for enemy, rect, _row_idx in enemy_slots:
            frozen_turns = int(max(0, enemy.state.frozen_turns))
            enemy_kind = self._enemy_kind(enemy)
            if frozen_turns > 0:
                self._draw_freeze_effect_back(rect)
            enemy_img = self._scaled_unit_image(enemy_kind, (rect.w, rect.h))
            if enemy_img is not None:
                self.screen.blit(enemy_img, rect.topleft)
            else:
                base_color = (120, 210, 255) if frozen_turns > 0 else ENEMY_COLOR
                self._draw_triangle(rect.center, 34, base_color)
            if frozen_turns > 0:
                self._draw_freeze_effect_front(rect)
            pygame.draw.rect(self.screen, BORDER, rect, width=2, border_radius=6)
            if selected_kind == "enemy" and selected_unit is enemy:
                sel = rect.inflate(14, 14)
                pygame.draw.rect(self.screen, (255, 224, 120), sel, width=4, border_radius=9)
            if enemy in pending_targets:
                hl = rect.inflate(10, 10)
                pygame.draw.rect(self.screen, ENEMY_HL, hl, width=3, border_radius=8)

            name_y = rect.bottom + 6
            name_surf = self.tiny.render(enemy.name, True, TEXT)
            self.screen.blit(name_surf, (rect.centerx - name_surf.get_width() // 2, name_y))

            e_hp_ratio = max(0.0, min(1.0, enemy.hp / enemy.max_hp))
            e_bar = pygame.Rect(rect.x - 4, name_y + 20, 96, 8)
            pygame.draw.rect(self.screen, (70, 36, 42), e_bar, border_radius=4)
            hp_w = int(e_bar.w * e_hp_ratio)
            pygame.draw.rect(self.screen, RED, (e_bar.x, e_bar.y, hp_w, e_bar.h), border_radius=4)

            bleed_turns = int(max(0, enemy.state.tags.get("bleed_turns", 0)))
            bleed_dmg = float(max(0.0, enemy.state.tags.get("bleed_damage", 0.0)))
            if bleed_turns > 0 and bleed_dmg > 0 and hp_w > 0:
                # 다음 턴 종료 시 출혈로 잃을 예상 체력을 현재 체력바 위에 별도 색으로 표시
                bleed_loss = max(0.0, min(enemy.hp, bleed_dmg))
                loss_w = int(e_bar.w * (bleed_loss / max(1.0, enemy.max_hp)))
                if loss_w > 0:
                    loss_w = min(loss_w, hp_w)
                    loss_x = e_bar.x + hp_w - loss_w
                    pygame.draw.rect(
                        self.screen,
                        (182, 82, 112),
                        (loss_x, e_bar.y, loss_w, e_bar.h),
                        border_radius=4,
                    )

            hp_num = f"{enemy.hp:.0f}/{enemy.max_hp:.0f}"
            hp_num_surf = self.tiny.render(hp_num, True, TEXT)
            self.screen.blit(hp_num_surf, (rect.centerx - hp_num_surf.get_width() // 2, e_bar.bottom + 2))

            status_y = e_bar.bottom + 20

            shield_now = max(0.0, float(enemy.state.shield))
            if shield_now > 0.0:
                s_ratio = max(0.0, min(1.0, shield_now / max(1.0, enemy.max_hp)))
                s_bar = pygame.Rect(e_bar.x, status_y, e_bar.w, 7)
                pygame.draw.rect(self.screen, (24, 44, 66), s_bar, border_radius=4)
                pygame.draw.rect(self.screen, (122, 214, 255), (s_bar.x, s_bar.y, int(s_bar.w * s_ratio), s_bar.h), border_radius=4)
                self.screen.blit(self.tiny.render(f"보호막 {shield_now:.1f}", True, MUTED), (s_bar.x, s_bar.y + 9))
                status_y += 24

            chip_x = e_bar.x
            if bleed_turns > 0:
                bcx = chip_x + 7
                bcy = status_y + 8
                pygame.draw.circle(self.screen, (220, 72, 72), (bcx, bcy), 8)
                bleed_num = self.tiny.render(str(bleed_turns), True, (255, 255, 255))
                bleed_num_rect = bleed_num.get_rect(center=(bcx, bcy))
                self.screen.blit(bleed_num, bleed_num_rect.topleft)
                chip_x += 22

            if frozen_turns > 0:
                freeze_txt = f"{frozen_turns}"
                freeze_w = max(56, self.tiny.size(freeze_txt)[0] + 10)
                freeze_rect = pygame.Rect(chip_x, status_y, freeze_w, 16)
                pygame.draw.rect(self.screen, (98, 188, 255), freeze_rect, border_radius=5)
                self.screen.blit(self.tiny.render(freeze_txt, True, (255, 255, 255)), (freeze_rect.x + 5, freeze_rect.y + 1))

        # 19번 칸 제거 연출과 동일한 톤: 좌우 흔들림 + 페이드 + 붉은 틴트
        now_ms = float(pygame.time.get_ticks())
        alive_remove_fx: List[Dict[str, Any]] = []
        for fx in self.enemy_remove_effects:
            duration = max(1.0, float(fx.get("duration_ms", ENEMY_REMOVE_EFFECT_MS)))
            elapsed = now_ms - float(fx.get("start_ms", now_ms))
            if elapsed < 0.0 or elapsed > duration:
                continue
            alive_remove_fx.append(fx)

            p = max(0.0, min(1.0, elapsed / duration))
            alpha = int(255 * (1.0 - p))
            size = max(24, int(float(fx.get("size", 88.0))))
            cx = float(fx.get("cx", 0.0))
            cy = float(fx.get("cy", 0.0))
            phase = float(fx.get("phase", 0.0))
            shake_x = math.sin((now_ms * 0.06) + phase) * (8.0 * (1.0 - p))

            draw_rect = pygame.Rect(0, 0, size, size)
            draw_rect.center = (int(round(cx + shake_x)), int(round(cy)))

            enemy_kind = str(fx.get("kind", "wolf"))
            enemy_img = self._scaled_unit_image(enemy_kind, (draw_rect.w, draw_rect.h))
            if enemy_img is not None:
                tmp = enemy_img.copy()
                # 이미지에 붉은 기운을 얹어 "처치" 느낌을 만든다.
                tint_p = 0.30 + (0.70 * p)
                add = pygame.Surface((draw_rect.w, draw_rect.h), pygame.SRCALPHA)
                add.fill((int(92 * tint_p), int(26 * tint_p), int(18 * tint_p), 0))
                tmp.blit(add, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
                sub = pygame.Surface((draw_rect.w, draw_rect.h), pygame.SRCALPHA)
                sub.fill((0, int(28 * tint_p), int(46 * tint_p), 0))
                tmp.blit(sub, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
                tmp.set_alpha(alpha)
                self.screen.blit(tmp, draw_rect.topleft)
            else:
                ghost = pygame.Surface((draw_rect.w, draw_rect.h), pygame.SRCALPHA)
                pygame.draw.rect(ghost, (255, 122, 122, alpha), ghost.get_rect(), border_radius=8)
                self.screen.blit(ghost, draw_rect.topleft)

        self.enemy_remove_effects = alive_remove_fx

    def _draw_cards(self) -> None:
        base_rects = self._display_card_rects()
        n = len(base_rects)
        if len(self.card_hover_anim) != n:
            old = self.card_hover_anim[:]
            self.card_hover_anim = [old[i] if i < len(old) else 0.0 for i in range(n)]

        hovered_idx = self._hovered_card_index()
        for i in range(n):
            target = 1.0 if hovered_idx == i else 0.0
            # ease-out: 초반 빠르게, 후반 천천히 수렴
            self.card_hover_anim[i] += (target - self.card_hover_anim[i]) * 0.28
            if abs(target - self.card_hover_anim[i]) < 0.002:
                self.card_hover_anim[i] = target

        rects: List[pygame.Rect] = []
        for i, r in enumerate(base_rects):
            t = max(0.0, min(1.0, self.card_hover_anim[i]))
            scale = 1.0 + 0.12 * t
            w = max(1, int(r.w * scale))
            h = max(1, int(r.h * scale))
            ar = pygame.Rect(0, 0, w, h)
            ar.center = r.center
            ar.y -= int(10 * t)
            rects.append(ar)

        selected_idx = self.selected_card_index
        cancel_rect = self._selected_card_cancel_rect(rects)
        draw_order = [i for i in range(len(rects)) if i != hovered_idx]
        if hovered_idx is not None and 0 <= hovered_idx < len(rects):
            draw_order.append(hovered_idx)

        for i in draw_order:
            rect = rects[i]
            card = self.rt.hand[i]
            if selected_idx == i:
                glow = rect.inflate(14, 14)
                pygame.draw.rect(self.screen, (255, 225, 120), glow, width=3, border_radius=14)
            self._draw_card_visual(card, rect)
            pygame.draw.rect(self.screen, BORDER, rect, width=2, border_radius=10)

            badge = pygame.Rect(rect.x + 8, rect.y + 8, 30, 22)
            pygame.draw.rect(self.screen, (12, 16, 24), badge, border_radius=6)
            pygame.draw.rect(self.screen, BORDER, badge, width=1, border_radius=6)
            self.screen.blit(self.tiny.render(str(i + 1), True, TEXT), (badge.x + 10, badge.y + 4))

        if cancel_rect is not None:
            pygame.draw.rect(self.screen, (70, 32, 32), cancel_rect, border_radius=6)
            pygame.draw.rect(self.screen, (230, 130, 130), cancel_rect, width=1, border_radius=6)
            self._draw_rect_x(cancel_rect, (255, 255, 255))

        # 카드 사용 시: 위로 올라가며 사라지는 애니메이션
        next_anims: List[Dict[str, Any]] = []
        for anim in self.card_use_anims:
            t = float(anim.get("t", 0.0))
            t += 0.10
            if t >= 1.0:
                continue
            p = max(0.0, min(1.0, t))
            ease = 1.0 - ((1.0 - p) * (1.0 - p))  # ease-out
            base: pygame.Rect = anim["rect"]
            scale = 1.0 + 0.10 * ease
            w = max(1, int(base.w * scale))
            h = max(1, int(base.h * scale))
            r = pygame.Rect(0, 0, w, h)
            r.centerx = base.centerx
            r.centery = base.centery - int(120 * ease)
            alpha = int(255 * (1.0 - p))
            self._draw_card_visual_alpha(anim["card"], r, alpha)
            anim["t"] = t
            next_anims.append(anim)
        self.card_use_anims = next_anims

    def _draw_logs(self) -> None:
        button = self._battle_log_button_rect()
        pygame.draw.rect(self.screen, PANEL, button, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, button, width=2, border_radius=8)
        self.screen.blit(self.tiny.render("전투 로그", True, TEXT), (button.x + 18, button.y + 9))

        sel_idx = self.selected_card_index
        hovered_idx = self._hovered_card_index()
        info_idx: Optional[int] = None
        if sel_idx is not None and 0 <= sel_idx < len(self.rt.hand):
            info_idx = sel_idx
        elif hovered_idx is not None and 0 <= hovered_idx < len(self.rt.hand):
            info_idx = hovered_idx

        if info_idx is not None and not self.battle_log_expanded:
            card = self.rt.hand[info_idx]
            panel_x = button.right + 10
            detail_parts = [p.strip() for p in describe_card(card, self.player).split(" / ") if p.strip()]
            if not detail_parts:
                detail_parts = [describe_card(card, self.player)]

            max_lines = 4
            lines = detail_parts[:max_lines]
            line_h = 18
            title_h = 18
            content_h = max(1, len(lines)) * line_h
            panel_h = 8 + title_h + 6 + content_h + 8

            max_panel_w = min(620, self.width - panel_x - 18)
            min_panel_w = 240
            text_w = self.tiny.size(card.name)[0]
            for part in lines:
                text_w = max(text_w, self.tiny.size(part)[0])
            panel_w = max(min_panel_w, min(max_panel_w, text_w + 24))
            panel = pygame.Rect(panel_x, button.y - 4, panel_w, panel_h)
            pygame.draw.rect(self.screen, PANEL, panel, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, panel, width=2, border_radius=8)
            self.screen.blit(
                self.tiny.render(card.name, True, _tier_color(_card_tier_index(card))),
                (panel.x + 10, panel.y + 8),
            )

            yy = panel.y + 8 + title_h + 6
            for part in lines:
                self._draw_wrapped(
                    part,
                    pygame.Rect(panel.x + 10, yy, panel.w - 18, 18),
                    self.tiny,
                    MUTED,
                    max_lines=1,
                )
                yy += 18

        if not self.battle_log_expanded:
            return

        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((8, 10, 14, 170))
        self.screen.blit(overlay, (0, 0))

        panel = self._battle_log_expanded_rect()
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, panel, width=2, border_radius=12)
        self.screen.blit(self.font.render("전투 로그 전체", True, TEXT), (panel.x + 14, panel.y + 10))

        close_rect = pygame.Rect(panel.right - 92, panel.y + 10, 76, 28)
        pygame.draw.rect(self.screen, (56, 64, 84), close_rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, close_rect, width=1, border_radius=8)
        self.screen.blit(self.tiny.render("닫기", True, TEXT), (close_rect.x + 26, close_rect.y + 7))

        content = pygame.Rect(panel.x + 14, panel.y + 48, panel.w - 28, panel.h - 62)
        pygame.draw.rect(self.screen, (22, 28, 39), content, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, content, width=1, border_radius=8)
        line_h = self.tiny.get_linesize() + 4
        self.battle_log_lines_per_page = max(1, content.h // line_h)
        self._clamp_battle_log_scroll()
        start = self.battle_log_scroll
        end = min(len(self.logs), start + self.battle_log_lines_per_page)

        yy = content.y
        for line in self.logs[start:end]:
            self.screen.blit(self.tiny.render(line, True, MUTED), (content.x, yy))
            yy += line_h

        if len(self.logs) > self.battle_log_lines_per_page:
            pos_txt = f"{start + 1}-{end} / {len(self.logs)}"
            self.screen.blit(self.tiny.render(pos_txt, True, MUTED), (panel.right - 160, panel.bottom - 22))

    def _draw_overlay(self) -> None:
        if self.pending_attack is not None:
            msg = f"공격 대상 선택: {self.pending_attack.card.name} ({self.pending_attack.hit_index}/{self.pending_attack.hit_count})"
            left_zone, right_zone, y_center = self._battlefield_bounds()
            mid_x = (left_zone.right + right_zone.left) // 2
            msg_surf = self.small.render(msg, True, ENEMY_HL)
            bg = pygame.Rect(0, 0, msg_surf.get_width() + 18, msg_surf.get_height() + 10)
            bg.center = (mid_x, y_center - 120)
            pygame.draw.rect(self.screen, (24, 30, 44), bg, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, bg, width=1, border_radius=8)
            self.screen.blit(msg_surf, (bg.x + 9, bg.y + 5))

        self._draw_unit_tooltip()

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((10, 12, 16, 165))
            self.screen.blit(overlay, (0, 0))
            title = "승리!" if self.win else "패배"
            color = GREEN if self.win else RED
            t = self.font.render(title, True, color)
            self.screen.blit(t, (self.width // 2 - t.get_width() // 2, self.height // 2 - 40))
            self.screen.blit(self.small.render("R 키로 재시작", True, TEXT), (self.width // 2 - 66, self.height // 2 + 4))


def main() -> None:
    ui = CombatUI()
    ui.run()


def run_meet_combat(
    enemy_count: int,
    enemy_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
    start_enhance_rolls: int = 10,
    initial_draw: float = 4.0,
    min_enhance_tier: int = 0,
    min_enhance_tier_rolls: int = 0,
    width: int = 1280,
    height: int = 800,
) -> Optional[bool]:
    """Meet 전투를 1회 실행하고 결과를 반환한다.

    Returns:
        True: 플레이어 승리
        False: 플레이어 패배
        None: 전투 화면이 닫혀 결과가 확정되지 않음
    """
    ui = CombatUI(
        width=width,
        height=height,
        seed=seed,
        encounter_enemy_count=max(1, int(enemy_count)),
        encounter_enemy_types=list(enemy_types or []),
        forced_start_enhance_rolls=max(0, int(start_enhance_rolls)),
        forced_initial_draw=max(0.0, float(initial_draw)),
        forced_min_enhance_tier=max(0, int(min_enhance_tier)),
        forced_min_enhance_tier_rolls=max(0, int(min_enhance_tier_rolls)),
    )
    return ui.run(quit_on_exit=False, stop_on_game_over=True)


def run_enhancement_choices(
    enhance_rolls: int,
    min_enhance_tier: int = 0,
    min_enhance_tier_rolls: int = 0,
    seed: Optional[int] = None,
    width: int = 1280,
    height: int = 800,
) -> List[str]:
    """강화 선택 페이즈만 실행하고, 덱 요약 라인을 반환한다."""
    ui = CombatUI(
        width=width,
        height=height,
        seed=seed,
        encounter_enemy_count=1,
        forced_start_enhance_rolls=max(0, int(enhance_rolls)),
        forced_min_enhance_tier=max(0, int(min_enhance_tier)),
        forced_min_enhance_tier_rolls=max(0, int(min_enhance_tier_rolls)),
    )
    ui.run(quit_on_exit=False, stop_on_enhance_complete=True)
    lines: List[str] = []
    if ui.enh_templates:
        lines.append(f"총 {len(ui.enh_templates) * 5}장")
        for card in ui.enh_templates:
            lines.append(f"{card.name} x5 | {describe_card(card, ui.player)}")
    return lines


if __name__ == "__main__":
    main()
