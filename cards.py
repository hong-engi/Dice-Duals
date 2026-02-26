# cards.py - Card state and EnhancementEngine (richer implementation)
from __future__ import annotations

import json
import os
import random
import math
import re
import copy
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, List, Tuple


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

TIER_LABELS_BY_NAME = {t.name: TIERS_NAME[t] for t in Tier}


DEFAULT_SHARED_VISUAL_CONFIG: Dict[str, Any] = {
    "frame_path": "images/card_images/card_frame.png",
    "image_anchor": "center",
    "image_box": {"x": 447, "y": 584, "w": 632, "h": 911},
    "name_anchor": "center",
    "name_box": {"x": 443, "y": 973, "w": 414, "h": 59},
    "name_color": "#FFFFFF",
    "name_outline_color": "#000000",
    "name_outline_width": 2,
    "font_path": "fonts/KERISKEDU_B.ttf",
    "name_min_font_size": 18,
    "name_max_font_size": 50,
    "tier_badge": {
        "enabled": True,
        "anchor": "center",
        "x": 445,
        "y": 1090,
        "size": 180,
        "path_template_unique": "images/tier_icons/t6_{option_id}.png",
        "path_template": "images/tier_icons/t{tier}.png",
    },
}


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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

    def attack_ratio_value(self) -> float:
        base = normalize_ratio_value(
            self.stats.get("attack_ratio", self.stats.get("base_attack", self.stats.get("attack", 1.0))),
            1.0,
        )
        bonus = safe_float(self.stats.get("attack_bonus_ratio", 0.0))
        return max(0.0, base * (1.0 + bonus))

    def defense_ratio_value(self) -> float:
        base = normalize_ratio_value(
            self.stats.get("defense_ratio", self.stats.get("base_shield", self.stats.get("shield", 1.0))),
            1.0,
        )
        bonus = safe_float(self.stats.get("shield_bonus_ratio", 0.0))
        return max(0.0, base * (1.0 + bonus))

    def compute_attack_damage(self, player_attack: float) -> float:
        dmg = max(0.0, safe_float(player_attack, 0.0)) * self.attack_ratio_value()
        if bool(self.effects.get("triple_attack", False)):
            dmg *= 3.0
        return dmg

    def compute_shield_amount(self, player_defense: float) -> float:
        return max(0.0, safe_float(player_defense, 0.0)) * self.defense_ratio_value()

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

    @staticmethod
    def _resolve_asset_path(path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (Path(__file__).resolve().parent / p).resolve()

    def _max_enhancement_tier_index(self) -> int:
        max_idx = -1
        for tier_name, cnt in (self.enhancement_counts or {}).items():
            if cnt and tier_name in Tier.__members__:
                max_idx = max(max_idx, Tier[tier_name].value)
        for rec in (self.enhancement_log or []):
            if not isinstance(rec, dict):
                continue
            tier_name = rec.get("tier") or rec.get("tier_used")
            if isinstance(tier_name, str) and tier_name in Tier.__members__:
                max_idx = max(max_idx, Tier[tier_name].value)
        return max_idx if max_idx >= 0 else Tier.COMMON.value

    def _last_unique_option_id(self) -> str:
        for rec in reversed(self.enhancement_log or []):
            if not isinstance(rec, dict):
                continue
            # overflow(rolled=UNIQUE, applied=MYTHICAL)는 UNIQUE 배지 후보에서 제외한다.
            applied_tier = rec.get("tier") or rec.get("tier_used")
            if applied_tier == Tier.UNIQUE.name:
                oid = rec.get("option_id")
                if isinstance(oid, str) and oid.strip():
                    return oid.strip()
        return ""

    def _resolve_tier_badge_path(
        self,
        badge_cfg: Dict[str, Any],
        tier_idx: int,
        unique_option_id: str = "",
    ) -> Path:
        base_dir = Path(__file__).resolve().parent

        template = str(badge_cfg.get("path_template", "images/tier_icons/t{tier}.png"))
        candidates: List[Path] = []

        # UNIQUE(t6)는 option_id별 배지를 우선 사용
        if tier_idx == Tier.UNIQUE.value and unique_option_id:
            unique_template = str(badge_cfg.get("path_template_unique", "images/tier_icons/t6_{option_id}.png"))
            try:
                unique_templated = unique_template.format(
                    tier=tier_idx,
                    t=tier_idx,
                    option_id=unique_option_id,
                )
            except Exception:
                unique_templated = unique_template
            utp = Path(unique_templated)
            candidates.append(utp if utp.is_absolute() else (base_dir / utp))

        # 1) 설정된 템플릿 경로
        try:
            templated = template.format(tier=tier_idx, t=tier_idx)
        except Exception:
            templated = template
        tp = Path(templated)
        candidates.append(tp if tp.is_absolute() else (base_dir / tp))

        # 2) 일반 fallback (png/jpg)
        candidates.append(base_dir / "images" / "tier_icons" / f"t{tier_idx}.png")
        candidates.append(base_dir / "images" / "tier_icons" / f"t{tier_idx}.jpg")

        # UNIQUE(t6) 기본 fallback 지정
        if tier_idx == Tier.UNIQUE.value:
            candidates.append(base_dir / "images" / "tier_icons" / "t6_atk_300.png")

        # 3) UNIQUE(t6) 등 변형 파일 fallback: t6_*.png/jpg
        candidates.extend(sorted((base_dir / "images" / "tier_icons").glob(f"t{tier_idx}*.png")))
        candidates.extend(sorted((base_dir / "images" / "tier_icons").glob(f"t{tier_idx}*.jpg")))

        for c in candidates:
            if c.exists():
                return c.resolve()
        raise FileNotFoundError(
            f"티어 배지 이미지를 찾을 수 없습니다: tier={tier_idx}, "
            f"template={template} (예: images/tier_icons/t{tier_idx}.png)"
        )

    @staticmethod
    def _load_font(size: int, font_path: Optional[str] = None):
        from PIL import ImageFont

        base_dir = Path(__file__).resolve().parent
        candidates: List[str] = []
        if font_path:
            fp = Path(font_path)
            if fp.is_absolute():
                candidates.append(str(fp))
            else:
                candidates.append(str((base_dir / fp).resolve()))
                candidates.append(str(fp))
        candidates.append(str((base_dir / "fonts" / "KERISKEDU_B.ttf").resolve()))
        candidates.extend(
            [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/AppleGothic.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:/Windows/Fonts/malgun.ttf",
            ]
        )
        for c in candidates:
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _fit_font(draw: Any, text: str, max_w: int, max_h: int, min_size: int, max_size: int, font_path: Optional[str]):
        best = CardState._load_font(min_size, font_path)
        lo, hi = min_size, max_size
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = CardState._load_font(mid, font_path)
            bbox = draw.textbbox((0, 0), text, font=cand)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= max_w and h <= max_h:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def render_visual(
        self,
        engine: "EnhancementEngine",
        image_path: Optional[str] = None,
        card_name: Optional[str] = None,
    ):
        """card_definitions.json visual 설정으로 카드 이미지를 렌더링해 PIL Image를 반환한다."""
        try:
            from PIL import Image, ImageDraw
        except Exception as e:
            raise RuntimeError("Pillow가 필요합니다. `pip install pillow` 후 다시 시도하세요.") from e

        visual_cfg = engine.get_visual_config(self.type)
        if not visual_cfg:
            raise ValueError(f"visual 설정이 없습니다: card_type={self.type}")

        frame_path = visual_cfg.get("frame_path")
        art_path = image_path or visual_cfg.get("image_path")
        image_box = visual_cfg.get("image_box", {})
        name_box = visual_cfg.get("name_box", {})
        if not frame_path or not art_path:
            raise ValueError("visual.frame_path / visual.image_path 설정이 필요합니다.")

        def _to_topleft(box: Dict[str, Any], default_anchor: str) -> Tuple[int, int, int, int]:
            x = int(box.get("x", 0))
            y = int(box.get("y", 0))
            w = int(box.get("w", 0))
            h = int(box.get("h", 0))
            anchor = str(box.get("anchor", default_anchor)).strip().lower()
            if anchor in ("center", "centre", "c"):
                x = x - (w // 2)
                y = y - (h // 2)
            elif anchor in ("topleft", "top_left", "left_top", "lt"):
                pass
            else:
                raise ValueError(f"지원하지 않는 anchor: {anchor}")
            return x, y, w, h

        image_anchor = str(visual_cfg.get("image_anchor", "topleft"))
        name_anchor = str(visual_cfg.get("name_anchor", "topleft"))
        ix, iy, iw, ih = _to_topleft(image_box, image_anchor)
        nx, ny, nw, nh = _to_topleft(name_box, name_anchor)
        if iw <= 0 or ih <= 0 or nw <= 0 or nh <= 0:
            raise ValueError("visual.image_box / visual.name_box의 크기(w,h)는 1 이상이어야 합니다.")

        frame_abs = self._resolve_asset_path(str(frame_path))
        art_abs = self._resolve_asset_path(str(art_path))
        if not frame_abs.exists():
            raise FileNotFoundError(f"frame 이미지가 없습니다: {frame_abs}")
        if not art_abs.exists():
            raise FileNotFoundError(f"card 이미지가 없습니다: {art_abs}")

        frame = Image.open(frame_abs).convert("RGBA")
        art = Image.open(art_abs).convert("RGBA").resize((iw, ih), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        canvas.alpha_composite(art, (ix, iy))
        canvas.alpha_composite(frame, (0, 0))

        draw = ImageDraw.Draw(canvas)
        default_name = engine.get_card_name(self.type)
        text = (card_name or default_name or self.name or "").strip()
        min_font = int(visual_cfg.get("name_min_font_size", 18))
        max_font = int(visual_cfg.get("name_max_font_size", 56))
        font_path = visual_cfg.get("font_path")
        font = self._fit_font(draw, text, nw, nh, min_font, max_font, font_path)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = nx + (nw - tw) // 2
        ty = ny + (nh - th) // 2

        fill = visual_cfg.get("name_color", "#FFFFFF")
        stroke_fill = visual_cfg.get("name_outline_color", "#000000")
        stroke_width = int(visual_cfg.get("name_outline_width", 2))
        draw.text((tx, ty), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

        badge_cfg = visual_cfg.get("tier_badge", {})
        if isinstance(badge_cfg, dict) and badge_cfg.get("enabled", True):
            tier_idx = self._max_enhancement_tier_index()
            bx = int(badge_cfg.get("x", 0))
            by = int(badge_cfg.get("y", 0))
            bsize = int(badge_cfg.get("size", 130))
            bw = int(badge_cfg.get("w", bsize))
            bh = int(badge_cfg.get("h", bsize))
            if bw > 0 and bh > 0:
                anchor = str(badge_cfg.get("anchor", "center")).strip().lower()
                if anchor in ("center", "centre", "c"):
                    bx = bx - (bw // 2)
                    by = by - (bh // 2)
                elif anchor not in ("topleft", "top_left", "left_top", "lt"):
                    raise ValueError(f"지원하지 않는 tier_badge.anchor: {anchor}")
                unique_option_id = self._last_unique_option_id() if tier_idx == Tier.UNIQUE.value else ""
                badge_path = self._resolve_tier_badge_path(
                    badge_cfg,
                    tier_idx,
                    unique_option_id=unique_option_id,
                )
                badge = Image.open(badge_path).convert("RGBA").resize((bw, bh), Image.Resampling.LANCZOS)
                canvas.alpha_composite(badge, (bx, by))

        return canvas

    def visualize(
        self,
        engine: "EnhancementEngine",
        output_path: Optional[str] = None,
        image_path: Optional[str] = None,
        card_name: Optional[str] = None,
    ) -> str:
        """카드 이미지를 렌더링 후 파일로 저장하고 경로를 반환한다."""
        frame = self.render_visual(engine=engine, image_path=image_path, card_name=card_name)
        if output_path is None:
            output_path = str((Path(__file__).resolve().parent / "images" / "card_images" / f"preview_{self.id}.png").resolve())
        out_abs = self._resolve_asset_path(output_path)
        out_abs.parent.mkdir(parents=True, exist_ok=True)
        frame.save(out_abs)
        return str(out_abs)

    def show_visual(
        self,
        engine: "EnhancementEngine",
        image_path: Optional[str] = None,
        card_name: Optional[str] = None,
    ) -> None:
        """카드 이미지를 렌더링해 기본 이미지 뷰어로 표시한다."""
        frame = self.render_visual(engine=engine, image_path=image_path, card_name=card_name)
        frame.show()


# small helpers used by engine (kept local to avoid depending on main.py)
def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def normalize_ratio_value(v: Any, default: float = 1.0) -> float:
    """레거시 절대값(예: 100)을 비율(1.0)로 정규화한다."""
    raw = safe_float(v, default)
    if raw < 0:
        return 0.0
    if raw > 5.0:
        return raw / 100.0
    return raw


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


def fmt_amount_1(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def _extract_player_attack(player: Any, default: float = 100.0) -> float:
    if player is None:
        return default
    attack_obj = getattr(player, "attack", None)
    if isinstance(attack_obj, (int, float)):
        return max(0.0, safe_float(attack_obj, default))
    if attack_obj is not None:
        power = getattr(attack_obj, "power", None)
        if power is not None:
            return max(0.0, safe_float(power, default))
    stats = getattr(player, "stats", None)
    if isinstance(stats, dict):
        return max(0.0, safe_float(stats.get("attack", default), default))
    return default


def _extract_player_defense(player: Any, default: float = 100.0) -> float:
    if player is None:
        return default
    defense_obj = getattr(player, "defense", None)
    if isinstance(defense_obj, (int, float)):
        return max(0.0, safe_float(defense_obj, default))
    if defense_obj is not None:
        defense_power = getattr(defense_obj, "defense_power", None)
        if defense_power is not None:
            return max(0.0, safe_float(defense_power, default))
        armor = getattr(defense_obj, "armor", None)
        if armor is not None:
            return max(0.0, safe_float(armor, default))
    stats = getattr(player, "stats", None)
    if isinstance(stats, dict):
        return max(0.0, safe_float(stats.get("defense", default), default))
    return default


class EnhancementEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.card_types_cfg: Dict[str, Any] = self._normalize_card_types_cfg(self.cfg)
        self.tiers_cfg: Dict[str, Any] = self._tiers_cfg_for_card("attack")
        self.tier_names = list(self.tiers_cfg.keys())
        for card_type, c in self.card_types_cfg.items():
            ct_tiers = c.get("tiers", {})
            if list(ct_tiers.keys()) != self.tier_names:
                raise ValueError(
                    f"tier 순서/구성이 attack과 다릅니다: card_type={card_type}"
                )
        proc_cfg = self.cfg.get("proc_rates", {})
        self.proc_efficiency_base = float(proc_cfg.get("efficiency_base", 0.10))
        self.proc_efficiency_with_higher = float(proc_cfg.get("efficiency_with_higher", 0.50))
        self.proc_lower_base = float(proc_cfg.get("lower_base", 0.10))
        self.proc_lower_with_higher = float(proc_cfg.get("lower_with_higher", 0.40))
        # gap(0~6) 기반 확률 테이블
        self.proc_efficiency_gap_rates = [0.15, 0.40, 0.70,1,1,1,1]
        self.proc_lower_gap_rates = [0.15, 0.30, 0.70,1,1,1,1]
        self.proc_efficiency_double_gap_rates = [0.05, 0.20, 0.50, 1.00, 1.00, 1.00, 1.00]
        self.proc_same_tier_bonus_gap_rates = [0.05, 0.15, 0.50, 1.00, 1.00, 1.00, 1.00]
        self.value_caps = self._load_value_caps(self.cfg.get("value_caps", {}))

    @staticmethod
    def _default_value_caps() -> Dict[str, Tuple[float, float, bool]]:
        return {
            "attack": (math.inf, 0.0, False),
            "range": (3.0, 0.0, True),
            "freeze_turns": (1.0, 1.0, False),
            "shield_ratio": (0.40, 0.60, False),
            "bleed_ratio": (1.50, 1.50, False),
            "armor_down_ratio": (0.40, 0.35, False),
            "draw_cards": (1.0, 1.0, False),
            "splash_other_ratio": (1.0, 0.0, False),
            "temp_armor_ratio": (1.0, 0.5, False),
            "on_hit_freeze_turns": (1.0, 1.0, False),
            "next_attack_bonus_ratio": (1.0, 2.0, False),
            "on_hit_bleed_ratio": (1.0, 2.0, False),
        }

    @staticmethod
    def _parse_cap_number(v: Any) -> float:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("inf", "+inf", "infinity", "+infinity"):
                return math.inf
        return float(v)

    def _load_value_caps(self, cfg_caps: Any) -> Dict[str, Tuple[float, float, bool]]:
        caps = dict(self._default_value_caps())
        if not isinstance(cfg_caps, dict):
            return caps

        for key, spec in cfg_caps.items():
            try:
                if isinstance(spec, dict):
                    a = self._parse_cap_number(spec.get("soft_cap", math.inf))
                    b = self._parse_cap_number(spec.get("breakthrough", 0.0))
                    is_int = bool(spec.get("is_int", False))
                elif isinstance(spec, (list, tuple)) and len(spec) >= 3:
                    a = self._parse_cap_number(spec[0])
                    b = self._parse_cap_number(spec[1])
                    is_int = bool(spec[2])
                else:
                    continue
                caps[str(key)] = (float(a), float(b), is_int)
            except Exception:
                continue
        return caps

    @staticmethod
    def _normalize_card_types_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """신규 스키마만 허용: {\"attack\": {\"tiers\": ...}, \"defense\": {\"tiers\": ...}}."""
        if "tiers" in cfg or "card_types" in cfg:
            raise ValueError(
                "구 스키마는 지원하지 않습니다. "
                "card_definitions.json은 {'attack': {'tiers': ...}, 'defense': {'tiers': ...}} 형식이어야 합니다."
            )

        direct = {
            k: v
            for k, v in cfg.items()
            if isinstance(v, dict) and isinstance(v.get("tiers"), dict) and v.get("tiers")
        }
        if "attack" not in direct or "defense" not in direct:
            raise ValueError("card_definitions.json에 attack/defense tiers가 필요합니다.")
        return direct

    def _tiers_cfg_for_card(self, card_type: str) -> Dict[str, Any]:
        """카드 타입에 해당하는 tiers를 반환(신스키마 전용)."""
        c = self.card_types_cfg.get(card_type)
        if not isinstance(c, dict):
            raise ValueError(f"알 수 없는 카드 타입: {card_type}")
        tiers = c.get("tiers")
        if not isinstance(tiers, dict) or not tiers:
            raise ValueError(f"tiers가 비어있습니다: card_type={card_type}")
        return tiers

    @staticmethod
    def _options_for_tier(tier_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """티어 설정의 옵션 목록(신스키마: options만 사용)."""
        options = tier_cfg.get("options", [])
        if not isinstance(options, list):
            return []
        return list(options)

    def get_visual_config(self, card_type: str) -> Dict[str, Any]:
        c = self.card_types_cfg.get(card_type)
        if not isinstance(c, dict):
            return copy.deepcopy(DEFAULT_SHARED_VISUAL_CONFIG)
        visual_override = c.get("visual", {})
        if not isinstance(visual_override, dict):
            visual_override = {}
        return deep_merge_dict(DEFAULT_SHARED_VISUAL_CONFIG, visual_override)

    def get_card_name(self, card_type: str) -> str:
        c = self.card_types_cfg.get(card_type)
        if not isinstance(c, dict):
            return ""
        name = c.get("card_name", "")
        return str(name).strip() if name is not None else ""

    def get_proc_rates(
        self, tier: str, higher_tier_gap: int = 0, higher_tier_exists: bool = False
    ) -> Tuple[float, float, float, float]:
        """현재 티어/상위 차이를 반영한 (eff_rate, lower_rate, eff2_rate, same_tier_rate) 반환."""
        try:
            tier_idx = self.tier_names.index(tier)
        except ValueError:
            tier_idx = -1

        gap = max(0, int(higher_tier_gap))
        if gap <= 0 and higher_tier_exists:
            gap = 1
        max_gap_idx = len(self.proc_efficiency_gap_rates) - 1
        gap_idx = min(gap, max_gap_idx)

        eff_rate = self.proc_efficiency_gap_rates[gap_idx]
        lower_rate = self.proc_lower_gap_rates[gap_idx]
        eff2_rate = self.proc_efficiency_double_gap_rates[gap_idx]
        same_tier_rate = self.proc_same_tier_bonus_gap_rates[gap_idx]

        # t4, t5, t6에서는 효율증폭 비활성화
        if tier_idx >= 4:
            eff_rate = 0.0
        # t6(UNIQUE)에서는 하위 보너스 비활성화
        if tier_idx == len(self.tier_names) - 1:
            lower_rate = 0.0

        return eff_rate, lower_rate, eff2_rate, same_tier_rate

    def _apply_with_caps(self, key: str, cur: float, delta: float) -> float:
        a, b, cap_is_int = self.value_caps.get(key, (math.inf, 0.0, False))

        # no cap
        if math.isinf(a) or (a is None):
            out = cur + delta
            return int(round(out)) if cap_is_int else out

        if delta <= 0:
            return int(round(cur)) if cap_is_int else cur

        out = cur
        leftover = delta

        # 1) soft cap(a)까지는 선형 증가
        if out < a:
            linear_inc = min(leftover, a - out)
            out += linear_inc
            leftover -= linear_inc

        # 2) soft cap 이후는 감쇠 증가
        #    b/2에 도달하려면 raw 증가량 b 필요,
        #    그 다음 b/4도 다시 raw 증가량 b 필요 ...
        #    => raw를 무한히 써야 a+b에 도달(이론상 상한)
        # over_eff(raw) = b * (1 - 2^(-raw / b))
        if leftover > 1e-12 and b > 0:
            over_eff_cur = max(0.0, out - a)
            if over_eff_cur < b:
                remain_ratio = 1.0 - (over_eff_cur / b)  # (0, 1]
                u_cur = -math.log2(max(1e-12, remain_ratio))
                u_new = u_cur + (leftover / b)
                over_eff_new = b * (1.0 - (2.0 ** (-u_new)))
                out = a + over_eff_new

        if cap_is_int:
            return int(round(out))
        return out

    @staticmethod
    def _normalize_effect_name(effect_name: str) -> str:
        # legacy 키 호환: temp_defense -> temp_armor
        if effect_name == "temp_defense":
            return "temp_armor"
        return effect_name

    @staticmethod
    def _value_spec_max(spec: Any) -> Optional[float]:
        if isinstance(spec, (int, float)):
            return float(spec)
        if isinstance(spec, dict) and spec.get("dist") == "uniform":
            try:
                return float(spec.get("max", 0.0))
            except Exception:
                return None
        return None

    def _is_hard_capped_now(self, key: str, cur: float) -> bool:
        a, b, _ = self.value_caps.get(key, (math.inf, 0.0, False))
        if math.isinf(a):
            return False
        return float(b) <= 0.0 and float(cur) >= float(a) - 1e-12

    def _option_blocked_by_hard_cap(self, card: CardState, opt: Dict[str, Any]) -> bool:
        """breakthrough=0이고 softcap 도달한 키를 더 올리려는 옵션은 후보에서 제외한다."""
        for eff in opt.get("effects", []):
            et = eff.get("type")
            raw_val = eff.get("value")
            vmax = self._value_spec_max(raw_val)

            if et == "stat.add":
                stat = str(eff.get("stat", ""))
                cur = safe_float(card.stats.get(stat, 0.0), 0.0)
                if self._is_hard_capped_now(stat, cur) and (vmax is None or vmax > 0.0):
                    return True

            elif et in ("stat.mul", "stat.mul_fixed"):
                stat = str(eff.get("stat", ""))
                if stat in ("attack", "shield"):
                    continue
                cur = safe_float(card.stats.get(stat, 0.0), 0.0)
                # 곱연산의 value>0은 증가 시도다.
                if self._is_hard_capped_now(stat, cur) and (vmax is None or vmax > 0.0):
                    return True

            elif et == "effect.additive":
                effect = self._normalize_effect_name(str(eff.get("effect", "")))
                field = str(eff.get("field", ""))
                key = f"{effect}_{field}"
                cur = safe_float(card.effects.get(key, 0.0), 0.0)
                if self._is_hard_capped_now(key, cur) and (vmax is None or vmax > 0.0):
                    return True

            elif et == "effect.max":
                effect = self._normalize_effect_name(str(eff.get("effect", "")))
                field = str(eff.get("field", ""))
                key = f"{effect}_{field}"
                cur = safe_float(card.effects.get(key, 0.0), 0.0)
                if self._is_hard_capped_now(key, cur) and (vmax is None or vmax > cur + 1e-12):
                    return True

        return False

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
            raw_val = eff.get("value")
            ratio = float(self._roll_value(eff["value"], rng))
            if stat == "attack":
                card.stats["attack_bonus_ratio"] = float(card.stats.get("attack_bonus_ratio", 0.0)) + ratio
                return f"{ratio*100:.1f}%"
            if stat == "shield":
                card.stats["shield_bonus_ratio"] = float(card.stats.get("shield_bonus_ratio", 0.0)) + ratio
                return f"{ratio*100:.1f}%"
            else:
                base = float(card.stats.get(stat, 0))
                new_val = base * (1.0 + ratio)
                # compute delta and apply caps
                delta = new_val - base
                nxt = self._apply_with_caps(stat, base, delta)
                card.stats[stat] = nxt
                # 샘플링형 강화(예: 2~3%)는 실제 확정 비율을 UI에 표시한다.
                if isinstance(raw_val, dict):
                    return f"{ratio*100:.1f}%"
                return None

        if et == "stat.mul_fixed":
            stat = eff["stat"]
            ratio = float(self._roll_value(eff.get("value"), rng))
            if stat == "attack":
                card.stats["attack_bonus_ratio"] = float(card.stats.get("attack_bonus_ratio", 0.0)) + ratio
            elif stat == "shield":
                card.stats["shield_bonus_ratio"] = float(card.stats.get("shield_bonus_ratio", 0.0)) + ratio
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
            effect = self._normalize_effect_name(eff["effect"])
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
            effect = self._normalize_effect_name(eff["effect"])
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
            effect = self._normalize_effect_name(eff["effect"])
            field = eff["field"]
            key = f"{effect}_{field}"
            raw_val = eff.get("value")
            r = float(self._roll_value(raw_val, rng))
            old = float(card.effects.get(key, 0.0))
            card.effects[key] = 1.0 - (1.0 - old) * (1.0 - r)
            return fmt_ratio_0(r) if isinstance(raw_val, dict) else None

        if et == "effect.set_if_none":
            effect = self._normalize_effect_name(eff["effect"])
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
        higher_tier_gap: int = 0,
        enable_bonus_procs: bool = True,
        forced_efficiency_proc: Optional[bool] = None,
        forced_efficiency_double_proc: Optional[bool] = None,
        forced_lower_bonus_proc: Optional[bool] = None,
        forced_same_tier_bonus_proc: Optional[bool] = None,
        forced_lower_selected_option: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if rng is None:
            rng = random
        card.ensure_counts(self.tier_names)

        tiers_cfg = self._tiers_cfg_for_card(card.type)
        rolled_tier = tier
        if tier not in tiers_cfg:
            raise ValueError(f"Unknown tier '{tier}' for card_type={card.type}")
        tier_cfg = tiers_cfg[tier]

        if tier_cfg.get("unique_once", False):
            if card.enhancement_counts.get(tier, 0) >= 1:
                overflow = tier_cfg.get("overflow_to", "MYTHICAL")
                tier = overflow
                if tier not in tiers_cfg:
                    raise ValueError(f"Unknown overflow tier '{tier}' for card_type={card.type}")
                tier_cfg = tiers_cfg[tier]

        options = self._options_for_tier(tier_cfg)
        if not options:
            raise ValueError(f"No options for tier={tier}, card_type={card.type}")

        def option_valid(opt: Dict[str, Any]) -> bool:
            if self._option_blocked_by_hard_cap(card, opt):
                return False
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
        if not valid:
            raise ValueError(f"No valid options for tier={tier}, card_type={card.type}")

        # If a specific option was provided (from preview), try to use it
        if selected_option is not None:
            # try to match by id first
            sid = selected_option.get("id") if isinstance(selected_option, dict) else None
            matched = None
            if sid:
                for o in valid:
                    if o.get("id") == sid:
                        matched = o
                        break
            # if no id match, and selected_option looks like an option dict, use it directly
            if matched is None and isinstance(selected_option, dict) and selected_option.get("id") is None:
                opt = selected_option if selected_option in valid else rng.choice(valid)
            else:
                opt = matched if matched is not None else rng.choice(valid)
        else:
            opt = rng.choice(valid)

        roll_parts: List[str] = []

        eff_rate, lower_rate, eff2_rate, same_tier_rate = self.get_proc_rates(
            tier=tier,
            higher_tier_gap=higher_tier_gap,
            higher_tier_exists=higher_tier_exists,
        )

        efficiency_proc = False
        efficiency_double_proc = False
        if forced_efficiency_proc is not None:
            efficiency_proc = bool(forced_efficiency_proc)
        elif enable_bonus_procs:
            has_eligible = any(self._is_efficiency_eligible(eff) for eff in opt.get("effects", []))
            efficiency_proc = has_eligible and (rng.random() < eff_rate)
        if forced_efficiency_double_proc is not None:
            efficiency_double_proc = bool(forced_efficiency_double_proc)
        elif enable_bonus_procs:
            has_eligible = any(self._is_efficiency_eligible(eff) for eff in opt.get("effects", []))
            efficiency_double_proc = has_eligible and (rng.random() < eff2_rate)
        # 우선순위: 효율 x2가 뜨면 x1.5는 덮어씀
        if efficiency_double_proc:
            efficiency_proc = False

        main_effects = []
        total_eff_ratio = 1.0
        if efficiency_proc:
            total_eff_ratio *= 1.5
        if efficiency_double_proc:
            total_eff_ratio *= 2.0
        for eff in opt.get("effects", []):
            main_effects.append(self._boost_effect(eff, total_eff_ratio) if total_eff_ratio > 1.0 else eff)

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
        same_tier_bonus_result: Optional[Dict[str, Any]] = None
        if enable_bonus_procs:
            try:
                tier_idx = self.tier_names.index(tier)
            except ValueError:
                tier_idx = -1
            lower_bonus_proc = False
            if forced_lower_bonus_proc is not None:
                lower_bonus_proc = (tier_idx > 0) and bool(forced_lower_bonus_proc)
            else:
                lower_bonus_proc = tier_idx > 0 and (rng.random() < lower_rate)
            same_tier_bonus_proc = False
            if forced_same_tier_bonus_proc is not None:
                same_tier_bonus_proc = bool(forced_same_tier_bonus_proc)
            else:
                same_tier_bonus_proc = rng.random() < same_tier_rate
            # 우선순위: 동일 티어 보너스가 뜨면 하위 보너스는 덮어씀
            if same_tier_bonus_proc:
                lower_bonus_proc = False
            if lower_bonus_proc:
                lower_tier = self.tier_names[tier_idx - 1]
                lower_bonus_result = self.apply_tier(
                    card,
                    lower_tier,
                    rng=rng,
                    selected_option=forced_lower_selected_option,
                    higher_tier_exists=False,
                    enable_bonus_procs=False,
                )
            if same_tier_bonus_proc:
                same_tier_bonus_result = self.apply_tier(
                    card,
                    tier,
                    rng=rng,
                    selected_option=None,
                    higher_tier_exists=False,
                    enable_bonus_procs=False,
                )

        def _token_before_boost(token: str, ratio: float) -> str:
            if ratio <= 1.0:
                return token
            m = re.match(r"^\s*(-?\d+(?:\.\d+)?)(\s*(?:%|턴|장)?)\s*$", token)
            if not m:
                return token
            num = float(m.group(1))
            unit = m.group(2)
            raw = num / ratio
            if unit.strip() == "%":
                out = f"{raw:.1f}".rstrip("0").rstrip(".")
            else:
                out = f"{raw:.2f}".rstrip("0").rstrip(".")
            return f"{out}{unit}"

        base_display = opt.get("display", opt.get("id", ""))
        display_applied = base_display
        main_roll_token = next((p for p in roll_parts if isinstance(p, str)), None)
        eff_suffix = ""
        if efficiency_double_proc:
            eff_suffix = " [x2]"
        elif efficiency_proc:
            eff_suffix = " [x1.5]"
        shown_main_roll_token = main_roll_token
        if main_roll_token is not None and total_eff_ratio > 1.0:
            shown_main_roll_token = _token_before_boost(main_roll_token, total_eff_ratio)
        consumed_main_roll = False
        consumed_eff_suffix = False
        if shown_main_roll_token is not None:
            # "2~4%" 같은 범위를 실제 적용 수치로 치환한다.
            range_pat = re.compile(r"\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?(?:%|턴|장)?")
            m = range_pat.search(base_display)
            if m:
                display_applied = base_display[:m.start()] + f"{shown_main_roll_token}{eff_suffix}" + base_display[m.end():]
                consumed_main_roll = True
                consumed_eff_suffix = bool(eff_suffix)
            elif eff_suffix and main_roll_token is not None and main_roll_token in base_display:
                display_applied = base_display.replace(main_roll_token, f"{shown_main_roll_token}{eff_suffix}", 1)
                consumed_eff_suffix = True
        if eff_suffix and not consumed_eff_suffix:
            # 고정 수치 옵션(예: 30%, 0.2턴)도 수치 옆에 배율을 붙인다.
            num_pat = re.compile(r"\d+(?:\.\d+)?\s*(?:%|턴|장)")
            n = num_pat.search(display_applied)
            if n:
                token = n.group(0)
                display_applied = display_applied[:n.start()] + f"{token}{eff_suffix}" + display_applied[n.end():]
            elif " 강화된다" in display_applied:
                display_applied = display_applied.replace(" 강화된다", f"{eff_suffix} 강화된다", 1)
            else:
                display_applied = display_applied + eff_suffix
            consumed_eff_suffix = True

        roll_text = ""
        base_roll_parts = list(roll_parts)
        if consumed_main_roll and main_roll_token is not None:
            removed = False
            kept_parts = []
            for p in base_roll_parts:
                if (not removed) and p == main_roll_token:
                    removed = True
                    continue
                kept_parts.append(p)
            base_roll_parts = kept_parts
        extra_parts = []
        if efficiency_proc and not consumed_eff_suffix:
            extra_parts.append("x1.5")
        if efficiency_double_proc and not consumed_eff_suffix:
            extra_parts.append("x2")
        if lower_bonus_result is not None:
            extra_parts.append(
                f"하위 티어 보너스 :{lower_bonus_result.get('display_applied', lower_bonus_result['display'])}{lower_bonus_result.get('roll_text', '')}"
            )
        if same_tier_bonus_result is not None:
            extra_parts.append(
                f"동일 티어 보너스 :{same_tier_bonus_result.get('display_applied', same_tier_bonus_result['display'])}{same_tier_bonus_result.get('roll_text', '')}"
            )
        all_parts = base_roll_parts + extra_parts
        if all_parts:
            roll_text = " [" + "] [".join(all_parts) + "]"

        return {
            "rolled_tier": rolled_tier,
            "tier_used": tier,
            "option_id": opt.get("id", ""),
            "display": base_display,
            "display_applied": display_applied,
            "roll_text": roll_text,
            "efficiency_proc": efficiency_proc,
            "efficiency_double_proc": efficiency_double_proc,
            "lower_bonus_proc": lower_bonus_result is not None,
            "lower_bonus_result": lower_bonus_result,
            "same_tier_bonus_proc": same_tier_bonus_result is not None,
            "same_tier_bonus_result": same_tier_bonus_result,
        }

    def valid_options_for_tier(self, card: CardState, tier: str) -> List[Dict[str, Any]]:
        """주어진 카드 상태에서 해당 `tier`에 대해 적용 가능한 옵션 리스트 반환(상태 변경 없음)."""
        tiers_cfg = self._tiers_cfg_for_card(card.type)
        if tier not in tiers_cfg:
            return []
        tier_cfg = tiers_cfg[tier]
        options = self._options_for_tier(tier_cfg)

        def option_valid(opt: Dict[str, Any]) -> bool:
            if self._option_blocked_by_hard_cap(card, opt):
                return False
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
        return valid

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


def ensure_json_ext(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    if not name.lower().endswith(".json"):
        name += ".json"
    return name


def _default_base_attack_card(card_no: int) -> Dict[str, Any]:
    return {
        "id": f"atk_{card_no:03d}",
        "name": "기본 공격",
        "type": "attack",
        "stats": {"attack_ratio": 1.0, "range": 1},
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
            "splash_other_ratio": 0.0,
            "summon_unit": None,
        },
    }


def _default_defense_card() -> Dict[str, Any]:
    return {
        "id": "def_001",
        "name": "방어",
        "type": "defense",
        "stats": {"defense_ratio": 1.0},
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
            "splash_other_ratio": 0.0,
            "temp_armor_ratio": 0.0,
            "on_hit_freeze_turns": 0.0,
            "next_attack_bonus_ratio": 0.0,
            "on_hit_bleed_ratio": 0.0,
            "on_hit_bleed_turns": 0,
            "on_hit_draw_cards": 0.0,
            "on_hit_frenzy_ratio": 0.0,
            "summon_unit": None,
        },
    }


def _normalize_card_dict(cd: Dict[str, Any]) -> CardState:
    cid = cd["id"]
    name = cd.get("name", cid)
    ctype = cd.get("type", "attack")

    stats = cd.get("stats")
    if stats is None:
        stats = {"attack": cd.get("attack", 0), "range": cd.get("range", 0)}
    stats = dict(stats)

    effects = dict(cd.get("effects", {}) or {})

    attack_ratio = normalize_ratio_value(stats.get("attack_ratio", stats.get("base_attack", stats.get("attack", 100.0))), 1.0)
    defense_ratio = normalize_ratio_value(
        stats.get("defense_ratio", stats.get("base_shield", stats.get("shield", 100.0))),
        1.0,
    )
    stats.setdefault("attack_bonus_ratio", 0.0)
    stats.setdefault("resource", 0)
    stats.setdefault("range", 1 if ctype == "attack" else 0)

    if ctype == "defense":
        stats["defense_ratio"] = defense_ratio
        stats["base_shield"] = defense_ratio
        stats["shield"] = defense_ratio
        stats.setdefault("attack_ratio", 0.0)
        stats.setdefault("base_attack", 0.0)
        stats.setdefault("attack", 0.0)
    else:
        stats["attack_ratio"] = attack_ratio
        stats["base_attack"] = attack_ratio
        stats["attack"] = attack_ratio
        stats.setdefault("defense_ratio", 0.0)
        stats.setdefault("base_shield", 0.0)
        stats.setdefault("shield", 0.0)
    stats.setdefault("shield_bonus_ratio", 0.0)

    effects.setdefault("ignore_defense", False)
    effects.setdefault("double_hit", False)
    effects.setdefault("aoe_all_enemies", False)
    effects.setdefault("freeze_turns", 0.0)
    effects.setdefault("shield_ratio", 0.0)
    effects.setdefault("bleed_ratio", 0.0)
    effects.setdefault("bleed_turns", 0)
    effects.setdefault("armor_down_ratio", 0.0)
    effects.setdefault("draw_cards", 0.0)
    effects.setdefault("splash_other_ratio", 0.0)
    # legacy 호환: temp_defense_ratio -> temp_armor_ratio
    if "temp_armor_ratio" not in effects and "temp_defense_ratio" in effects:
        effects["temp_armor_ratio"] = effects.get("temp_defense_ratio", 0.0)
    effects.setdefault("temp_armor_ratio", 0.0)
    effects.setdefault("on_hit_freeze_turns", 0.0)
    effects.setdefault("next_attack_bonus_ratio", 0.0)
    effects.setdefault("on_hit_bleed_ratio", 0.0)
    effects.setdefault("on_hit_bleed_turns", 0)
    effects.setdefault("on_hit_draw_cards", 0.0)
    effects.setdefault("on_hit_frenzy_ratio", 0.0)
    effects.setdefault("summon_unit", None)
    effects.setdefault("triple_attack", False)

    return CardState(id=cid, name=name, type=ctype, stats=stats, effects=effects)


def load_cards(path: str = "cards.json") -> List[CardState]:
    if not os.path.exists(path):
        data = {
            "cards": [
                _default_base_attack_card(1),
                _default_defense_card(),
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_cards = data.get("cards", [])
        if not raw_cards and {"id", "stats", "effects"} & set(data.keys()):
            raw_cards = [data]
    elif isinstance(data, list):
        raw_cards = data
    else:
        raw_cards = []

    if not raw_cards:
        raise ValueError("cards.json에 cards가 없습니다.")

    return [_normalize_card_dict(cd) for cd in raw_cards]


def load_first_card(path: str = "cards.json") -> CardState:
    cards = load_cards(path)
    return cards[0]


def describe_card(card: CardState, player: Optional[Any] = None) -> str:
    parts: List[str] = []
    eff = card.effects
    resource = safe_int(card.stats.get("resource", 0))
    player_attack = _extract_player_attack(player, 100.0)
    player_defense = _extract_player_defense(player, 100.0)
    if card.type == "attack":
        atk = card.attack_ratio_value()
        base_damage = max(0.0, safe_float(player_attack, 0.0)) * atk
        total_damage = card.compute_attack_damage(player_attack)
        rngv = safe_int(card.stats.get("range", 0))
        if bool(eff.get("triple_attack", False)):
            parts.append(f"피해를 {fmt_amount_1(total_damage)} 줍니다.")
        else:
            parts.append(f"피해를 {fmt_amount_1(base_damage)} 줍니다.")
        parts.append(f"거리({rngv})")
    elif card.type == "defense":
        shield_amount = card.compute_shield_amount(player_defense)
        parts.append(f"보호막을 {fmt_amount_1(shield_amount)} 얻습니다.")
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
        eff_parts.append("전체 피해가 300% 강화된다")

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
        eff_parts.append(f"출혈(피해의 {fmt_ratio_0(br)}) {bt}턴")

    ar = safe_float(eff.get("armor_down_ratio", 0.0))
    if ar > 0:
        eff_parts.append(f"상대의 방어력을 {fmt_ratio_0(ar)} 감소시킨다")

    spr = safe_float(eff.get("splash_other_ratio", 0.0))
    if spr > 0:
        eff_parts.append(f"공격 대상 외 모든 대상에게 {fmt_ratio_0(spr)} 데미지를 입힌다")

    tdr = safe_float(eff.get("temp_armor_ratio", 0.0)) + safe_float(eff.get("temp_defense_ratio", 0.0))
    if tdr > 0:
        eff_parts.append(f"일시 갑옷 {fmt_ratio_0(tdr)}")

    ohf = safe_float(eff.get("on_hit_freeze_turns", 0.0))
    if ohf > 0:
        eff_parts.append(f"피격 시 빙결 {ohf:.2f}턴")

    nab = safe_float(eff.get("next_attack_bonus_ratio", 0.0))
    if nab > 0:
        eff_parts.append(f"다음 공격 피해 {fmt_ratio_0(nab)} 추가")

    ohbr = safe_float(eff.get("on_hit_bleed_ratio", 0.0))
    ohbt = safe_int(eff.get("on_hit_bleed_turns", 0))
    if ohbr > 0 and ohbt > 0:
        eff_parts.append(f"피격 시 출혈({fmt_ratio_0(ohbr)}) {ohbt}턴")

    ohdc = safe_float(eff.get("on_hit_draw_cards", 0.0))
    if ohdc > 0:
        eff_parts.append(f"피격 시 카드 {ohdc:.2f}장")

    ohfr = safe_float(eff.get("on_hit_frenzy_ratio", 0.0))
    if ohfr > 0:
        eff_parts.append(f"피격 시 광란 {fmt_ratio_0(ohfr)}")

    if resource > 0:
        eff_parts.append(f"자원 +{resource}")

    su = eff.get("summon_unit", None)
    if su:
        if su == "wolf":
            eff_parts.append("늑대를 소환한다")
        elif su == "barricade":
            eff_parts.append("바리케이드를 소환한다")
        else:
            eff_parts.append(f"{su}을(를) 소환한다")

    if eff_parts:
        parts.append("/ " + " / ".join(eff_parts))

    return " ".join(parts).replace("  ", " ").strip()


STAT_LABEL = {
    "attack": "피해",
    "attack_ratio": "피해 비율",
    "range": "거리",
    "resource": "자원",
    "shield": "보호막",
    "defense_ratio": "방어력 비율",
}
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
    "splash_other_ratio": "추가 광역 피해",
    "temp_armor_ratio": "일시 갑옷",
    "on_hit_freeze_turns": "피격 시 빙결",
    "next_attack_bonus_ratio": "다음 공격 피해 증가",
    "on_hit_bleed_ratio": "피격 시 출혈 비율",
    "on_hit_bleed_turns": "피격 시 출혈 턴",
    "on_hit_draw_cards": "피격 시 드로우",
    "on_hit_frenzy_ratio": "피격 시 광란",
    "summon_unit": "소환",
    "triple_attack": "3배 공격",
}


def format_field(key: str, v: Any) -> str:
    if key in (
        "shield_ratio",
        "bleed_ratio",
        "armor_down_ratio",
        "splash_other_ratio",
        "temp_armor_ratio",
        "next_attack_bonus_ratio",
        "on_hit_bleed_ratio",
        "on_hit_frenzy_ratio",
    ):
        return fmt_ratio_2(safe_float(v))
    if key in ("freeze_turns", "on_hit_freeze_turns"):
        return f"{safe_float(v):.2f}턴"
    if key in ("draw_cards", "on_hit_draw_cards"):
        return f"{safe_float(v):.2f}장"
    if key in ("bleed_turns", "on_hit_bleed_turns"):
        return f"{safe_int(v)}턴"
    return str(v)


def diff_card(before: CardState, after: CardState) -> List[str]:
    def fmt_num(v: Any, dec: int = 2) -> str:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:.{dec}f}".rstrip("0").rstrip(".")
        return str(v)

    def calc_attack(c: CardState) -> float:
        return c.attack_ratio_value()

    def calc_shield(c: CardState) -> float:
        return c.defense_ratio_value()

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
            if k in ("attack_ratio", "defense_ratio"):
                delta_s = ("+" if delta >= 0 else "-") + fmt_ratio_1(abs(delta))
                lines.append(f"- {label}: {fmt_ratio_1(float(b))} -> {fmt_ratio_1(float(a))} ({delta_s})")
                continue
            if k == "attack_bonus_ratio":
                b_atk = calc_attack(before)
                a_atk = calc_attack(after)
                if b_atk == a_atk:
                    continue
                delta = a_atk - b_atk
                delta_s = ("+" if delta >= 0 else "-") + fmt_ratio_1(abs(delta))
                lines.append(f"- 피해 비율: {fmt_ratio_1(b_atk)} -> {fmt_ratio_1(a_atk)} ({delta_s})")
                continue
            if k == "shield_bonus_ratio":
                b_sh = calc_shield(before)
                a_sh = calc_shield(after)
                if b_sh == a_sh:
                    continue
                delta = a_sh - b_sh
                delta_s = ("+" if delta >= 0 else "-") + fmt_ratio_1(abs(delta))
                lines.append(f"- 방어력 비율: {fmt_ratio_1(b_sh)} -> {fmt_ratio_1(a_sh)} ({delta_s})")
                continue
            delta_s = ("+" if delta >= 0 else "-") + fmt_num(abs(delta))
            lines.append(f"- {label}: {fmt_num(b)} -> {fmt_num(a)} ({delta_s})")
        else:
            lines.append(f"- {label}: {b} -> {a}")

    e_keys = sorted(set(before.effects.keys()) | set(after.effects.keys()))
    for k in e_keys:
        b = before.effects.get(k, None)
        a = after.effects.get(k, None)
        if b == a:
            continue
        label = EFFECT_LABEL.get(k, k)
        lines.append(f"- {label}: {format_field(k, b)} -> {format_field(k, a)}")

    return lines


def enhancement_counts_line(
    card: CardState,
    tier_order: Optional[List[str]] = None,
    tier_labels: Optional[Dict[str, str]] = None,
) -> str:
    tier_labels = tier_labels or {}
    items: List[str] = []
    if tier_order is None:
        tier_order = sorted(card.enhancement_counts.keys())
    for tier_name in tier_order:
        c = card.enhancement_counts.get(tier_name, 0)
        if c:
            label = tier_labels.get(tier_name, tier_name)
            items.append(f"{label}×{c}")
    return ", ".join(items) if items else "없음"


def save_card(card: CardState, filename: str) -> str:
    return save_cards([card], filename)


def save_cards(cards: List[CardState], filename: str) -> str:
    filename = ensure_json_ext(filename)
    if not filename:
        return ""
    data = {"cards": [c.to_dict() for c in cards]}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename
