import argparse
import math
import random
from pathlib import Path

import pygame as pg
from combat_ui import run_enhancement_choices, run_meet_combat


# Theme
BG_TOP = (34, 24, 17)
BG_BOTTOM = (18, 12, 9)
PANEL = (28, 23, 19)
PANEL_EDGE = (134, 98, 58)
WHITE = (244, 236, 221)
MUTED = (205, 182, 146)
BAR_BG = (91, 77, 60)
BAR_FILL = (204, 152, 78)
BUTTON = (125, 90, 56)
BUTTON_HOVER = (147, 107, 66)
BUTTON_PRESS = (105, 75, 46)
BUTTON_EDGE = (216, 176, 124)
BLACK = (12, 16, 24)
LAST_ROLL_RESULT: dict[str, int | bool] | None = None
ROLL_HISTORY: list[dict[str, int | bool]] = []
ANIMATION_SPEED = 0.7
FAST_ANIMATION_SPEED = 120.0

# 원본(1024x1024) 보드 좌표: 시작칸 -> 마지막칸, 마지막 다음은 시작칸으로 순환
BOARD_PATH_ORIGINAL: list[tuple[int, int]] = [
    (97, 894),
    (247, 884),
    (369, 887),
    (510, 877),
    (654, 868),
    (784, 878),
    (933, 892),
    (927, 721),
    (933, 593),
    (933, 467),
    (928, 346),
    (928, 235),
    (919, 106),
    (771, 90),
    (636, 91),
    (490, 94),
    (349, 93),
    (235, 96),
    (102, 105),
    (85, 234),
    (96, 352),
    (96, 464),
    (91, 581),
    (97, 710),
]

# 원본(1024x1024) 내부 루프 좌표
INNER_PATH_ORIGINAL: list[tuple[int, int]] = [
    (255, 721),
    (378, 727),
    (511, 708),
    (649, 708),
    (775, 715),
    (786, 596),
    (784, 466),
    (777, 349),
    (771, 235),
    (636, 228),
    (495, 229),
    (361, 232),
    (237, 241),
    (237, 352),
    (243, 469),
    (241, 595),
]

# (플레이어 보드 1~24) <=> (적 보드 1~16) meet 칸 매핑 (1-based)
MEET_PAIRS_1B: list[tuple[int, int]] = [
    (2, 1), (3, 2), (4, 3), (5, 4), (6, 5),
    (8, 5), (9, 6), (10, 7), (11, 8), (12, 9),
    (14, 9), (15, 10), (16, 11), (17, 12), (18, 13),
    (20, 13), (21, 14), (22, 15), (23, 16), (24, 1),
]
MEET_PAIRS_0B: set[tuple[int, int]] = {(p - 1, b - 1) for p, b in MEET_PAIRS_1B}
ABILITY_STUN_CELL_1B = 4
ABILITY_ADD_BLUE_CELL_1B = 6
ABILITY_BARRICADE_CELL_1B = 7
ABILITY_T2_ENHANCE_CELL_1B = 13
ABILITY_REMOVE_CELL_1B = 19
COMBAT_START_HAND = 4.0
COMBAT_PRE_GAME_ENHANCE_ROLLS = 1000
COMBAT_MEET_ENHANCE_ROLLS = 0
COMBAT_ENHANCE_ROLLS_PER_KILL = 3
COMBAT_T2_PLUS_TIER = 2
COMBAT_MAX_ENCOUNTER_ENEMIES = 8
STUN_SPIN_CYCLE_MS = 4500.0
STUN_HAMMER_SWING_BASE_MS = 120
STUN_HAMMER_HIT_HOLD_BASE_MS = 240
BARRICADE_BOUNCE_BASE_MS = 280
BARRICADE_BOUNCE_REACH_T = 0.58
HAMMER_HITTING_IMAGE_PATH = Path("images/effects/hammer_hitting.png")
HAMMER_HIT_IMAGE_PATH = Path("images/effects/hammer_hit.png")
UNIT_ARROW_IMAGE_PATH = Path("images/effects/unit_arrow.png")
HAMMER_REF_SIZE = (1536.0, 1024.0)
HAMMER_IMPACT_CENTER_REF = (296.0, 778.0)
HAMMER_PIVOT_REF = (1236.0, 473.0)
HAMMER_IMPACT_WIDTH_REF = 450.0
HAMMER_SWING_START_DEG = 60.0
HAMMER_DEBUG_OVERLAY = True
MARKER_REF_BOARD_SIZE = 700.0
T2_MARKER_CENTER_REF = (671.0, 100.0)
T2_MARKER_RADIUS_REF = 11.0
COMBAT_PLAYER_PROFILE: dict[str, float | str] = {
    "unit_id": "Player",
    "unit_key": "Player",
    "name": "플레이어",
    "max_hp": 500.0,
    "attack": 100.0,
    "armor": 50.0,
    "shield_power": 30.0,
}
COMBAT_UNIT_ARCHETYPES: dict[str, dict[str, float | str]] = {
    "Wolf": {"name": "늑대", "max_hp": 200.0, "attack": 70.0, "armor": 50.0},
    "Archer": {"name": "궁수", "max_hp": 150.0, "attack": 100.0, "armor": 30.0},
    "Mage": {"name": "마법사", "max_hp": 100.0, "attack": 150.0, "armor": 20.0},
}
ENEMY_UNIT_TYPES = tuple(COMBAT_UNIT_ARCHETYPES.keys())
ENEMY_UNIT_WEIGHTS: tuple[float, ...] = tuple(2.0 if t == "Wolf" else 1.0 for t in ENEMY_UNIT_TYPES)
ENEMY_START_COUNT = 50
ENEMY_ADD_PER_LAP = 8
ENEMY_ANIM_REFERENCE_COUNT = 5
ENEMY_ANIM_BASE_MULTIPLIER = 2.0
ENEMY_SUMMON_BASE_MS = 560
UNIT_FALLBACK_COLORS: dict[str, tuple[int, int, int]] = {
    "Player": (226, 114, 108),
    "Wolf": (118, 194, 255),
    "Archer": (247, 171, 108),
    "Mage": (194, 148, 255),
}
AURA_TEXTURE_PATH = Path("images/text_effects/aura1.png")
AURA_DOWNSCALE = 0.28
_AURA_TEXTURE_CACHE: pg.Surface | None = None
_AURA_PATCH_CACHE: dict[tuple[int, int], pg.Surface] = {}
_AURA_RUN_VARIANT_SEED = random.randrange(1 << 30)


def roll_weighted_total(charge_ratio: float, sigma: float = 1.2) -> int:
    """0~1 충전 비율을 0~12로 보고, 0/1은 2로 보정해 합계를 가중치로 뽑는다."""
    faces = list(range(2, 13))
    weights = []
    gauge_level = charge_ratio * 12.0
    if gauge_level < 2.0:
        gauge_level = 2.0

    for x in faces:
        weight = math.exp(-((x - gauge_level) ** 2) / (2 * sigma ** 2))
        weights.append(weight)

    return random.choices(faces, weights=weights)[0]


def _split_total_to_dice(total: int) -> tuple[int, int]:
    pairs: list[tuple[int, int]] = []
    for a in range(1, 7):
        b = total - a
        if 1 <= b <= 6:
            pairs.append((a, b))
    if not pairs:
        return 1, 1
    d1, d2 = random.choice(pairs)
    if random.random() < 0.5:
        return d1, d2
    return d2, d1


def throw_dice(charge_ratio: float, sigma: float = 1.2) -> tuple[int, int, int, bool]:
    """가중치 기반으로 주사위 2개를 굴리고 (주사위1, 주사위2, 합계, 더블여부)를 반환한다."""
    total = roll_weighted_total(charge_ratio, sigma=sigma)
    dice1, dice2 = _split_total_to_dice(total)
    return dice1, dice2, total, (dice1 == dice2)


def throw_dice_forced_total(total_hint: int) -> tuple[int, int, int, bool]:
    """디버그용: 지정 합계(1~12)를 강제로 던진다. 0/1은 2로 보정."""
    total = max(2, min(12, int(total_hint)))
    dice1, dice2 = _split_total_to_dice(total)
    return dice1, dice2, total, (dice1 == dice2)


def load_scaled_image(path: Path, size: tuple[int, int], fallback_color: tuple[int, int, int]) -> pg.Surface:
    try:
        img = pg.image.load(str(path)).convert_alpha()
        return pg.transform.smoothscale(img, size)
    except Exception:
        surf = pg.Surface(size, pg.SRCALPHA)
        surf.fill((*fallback_color, 255))
        pg.draw.rect(surf, (230, 236, 244), surf.get_rect(), width=3, border_radius=12)
        return surf


def load_aura_texture() -> pg.Surface | None:
    global _AURA_TEXTURE_CACHE
    if _AURA_TEXTURE_CACHE is not None:
        return _AURA_TEXTURE_CACHE
    if not AURA_TEXTURE_PATH.exists():
        return None
    _AURA_TEXTURE_CACHE = pg.image.load(str(AURA_TEXTURE_PATH)).convert_alpha()
    return _AURA_TEXTURE_CACHE


def get_run_variant_aura_patch(w: int, h: int) -> pg.Surface | None:
    key = (w, h)
    cached = _AURA_PATCH_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    texture = load_aura_texture()
    if texture is None:
        return None

    tw, th = texture.get_size()
    small_w = max(1, int(round(tw * AURA_DOWNSCALE)))
    small_h = max(1, int(round(th * AURA_DOWNSCALE)))
    small = pg.transform.smoothscale(texture, (small_w, small_h))

    tile_cols = max(2, (w // max(1, small_w)) + 3)
    tile_rows = max(2, (h // max(1, small_h)) + 3)
    tiled = pg.Surface((small_w * tile_cols, small_h * tile_rows), pg.SRCALPHA)
    for ty in range(tile_rows):
        for tx in range(tile_cols):
            tiled.blit(small, (tx * small_w, ty * small_h))

    seed = _AURA_RUN_VARIANT_SEED ^ (w * 92821) ^ (h * 68917) ^ 0xA47C
    rng = random.Random(seed)
    max_x = max(0, tiled.get_width() - w)
    max_y = max(0, tiled.get_height() - h)
    ox = rng.randint(0, max_x) if max_x > 0 else 0
    oy = rng.randint(0, max_y) if max_y > 0 else 0
    patch = tiled.subsurface(pg.Rect(ox, oy, w, h)).copy()
    _AURA_PATCH_CACHE[key] = patch
    return patch.copy()


def render_aura_counter_text(text: str, font: pg.font.Font) -> pg.Surface | None:
    mask = font.render(text, True, (255, 255, 255)).convert_alpha()
    bounds = mask.get_bounding_rect(min_alpha=1)
    if bounds.w <= 0 or bounds.h <= 0:
        return None
    mask = mask.subsurface(bounds).copy()
    w, h = mask.get_size()

    patch = get_run_variant_aura_patch(w, h)
    if patch is None:
        patch = pg.Surface((w, h), pg.SRCALPHA)
        patch.fill((255, 145, 96, 255))

    patch.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
    return patch


def load_font(size: int, family: str) -> pg.font.Font:
    font_map = {
        "keriskedu_bold": Path("fonts/KERISKEDU_B.ttf"),
        "keriskedu_regular": Path("fonts/KERISKEDU_R.ttf"),
        "katuri": Path("fonts/Katuri.ttf"),
    }
    path = font_map.get(family)
    if path is not None and path.exists():
        return pg.font.Font(str(path), size)
    return pg.font.SysFont("arial", size)


def draw_gradient(screen: pg.Surface, width: int, height: int) -> None:
    top = pg.Color(*BG_TOP)
    bottom = pg.Color(*BG_BOTTOM)
    for y in range(height):
        t = y / max(1, height - 1)
        c = top.lerp(bottom, t)
        pg.draw.line(screen, c, (0, y), (width, y))


def draw_rounded_panel(screen: pg.Surface, rect: pg.Rect, fill: tuple[int, int, int], edge: tuple[int, int, int], radius: int = 16) -> None:
    pg.draw.rect(screen, fill, rect, border_radius=radius)
    pg.draw.rect(screen, edge, rect, width=2, border_radius=radius)


def blit_rounded_image(screen: pg.Surface, image: pg.Surface, rect: pg.Rect, radius: int) -> None:
    clipped = pg.Surface((rect.w, rect.h), pg.SRCALPHA)
    clipped.blit(image, (0, 0))

    mask = pg.Surface((rect.w, rect.h), pg.SRCALPHA)
    mask.fill((255, 255, 255, 0))
    pg.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
    clipped.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)

    screen.blit(clipped, rect.topleft)


def draw_button(
    screen: pg.Surface,
    rect: pg.Rect,
    text: str,
    font: pg.font.Font,
    hovered: bool,
    pressed: bool,
    disabled: bool = False,
) -> None:
    if disabled:
        color = (74, 62, 48)
        edge = (132, 108, 76)
        txt_color = (188, 168, 139)
    else:
        color = BUTTON_PRESS if pressed else (BUTTON_HOVER if hovered else BUTTON)
        edge = BUTTON_EDGE
        txt_color = WHITE
    draw_rounded_panel(screen, rect, color, edge, radius=14)
    txt = font.render(text, True, txt_color)
    txt_rect = txt.get_rect(center=rect.center)
    screen.blit(txt, txt_rect)


def scale_board_path(
    source_points: list[tuple[int, int]],
    target_rect: pg.Rect,
    source_size: tuple[int, int] = (1024, 1024),
) -> list[tuple[float, float]]:
    sx = target_rect.w / float(source_size[0])
    sy = target_rect.h / float(source_size[1])
    return [
        (target_rect.x + (x * sx), target_rect.y + (y * sy))
        for x, y in source_points
    ]


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - ((1.0 - t) ** 3)


def ease_in_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + ((b - a) * t)


def compute_step_duration_ms(total_steps: int, steps_left: int) -> int:
    del total_steps, steps_left
    # 전체 칸 이동 속도는 고정(가속/감속 없음)
    return int(170 / ANIMATION_SPEED)


def parabolic_step_position(
    from_pos: tuple[float, float],
    to_pos: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """칸 사이 이동에서 포물선(상향 아크) + 약간의 측면 흔들림을 준다."""
    from_x, from_y = from_pos
    to_x, to_y = to_pos
    base_x = lerp(from_x, to_x, t)
    base_y = lerp(from_y, to_y, t)

    dx = to_x - from_x
    dy = to_y - from_y
    distance = math.hypot(dx, dy)
    jump = max(14.0, min(36.0, distance * 0.2))

    # 포물선: 4t(1-t)
    arc = 4.0 * t * (1.0 - t)
    y = base_y - (jump * arc)

    # 귀엽게 보이도록 진행 중 좌우 흔들림(시작/끝 0)
    wobble = math.sin(t * math.pi) * jump * 0.08
    if distance > 1e-6:
        nx = -dy / distance
        ny = dx / distance
        x = base_x + (nx * wobble)
        y = y + (ny * wobble * 0.5)
    else:
        x = base_x

    return x, y


def inner_route_points(
    path: list[tuple[float, float]],
    start_idx: int,
    steps: int,
) -> list[tuple[float, float]]:
    if not path:
        return []
    pts: list[tuple[float, float]] = [path[start_idx % len(path)]]
    idx = start_idx % len(path)
    for _ in range(max(0, steps)):
        idx = (idx + 1) % len(path)
        pts.append(path[idx])
    return pts


def main(*, fast: bool = False) -> int | None:
    global LAST_ROLL_RESULT, ANIMATION_SPEED
    fast_mode = bool(fast)
    if fast:
        ANIMATION_SPEED = FAST_ANIMATION_SPEED
    pg.init()

    screen_w, screen_h = 1200, 820
    screen = pg.display.set_mode((screen_w, screen_h))
    pg.display.set_caption("Dice Duals - Dice Visualization")
    clock = pg.time.Clock()

    font_button = load_font(34, "keriskedu_regular")
    font_button_bold = load_font(34, "keriskedu_bold")
    font_path_num = load_font(26, "keriskedu_bold")
    font_path_num_corner = load_font(34, "keriskedu_bold")
    font_deck_button = load_font(24, "keriskedu_bold")
    font_deck_title = load_font(30, "keriskedu_bold")
    font_deck_line = load_font(19, "keriskedu_regular")
    font_t2_counter = load_font(29, "keriskedu_bold")

    def draw_loading_overlay(text: str = "Loading...") -> None:
        # 전투/강화 UI 전환 중 검은 공백 대신 로딩 문구를 고정 표시한다.
        screen.fill((8, 10, 14))
        loading_surf = font_button_bold.render(text, True, (245, 245, 245))
        screen.blit(loading_surf, loading_surf.get_rect(center=(screen_w // 2, screen_h // 2)))
        pg.display.flip()
        pg.event.pump()

    # Layout
    board_size = min(int(screen_h * 0.92), 770)
    board_rect = pg.Rect(0, 0, board_size, board_size)
    board_rect.center = (screen_w // 2, screen_h // 2)
    board_path = scale_board_path(BOARD_PATH_ORIGINAL, board_rect)
    inner_path = scale_board_path(INNER_PATH_ORIGINAL, board_rect)

    cluster_rect = pg.Rect(0, 0, 300, 272)
    cluster_rect.center = (board_rect.centerx, board_rect.centery - 24)

    cluster_pad_x = 12
    cluster_pad_y = 24
    stack_gap = 10

    dice_size = (132, 132)
    dice1_rect = pg.Rect(0, 0, *dice_size)
    dice2_rect = pg.Rect(0, 0, *dice_size)
    dice_gap = 12
    total_dice_w = dice_size[0] * 2 + dice_gap
    dice1_rect.x = cluster_rect.centerx - total_dice_w // 2
    dice2_rect.x = dice1_rect.right + dice_gap
    dice1_rect.y = cluster_rect.y + cluster_pad_y
    dice2_rect.y = dice1_rect.y

    gauge_rect = pg.Rect(0, 0, cluster_rect.w - (cluster_pad_x * 2), 20)
    gauge_rect.centerx = cluster_rect.centerx
    gauge_rect.y = dice1_rect.bottom + stack_gap

    button_rect = pg.Rect(0, 0, cluster_rect.w - (cluster_pad_x * 2), 52)
    button_rect.centerx = cluster_rect.centerx
    button_rect.y = gauge_rect.bottom + stack_gap
    deck_button_rect = pg.Rect(0, 0, 154, 46)
    deck_button_rect.x = min(screen_w - 16 - deck_button_rect.w, board_rect.right + 14)
    deck_button_rect.y = board_rect.y + 16
    skip_button_rect = pg.Rect(0, 0, 154, 46)
    skip_button_rect.x = min(screen_w - 16 - skip_button_rect.w, board_rect.right + 14)
    skip_button_rect.y = screen_h - 16 - skip_button_rect.h
    deck_panel_rect = pg.Rect(0, 0, min(760, screen_w - 72), min(620, screen_h - 72))
    deck_panel_rect.center = (screen_w // 2, screen_h // 2)

    # Assets
    board_image = load_scaled_image(
        Path("images/board/board2.png"),
        (board_rect.w, board_rect.h),
        (40, 50, 64),
    )

    dice_cache: dict[int, pg.Surface] = {}

    def get_dice(v: int) -> pg.Surface:
        if v in dice_cache:
            return dice_cache[v]
        surf = load_scaled_image(
            Path(f"images/dice_images/dice{v}.png"),
            dice_size,
            (90, 96, 112),
        )
        dice_cache[v] = surf
        return surf

    def _wrap_lines(text: str, font: pg.font.Font, max_width: int) -> list[str]:
        words = str(text).split(" ")
        if not words:
            return [""]
        lines: list[str] = []
        cur = words[0]
        for w in words[1:]:
            nxt = f"{cur} {w}"
            if font.size(nxt)[0] <= max_width:
                cur = nxt
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def _random_enemy_unit_type() -> str:
        if not ENEMY_UNIT_TYPES:
            return "Wolf"
        return random.choices(ENEMY_UNIT_TYPES, weights=ENEMY_UNIT_WEIGHTS, k=1)[0]

    unit_image_cache: dict[tuple[str, int], pg.Surface] = {}
    unit_center_offset_cache: dict[tuple[str, int], tuple[float, float]] = {}
    unit_raw_cache: dict[str, pg.Surface] = {}
    unit_arrow_cache: dict[int, pg.Surface] = {}
    unit_arrow_raw: pg.Surface | None = None
    stun_effect_cache: dict[int, pg.Surface] = {}
    stun_raw_image: pg.Surface | None = None
    hammer_effect_cache: dict[tuple[str, int, int], pg.Surface] = {}
    hammer_raw_images: dict[str, pg.Surface] = {}

    def trim_alpha_bounds(image: pg.Surface, pad: int = 2) -> pg.Surface:
        bounds = image.get_bounding_rect(min_alpha=1)
        if bounds.w <= 0 or bounds.h <= 0:
            return image
        x = max(0, bounds.x - pad)
        y = max(0, bounds.y - pad)
        w = min(image.get_width() - x, bounds.w + (pad * 2))
        h = min(image.get_height() - y, bounds.h + (pad * 2))
        cropped = pg.Surface((w, h), pg.SRCALPHA)
        cropped.blit(image, (0, 0), area=pg.Rect(x, y, w, h))
        return cropped

    def get_unit_image(unit_key: str, size_px: int) -> pg.Surface:
        size = max(8, int(size_px))
        key = (unit_key, size)
        if key in unit_image_cache:
            return unit_image_cache[key]

        raw = unit_raw_cache.get(unit_key)
        if raw is None:
            path = Path(f"images/units/{unit_key}.png")
            try:
                raw = trim_alpha_bounds(pg.image.load(str(path)).convert_alpha())
            except Exception:
                fallback = UNIT_FALLBACK_COLORS.get(unit_key, (132, 136, 150))
                raw = pg.Surface((128, 128), pg.SRCALPHA)
                raw.fill((*fallback, 255))
            unit_raw_cache[unit_key] = raw

        raw_w = max(1, raw.get_width())
        raw_h = max(1, raw.get_height())
        fit_scale = min(size / float(raw_w), size / float(raw_h))
        dst_w = max(1, int(round(raw_w * fit_scale)))
        dst_h = max(1, int(round(raw_h * fit_scale)))
        scaled = pg.transform.smoothscale(raw, (dst_w, dst_h))
        surf = pg.Surface((size, size), pg.SRCALPHA)
        surf.blit(scaled, scaled.get_rect(center=(size // 2, size // 2)))
        unit_image_cache[key] = surf
        return surf

    def get_unit_arrow_image(size_px: int) -> pg.Surface:
        nonlocal unit_arrow_raw
        size = max(8, int(size_px))
        cached = unit_arrow_cache.get(size)
        if cached is not None:
            return cached

        if unit_arrow_raw is None:
            try:
                unit_arrow_raw = trim_alpha_bounds(pg.image.load(str(UNIT_ARROW_IMAGE_PATH)).convert_alpha())
            except Exception:
                unit_arrow_raw = pg.Surface((96, 96), pg.SRCALPHA)
                pts = [(48, 8), (88, 62), (64, 62), (64, 88), (32, 88), (32, 62), (8, 62)]
                pg.draw.polygon(unit_arrow_raw, (255, 240, 170), pts)
                pg.draw.polygon(unit_arrow_raw, (42, 30, 16), pts, width=4)

        raw_w = max(1, unit_arrow_raw.get_width())
        raw_h = max(1, unit_arrow_raw.get_height())
        fit_scale = min(size / float(raw_w), size / float(raw_h))
        dst_w = max(1, int(round(raw_w * fit_scale)))
        dst_h = max(1, int(round(raw_h * fit_scale)))
        scaled = pg.transform.smoothscale(unit_arrow_raw, (dst_w, dst_h))
        surf = pg.Surface((size, size), pg.SRCALPHA)
        surf.blit(scaled, scaled.get_rect(center=(size // 2, size // 2)))
        unit_arrow_cache[size] = surf
        return surf

    def get_unit_center_offset(unit_key: str, size_px: int) -> tuple[float, float]:
        size = max(8, int(size_px))
        key = (unit_key, size)
        cached = unit_center_offset_cache.get(key)
        if cached is not None:
            return cached

        surf = get_unit_image(unit_key, size)
        mask = pg.mask.from_surface(surf)
        centroid = mask.centroid() if mask.count() > 0 else None
        if centroid is None:
            offset = (0.0, 0.0)
        else:
            target_x = surf.get_width() * 0.5
            target_y = surf.get_height() * 0.5
            offset = (target_x - (centroid[0] + 0.5), target_y - (centroid[1] + 0.5))

        unit_center_offset_cache[key] = offset
        return offset

    def get_stun_effect_image(size_px: int) -> pg.Surface:
        nonlocal stun_raw_image
        size = max(8, int(size_px))
        if size in stun_effect_cache:
            return stun_effect_cache[size]
        if stun_raw_image is None:
            try:
                stun_raw_image = trim_alpha_bounds(pg.image.load("images/effects/stun.png").convert_alpha())
            except Exception:
                stun_raw_image = pg.Surface((96, 96), pg.SRCALPHA)
                pg.draw.circle(stun_raw_image, (186, 136, 255), (48, 48), 34)
        raw_w = max(1, stun_raw_image.get_width())
        raw_h = max(1, stun_raw_image.get_height())
        fit_scale = min(size / float(raw_w), size / float(raw_h))
        dst_w = max(1, int(round(raw_w * fit_scale)))
        dst_h = max(1, int(round(raw_h * fit_scale)))
        scaled = pg.transform.smoothscale(stun_raw_image, (dst_w, dst_h))
        surf = pg.Surface((size, size), pg.SRCALPHA)
        surf.blit(scaled, scaled.get_rect(center=(size // 2, size // 2)))
        stun_effect_cache[size] = surf
        return surf

    def get_hammer_effect_image(kind: str, impact_width_px: float) -> pg.Surface:
        raw = hammer_raw_images.get(kind)
        if raw is None:
            path = HAMMER_HITTING_IMAGE_PATH if kind == "hitting" else HAMMER_HIT_IMAGE_PATH
            try:
                raw = pg.image.load(str(path)).convert_alpha()
            except Exception:
                raw = pg.Surface((int(HAMMER_REF_SIZE[0]), int(HAMMER_REF_SIZE[1])), pg.SRCALPHA)
                pg.draw.rect(raw, (206, 172, 118), (540, 420, 430, 210), border_radius=28)
            hammer_raw_images[kind] = raw

        raw_w = max(1, raw.get_width())
        raw_h = max(1, raw.get_height())
        impact_width_norm = HAMMER_IMPACT_WIDTH_REF / max(1.0, HAMMER_REF_SIZE[0])
        target_w = max(32, int(round(float(impact_width_px) / max(1e-6, impact_width_norm))))
        target_h = max(32, int(round(target_w * (raw_h / float(raw_w)))))
        key = (kind, target_w, target_h)
        cached = hammer_effect_cache.get(key)
        if cached is not None:
            return cached
        scaled = pg.transform.smoothscale(raw, (target_w, target_h))
        hammer_effect_cache[key] = scaled
        return scaled

    def blit_hammer_swing(
        kind: str,
        impact_xy: tuple[float, float],
        impact_width_px: float,
        angle_deg: float,
        alpha: int = 255,
    ) -> None:
        hammer = get_hammer_effect_image(kind, impact_width_px)
        hw, hh = hammer.get_size()

        # 원본(1536x1024) 기준 좌표를 현재 리사이즈된 해머 로컬 좌표로 변환
        impact_local = pg.Vector2(
            hw * (HAMMER_IMPACT_CENTER_REF[0] / HAMMER_REF_SIZE[0]),
            hh * (HAMMER_IMPACT_CENTER_REF[1] / HAMMER_REF_SIZE[1]),
        )
        pivot_local = pg.Vector2(
            hw * (HAMMER_PIVOT_REF[0] / HAMMER_REF_SIZE[0]),
            hh * (HAMMER_PIVOT_REF[1] / HAMMER_REF_SIZE[1]),
        )

        impact_world = pg.Vector2(float(impact_xy[0]), float(impact_xy[1]))
        pivot_to_impact = impact_local - pivot_local
        # 0도(원본 각도)에서 impact_local이 impact_world에 정확히 맞도록 pivot 월드 좌표를 고정한다.
        pivot_world = impact_world - pivot_to_impact

        # pygame.transform.rotozoom(+deg)는 시각적으로 반시계(CCW) 회전,
        # Vector2.rotate(+deg)는 화면 좌표에서 시계(CW) 방향이므로 부호를 반대로 맞춘다.
        vec_rot_deg = -angle_deg

        # pivot 기준으로 center의 월드 좌표를 계산해, 회전 후 이미지를 정확히 배치한다.
        center_local = pg.Vector2(hw * 0.5, hh * 0.5)
        pivot_to_center = center_local - pivot_local
        center_world = pivot_world + pivot_to_center.rotate(vec_rot_deg)

        rotated = pg.transform.rotozoom(hammer, angle_deg, 1.0)
        if alpha < 255:
            rotated = rotated.copy()
            rotated.set_alpha(max(0, min(255, int(alpha))))

        rect = rotated.get_rect(center=(int(round(center_world.x)), int(round(center_world.y))))
        screen.blit(rotated, rect.topleft)

        if HAMMER_DEBUG_OVERLAY:
            pg.draw.rect(screen, (250, 92, 92), rect, width=2)

            # 목표점 (노랑)
            pg.draw.circle(screen, (255, 236, 122),
                        (int(round(impact_world.x)), int(round(impact_world.y))), 6, width=2)

            # 피봇 (파랑) - 항상 고정
            pg.draw.circle(screen, (108, 220, 255),
                        (int(round(pivot_world.x)), int(round(pivot_world.y))), 7, width=2)

            # 현재 타격점 (주황) - 파랑 중심 원호
            current_impact_world = pivot_world + pivot_to_impact.rotate(vec_rot_deg)
            pg.draw.circle(screen, (255, 148, 92),
                        (int(round(current_impact_world.x)), int(round(current_impact_world.y))), 5, width=2)
    # Gauge state
    max_gauge = 100
    current_gauge = 0
    # 파워바 왕복 속도(값이 작을수록 느림)
    gauge_differential = 3
    gauge_dir = 1
    is_charging = False
    shuffle_interval_ms = int(70 / ANIMATION_SPEED)
    next_shuffle_ms = 0
    resolve_duration_ms = int(3000 / ANIMATION_SPEED)
    is_resolving = False
    resolve_start_ms = 0
    resolve_next_shuffle_ms = 0
    final_dice1 = 1
    final_dice2 = 1

    # Dice state
    dice1_value = 1
    dice2_value = 1

    # Move state
    current_phase = "Throw"
    player_pre_move_ms = int(1000 / ANIMATION_SPEED)
    player_post_move_ms = int(500 / ANIMATION_SPEED)
    enemy_pre_move_base_ms = int(500 / ANIMATION_SPEED)
    enemy_post_move_base_ms = int(500 / ANIMATION_SPEED)
    phase_timer_start_ms = 0
    token_position = 0
    move_total = 0
    move_left = 0
    pending_move_total = 0
    step_from_position = 0
    step_to_position = 0
    step_start_ms = 0
    step_duration_ms = 220
    player_bounce_start_ms = 0
    player_bounce_duration_ms = int(BARRICADE_BOUNCE_BASE_MS / ANIMATION_SPEED)
    player_bounce_from_position = 0
    player_bounce_to_position = 0
    player_lap_count = 0
    roll_result_pause_ms = int(520 / ANIMATION_SPEED)
    roll_result_start_ms = 0

    # 내부 루프 적 말(독립 이동): 시작 시 8마리 배치
    blue_tokens: list[int] = [random.randrange(len(inner_path)) for _ in range(ENEMY_START_COUNT)]
    blue_token_types: list[str] = [_random_enemy_unit_type() for _ in blue_tokens]
    blue_stun_turns: list[int] = [0 for _ in blue_tokens]
    blue_move_queue: list[dict[str, int | str]] = []
    blue_active_token_i: int | None = None
    blue_active_order = 0
    blue_active_steps_left = 0
    blue_active_mode = "move"
    blue_active_from = 0
    blue_active_to = 0
    blue_active_start_ms = 0
    enemy_move_base_ms = int(155 / ANIMATION_SPEED)
    enemy_sleep_base_ms = int(1000 / ANIMATION_SPEED)
    enemy_stun_base_ms = int(700 / ANIMATION_SPEED)
    enemy_remove_base_ms = int(650 / ANIMATION_SPEED)
    enemy_spawn_base_ms = int(ENEMY_SUMMON_BASE_MS / ANIMATION_SPEED)
    blue_active_duration_ms = enemy_move_base_ms
    blue_sleep_duration_ms = enemy_sleep_base_ms
    blue_stun_duration_ms = enemy_stun_base_ms
    blue_remove_duration_ms = enemy_remove_base_ms
    blue_sleep_batch_token_ids: list[int] = []
    blue_stun_recovered_this_turn: set[int] = set()
    blue_spawn_effects: list[dict[str, float]] = []
    meet_banner_until_ms = 0
    meet_effects: list[dict[str, float]] = []
    current_meeting_blue_ids: set[int] = set()
    meet_lightning_duration_ms = 60 if fast_mode else 2000
    meet_lightning_refresh_ms = 8 if fast_mode else 300
    meet_lightning_start_ms = 0
    meet_lightning_next_refresh_ms = 0
    meet_pending_blue_ids: set[int] = set()
    meet_return_phase = "PlayerPostMove"
    meet_lightning_paths: dict[int, list[tuple[list[tuple[float, float]], bool]]] = {}
    inner_pick_radius = max(16, int(board_rect.w * 0.03))
    outer_pick_radius = max(16, int(board_rect.w * 0.028))
    # i는 i번째 칸 -> i+1번째 칸 사이(순환)의 벽을 의미
    barricade_edges: set[int] = set()
    blue_remove_effects: list[dict[str, float]] = []
    pending_remove_token_ids: set[int] = set()
    blue_remove_return_phase = "PlayerPostMove"
    stun_hammer_swing_ms = int(STUN_HAMMER_SWING_BASE_MS / ANIMATION_SPEED)
    stun_hammer_hit_hold_ms = int(STUN_HAMMER_HIT_HOLD_BASE_MS / ANIMATION_SPEED)
    stun_hammer_total_ms = stun_hammer_swing_ms + stun_hammer_hit_hold_ms
    stun_hammer_effect: dict[str, float] | None = None
    board_game_over = False
    board_game_over_text = ""
    skip_confirm_armed = False
    t2_plus_counter = 1
    t2_plus_banner_until_ms = 0
    t2_plus_banner_text = ""
    deck_panel_open = False
    deck_preview_lines: list[str] = []
    combat_template_state: list[dict[str, object]] = []
    combat_enhance_gacha_state: dict[str, object] = {}
    board_player_hp = float(COMBAT_PLAYER_PROFILE.get("max_hp", 500.0))

    def _enemy_anim_speed_multiplier() -> float:
        enemy_count = max(1, len(blue_tokens))
        reference = float(max(1, ENEMY_ANIM_REFERENCE_COUNT))
        proportional_mul = ENEMY_ANIM_BASE_MULTIPLIER * (enemy_count / reference)
        return max(ENEMY_ANIM_BASE_MULTIPLIER, proportional_mul)

    def _enemy_phase_duration(base_ms: int, min_ms: int = 40) -> int:
        del min_ms
        speed_mul = max(1.0, _enemy_anim_speed_multiplier())
        return max(1, int(round(base_ms / speed_mul)))

    def _register_blue_spawn(token_i: int, now_ms: int, stagger_ms: int = 0) -> None:
        if not (0 <= token_i < len(blue_tokens)):
            return
        blue_spawn_effects.append(
            {
                "token_i": float(token_i),
                "start_ms": float(now_ms + max(0, int(stagger_ms))),
                "duration_ms": float(_enemy_phase_duration(enemy_spawn_base_ms, min_ms=120)),
            }
        )

    _initial_spawn_ms = pg.time.get_ticks()
    for _spawn_i in range(len(blue_tokens)):
        _register_blue_spawn(_spawn_i, _initial_spawn_ms, stagger_ms=36 * _spawn_i)

    def _begin_player_pre_move(now_ms: int) -> None:
        nonlocal current_phase, move_total, move_left, pending_move_total, phase_timer_start_ms, is_charging
        if pending_move_total > 0:
            current_phase = "PlayerPreMove"
            move_total = pending_move_total
            move_left = pending_move_total
            pending_move_total = 0
            phase_timer_start_ms = now_ms
            is_charging = False
        else:
            current_phase = "Throw"

    def _is_enemy_animation_phase(phase_name: str) -> bool:
        return phase_name in ("BluePreMove", "InnerMove", "BlueSleep", "BluePostMove", "BlueStun")

    def _skip_enemy_animation_now(now_ms: int) -> None:
        nonlocal current_phase, phase_timer_start_ms
        nonlocal blue_active_from, blue_active_to, blue_active_start_ms, blue_active_duration_ms
        nonlocal blue_active_steps_left

        guard = 0
        while guard < 4096:
            guard += 1
            if current_phase == "BluePreMove":
                if blue_active_token_i is None:
                    _begin_next_blue_pre_move(now_ms)
                    if current_phase == "Throw":
                        _finalize_enemy_turn_with_meet(now_ms)
                        return
                    continue
                if blue_active_mode == "sleep":
                    current_phase = "BluePostMove"
                    phase_timer_start_ms = now_ms
                    continue
                current_phase = "InnerMove"
                blue_active_from = blue_tokens[blue_active_token_i]
                blue_active_to = (blue_active_from + 1) % len(inner_path)
                blue_active_start_ms = now_ms - 1
                blue_active_duration_ms = 1
                continue

            if current_phase == "BlueSleep":
                current_phase = "BluePostMove"
                phase_timer_start_ms = now_ms
                continue

            if current_phase == "BlueStun":
                _begin_next_blue_pre_move(now_ms)
                if current_phase == "Throw":
                    _finalize_enemy_turn_with_meet(now_ms)
                    return
                continue

            if current_phase == "InnerMove" and blue_active_token_i is not None:
                while blue_active_steps_left > 0:
                    blue_tokens[blue_active_token_i] = blue_active_to
                    blue_active_steps_left -= 1
                    if blue_active_steps_left > 0:
                        blue_active_from = blue_tokens[blue_active_token_i]
                        blue_active_to = (blue_active_from + 1) % len(inner_path)
                current_phase = "BluePostMove"
                phase_timer_start_ms = now_ms
                continue

            if current_phase == "BluePostMove":
                _begin_next_blue_pre_move(now_ms)
                if current_phase == "Throw":
                    _finalize_enemy_turn_with_meet(now_ms)
                    return
                continue
            return

    def _begin_next_blue_pre_move(now_ms: int) -> None:
        nonlocal current_phase, phase_timer_start_ms
        nonlocal blue_active_token_i, blue_active_order, blue_active_steps_left, blue_active_mode
        nonlocal blue_sleep_duration_ms, blue_sleep_batch_token_ids, blue_stun_recovered_this_turn
        while blue_move_queue:
            head = blue_move_queue.pop(0)
            mode = str(head["mode"])
            # 기절은 적 턴에서 표시/대기 없이 즉시 소모한다.
            if mode == "stun":
                ti = int(head["token_i"])
                if 0 <= ti < len(blue_tokens):
                    blue_stun_recovered_this_turn.add(ti)
                continue
            if mode == "sleep":
                sleep_batch = [int(head["token_i"])]
                remain_queue: list[dict[str, int | str]] = []
                for item in blue_move_queue:
                    if str(item.get("mode", "")) == "sleep":
                        sleep_batch.append(int(item["token_i"]))
                    else:
                        remain_queue.append(item)
                blue_move_queue[:] = remain_queue
                blue_sleep_batch_token_ids = [i for i in sleep_batch if 0 <= i < len(blue_tokens)]
                if not blue_sleep_batch_token_ids:
                    continue
                blue_active_token_i = int(blue_sleep_batch_token_ids[0])
                blue_active_order = int(head["order"])
                blue_active_mode = mode
                blue_active_steps_left = 0
                blue_sleep_duration_ms = _enemy_phase_duration(enemy_sleep_base_ms, min_ms=120)
                current_phase = "BlueSleep"
                phase_timer_start_ms = now_ms
                return
            blue_sleep_batch_token_ids = []
            blue_active_token_i = int(head["token_i"])
            blue_active_order = int(head["order"])
            blue_active_mode = mode
            blue_active_steps_left = int(head["steps"])
            current_phase = "BluePreMove"
            phase_timer_start_ms = now_ms
            return
        blue_active_token_i = None
        blue_active_order = 0
        blue_active_steps_left = 0
        blue_active_mode = "move"
        blue_sleep_batch_token_ids = []
        current_phase = "Throw"

    def _build_blue_action(order: int, token_i: int, force_move: bool = False) -> dict[str, int | str]:
        if force_move:
            return {"order": order, "token_i": token_i, "mode": "move", "steps": random.choice((1, 2, 3))}
        if 0 <= token_i < len(blue_stun_turns) and blue_stun_turns[token_i] > 0:
            blue_stun_turns[token_i] -= 1
            return {"order": order, "token_i": token_i, "mode": "stun", "steps": 0}
        if random.random() < 0.5:
            return {"order": order, "token_i": token_i, "mode": "sleep", "steps": 0}
        return {"order": order, "token_i": token_i, "mode": "move", "steps": random.choice((1, 2, 3))}

    def _next_blue_order() -> int:
        return max((int(x["order"]) for x in blue_move_queue), default=0) + 1

    def _enqueue_blue_for_current_turn(token_i: int, force_move: bool = True) -> None:
        blue_move_queue.append(_build_blue_action(_next_blue_order(), token_i, force_move=force_move))

    def _pick_inner_cell(mouse: tuple[int, int]) -> int | None:
        mx, my = mouse
        best_idx = None
        best_d2 = float("inf")
        for i, (x, y) in enumerate(inner_path):
            dx = mx - x
            dy = my - y
            d2 = (dx * dx) + (dy * dy)
            if d2 < best_d2:
                best_idx = i
                best_d2 = d2
        if best_idx is None:
            return None
        if best_d2 <= float(inner_pick_radius * inner_pick_radius):
            return int(best_idx)
        return None

    def _edge_midpoint(edge_idx: int) -> tuple[float, float]:
        a = board_path[edge_idx % len(board_path)]
        b = board_path[(edge_idx + 1) % len(board_path)]
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    def _pick_outer_edge(mouse: tuple[int, int]) -> int | None:
        mx, my = mouse
        best_idx = None
        best_d2 = float("inf")
        for i in range(len(board_path)):
            x, y = _edge_midpoint(i)
            dx = mx - x
            dy = my - y
            d2 = (dx * dx) + (dy * dy)
            if d2 < best_d2:
                best_idx = i
                best_d2 = d2
        if best_idx is None:
            return None
        if best_d2 <= float(outer_pick_radius * outer_pick_radius):
            return int(best_idx)
        return None

    def _retarget_queue_to_stun(token_i: int) -> None:
        for item in blue_move_queue:
            if int(item["token_i"]) == token_i:
                item["mode"] = "stun"
                item["steps"] = 0

    def _remove_blue_by_index(token_i: int) -> None:
        nonlocal blue_active_token_i, current_meeting_blue_ids, blue_spawn_effects, blue_sleep_batch_token_ids
        nonlocal blue_stun_recovered_this_turn
        if not (0 <= token_i < len(blue_tokens)):
            return
        blue_tokens.pop(token_i)
        if 0 <= token_i < len(blue_token_types):
            blue_token_types.pop(token_i)
        blue_stun_turns.pop(token_i)

        fixed_queue: list[dict[str, int | str]] = []
        for item in blue_move_queue:
            ti = int(item["token_i"])
            if ti == token_i:
                continue
            if ti > token_i:
                item["token_i"] = ti - 1
            fixed_queue.append(item)
        blue_move_queue[:] = fixed_queue

        if blue_active_token_i is not None:
            if blue_active_token_i == token_i:
                blue_active_token_i = None
            elif blue_active_token_i > token_i:
                blue_active_token_i -= 1

        fixed_sleep_batch: list[int] = []
        for ti in blue_sleep_batch_token_ids:
            if ti == token_i:
                continue
            fixed_sleep_batch.append(ti - 1 if ti > token_i else ti)
        blue_sleep_batch_token_ids = fixed_sleep_batch

        fixed_stun_recovered: set[int] = set()
        for ti in blue_stun_recovered_this_turn:
            if ti == token_i:
                continue
            fixed_stun_recovered.add(ti - 1 if ti > token_i else ti)
        blue_stun_recovered_this_turn = fixed_stun_recovered

        fixed_meeting: set[int] = set()
        for i in current_meeting_blue_ids:
            if i == token_i:
                continue
            fixed_meeting.add(i - 1 if i > token_i else i)
        current_meeting_blue_ids = fixed_meeting

        fixed_effects: list[dict[str, float]] = []
        for e in meet_effects:
            bi = int(e.get("blue_i", -1))
            if bi == token_i:
                continue
            if bi > token_i:
                e["blue_i"] = float(bi - 1)
            fixed_effects.append(e)
        meet_effects[:] = fixed_effects

        fixed_spawn_effects: list[dict[str, float]] = []
        for e in blue_spawn_effects:
            bi = int(e.get("token_i", -1))
            if bi == token_i:
                continue
            if bi > token_i:
                e["token_i"] = float(bi - 1)
            fixed_spawn_effects.append(e)
        blue_spawn_effects = fixed_spawn_effects

    def _apply_cell_stun(cell_idx: int) -> None:
        targets = [i for i, p in enumerate(blue_tokens) if p == cell_idx]
        if not targets:
            return
        for target_i in targets:
            blue_stun_turns[target_i] = max(blue_stun_turns[target_i], 1)
            _retarget_queue_to_stun(target_i)

    def _start_stun_hammer_effect(cell_idx: int, now_ms: int) -> None:
        nonlocal current_phase, phase_timer_start_ms, stun_hammer_effect
        stun_hammer_effect = {
            "cell_idx": float(cell_idx),
            "start_ms": float(now_ms),
        }
        current_phase = "StunHammer"
        phase_timer_start_ms = now_ms

    def _start_remove_effect(cell_idx: int, now_ms: int) -> bool:
        nonlocal current_phase, phase_timer_start_ms
        nonlocal blue_remove_effects, pending_remove_token_ids, blue_remove_return_phase
        nonlocal blue_remove_duration_ms
        targets = [i for i, p in enumerate(blue_tokens) if p == cell_idx]
        if not targets:
            return False
        pending_remove_token_ids = set(targets)
        blue_remove_return_phase = "PlayerPostMove"
        blue_remove_duration_ms = _enemy_phase_duration(enemy_remove_base_ms, min_ms=110)
        blue_remove_effects = [
            {
                "token_i": float(i),
                "start_ms": float(now_ms),
                "duration_ms": float(blue_remove_duration_ms),
            }
            for i in targets
        ]
        current_phase = "BlueRemoveEffect"
        phase_timer_start_ms = now_ms
        return True

    def _is_blue_stunned_visual(token_i: int) -> bool:
        if 0 <= token_i < len(blue_stun_turns) and blue_stun_turns[token_i] > 0:
            return True
        if token_i == blue_active_token_i and current_phase == "BlueStun":
            return True
        for item in blue_move_queue:
            if int(item["token_i"]) == token_i and str(item["mode"]) == "stun":
                return True
        return False

    def _apply_edge_barricade(edge_idx: int) -> None:
        barricade_edges.add(edge_idx % len(board_path))

    def _current_meeting_blue_ids(excluded_ids: set[int] | None = None) -> set[int]:
        blocked = excluded_ids if excluded_ids is not None else set()
        return {
            i
            for i, bpos in enumerate(blue_tokens)
            if (token_position, bpos) in MEET_PAIRS_0B
            and (i not in blocked)
            and (not _is_blue_stunned_visual(i))
        }

    def _build_lightning_path(
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        segments: int,
        jitter: float,
    ) -> list[tuple[float, float]]:
        sx, sy = start_xy
        ex, ey = end_xy
        dx = ex - sx
        dy = ey - sy
        length = max(1.0, math.hypot(dx, dy))
        nx = -dy / length
        ny = dx / length
        seg_count = max(4, int(segments))
        points: list[tuple[float, float]] = [(sx, sy)]
        for s in range(1, seg_count):
            t = s / float(seg_count)
            base_x = sx + (dx * t)
            base_y = sy + (dy * t)
            taper = 1.0 - abs((t - 0.5) * 1.4)
            amp = jitter * max(0.28, taper)
            off = (random.random() * 2.0 - 1.0) * amp
            points.append((base_x + (nx * off), base_y + (ny * off)))
        points.append((ex, ey))
        return points

    def _build_lightning_bundle(
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        segments: int,
        jitter: float,
    ) -> list[tuple[list[tuple[float, float]], bool]]:
        sx, sy = start_xy
        ex, ey = end_xy
        dx = ex - sx
        dy = ey - sy
        length = max(1.0, math.hypot(dx, dy))
        nx = -dy / length
        ny = dx / length

        bolts: list[tuple[list[tuple[float, float]], bool]] = []
        bolts.append((_build_lightning_path(start_xy, end_xy, segments, jitter), False))

        heavy_extra_idx = random.randrange(3)
        for extra_i in range(3):
            start_off = (random.random() * 2.0 - 1.0) * min(10.0, length * 0.04)
            end_off = (random.random() * 2.0 - 1.0) * min(24.0, length * 0.11)
            end_jitter_x = (random.random() * 2.0 - 1.0) * min(9.0, length * 0.04)
            end_jitter_y = (random.random() * 2.0 - 1.0) * min(9.0, length * 0.04)
            extra_start = (sx + (nx * start_off), sy + (ny * start_off))
            extra_end = (
                ex + (nx * end_off) + end_jitter_x,
                ey + (ny * end_off) + end_jitter_y,
            )
            extra_segments = max(5, int(segments + random.choice((-1, 0, 1, 2))))
            extra_jitter = jitter * random.uniform(0.85, 1.35)
            bolts.append(
                (
                    _build_lightning_path(extra_start, extra_end, extra_segments, extra_jitter),
                    extra_i == heavy_extra_idx,
                )
            )
        return bolts

    def _begin_meet_lightning(new_meets: set[int], now_ms: int, return_phase: str) -> bool:
        nonlocal current_phase, phase_timer_start_ms
        nonlocal meet_lightning_start_ms, meet_lightning_next_refresh_ms, meet_pending_blue_ids
        nonlocal meet_return_phase, meet_lightning_paths
        if not new_meets:
            return False
        meet_pending_blue_ids = set(new_meets)
        meet_return_phase = return_phase
        meet_lightning_paths = {}
        meet_lightning_start_ms = now_ms
        meet_lightning_next_refresh_ms = 0
        current_phase = "MeetLightning"
        phase_timer_start_ms = now_ms
        return True

    def _start_meet_combat(new_meets: set[int], now_ms: int) -> bool:
        nonlocal current_phase, phase_timer_start_ms, screen
        nonlocal current_meeting_blue_ids, meet_banner_until_ms
        nonlocal board_game_over, board_game_over_text
        nonlocal is_charging, is_resolving, current_gauge, gauge_dir
        nonlocal blue_remove_effects, pending_remove_token_ids, blue_remove_return_phase
        nonlocal blue_remove_duration_ms
        nonlocal deck_preview_lines, combat_template_state, board_player_hp
        if not new_meets:
            return False
        candidate_ids = sorted(i for i in new_meets if 0 <= i < len(blue_tokens))
        if not candidate_ids:
            return False
        if len(candidate_ids) > COMBAT_MAX_ENCOUNTER_ENEMIES:
            encounter_ids = sorted(random.sample(candidate_ids, COMBAT_MAX_ENCOUNTER_ENEMIES))
        else:
            encounter_ids = candidate_ids
        encounter_types = [
            blue_token_types[i] if 0 <= i < len(blue_token_types) else _random_enemy_unit_type()
            for i in encounter_ids
        ]
        combat_player_profile = dict(COMBAT_PLAYER_PROFILE)
        max_hp = float(combat_player_profile.get("max_hp", 500.0) or 500.0)
        combat_player_profile["current_hp"] = max(0.0, min(max_hp, float(board_player_hp)))
        player_state_out: dict[str, object] = {}
        draw_loading_overlay("Loading...")
        result = run_meet_combat(
            enemy_count=len(encounter_ids),
            enemy_types=encounter_types,
            seed=random.randrange(1 << 30),
            start_enhance_rolls=COMBAT_MEET_ENHANCE_ROLLS,
            initial_draw=COMBAT_START_HAND,
            min_enhance_tier=0,
            min_enhance_tier_rolls=0,
            width=screen_w,
            height=screen_h,
            player_profile=combat_player_profile,
            unit_archetypes=COMBAT_UNIT_ARCHETYPES,
            template_state=combat_template_state,
            template_state_out=combat_template_state,
            player_state_out=player_state_out,
            enhance_gacha_state=combat_enhance_gacha_state,
            enhance_gacha_state_out=combat_enhance_gacha_state,
        )
        if "hp" in player_state_out:
            board_player_hp = max(0.0, float(player_state_out["hp"]))

        # 전투 UI가 display mode를 바꾸므로 보드 화면을 복구한다.
        screen = pg.display.set_mode((screen_w, screen_h))
        pg.display.set_caption("Dice Duals - Dice Visualization")
        pg.event.clear()

        if result is True:
            remove_targets = [i for i in encounter_ids if 0 <= i < len(blue_tokens)]
            reward_rolls = max(0, len(remove_targets) * COMBAT_ENHANCE_ROLLS_PER_KILL)
            if reward_rolls > 0:
                draw_loading_overlay("Loading...")
                updated_deck = run_enhancement_choices(
                    enhance_rolls=reward_rolls,
                    min_enhance_tier=0,
                    min_enhance_tier_rolls=0,
                    seed=random.randrange(1 << 30),
                    width=screen_w,
                    height=screen_h,
                    player_profile=COMBAT_PLAYER_PROFILE,
                    unit_archetypes=COMBAT_UNIT_ARCHETYPES,
                    template_state=combat_template_state,
                    template_state_out=combat_template_state,
                    enhance_gacha_state=combat_enhance_gacha_state,
                    enhance_gacha_state_out=combat_enhance_gacha_state,
                )
                if updated_deck:
                    deck_preview_lines = list(updated_deck)
                # 강화 UI가 display mode를 바꾸므로 보드 화면을 복구한다.
                screen = pg.display.set_mode((screen_w, screen_h))
                pg.display.set_caption("Dice Duals - Dice Visualization")
                pg.event.clear()
            if remove_targets:
                pending_remove_token_ids = set(remove_targets)
                blue_remove_duration_ms = _enemy_phase_duration(enemy_remove_base_ms, min_ms=110)
                blue_remove_effects = [
                    {
                        "token_i": float(i),
                        "start_ms": float(now_ms),
                        "duration_ms": float(blue_remove_duration_ms),
                    }
                    for i in remove_targets
                ]
                blue_remove_return_phase = meet_return_phase
                current_phase = "BlueRemoveEffect"
                phase_timer_start_ms = now_ms
                meet_banner_until_ms = now_ms + int(900 / ANIMATION_SPEED)
                return True
            current_meeting_blue_ids = _current_meeting_blue_ids()
            meet_banner_until_ms = now_ms + int(900 / ANIMATION_SPEED)
            return False

        board_game_over = True
        board_game_over_text = "Defeat"
        current_phase = "GameOver"
        phase_timer_start_ms = now_ms
        is_charging = False
        is_resolving = False
        current_gauge = 0
        gauge_dir = 1
        return True

    def _update_meet_events(now_ms: int, excluded_ids: set[int] | None = None) -> set[int]:
        nonlocal meet_banner_until_ms, current_meeting_blue_ids
        meeting_now = _current_meeting_blue_ids(excluded_ids=excluded_ids)
        new_meets = meeting_now - current_meeting_blue_ids
        if new_meets:
            meet_banner_until_ms = now_ms + int(900 / ANIMATION_SPEED)
            for i in new_meets:
                meet_effects.append(
                    {
                        "start_ms": float(now_ms),
                        "duration_ms": float(int(850 / ANIMATION_SPEED)),
                        "blue_i": float(i),
                    }
                )
        current_meeting_blue_ids = meeting_now
        return new_meets

    def _try_begin_meet(now_ms: int, return_phase: str, excluded_ids: set[int] | None = None) -> bool:
        return _begin_meet_lightning(_update_meet_events(now_ms, excluded_ids=excluded_ids), now_ms, return_phase)

    def _try_begin_player_meet(now_ms: int) -> bool:
        return _try_begin_meet(now_ms, "PlayerPostMove")

    def _try_begin_enemy_meet(now_ms: int) -> bool:
        # 적 턴 종료 후 일괄 교전. 이번 턴에 기절이 막 풀린 적은 즉시 교전에서 제외한다.
        return _try_begin_meet(now_ms, "Throw", excluded_ids=set(blue_stun_recovered_this_turn))

    def _finalize_enemy_turn_with_meet(now_ms: int) -> bool:
        started = _try_begin_enemy_meet(now_ms)
        blue_stun_recovered_this_turn.clear()
        return started

    def _resolve_player_landed_cell(now_ms: int) -> None:
        nonlocal current_phase, phase_timer_start_ms, screen
        nonlocal t2_plus_counter, t2_plus_banner_until_ms, t2_plus_banner_text
        nonlocal deck_preview_lines, combat_template_state
        landed_cell = token_position + 1  # 1-based
        if landed_cell == ABILITY_T2_ENHANCE_CELL_1B:
            gain_rolls = t2_plus_counter
            draw_loading_overlay("Loading...")
            updated_deck = run_enhancement_choices(
                enhance_rolls=gain_rolls,
                min_enhance_tier=COMBAT_T2_PLUS_TIER,
                min_enhance_tier_rolls=gain_rolls,
                seed=random.randrange(1 << 30),
                width=screen_w,
                height=screen_h,
                player_profile=COMBAT_PLAYER_PROFILE,
                unit_archetypes=COMBAT_UNIT_ARCHETYPES,
                template_state=combat_template_state,
                template_state_out=combat_template_state,
                enhance_gacha_state=combat_enhance_gacha_state,
                enhance_gacha_state_out=combat_enhance_gacha_state,
            )
            if updated_deck:
                deck_preview_lines = list(updated_deck)
            # 강화 UI가 display mode를 바꾸므로 보드 화면을 복구한다.
            screen = pg.display.set_mode((screen_w, screen_h))
            pg.display.set_caption("Dice Duals - Dice Visualization")
            pg.event.clear()
            t2_plus_banner_text = f"T2+ 강화 선택 {gain_rolls}회"
            t2_plus_banner_until_ms = now_ms + int(1200 / ANIMATION_SPEED)
            t2_plus_counter += 1

        if landed_cell == ABILITY_ADD_BLUE_CELL_1B:
            blue_tokens.append(random.randrange(len(inner_path)))
            blue_token_types.append(_random_enemy_unit_type())
            blue_stun_turns.append(0)
            new_blue_i = len(blue_tokens) - 1
            _register_blue_spawn(new_blue_i, now_ms)
            _enqueue_blue_for_current_turn(new_blue_i, force_move=True)

        if landed_cell == ABILITY_STUN_CELL_1B and len(blue_tokens) > 0:
            current_phase = "AbilitySelectStun"
            return
        elif landed_cell == ABILITY_BARRICADE_CELL_1B:
            current_phase = "AbilitySelectBarricade"
            return
        elif landed_cell == ABILITY_REMOVE_CELL_1B and len(blue_tokens) > 0:
            current_phase = "AbilitySelectRemove"
            return

        if _try_begin_player_meet(now_ms):
            return
        current_phase = "PlayerPostMove"
        phase_timer_start_ms = now_ms

    # 게임 시작 전에 강화 선택 10회를 먼저 진행한다.
    draw_loading_overlay("Loading...")
    deck_preview_lines = run_enhancement_choices(
        enhance_rolls=COMBAT_PRE_GAME_ENHANCE_ROLLS,
        seed=random.randrange(1 << 30),
        width=screen_w,
        height=screen_h,
        player_profile=COMBAT_PLAYER_PROFILE,
        unit_archetypes=COMBAT_UNIT_ARCHETYPES,
        template_state=combat_template_state,
        template_state_out=combat_template_state,
        enhance_gacha_state=combat_enhance_gacha_state,
        enhance_gacha_state_out=combat_enhance_gacha_state,
    )
    if not deck_preview_lines:
        deck_preview_lines = ["덱 정보가 없습니다."]
    # 강화 UI가 display mode를 바꾸므로 보드 화면을 복구한다.
    screen = pg.display.set_mode((screen_w, screen_h))
    pg.display.set_caption("Dice Duals - Dice Visualization")
    pg.event.clear()

    running = True
    last_total: int | None = None
    last_is_double = False
    while running:
        mouse_pos = pg.mouse.get_pos()
        can_throw = current_phase == "Throw" and (not is_resolving) and move_left == 0
        hovered = button_rect.collidepoint(mouse_pos) and can_throw
        deck_hovered = deck_button_rect.collidepoint(mouse_pos)
        enemy_anim_active = _is_enemy_animation_phase(current_phase)
        if (not enemy_anim_active) and skip_confirm_armed:
            skip_confirm_armed = False
        skip_hovered = (
            skip_button_rect.collidepoint(mouse_pos)
            and enemy_anim_active
            and (not board_game_over)
            and (not deck_panel_open)
        )

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE and deck_panel_open:
                    deck_panel_open = False
                    continue
                if deck_panel_open:
                    continue
                if not (can_throw and (not is_charging)):
                    continue
                debug_key_map = {
                    pg.K_1: 1,
                    pg.K_2: 2,
                    pg.K_3: 3,
                    pg.K_4: 4,
                    pg.K_5: 5,
                    pg.K_6: 6,
                    pg.K_7: 7,
                    pg.K_8: 8,
                    pg.K_9: 9,
                    pg.K_0: 10,
                    pg.K_MINUS: 11,
                    pg.K_EQUALS: 12,
                }
                if event.key in debug_key_map:
                    forced_total = debug_key_map[event.key]
                    final_dice1, final_dice2, last_total, last_is_double = throw_dice_forced_total(forced_total)
                    LAST_ROLL_RESULT = {
                        "dice1": final_dice1,
                        "dice2": final_dice2,
                        "total": last_total,
                        "is_double": last_is_double,
                    }
                    ROLL_HISTORY.append(LAST_ROLL_RESULT.copy())
                    pending_move_total = last_total or 0
                    is_resolving = True
                    resolve_start_ms = pg.time.get_ticks()
                    resolve_next_shuffle_ms = resolve_start_ms
                    continue
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if skip_button_rect.collidepoint(event.pos) and (not deck_panel_open):
                    if enemy_anim_active and (not board_game_over):
                        now_skip = pg.time.get_ticks()
                        if skip_confirm_armed:
                            _skip_enemy_animation_now(now_skip)
                            skip_confirm_armed = False
                        else:
                            skip_confirm_armed = True
                    else:
                        skip_confirm_armed = False
                    continue
                if skip_confirm_armed:
                    skip_confirm_armed = False
                if deck_button_rect.collidepoint(event.pos):
                    deck_panel_open = not deck_panel_open
                    if deck_panel_open:
                        is_charging = False
                    continue
                if deck_panel_open:
                    continue
                if current_phase == "AbilitySelectStun":
                    picked = _pick_inner_cell(event.pos)
                    if picked is not None:
                        now_pick = pg.time.get_ticks()
                        _apply_cell_stun(picked)
                        _start_stun_hammer_effect(picked, now_pick)
                    continue
                if current_phase == "AbilitySelectRemove":
                    picked = _pick_inner_cell(event.pos)
                    if picked is not None:
                        now_pick = pg.time.get_ticks()
                        if not _start_remove_effect(picked, now_pick):
                            if not _try_begin_player_meet(now_pick):
                                current_phase = "PlayerPostMove"
                                phase_timer_start_ms = now_pick
                    continue
                if current_phase == "AbilitySelectBarricade":
                    picked = _pick_outer_edge(event.pos)
                    if picked is not None:
                        _apply_edge_barricade(picked)
                        now_pick = pg.time.get_ticks()
                        if not _try_begin_player_meet(now_pick):
                            current_phase = "PlayerPostMove"
                            phase_timer_start_ms = now_pick
                    continue
                if hovered:
                    is_charging = True
                    next_shuffle_ms = 0
            elif (
                event.type == pg.MOUSEBUTTONUP
                and event.button == 1
                and is_charging
                and can_throw
                and (not deck_panel_open)
            ):
                is_charging = False
                charge_ratio = min(max(current_gauge / max_gauge, 0.0), 1.0)
                final_dice1, final_dice2, last_total, last_is_double = throw_dice(charge_ratio)
                LAST_ROLL_RESULT = {
                    "dice1": final_dice1,
                    "dice2": final_dice2,
                    "total": last_total,
                    "is_double": last_is_double,
                }
                ROLL_HISTORY.append(LAST_ROLL_RESULT.copy())
                pending_move_total = last_total or 0
                is_resolving = True
                resolve_start_ms = pg.time.get_ticks()
                resolve_next_shuffle_ms = resolve_start_ms

        if is_charging and can_throw:
            current_gauge += gauge_differential * gauge_dir
            if current_gauge >= max_gauge:
                current_gauge = max_gauge
                gauge_dir = -1
            elif current_gauge <= 0:
                current_gauge = 0
                gauge_dir = 1
            now_ms = pg.time.get_ticks()
            if now_ms >= next_shuffle_ms:
                dice1_value = random.randint(1, 6)
                dice2_value = random.randint(1, 6)
                next_shuffle_ms = now_ms + shuffle_interval_ms

        if is_resolving:
            now_ms = pg.time.get_ticks()
            elapsed = max(0, now_ms - resolve_start_ms)
            p = min(1.0, elapsed / resolve_duration_ms)
            # 처음에는 빠르게 바뀌고, 끝으로 갈수록 느려지게 간격을 증가시킨다.
            interval = int((30 + (470 * (p ** 2.2))) / ANIMATION_SPEED)
            if now_ms >= resolve_next_shuffle_ms and p < 0.98:
                dice1_value = random.randint(1, 6)
                dice2_value = random.randint(1, 6)
                resolve_next_shuffle_ms = now_ms + interval
            if p >= 1.0:
                dice1_value = final_dice1
                dice2_value = final_dice2
                is_resolving = False
                current_gauge = 0
                gauge_dir = 1

                # 플레이어가 주사위를 1회 던질 때마다 파란 말 이동 계획(실제 이동은 플레이어 이후).
                blue_stun_recovered_this_turn.clear()
                if blue_tokens:
                    ordered = sorted(range(len(blue_tokens)), key=lambda i: blue_tokens[i], reverse=True)
                    blue_move_queue = [
                        _build_blue_action(order=k + 1, token_i=i)
                        for k, i in enumerate(ordered)
                    ]
                else:
                    blue_move_queue = []
                current_phase = "RollResultDelay"
                roll_result_start_ms = now_ms

        if current_phase == "RollResultDelay":
            now_ms = pg.time.get_ticks()
            if now_ms - roll_result_start_ms >= roll_result_pause_ms:
                _begin_player_pre_move(now_ms)

        if current_phase == "PlayerPreMove":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= player_pre_move_ms:
                current_phase = "Move"
                step_from_position = token_position
                step_to_position = (token_position + 1) % len(board_path)
                step_start_ms = now_ms
                step_duration_ms = compute_step_duration_ms(move_total, move_left)

        if current_phase == "Move" and move_left > 0:
            now_ms = pg.time.get_ticks()
            # 바리케이드는 칸 사이 벽이며 1회용: 통과를 시도하면 소모되고 즉시 정지
            if step_from_position in barricade_edges:
                barricade_edges.discard(step_from_position)
                move_left = 0
                player_bounce_from_position = step_from_position
                player_bounce_to_position = step_to_position
                player_bounce_start_ms = now_ms
                player_bounce_duration_ms = max(1, int(BARRICADE_BOUNCE_BASE_MS / ANIMATION_SPEED))
                current_phase = "PlayerBarricadeBounce"
                phase_timer_start_ms = now_ms
                continue
            elapsed = max(0, now_ms - step_start_ms)
            step_t = min(1.0, elapsed / float(max(1, step_duration_ms)))
            if step_t >= 1.0:
                prev_position = token_position
                token_position = step_to_position

                # 외곽 루프 한 바퀴 완주 시 내부 루프에 적 말 8마리 추가
                if prev_position == (len(board_path) - 1) and token_position == 0:
                    player_lap_count += 1
                    for _ in range(ENEMY_ADD_PER_LAP):
                        blue_tokens.append(random.randrange(len(inner_path)))
                        blue_token_types.append(_random_enemy_unit_type())
                        blue_stun_turns.append(0)
                        # 이번 턴 블루 행동 큐에도 즉시 포함시켜서 반드시 이동하게 한다.
                        new_blue_i = len(blue_tokens) - 1
                        _register_blue_spawn(new_blue_i, now_ms, stagger_ms=30 * _)
                        _enqueue_blue_for_current_turn(new_blue_i, force_move=True)

                move_left -= 1
                if move_left <= 0:
                    _resolve_player_landed_cell(now_ms)
                else:
                    step_from_position = token_position
                    step_to_position = (token_position + 1) % len(board_path)
                    step_start_ms = now_ms
                    step_duration_ms = compute_step_duration_ms(move_total, move_left)

        if current_phase == "PlayerBarricadeBounce":
            now_ms = pg.time.get_ticks()
            if now_ms - player_bounce_start_ms >= player_bounce_duration_ms:
                token_position = player_bounce_from_position
                _resolve_player_landed_cell(now_ms)

        if current_phase == "PlayerPostMove":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= player_post_move_ms:
                _begin_next_blue_pre_move(now_ms)

        if current_phase == "BluePreMove":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= _enemy_phase_duration(enemy_pre_move_base_ms, min_ms=80):
                if blue_active_token_i is None:
                    _begin_next_blue_pre_move(now_ms)
                    if current_phase == "Throw":
                        _finalize_enemy_turn_with_meet(now_ms)
                else:
                    current_phase = "InnerMove"
                    blue_active_from = blue_tokens[blue_active_token_i]
                    blue_active_to = (blue_active_from + 1) % len(inner_path)
                    blue_active_start_ms = now_ms
                    blue_active_duration_ms = _enemy_phase_duration(enemy_move_base_ms, min_ms=36)

        if current_phase == "BlueSleep":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= blue_sleep_duration_ms:
                current_phase = "BluePostMove"
                phase_timer_start_ms = now_ms

        if current_phase == "BlueStun":
            now_ms = pg.time.get_ticks()
            _begin_next_blue_pre_move(now_ms)
            if current_phase == "Throw":
                _finalize_enemy_turn_with_meet(now_ms)

        if current_phase == "BlueRemoveEffect":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= blue_remove_duration_ms:
                for remove_i in sorted(pending_remove_token_ids, reverse=True):
                    _remove_blue_by_index(remove_i)
                pending_remove_token_ids.clear()
                blue_remove_effects.clear()
                if not _try_begin_meet(now_ms, blue_remove_return_phase):
                    current_phase = blue_remove_return_phase
                    phase_timer_start_ms = now_ms

        if current_phase == "StunHammer":
            now_ms = pg.time.get_ticks()
            start_ms = int(stun_hammer_effect["start_ms"]) if stun_hammer_effect is not None else now_ms
            if now_ms - start_ms >= stun_hammer_total_ms:
                stun_hammer_effect = None
                if not _try_begin_player_meet(now_ms):
                    current_phase = "PlayerPostMove"
                    phase_timer_start_ms = now_ms

        if current_phase == "MeetLightning":
            now_ms = pg.time.get_ticks()
            if now_ms - meet_lightning_start_ms >= meet_lightning_duration_ms:
                target_ids = set(meet_pending_blue_ids)
                meet_pending_blue_ids.clear()
                meet_lightning_paths.clear()
                if not _start_meet_combat(target_ids, now_ms):
                    current_phase = meet_return_phase
                    phase_timer_start_ms = now_ms

        if current_phase == "InnerMove" and blue_active_token_i is not None:
            now_ms = pg.time.get_ticks()
            elapsed = max(0, now_ms - blue_active_start_ms)
            t = min(1.0, elapsed / float(max(1, blue_active_duration_ms)))
            if t >= 1.0:
                blue_tokens[blue_active_token_i] = blue_active_to
                blue_active_steps_left -= 1
                if blue_active_steps_left > 0:
                    blue_active_from = blue_tokens[blue_active_token_i]
                    blue_active_to = (blue_active_from + 1) % len(inner_path)
                    blue_active_start_ms = now_ms
                    blue_active_duration_ms = _enemy_phase_duration(enemy_move_base_ms, min_ms=36)
                else:
                    current_phase = "BluePostMove"
                    phase_timer_start_ms = now_ms

        if current_phase == "BluePostMove":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= _enemy_phase_duration(enemy_post_move_base_ms, min_ms=60):
                _begin_next_blue_pre_move(now_ms)
                if current_phase == "Throw":
                    _finalize_enemy_turn_with_meet(now_ms)

        # Draw
        draw_gradient(screen, screen_w, screen_h)

        # 보드 자체를 라운드 마스크로 잘라서 그린다.
        board_radius = 28
        blit_rounded_image(screen, board_image, board_rect, board_radius)
        pg.draw.rect(screen, PANEL_EDGE, board_rect, width=2, border_radius=board_radius)
        # 보드 기준(700x700) 좌표의 T2 강화 칸에 카운터 숫자를 표시한다.
        marker_x = board_rect.x + (T2_MARKER_CENTER_REF[0] * (board_rect.w / MARKER_REF_BOARD_SIZE))
        marker_y = board_rect.y + (T2_MARKER_CENTER_REF[1] * (board_rect.h / MARKER_REF_BOARD_SIZE))
        marker_center = (int(marker_x), int(marker_y))
        counter_value = str(max(1, int(t2_plus_counter)))
        counter_surface = render_aura_counter_text(counter_value, font_t2_counter)
        if counter_surface is not None:
            screen.blit(counter_surface, counter_surface.get_rect(center=marker_center))

        # 바리케이드 표시 (플레이어 외곽 보드)
        if barricade_edges:
            for edge_i in sorted(barricade_edges):
                a = board_path[edge_i % len(board_path)]
                b = board_path[(edge_i + 1) % len(board_path)]
                mx = (a[0] + b[0]) * 0.5
                my = (a[1] + b[1]) * 0.5
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                length = math.hypot(dx, dy)
                if length <= 1e-6:
                    continue
                nx = -dy / length
                ny = dx / length
                half = max(32.0, min(72.0, board_rect.w * 0.072))
                thickness = max(18.0, min(32.0, board_rect.w * 0.031))
                # 칸 사이 벽을 화면 좌표로 계산
                p1 = (mx - (nx * half), my - (ny * half))
                p2 = (mx + (nx * half), my + (ny * half))
                tx, ty = (p2[0] - p1[0], p2[1] - p1[1])
                tlen = math.hypot(tx, ty)
                if tlen <= 1e-6:
                    continue
                # 로컬 직사각형(흰 바탕) 내부에 작은 체크무늬를 채운 후 회전 배치
                w = int(max(26, tlen))
                h = int(max(16, thickness))
                pad = 6
                surf = pg.Surface((w + pad * 2, h + pad * 2), pg.SRCALPHA)
                rect = pg.Rect(pad, pad, w, h)
                pg.draw.rect(surf, (250, 250, 250), rect, border_radius=max(4, h // 4))

                # 체크는 더 작게, 그리고 흰 사각형 안쪽으로만
                inner = rect.inflate(-6, -6)
                check = max(4, int(h * 0.28))
                rows = max(1, inner.h // check + 1)
                cols = max(1, inner.w // check + 1)
                for ry in range(rows):
                    for cx in range(cols):
                        if (ry + cx) % 2 != 0:
                            continue
                        x = inner.x + (cx * check)
                        y = inner.y + (ry * check)
                        sq = pg.Rect(x, y, check, check).clip(inner)
                        if sq.w > 0 and sq.h > 0:
                            pg.draw.rect(surf, (22, 22, 22), sq)

                pg.draw.rect(surf, (20, 20, 20), rect, width=2, border_radius=max(4, h // 4))

                angle = -math.degrees(math.atan2(ty, tx))
                rot = pg.transform.rotate(surf, angle)
                screen.blit(rot, rot.get_rect(center=(int(mx), int(my))))

        # 능력칸 선택 하이라이트
        if current_phase in ("AbilitySelectStun", "AbilitySelectRemove", "AbilitySelectBarricade"):
            overlay = pg.Surface((screen_w, screen_h), pg.SRCALPHA)
            if current_phase == "AbilitySelectStun":
                ring_color = (122, 199, 255, 120)
                dot_color = (193, 229, 255, 180)
                for x, y in inner_path:
                    pg.draw.circle(overlay, ring_color, (int(x), int(y)), inner_pick_radius, width=2)
                    pg.draw.circle(overlay, dot_color, (int(x), int(y)), 4)
            elif current_phase == "AbilitySelectRemove":
                ring_color = (255, 140, 140, 120)
                dot_color = (255, 198, 198, 180)
                for x, y in inner_path:
                    pg.draw.circle(overlay, ring_color, (int(x), int(y)), inner_pick_radius, width=2)
                    pg.draw.circle(overlay, dot_color, (int(x), int(y)), 4)
            else:
                ring_color = (255, 191, 118, 125)
                dot_color = (255, 221, 170, 185)
                for i in range(len(board_path)):
                    x, y = _edge_midpoint(i)
                    pg.draw.circle(overlay, ring_color, (int(x), int(y)), outer_pick_radius, width=2)
                    pg.draw.circle(overlay, dot_color, (int(x), int(y)), 4)
            screen.blit(overlay, (0, 0))

        # 블루 활성 말 이동 trajectory (반투명 파란 선 + 종착지 마커)
        # 이동 중에는 현재 1칸을 제외한 "남은 경로"만 표시한다.
        route_start = -1
        remaining = 0
        if current_phase == "BluePreMove" and blue_active_token_i is not None and blue_active_steps_left > 0:
            route_start = blue_tokens[blue_active_token_i]
            remaining = blue_active_steps_left
        elif current_phase == "InnerMove" and blue_active_token_i is not None and blue_active_steps_left > 1:
            route_start = blue_active_to
            remaining = blue_active_steps_left - 1

        if route_start >= 0 and remaining > 0:
            route = inner_route_points(inner_path, route_start, remaining)
            if len(route) >= 2:
                fx = pg.Surface((screen_w, screen_h), pg.SRCALPHA)
                pg.draw.lines(fx, (98, 192, 255, 125), False, [(int(x), int(y)) for x, y in route], 5)
                end_x, end_y = route[-1]
                pg.draw.circle(fx, (102, 210, 255, 165), (int(end_x), int(end_y)), 14, width=3)
                pg.draw.circle(fx, (182, 238, 255, 140), (int(end_x), int(end_y)), 6)
                screen.blit(fx, (0, 0))

        # 이동 예정 칸 번호 표시 (반투명, 총 이동 수 기준 고정 번호)
        preview_steps = 0
        label_start = 1
        if current_phase in ("PlayerPreMove", "Move") and move_left > 0:
            preview_steps = move_left
            moved_steps = max(0, move_total - move_left)
            label_start = moved_steps + 1
        elif current_phase == "Throw" and can_throw and (hovered or is_charging):
            # 던지기 준비/차징 중에는 가능한 합(1~12) 전체를 미리 표시한다.
            preview_steps = 12
            label_start = 1

        if preview_steps > 0:
            corner_cells_1b = {1, 7, 13, 19}
            for offset in range(1, preview_steps + 1):
                idx = (token_position + offset) % len(board_path)
                label = label_start + (offset - 1)
                px, py = board_path[idx]
                is_corner = (idx + 1) in corner_cells_1b
                label_font = font_path_num_corner if is_corner else font_path_num
                label_text = str(label)
                txt = label_font.render(label_text, True, (255, 245, 225))
                txt_alpha = 210 if is_corner else 192
                txt.set_alpha(txt_alpha)
                txt_rect = txt.get_rect(center=(int(px), int(py)))

                outline = label_font.render(label_text, True, (12, 12, 12))
                outline.set_alpha(min(255, txt_alpha + 24))
                for ox, oy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    o_rect = outline.get_rect(center=(int(px + ox), int(py + oy)))
                    screen.blit(outline, o_rect)

                screen.blit(txt, txt_rect)

        # Token (이동 중에는 칸 단위 ease-out으로 보간)
        if current_phase == "Move" and move_left > 0:
            now_ms = pg.time.get_ticks()
            elapsed = max(0, now_ms - step_start_ms)
            raw_t = min(1.0, elapsed / float(max(1, step_duration_ms)))
            t = ease_out_cubic(raw_t)
            from_xy = board_path[step_from_position]
            to_xy = board_path[step_to_position]
            token_x, token_y = parabolic_step_position(from_xy, to_xy, t)
        elif current_phase == "PlayerBarricadeBounce":
            now_ms = pg.time.get_ticks()
            elapsed = max(0, now_ms - player_bounce_start_ms)
            p = min(1.0, elapsed / float(max(1, player_bounce_duration_ms)))
            if p <= 0.5:
                q = ease_out_cubic(p / 0.5)
                travel_t = BARRICADE_BOUNCE_REACH_T * q
            else:
                q = ease_in_cubic((p - 0.5) / 0.5)
                travel_t = BARRICADE_BOUNCE_REACH_T * (1.0 - q)
            from_xy = board_path[player_bounce_from_position]
            to_xy = board_path[player_bounce_to_position]
            token_x, token_y = parabolic_step_position(from_xy, to_xy, travel_t)
        else:
            token_x, token_y = board_path[token_position]
        now_ms_arrow = pg.time.get_ticks()
        # 보드 위 플레이어 말을 더 키우고, 지정 좌표의 중심에 정확히 맞춘다.
        player_size = max(93, int(board_rect.w * 0.1575))
        player_img = get_unit_image("Player", player_size)
        player_rect = player_img.get_rect()
        player_visual = player_img.get_bounding_rect(min_alpha=1)
        if player_visual.w > 0 and player_visual.h > 0:
            # 실제 보이는 픽셀 중심이 보드 칸 중심과 일치하도록 배치한다.
            player_rect.x = int(round(token_x - player_visual.centerx))
            player_rect.y = int(round(token_y - player_visual.centery))
        else:
            player_rect.center = (int(round(token_x)), int(round(token_y)))
        screen.blit(player_img, player_rect.topleft)
        if (current_phase in ("PlayerPreMove", "Move")) and move_left > 0:
            arrow_size = max(18, int(player_size * 0.34))
            arrow_img = get_unit_arrow_image(arrow_size)
            bob = math.sin(now_ms_arrow * 0.0105) * 6.0
            arrow_rect = arrow_img.get_rect(
                midbottom=(int(round(token_x)), int(round(player_rect.top - 6 + bob)))
            )
            screen.blit(arrow_img, arrow_rect.topleft)

        # Center cluster panel (button, bar, dice) - semi-transparent
        cluster_panel = pg.Surface(cluster_rect.size, pg.SRCALPHA)
        draw_rounded_panel(
            cluster_panel,
            cluster_panel.get_rect(),
            (24, 20, 17, 185),
            (142, 104, 62, 220),
            radius=18,
        )
        screen.blit(cluster_panel, cluster_rect.topleft)

        # 내부 루프 적 말
        blue_active_pos: tuple[float, float] | None = None
        if current_phase == "InnerMove" and blue_active_token_i is not None:
            now_ms = pg.time.get_ticks()
            elapsed = max(0, now_ms - blue_active_start_ms)
            raw_t = min(1.0, elapsed / float(max(1, blue_active_duration_ms)))
            t = ease_out_cubic(raw_t)
            blue_active_pos = parabolic_step_position(
                inner_path[blue_active_from],
                inner_path[blue_active_to],
                t,
            )

        # 내부 루프 적 말을 더 키우고, 같은 칸에서는 축소 + 균등 배치를 적용한다.
        blue_size = max(54, int(board_rect.w * 0.086))
        blue_draw_centers: dict[int, tuple[float, float]] = {}
        stationary_groups: dict[int, list[int]] = {}
        for i, idx in enumerate(blue_tokens):
            if blue_active_pos is not None and i == blue_active_token_i:
                continue
            stationary_groups.setdefault(idx, []).append(i)

        blue_layout: dict[int, tuple[float, float, int]] = {}
        for idx, token_ids in stationary_groups.items():
            cx, cy = inner_path[idx]
            count = len(token_ids)
            if count <= 1:
                blue_layout[token_ids[0]] = (cx, cy, blue_size)
                continue

            shrink_ratio = max(0.68, 1.0 - ((count - 1) * 0.10))
            stacked_size = max(28, int(blue_size * shrink_ratio))
            spread_radius = max(
                10.0,
                (stacked_size * 0.34) + ((count - 1) * (stacked_size * 0.07)),
            )
            start_angle = (-math.pi / 2.0) - (math.pi / count if (count % 2 == 0) else 0.0)

            for order, token_i in enumerate(token_ids):
                angle = start_angle + ((2.0 * math.pi * order) / count)
                px = cx + (math.cos(angle) * spread_radius)
                py = cy + (math.sin(angle) * spread_radius)
                blue_layout[token_i] = (px, py, stacked_size)

        now_ms_draw = pg.time.get_ticks()
        alive_remove_effects: list[dict[str, float]] = []
        remove_effect_progress: dict[int, float] = {}
        for e in blue_remove_effects:
            elapsed = now_ms_draw - e["start_ms"]
            duration = max(1.0, e["duration_ms"])
            if elapsed < 0 or elapsed > duration:
                continue
            alive_remove_effects.append(e)
            remove_effect_progress[int(e["token_i"])] = float(elapsed / duration)
        blue_remove_effects = alive_remove_effects
        alive_spawn_effects: list[dict[str, float]] = []
        spawn_effect_progress: dict[int, float] = {}
        for e in blue_spawn_effects:
            elapsed = now_ms_draw - e["start_ms"]
            duration = max(1.0, e["duration_ms"])
            if elapsed < 0 or elapsed > duration:
                continue
            alive_spawn_effects.append(e)
            spawn_effect_progress[int(e["token_i"])] = float(elapsed / duration)
        blue_spawn_effects = alive_spawn_effects
        summoning_active = bool(spawn_effect_progress)

        for i, idx in enumerate(blue_tokens):
            is_stunned_visual = _is_blue_stunned_visual(i)
            base_x, base_y = inner_path[idx]
            if blue_active_pos is not None and i == blue_active_token_i:
                bx, by = blue_active_pos
                draw_size = blue_size
            else:
                bx, by, draw_size = blue_layout.get(i, (base_x, base_y, blue_size))

            remove_p = remove_effect_progress.get(i)
            if remove_p is not None:
                shake_x = math.sin((now_ms_draw * 0.06) + (i * 1.7)) * (8.0 * (1.0 - remove_p))
                bx += shake_x

            blue_draw_centers[i] = (bx, by)
            unit_kind = blue_token_types[i] if 0 <= i < len(blue_token_types) else _random_enemy_unit_type()
            unit_img = get_unit_image(unit_kind, draw_size)
            spawn_p = spawn_effect_progress.get(i)
            token_rect = pg.Rect(0, 0, draw_size, draw_size)
            token_rect.center = (int(round(bx)), int(round(by)))
            draw_img = unit_img
            if remove_p is None and spawn_p is not None:
                p = max(0.0, min(1.0, spawn_p))
                eased = ease_out_cubic(p)
                summon_h = max(1, int(round(draw_size * eased)))
                draw_img = pg.transform.smoothscale(unit_img, (draw_size, summon_h))
                token_rect = draw_img.get_rect(
                    midbottom=(int(round(bx)), int(round(by + (draw_size * 0.5))))
                )

            if remove_p is not None:
                rp = max(0.0, min(1.0, remove_p))
                alpha = int(255 * (1.0 - rp))
                tmp = unit_img.copy()
                # 제거 연출 동안 이미지 자체에 붉은 기운을 점점 강하게 얹는다.
                tint_p = 0.30 + (0.70 * rp)
                tint_add = pg.Surface((token_rect.w, token_rect.h), pg.SRCALPHA)
                tint_add.fill((int(92 * tint_p), int(26 * tint_p), int(18 * tint_p), 0))
                tmp.blit(tint_add, (0, 0), special_flags=pg.BLEND_RGBA_ADD)
                tint_sub = pg.Surface((token_rect.w, token_rect.h), pg.SRCALPHA)
                tint_sub.fill((0, int(24 * tint_p), int(40 * tint_p), 0))
                tmp.blit(tint_sub, (0, 0), special_flags=pg.BLEND_RGBA_SUB)
                tmp.set_alpha(alpha)
                screen.blit(tmp, token_rect.topleft)
            else:
                screen.blit(draw_img, token_rect.topleft)
                if is_stunned_visual:
                    stun_img = get_stun_effect_image(max(16, int(draw_size * 0.92)))
                    cycle_ms = STUN_SPIN_CYCLE_MS
                    turns_per_cycle = 2  # 사이클당 정수 바퀴 회전으로 경계 이음새를 없앤다.
                    phase_ms = float((i * 137) % int(cycle_ms))
                    cycle_p = ((now_ms_draw + phase_ms) % cycle_ms) / cycle_ms
                    spin_angle = cycle_p * (360.0 * turns_per_cycle)
                    spin = pg.transform.rotozoom(stun_img, -spin_angle, 1.0)
                    spin.set_alpha(172)
                    screen.blit(spin, spin.get_rect(center=token_rect.center))

        if current_phase in ("BluePreMove", "InnerMove") and blue_active_token_i is not None and (0 <= blue_active_token_i < len(blue_tokens)):
            if current_phase == "InnerMove" and blue_active_pos is not None:
                bx, by = blue_active_pos
                active_size = blue_size
            else:
                bx, by, active_size = blue_layout.get(
                    blue_active_token_i,
                    (inner_path[blue_tokens[blue_active_token_i]][0], inner_path[blue_tokens[blue_active_token_i]][1], blue_size),
                )
            arrow_size = max(14, int(active_size * 0.44))
            arrow_img = get_unit_arrow_image(arrow_size)
            bob = math.sin((now_ms_arrow * 0.0105) + (blue_active_token_i * 0.45)) * 6.0
            arrow_rect = arrow_img.get_rect(
                midbottom=(int(round(bx)), int(round(by - (active_size * 0.56) + bob)))
            )
            screen.blit(arrow_img, arrow_rect.topleft)

        if current_phase == "StunHammer" and stun_hammer_effect is not None:
            picked_cell = int(stun_hammer_effect.get("cell_idx", -1))
            if 0 <= picked_cell < len(inner_path):
                hx, hy = inner_path[picked_cell]
                elapsed = max(0.0, now_ms_draw - float(stun_hammer_effect.get("start_ms", now_ms_draw)))
                impact_width_px = max(76.0, board_rect.w * 0.118)
                if elapsed < float(max(1, stun_hammer_swing_ms)):
                    p = min(1.0, elapsed / float(max(1, stun_hammer_swing_ms)))
                    # 천천히 시작해서 점점 빨라지며 내려치도록 가속 보간을 사용한다.
                    eased = ease_in_cubic(p)
                    # 60도 위에서 시작해 0도로 내려친다.
                    swing_angle = -HAMMER_SWING_START_DEG * (1.0 - eased)
                    blit_hammer_swing(
                        "hitting",
                        (hx, hy),
                        impact_width_px=impact_width_px,
                        angle_deg=swing_angle,
                        alpha=int(208 + (47 * eased)),
                    )
                else:
                    hold_p = min(
                        1.0,
                        (elapsed - float(stun_hammer_swing_ms)) / float(max(1, stun_hammer_hit_hold_ms)),
                    )
                    blit_hammer_swing(
                        "hit",
                        (hx, hy),
                        impact_width_px=impact_width_px,
                        angle_deg=0.0,
                        alpha=255,
                    )
                    shock = pg.Surface((screen_w, screen_h), pg.SRCALPHA)
                    shock_alpha = int(132 * (1.0 - hold_p))
                    if shock_alpha > 0:
                        pg.draw.circle(
                            shock,
                            (255, 246, 216, shock_alpha),
                            (int(hx), int(hy)),
                            max(16, int(board_rect.w * 0.028)),
                            width=2,
                        )
                        screen.blit(shock, (0, 0))

        # Meet 이펙트 렌더링 (캐릭터 기준)
        alive_effects: list[dict[str, float]] = []
        for e in meet_effects:
            elapsed = now_ms_draw - e["start_ms"]
            if elapsed < 0 or elapsed > e["duration_ms"]:
                continue
            blue_i = int(e.get("blue_i", -1))
            if blue_i not in blue_draw_centers:
                continue
            alive_effects.append(e)
            p = max(0.0, min(1.0, elapsed / e["duration_ms"]))
            alpha = int(220 * (1.0 - p))
            ring_r = int(14 + (34 * p))

            px, py = token_x, token_y
            bx, by = blue_draw_centers[blue_i]
            fx = pg.Surface((screen_w, screen_h), pg.SRCALPHA)
            line_color = (255, 214, 124, max(30, alpha // 2))
            ring_color = (255, 239, 171, alpha)
            pg.draw.line(fx, line_color, (px, py), (bx, by), width=2)
            pg.draw.circle(fx, ring_color, (int(px), int(py)), ring_r, width=3)
            pg.draw.circle(fx, ring_color, (int(bx), int(by)), ring_r, width=3)
            screen.blit(fx, (0, 0))
        meet_effects = alive_effects

        # Meet 번개 연출 (2초, 0.3초마다 랜덤 지그재그 갱신)
        if current_phase == "MeetLightning" and meet_pending_blue_ids:
            if now_ms_draw >= meet_lightning_next_refresh_ms:
                refreshed_paths: dict[int, list[tuple[list[tuple[float, float]], bool]]] = {}
                start_xy = (token_x, token_y)
                for blue_i in sorted(meet_pending_blue_ids):
                    if blue_i not in blue_draw_centers:
                        continue
                    end_xy = blue_draw_centers[blue_i]
                    dist = max(1.0, math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
                    segs = max(5, min(11, int(dist / 28.0)))
                    jitter = max(8.0, min(24.0, dist * 0.10))
                    refreshed_paths[blue_i] = _build_lightning_bundle(start_xy, end_xy, segs, jitter)
                meet_lightning_paths = refreshed_paths
                meet_lightning_next_refresh_ms = now_ms_draw + meet_lightning_refresh_ms

            lightning_fx = pg.Surface((screen_w, screen_h), pg.SRCALPHA)
            for bolts in meet_lightning_paths.values():
                for pts, is_heavy in bolts:
                    if len(pts) < 2:
                        continue
                    int_pts = [(int(x), int(y)) for x, y in pts]
                    if is_heavy:
                        pg.draw.lines(lightning_fx, (255, 188, 22, 96), False, int_pts, 11)
                        pg.draw.lines(lightning_fx, (255, 222, 96, 208), False, int_pts, 6)
                        pg.draw.lines(lightning_fx, (255, 248, 212, 236), False, int_pts, 3)
                    else:
                        pg.draw.lines(lightning_fx, (255, 194, 38, 84), False, int_pts, 7)
                        pg.draw.lines(lightning_fx, (255, 226, 118, 192), False, int_pts, 4)
                        pg.draw.lines(lightning_fx, (255, 250, 214, 232), False, int_pts, 2)
            screen.blit(lightning_fx, (0, 0))

        # 블루 수면 Z 애니메이션
        if current_phase == "BlueSleep":
            sleep_elapsed = max(0, pg.time.get_ticks() - phase_timer_start_ms)
            sleep_p = min(1.0, sleep_elapsed / float(max(1, blue_sleep_duration_ms)))
            sleep_targets = [ti for ti in blue_sleep_batch_token_ids if 0 <= ti < len(blue_tokens)]
            if (not sleep_targets) and blue_active_token_i is not None and (0 <= blue_active_token_i < len(blue_tokens)):
                sleep_targets = [blue_active_token_i]
            for order, token_i in enumerate(sleep_targets):
                bx, by = inner_path[blue_tokens[token_i]]
                phase_shift = (order % 3) * 0.08
                for i, phase in enumerate((0.0, 0.32, 0.64)):
                    p = sleep_p - phase - phase_shift
                    if p <= 0.0:
                        continue
                    p = min(1.0, p / 0.36)
                    zz_alpha = int(170 * (1.0 - p))
                    zz_size = int(16 + (12 * p))
                    zz = load_font(zz_size, "keriskedu_bold").render("Z", True, (220, 239, 255))
                    zz.set_alpha(max(0, zz_alpha))
                    z_y = by - blue_size - (26 * p) - (i * 3)
                    z_x = bx + (i * 8)
                    screen.blit(zz, zz.get_rect(center=(int(z_x), int(z_y))))

        # Gauge
        pg.draw.rect(screen, BAR_BG, gauge_rect, border_radius=8)
        fill_w = int(gauge_rect.w * (current_gauge / max_gauge))
        if fill_w > 0:
            fill_rect = pg.Rect(gauge_rect.x, gauge_rect.y, fill_w, gauge_rect.h)
            pg.draw.rect(screen, BAR_FILL, fill_rect, border_radius=8)
        pg.draw.rect(screen, (186, 154, 104), gauge_rect, width=2, border_radius=8)

        # Button
        meet_active = (not board_game_over) and (pg.time.get_ticks() <= meet_banner_until_ms)
        t2_banner_active = (not board_game_over) and (pg.time.get_ticks() <= t2_plus_banner_until_ms)
        if board_game_over:
            button_text = board_game_over_text
            button_font = font_button_bold
        elif t2_banner_active:
            button_text = t2_plus_banner_text
            button_font = font_button_bold
        elif meet_active:
            button_text = "!!ENGAGE!!"
            button_font = font_button_bold
        elif summoning_active:
            button_text = "Summoning..."
            button_font = font_button_bold
        else:
            active_enemy_name = "enemy"
            if blue_active_token_i is not None and 0 <= blue_active_token_i < len(blue_token_types):
                active_enemy_name = str(blue_token_types[blue_active_token_i])
            if current_phase == "Move":
                button_text = "Moving..."
            elif current_phase == "PlayerBarricadeBounce":
                button_text = "Blocked!"
            elif current_phase == "PlayerPreMove":
                button_text = "Player Ready..."
            elif current_phase == "PlayerPostMove":
                button_text = "Player Done..."
            elif current_phase == "MeetLightning":
                button_text = "Fighting..."
            elif current_phase == "InnerMove":
                button_text = f"{active_enemy_name} Moving..."
            elif current_phase == "BlueSleep":
                sleep_count = len([ti for ti in blue_sleep_batch_token_ids if 0 <= ti < len(blue_tokens)])
                if sleep_count <= 0:
                    sleep_count = 1 if (blue_active_token_i is not None and 0 <= blue_active_token_i < len(blue_tokens)) else 0
                sleep_count = max(1, sleep_count)
                button_text = f"{sleep_count} sleeping..."
            elif current_phase == "BlueStun":
                button_text = "..."
            elif current_phase == "BlueRemoveEffect":
                button_text = f"{active_enemy_name} removing..."
            elif current_phase == "BluePreMove":
                button_text = f"{active_enemy_name} Ready..."
            elif current_phase == "BluePostMove":
                button_text = f"..."
            elif current_phase == "AbilitySelectStun":
                button_text = "HAMMERTIME!!!"
            elif current_phase == "AbilitySelectRemove":
                button_text = "Remove all"
            elif current_phase == "AbilitySelectBarricade":
                button_text = "Place Barricade"
            elif current_phase == "StunHammer":
                button_text = "Smash!"
            elif current_phase == "RollResultDelay":
                button_text = str(last_total or "")
            elif is_resolving:
                button_text = "Rolling..."
            else:
                if is_charging:
                    button_text = "Release"
                else:
                    button_text = "Roll Dice"
            button_font = font_button
        draw_button(
            screen,
            button_rect,
            button_text,
            button_font,
            hovered,
            is_charging,
            disabled=board_game_over or is_resolving or current_phase != "Throw",
        )
        draw_button(
            screen,
            deck_button_rect,
            "덱 확인",
            font_deck_button,
            deck_hovered,
            False,
            disabled=False,
        )
        draw_button(
            screen,
            skip_button_rect,
            ("Skip?" if skip_confirm_armed else "Skip"),
            font_deck_button,
            skip_hovered,
            False,
            disabled=(not enemy_anim_active) or board_game_over or deck_panel_open,
        )

        # Dice
        for rect, value in ((dice1_rect, dice1_value), (dice2_rect, dice2_value)):
            screen.blit(get_dice(value), rect.topleft)

        if deck_panel_open:
            dim = pg.Surface((screen_w, screen_h), pg.SRCALPHA)
            dim.fill((10, 10, 14, 165))
            screen.blit(dim, (0, 0))
            draw_rounded_panel(screen, deck_panel_rect, (29, 25, 21), (168, 132, 84), radius=16)

            title = font_deck_title.render("덱 확인", True, (244, 236, 221))
            screen.blit(title, (deck_panel_rect.x + 22, deck_panel_rect.y + 18))

            content_rect = pg.Rect(
                deck_panel_rect.x + 20,
                deck_panel_rect.y + 64,
                deck_panel_rect.w - 40,
                deck_panel_rect.h - 84,
            )
            pg.draw.rect(screen, (22, 19, 16), content_rect, border_radius=10)
            pg.draw.rect(screen, (118, 93, 64), content_rect, width=1, border_radius=10)

            lines = deck_preview_lines or ["덱 정보가 없습니다."]
            yy = content_rect.y + 10
            line_h = font_deck_line.get_linesize() + 2
            truncated = False
            for entry in lines:
                for wrapped in _wrap_lines(entry, font_deck_line, content_rect.w - 18):
                    if yy + line_h > content_rect.bottom - 8:
                        truncated = True
                        break
                    txt = font_deck_line.render(wrapped, True, (216, 201, 176))
                    screen.blit(txt, (content_rect.x + 9, yy))
                    yy += line_h
                if truncated:
                    break
            if truncated:
                more = font_deck_line.render("...", True, (188, 168, 139))
                screen.blit(more, (content_rect.x + 9, content_rect.bottom - line_h))

        if board_game_over:
            over = pg.Surface((screen_w, screen_h), pg.SRCALPHA)
            over.fill((6, 8, 12, 150))
            screen.blit(over, (0, 0))
            title = font_button_bold.render("You Lose", True, (255, 172, 160))
            hint = font_button.render("Close window to exit", True, (244, 236, 221))
            title_rect = title.get_rect(center=(screen_w // 2, screen_h // 2 - 28))
            hint_rect = hint.get_rect(center=(screen_w // 2, screen_h // 2 + 22))
            screen.blit(title, title_rect)
            screen.blit(hint, hint_rect)

        pg.display.flip()
        clock.tick(60)

    pg.quit()
    return last_total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dice Duals board game")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="애니메이션 시간을 거의 0초 수준으로 줄입니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(fast=bool(args.fast))
