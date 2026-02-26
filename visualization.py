import math
import random
from pathlib import Path

import pygame as pg


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
ABILITY_REMOVE_CELL_1B = 19


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


def main() -> int | None:
    global LAST_ROLL_RESULT
    pg.init()

    screen_w, screen_h = 1200, 820
    screen = pg.display.set_mode((screen_w, screen_h))
    pg.display.set_caption("Dice Duals - Dice Visualization")
    clock = pg.time.Clock()

    font_button = load_font(34, "keriskedu_regular")
    font_button_bold = load_font(34, "keriskedu_bold")
    font_path_num = load_font(26, "keriskedu_bold")
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
    pre_move_ms = int(1000 / ANIMATION_SPEED)
    post_move_ms = int(500 / ANIMATION_SPEED)
    phase_timer_start_ms = 0
    token_position = 0
    move_total = 0
    move_left = 0
    pending_move_total = 0
    step_from_position = 0
    step_to_position = 0
    step_start_ms = 0
    step_duration_ms = 220
    player_lap_count = 0
    roll_result_pause_ms = int(520 / ANIMATION_SPEED)
    roll_result_start_ms = 0

    # 내부 루프 파란 말(독립 이동)
    blue_tokens: list[int] = [random.randrange(len(inner_path)), random.randrange(len(inner_path))]
    blue_stun_turns: list[int] = [0 for _ in blue_tokens]
    blue_move_queue: list[dict[str, int | str]] = []
    blue_active_token_i: int | None = None
    blue_active_order = 0
    blue_active_steps_left = 0
    blue_active_mode = "move"
    blue_active_from = 0
    blue_active_to = 0
    blue_active_start_ms = 0
    blue_active_duration_ms = int(155 / ANIMATION_SPEED)
    blue_sleep_duration_ms = int(1000 / ANIMATION_SPEED)
    blue_stun_duration_ms = int(700 / ANIMATION_SPEED)
    blue_remove_duration_ms = int(650 / ANIMATION_SPEED)
    meet_banner_until_ms = 0
    meet_effects: list[dict[str, float]] = []
    current_meeting_blue_ids: set[int] = set()
    inner_pick_radius = max(16, int(board_rect.w * 0.03))
    outer_pick_radius = max(16, int(board_rect.w * 0.028))
    # i는 i번째 칸 -> i+1번째 칸 사이(순환)의 벽을 의미
    barricade_edges: set[int] = set()
    blue_remove_effects: list[dict[str, float]] = []
    pending_remove_token_ids: set[int] = set()

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

    def _begin_next_blue_pre_move(now_ms: int) -> None:
        nonlocal current_phase, phase_timer_start_ms
        nonlocal blue_active_token_i, blue_active_order, blue_active_steps_left, blue_active_mode
        if not blue_move_queue:
            blue_active_token_i = None
            blue_active_order = 0
            blue_active_steps_left = 0
            blue_active_mode = "move"
            current_phase = "Throw"
            return
        head = blue_move_queue.pop(0)
        blue_active_token_i = head["token_i"]
        blue_active_order = head["order"]
        blue_active_mode = head["mode"]
        blue_active_steps_left = head["steps"]
        if blue_active_mode == "sleep":
            current_phase = "BlueSleep"
            phase_timer_start_ms = now_ms
        elif blue_active_mode == "stun":
            current_phase = "BlueStun"
            phase_timer_start_ms = now_ms
        else:
            current_phase = "BluePreMove"
            phase_timer_start_ms = now_ms

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
        nonlocal blue_active_token_i, current_meeting_blue_ids
        if not (0 <= token_i < len(blue_tokens)):
            return
        blue_tokens.pop(token_i)
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

    def _apply_cell_stun(cell_idx: int) -> None:
        targets = [i for i, p in enumerate(blue_tokens) if p == cell_idx]
        if not targets:
            return
        for target_i in targets:
            blue_stun_turns[target_i] = max(blue_stun_turns[target_i], 1)
            _retarget_queue_to_stun(target_i)

    def _start_remove_effect(cell_idx: int, now_ms: int) -> bool:
        nonlocal current_phase, phase_timer_start_ms
        nonlocal blue_remove_effects, pending_remove_token_ids
        targets = [i for i, p in enumerate(blue_tokens) if p == cell_idx]
        if not targets:
            return False
        pending_remove_token_ids = set(targets)
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

    def _update_meet_events(now_ms: int) -> None:
        nonlocal meet_banner_until_ms, current_meeting_blue_ids
        meeting_now: set[int] = set()
        for i, bpos in enumerate(blue_tokens):
            if (token_position, bpos) in MEET_PAIRS_0B:
                meeting_now.add(i)

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

    def _resolve_player_landed_cell(now_ms: int) -> None:
        nonlocal current_phase, phase_timer_start_ms
        _update_meet_events(now_ms)
        landed_cell = token_position + 1  # 1-based
        if landed_cell == ABILITY_ADD_BLUE_CELL_1B:
            blue_tokens.append(random.randrange(len(inner_path)))
            blue_stun_turns.append(0)
            _enqueue_blue_for_current_turn(len(blue_tokens) - 1, force_move=True)

        if landed_cell == ABILITY_STUN_CELL_1B and len(blue_tokens) > 0:
            current_phase = "AbilitySelectStun"
        elif landed_cell == ABILITY_BARRICADE_CELL_1B:
            current_phase = "AbilitySelectBarricade"
        elif landed_cell == ABILITY_REMOVE_CELL_1B and len(blue_tokens) > 0:
            current_phase = "AbilitySelectRemove"
        else:
            current_phase = "PlayerPostMove"
            phase_timer_start_ms = now_ms

    running = True
    last_total: int | None = None
    last_is_double = False
    while running:
        mouse_pos = pg.mouse.get_pos()
        can_throw = current_phase == "Throw" and (not is_resolving) and move_left == 0
        hovered = button_rect.collidepoint(mouse_pos) and can_throw

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and can_throw and (not is_charging):
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
                if current_phase == "AbilitySelectStun":
                    picked = _pick_inner_cell(event.pos)
                    if picked is not None:
                        _apply_cell_stun(picked)
                        current_phase = "PlayerPostMove"
                        phase_timer_start_ms = pg.time.get_ticks()
                    continue
                if current_phase == "AbilitySelectRemove":
                    picked = _pick_inner_cell(event.pos)
                    if picked is not None:
                        now_pick = pg.time.get_ticks()
                        if not _start_remove_effect(picked, now_pick):
                            current_phase = "PlayerPostMove"
                            phase_timer_start_ms = now_pick
                    continue
                if current_phase == "AbilitySelectBarricade":
                    picked = _pick_outer_edge(event.pos)
                    if picked is not None:
                        _apply_edge_barricade(picked)
                        current_phase = "PlayerPostMove"
                        phase_timer_start_ms = pg.time.get_ticks()
                    continue
                if hovered:
                    is_charging = True
                    next_shuffle_ms = 0
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1 and is_charging and can_throw:
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
            if now_ms - phase_timer_start_ms >= pre_move_ms:
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
                _resolve_player_landed_cell(now_ms)
                continue
            elapsed = max(0, now_ms - step_start_ms)
            step_t = min(1.0, elapsed / float(max(1, step_duration_ms)))
            if step_t >= 1.0:
                prev_position = token_position
                token_position = step_to_position

                # 외곽 루프 한 바퀴 완주 시 내부 루프에 파란 말 1개 추가
                if prev_position == (len(board_path) - 1) and token_position == 0:
                    player_lap_count += 1
                    blue_tokens.append(random.randrange(len(inner_path)))
                    blue_stun_turns.append(0)
                    # 이번 턴 블루 행동 큐에도 즉시 포함시켜서 반드시 이동하게 한다.
                    new_blue_i = len(blue_tokens) - 1
                    _enqueue_blue_for_current_turn(new_blue_i, force_move=True)

                move_left -= 1
                if move_left <= 0:
                    _resolve_player_landed_cell(now_ms)
                else:
                    step_from_position = token_position
                    step_to_position = (token_position + 1) % len(board_path)
                    step_start_ms = now_ms
                    step_duration_ms = compute_step_duration_ms(move_total, move_left)

        if current_phase == "PlayerPostMove":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= post_move_ms:
                _begin_next_blue_pre_move(now_ms)

        if current_phase == "BluePreMove":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= pre_move_ms:
                if blue_active_token_i is None:
                    _begin_next_blue_pre_move(now_ms)
                else:
                    current_phase = "InnerMove"
                    blue_active_from = blue_tokens[blue_active_token_i]
                    blue_active_to = (blue_active_from + 1) % len(inner_path)
                    blue_active_start_ms = now_ms

        if current_phase == "BlueSleep":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= blue_sleep_duration_ms:
                current_phase = "BluePostMove"
                phase_timer_start_ms = now_ms

        if current_phase == "BlueStun":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= blue_stun_duration_ms:
                current_phase = "BluePostMove"
                phase_timer_start_ms = now_ms

        if current_phase == "BlueRemoveEffect":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= blue_remove_duration_ms:
                for remove_i in sorted(pending_remove_token_ids, reverse=True):
                    _remove_blue_by_index(remove_i)
                pending_remove_token_ids.clear()
                blue_remove_effects.clear()
                current_phase = "PlayerPostMove"
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
                else:
                    _update_meet_events(now_ms)
                    current_phase = "BluePostMove"
                    phase_timer_start_ms = now_ms

        if current_phase == "BluePostMove":
            now_ms = pg.time.get_ticks()
            if now_ms - phase_timer_start_ms >= post_move_ms:
                _begin_next_blue_pre_move(now_ms)

        # Draw
        draw_gradient(screen, screen_w, screen_h)

        # 보드 자체를 라운드 마스크로 잘라서 그린다.
        board_radius = 28
        blit_rounded_image(screen, board_image, board_rect, board_radius)
        pg.draw.rect(screen, PANEL_EDGE, board_rect, width=2, border_radius=board_radius)

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
        if current_phase in ("PlayerPreMove", "Move") and move_left > 0:
            moved_steps = max(0, move_total - move_left)
            start_label = moved_steps + 1
            for offset in range(1, move_left + 1):
                idx = (token_position + offset) % len(board_path)
                label = start_label + (offset - 1)
                px, py = board_path[idx]
                txt = font_path_num.render(str(label), True, (255, 245, 225))
                txt.set_alpha(120)
                txt_rect = txt.get_rect(center=(int(px), int(py)))
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
        else:
            token_x, token_y = board_path[token_position]
        player_highlight = current_phase == "PlayerPreMove"
        token_radius = max(12, int(board_rect.w * 0.02))
        player_fill = (255, 124, 108) if player_highlight else (239, 96, 96)
        player_edge = (255, 250, 182) if player_highlight else (255, 235, 214)
        pg.draw.circle(screen, player_fill, (int(token_x), int(token_y)), token_radius)
        pg.draw.circle(screen, player_edge, (int(token_x), int(token_y)), token_radius, width=3)
        if player_highlight:
            pg.draw.circle(screen, (255, 223, 140), (int(token_x), int(token_y)), token_radius + 4, width=2)

        # Center cluster panel (button, bar, dice)
        draw_rounded_panel(screen, cluster_rect, (24, 20, 17), (142, 104, 62), radius=18)

        # 내부 루프 파란 네모 말
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

        blue_size = max(16, int(board_rect.w * 0.028))
        blue_border = max(2, int(blue_size * 0.14))
        stack_counts: dict[int, int] = {}
        blue_draw_centers: dict[int, tuple[float, float]] = {}
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

        for i, idx in enumerate(blue_tokens):
            is_blue_active = i == blue_active_token_i and current_phase == "BluePreMove"
            is_stunned_visual = _is_blue_stunned_visual(i)
            if blue_active_pos is not None and i == blue_active_token_i:
                bx, by = blue_active_pos
                ox = 0.0
                oy = 0.0
            else:
                order = stack_counts.get(idx, 0)
                stack_counts[idx] = order + 1
                bx, by = inner_path[idx]
                ox = ((order % 3) - 1) * (blue_size * 0.5)
                oy = (order // 3) * (blue_size * 0.45)

            remove_p = remove_effect_progress.get(i)
            if remove_p is not None:
                shake_x = math.sin((now_ms_draw * 0.06) + (i * 1.7)) * (8.0 * (1.0 - remove_p))
                bx += shake_x
            elif is_stunned_visual:
                stun_shake_x = math.sin((now_ms_draw * 0.085) + (i * 1.9)) * 4.5
                bx += stun_shake_x

            blue_draw_centers[i] = (bx + ox, by + oy)
            token_rect = pg.Rect(0, 0, blue_size, blue_size)
            token_rect.center = (int(bx + ox), int(by + oy))

            if remove_p is not None:
                rp = max(0.0, min(1.0, remove_p))
                alpha = int(255 * (1.0 - rp))
                fill = (
                    int(92 + ((236 - 92) * rp)),
                    int(162 + ((78 - 162) * rp)),
                    int(246 + ((78 - 246) * rp)),
                    alpha,
                )
                edge = (255, 206, 206, alpha)
                tmp = pg.Surface((blue_size + 14, blue_size + 14), pg.SRCALPHA)
                local_rect = pg.Rect(0, 0, blue_size, blue_size)
                local_rect.center = (tmp.get_width() // 2, tmp.get_height() // 2)
                pg.draw.rect(tmp, fill, local_rect, border_radius=max(3, blue_size // 5))
                pg.draw.rect(tmp, edge, local_rect, width=blue_border, border_radius=max(3, blue_size // 5))
                screen.blit(tmp, tmp.get_rect(center=token_rect.center))
            else:
                fill = (118, 194, 255) if is_blue_active else (92, 162, 246)
                edge = (255, 248, 176) if is_blue_active else (223, 243, 255)
                pg.draw.rect(screen, fill, token_rect, border_radius=max(3, blue_size // 5))
                pg.draw.rect(screen, edge, token_rect, width=blue_border, border_radius=max(3, blue_size // 5))
                if is_blue_active:
                    glow_rect = token_rect.inflate(8, 8)
                    pg.draw.rect(screen, (255, 223, 140), glow_rect, width=2, border_radius=max(4, blue_size // 4))
                if is_stunned_visual:
                    center_r = max(7, blue_size // 3)
                    pg.draw.circle(screen, (178, 114, 255), token_rect.center, center_r)
                    pg.draw.circle(screen, (238, 219, 255), token_rect.center, center_r, width=2)

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

        # 블루 수면 Z 애니메이션
        if current_phase == "BlueSleep" and blue_active_token_i is not None:
            bx, by = inner_path[blue_tokens[blue_active_token_i]]
            sleep_elapsed = max(0, pg.time.get_ticks() - phase_timer_start_ms)
            sleep_p = min(1.0, sleep_elapsed / float(max(1, blue_sleep_duration_ms)))
            for i, phase in enumerate((0.0, 0.32, 0.64)):
                p = sleep_p - phase
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
        meet_active = pg.time.get_ticks() <= meet_banner_until_ms
        if meet_active:
            button_text = "MEET!"
            button_font = font_button_bold
        else:
            if current_phase == "Move":
                button_text = "Moving..."
            elif current_phase == "PlayerPreMove":
                button_text = "Player Ready..."
            elif current_phase == "PlayerPostMove":
                button_text = "Player Done..."
            elif current_phase == "InnerMove":
                if blue_active_order > 0:
                    button_text = f"Blue{blue_active_order} Moving..."
                else:
                    button_text = "Blue Waiting..."
            elif current_phase == "BlueSleep":
                button_text = f"blue {blue_active_order} sleeping..." if blue_active_order > 0 else "blue sleeping..."
            elif current_phase == "BlueStun":
                button_text = f"blue {blue_active_order} stunned..." if blue_active_order > 0 else "blue stunned..."
            elif current_phase == "BlueRemoveEffect":
                button_text = "blue removing..."
            elif current_phase == "BluePreMove":
                button_text = f"Blue{blue_active_order} Ready..." if blue_active_order > 0 else "Blue Ready..."
            elif current_phase == "BluePostMove":
                button_text = f"Blue{blue_active_order} Done..." if blue_active_order > 0 else "Blue Done..."
            elif current_phase == "AbilitySelectStun":
                button_text = "Select Inner Cell (Stun All)"
            elif current_phase == "AbilitySelectRemove":
                button_text = "Select Inner Cell (Remove All)"
            elif current_phase == "AbilitySelectBarricade":
                button_text = "Place Barricade"
            elif current_phase == "RollResultDelay":
                button_text = str(last_total or "")
            elif is_resolving:
                button_text = "Rolling..."
            else:
                button_text = "Release" if is_charging else "Roll Dice"
            button_font = font_button
        draw_button(
            screen,
            button_rect,
            button_text,
            button_font,
            hovered,
            is_charging,
            disabled=is_resolving or current_phase != "Throw",
        )

        # Dice
        for rect, value in ((dice1_rect, dice1_value), (dice2_rect, dice2_value)):
            screen.blit(get_dice(value), rect.topleft)

        pg.display.flip()
        clock.tick(60)

    pg.quit()
    return last_total


if __name__ == "__main__":
    main()
