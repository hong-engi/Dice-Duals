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


def roll_weighted_die(charge_ratio: float, sigma: float = 1.0) -> int:
    """0~1 충전 비율에 따라 1~6의 주사위를 가중치로 뽑는다."""
    faces = [1, 2, 3, 4, 5, 6]
    weights = []
    gauge_level = 1 + (charge_ratio * 5)  # 1~6 범위

    for x in faces:
        weight = math.exp(-((x - gauge_level) ** 2) / (2 * sigma ** 2))
        weights.append(weight)

    return random.choices(faces, weights=weights)[0]


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


def main() -> None:
    pg.init()

    screen_w, screen_h = 1200, 820
    screen = pg.display.set_mode((screen_w, screen_h))
    pg.display.set_caption("Dice Duals - Dice Visualization")
    clock = pg.time.Clock()

    font_title = load_font(50, "keriskedu_bold")
    font_button = load_font(34, "keriskedu_regular")
    font_small = load_font(20, "katuri")

    # Layout
    board_size = min(int(screen_h * 0.92), 770)
    board_rect = pg.Rect(0, 0, board_size, board_size)
    board_rect.center = (screen_w // 2, screen_h // 2)

    cluster_rect = pg.Rect(0, 0, 300, 272)
    cluster_rect.center = (board_rect.centerx, board_rect.centery - 24)

    gauge_rect = pg.Rect(0, 0, 270, 20)
    gauge_rect.centerx = cluster_rect.centerx
    gauge_rect.y = cluster_rect.y + 60

    button_rect = pg.Rect(0, 0, 220, 52)
    button_rect.centerx = cluster_rect.centerx
    button_rect.y = gauge_rect.bottom+12
 
    dice_size = (104, 104)
    dice1_rect = pg.Rect(0, 0, *dice_size)
    dice2_rect = pg.Rect(0, 0, *dice_size)
    dice_gap = 22
    total_dice_w = dice_size[0] * 2 + dice_gap
    dice1_rect.x = cluster_rect.centerx - total_dice_w // 2
    dice2_rect.x = dice1_rect.right + dice_gap
    dice1_rect.y = button_rect.bottom + 12
    dice2_rect.y = dice1_rect.y

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
    gauge_differential = 5
    gauge_dir = 1
    is_charging = False
    shuffle_interval_ms = 70
    next_shuffle_ms = 0
    resolve_duration_ms = 3000
    is_resolving = False
    resolve_start_ms = 0
    resolve_next_shuffle_ms = 0
    final_dice1 = 1
    final_dice2 = 1

    # Dice state
    dice1_value = 1
    dice2_value = 1

    running = True
    while running:
        mouse_pos = pg.mouse.get_pos()
        hovered = button_rect.collidepoint(mouse_pos) and (not is_resolving)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and hovered and (not is_resolving):
                is_charging = True
                next_shuffle_ms = 0
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1 and is_charging and (not is_resolving):
                is_charging = False
                charge_ratio = min(max(current_gauge / max_gauge, 0.0), 1.0)
                final_dice1 = roll_weighted_die(charge_ratio)
                final_dice2 = roll_weighted_die(charge_ratio)
                is_resolving = True
                resolve_start_ms = pg.time.get_ticks()
                resolve_next_shuffle_ms = resolve_start_ms

        if is_charging:
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
            interval = int(30 + (470 * (p ** 2.2)))
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

        # Draw
        draw_gradient(screen, screen_w, screen_h)

        # 보드 자체를 라운드 마스크로 잘라서 그린다.
        board_radius = 28
        blit_rounded_image(screen, board_image, board_rect, board_radius)
        pg.draw.rect(screen, PANEL_EDGE, board_rect, width=2, border_radius=board_radius)

        # Center cluster panel (button, bar, dice)
        draw_rounded_panel(screen, cluster_rect, (24, 20, 17), (142, 104, 62), radius=18)

        title = font_title.render("Dice Roll", True, WHITE)
        screen.blit(title, title.get_rect(center=(cluster_rect.centerx, cluster_rect.y + 35)))

        # Gauge
        pg.draw.rect(screen, BAR_BG, gauge_rect, border_radius=8)
        fill_w = int(gauge_rect.w * (current_gauge / max_gauge))
        if fill_w > 0:
            fill_rect = pg.Rect(gauge_rect.x, gauge_rect.y, fill_w, gauge_rect.h)
            pg.draw.rect(screen, BAR_FILL, fill_rect, border_radius=8)
        pg.draw.rect(screen, (186, 154, 104), gauge_rect, width=2, border_radius=8)

        # Button
        if is_resolving:
            button_text = "Rolling..."
        else:
            button_text = "Release" if is_charging else "Roll Dice"
        draw_button(screen, button_rect, button_text, font_button, hovered, is_charging, disabled=is_resolving)

        # Dice
        for rect, value in ((dice1_rect, dice1_value), (dice2_rect, dice2_value)):
            screen.blit(get_dice(value), rect.topleft)

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()
