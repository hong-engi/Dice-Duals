import pygame as pg
import random
from pathlib import Path


def load_font(size: int) -> pg.font.Font:
    bold_path = Path("fonts/KERISKEDU_B.ttf")
    if bold_path.exists():
        return pg.font.Font(str(bold_path), size)
    return pg.font.SysFont("arial", size, bold=True)


AURA_TEXTURE_PATH = Path("images/text_effects/aura1.png")
_AURA_TEXTURE_CACHE: pg.Surface | None = None
_AURA_PATCH_CACHE: dict[tuple[int, int], pg.Surface] = {}
RUN_VARIANT_SEED = random.randrange(1 << 30)
AURA_DOWNSCALE = 0.28


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

    seed = RUN_VARIANT_SEED ^ (w * 92821) ^ (h * 68917) ^ 0xA47C
    rng = random.Random(seed)
    max_x = max(0, tiled.get_width() - w)
    max_y = max(0, tiled.get_height() - h)
    ox = rng.randint(0, max_x) if max_x > 0 else 0
    oy = rng.randint(0, max_y) if max_y > 0 else 0
    patch = tiled.subsurface(pg.Rect(ox, oy, w, h)).copy()

    _AURA_PATCH_CACHE[key] = patch
    return patch.copy()


def draw_clean_colored_one(screen, center, font):
    counter_value = "1"
    # 1. 숫자 모양 틀(알파 마스크)
    mask = font.render(counter_value, True, (255, 255, 255)).convert_alpha()
    bounds = mask.get_bounding_rect(min_alpha=1)
    if bounds.w <= 0 or bounds.h <= 0:
        return
    mask = mask.subsurface(bounds).copy()
    w, h = mask.get_size()

    patch = get_run_variant_aura_patch(w, h)
    if patch is None:
        fallback = pg.Surface((w, h), pg.SRCALPHA)
        fallback.fill((255, 145, 96, 255))
        fallback.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        screen.blit(fallback, fallback.get_rect(center=center))
        return

    # 2. 텍스처를 글자 알파로만 잘라낸다.
    patch.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)

    # 3. 화면에 그리기 (배경은 흰색 유지)
    rect = patch.get_rect(center=center)
    screen.blit(patch, rect)


def draw_mystic_one(screen: pg.Surface, center: tuple[int, int], font: pg.font.Font) -> None:
    counter_value = "1"
    glyph = font.render(counter_value, True, (255, 255, 255)).convert_alpha()
    glyph_bounds = glyph.get_bounding_rect(min_alpha=1)
    if glyph_bounds.w <= 0 or glyph_bounds.h <= 0:
        return
    glyph = glyph.subsurface(glyph_bounds).copy()
    text_w, text_h = glyph.get_size()

    gradient = pg.Surface((text_w, text_h), pg.SRCALPHA)
    for y in range(text_h):
        ty = y / max(1, text_h - 1)
        if ty < 0.5:
            p = ty / 0.5
            c = pg.Color(255, 124, 198).lerp(pg.Color(241, 91, 67), p)
        else:
            p = (ty - 0.5) / 0.5
            c = pg.Color(241, 91, 67).lerp(pg.Color(255, 181, 98), p)
        pg.draw.line(gradient, c, (0, y), (text_w - 1, y))
    highlight = pg.Surface((text_w, text_h), pg.SRCALPHA)
    pg.draw.ellipse(highlight, (255, 232, 214, 86), (-2, -6, text_w + 4, int(text_h * 0.62)))
    gradient.blit(highlight, (0, 0))
    gradient.blit(glyph, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
    gradient.blit(glyph, (0, 0), special_flags=pg.BLEND_RGBA_MIN)

    text_rect = gradient.get_rect(center=center)
    top_glint = glyph.copy()
    top_glint.fill((255, 238, 226, 90), special_flags=pg.BLEND_RGBA_MULT)
    screen.blit(top_glint, top_glint.get_rect(center=(center[0], center[1] - 1)))
    screen.blit(gradient, text_rect)


def main() -> None:
    pg.init()
    screen = pg.display.set_mode((700, 500))
    pg.display.set_caption("Mystic Counter Test")
    clock = pg.time.Clock()
    font = load_font(160)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False

        screen.fill((255, 255, 255))
        draw_clean_colored_one(screen, (350, 250), font)
        # draw_mystic_one(screen, (350, 250), font)
        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()
