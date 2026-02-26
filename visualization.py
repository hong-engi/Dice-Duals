import pygame as pg
import random


# 색상 및 버튼 정의
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
button_rect = pg.Rect(10, 600, 200, 50) # x, y, width, height

# 게이지 변수
max_gauge = 100
current_gauge = 0
gauge_differential = 5
gauge_dir = 1 # 1: 증가, -1: 감소
bar_width = 200
bar_height = 20

def roll_weighted_die(charge_ratio):
    """
    charge_ratio: 0.0 ~ 1.0
    """

    # 기본 가중치
    weights = [1, 1, 1, 1, 1, 1]

    # 높은 눈 가중치 증가
    weights[4] += charge_ratio * 3   # 5
    weights[5] += charge_ratio * 5   # 6

    # 가중치 기반 랜덤 선택
    result = random.choices(
        population=[1,2,3,4,5,6],
        weights=weights
    )[0]

    return result

def draw_button(screen, rect, text, is_hovered, is_clicked):
    color = DARK_GRAY if is_clicked else GRAY
    rect = pg.Rect(rect).inflate(10, 10) if is_hovered or is_clicked else pg.Rect(rect)
    pg.draw.rect(screen, color, rect)
    text_surf = font.render(text, True, BLACK)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

# Pygame 초기화
pg.init()
screen = pg.display.set_mode((500, 700))
clock = pg.time.Clock()
font = pg.font.SysFont("Arial", 50)
background = pg.image.load("images/board/board.png") # 배경 이미지 로드
background = pg.transform.scale(background, (500, 500)) # 배경 이미지 크기 조정

dice1_value = 1
dice2_value = 1
dice_value = dice1_value + dice2_value

is_clicked = False
running = True

while running:
    screen.fill(WHITE) # 화면 초기화
    screen.blit(background, (0, 0)) # 배경 이미지 그리기

    mouse_pos = pg.mouse.get_pos()
    is_hovered = button_rect.collidepoint(mouse_pos)
    draw_button(screen, button_rect, "Roll Dice", is_hovered, is_clicked)

    if is_clicked:
        current_gauge += gauge_differential * gauge_dir
        if current_gauge >= max_gauge:
            current_gauge = max_gauge
            gauge_dir = -1
        elif current_gauge <= 0:
            current_gauge = 0
            gauge_dir = 1
    
    # 게이지 바 그리기
    pg.draw.rect(screen, GRAY, (10, 560, bar_width, bar_height)) # 배경 바
    pg.draw.rect(screen, DARK_GRAY, (10, 560, int(bar_width * (current_gauge / max_gauge)), bar_height)) # 현재 게이지 바

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        # 마우스 클릭 시 주사위 굴리기
        if event.type == pg.MOUSEBUTTONDOWN and is_hovered:
            is_clicked = True
        
        if event.type == pg.MOUSEBUTTONUP and is_clicked:
            is_clicked = False

            charge_ratio = min(max(current_gauge / max_gauge, 0), 1) # 0.0 ~ 1.0 범위로 정규화
            current_gauge = 0

            dice1_value = roll_weighted_die(charge_ratio)
            dice2_value = roll_weighted_die(charge_ratio)
            dice_value = dice1_value + dice2_value

    # 주사위 숫자 표시
    dice1 = pg.image.load(f"images/dice_images/{dice1_value}.png") # 주사위 이미지 로드
    dice2 = pg.image.load(f"images/dice_images/{dice2_value}.png")

    dice1 = pg.transform.scale(dice1, (100, 100)) # 주사위 이미지 크기 조정
    dice2 = pg.transform.scale(dice2, (100, 100))
    
    screen.blit(dice1, (250, 500))
    screen.blit(dice2, (350, 500))
    
    text = font.render(str(dice_value), True, (0, 0, 0))
    screen.blit(text, (250, 600))

    pg.display.flip()
    clock.tick(60)

pg.quit()
