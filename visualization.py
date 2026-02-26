import pygame as pg
import random
import math


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

# TODO: 보드판 이미지와 칸 위치는 실제 게임에 맞게 조정 필요
# 보드판 각 칸 위치
board_positions = [
    (430, 410), (430, 330), (430, 265), (430, 200), (430, 140), (430, 90),
    (430, 30), (340, 30), (280, 30), (450, 30), (450, 30),
    (350, 450), (250, 450), (150, 450), (50, 450),
    (50, 350), (50, 250), (50, 150)
]

# 게임 단계 변수
current_phase = "Throw"

# 가중치가 적용된 주사위 굴리기 함수
def roll_weighted_die(charge_ratio, sigma=1.0):
    """
    charge_ratio: 0.0 ~ 1.0 범위로 정규화된 충전 비율
    sigma: 분포 퍼짐 정도
    """

    faces = [1, 2, 3, 4, 5, 6]
    weights = []

    gauge_level = 1 + (charge_ratio * 5)  # 1~6 범위로 변환

    for x in faces:
        weight = math.exp(-((x - gauge_level) ** 2) / (2 * sigma ** 2))
        weights.append(weight)

    result = random.choices(faces, weights=weights)[0]
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
background = pg.image.load("images/board2.png") # 배경 이미지 로드
background = pg.transform.scale(background, (500, 500)) # 배경 이미지 크기 조정

dice1_value = 1
dice2_value = 1
dice_value = dice1_value + dice2_value

token_position = 0
move_left = 0

is_clicked = False
running = True

while running:
    screen.fill(WHITE) # 화면 초기화
    screen.blit(background, (0, 0)) # 배경 이미지 그리기

    if current_phase == "Throw":
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

        if current_phase == "Throw":
            # 마우스 클릭 시 주사위 굴리기
            if event.type == pg.MOUSEBUTTONDOWN and is_hovered and move_left == 0:
                is_clicked = True
            
            if event.type == pg.MOUSEBUTTONUP and is_clicked:
                is_clicked = False

                charge_ratio = min(max(current_gauge / max_gauge, 0), 1) # 0.0 ~ 1.0 범위로 정규화
                current_gauge = 0

                dice1_value = roll_weighted_die(charge_ratio)
                dice2_value = roll_weighted_die(charge_ratio)
                dice_value = dice1_value + dice2_value
                move_left = dice_value

    # 주사위 숫자 표시
    dice1 = pg.image.load(f"images/dice{dice1_value}.png") # 주사위 이미지 로드
    dice2 = pg.image.load(f"images/dice{dice2_value}.png")

    dice1 = pg.transform.scale(dice1, (100, 100)) # 주사위 이미지 크기 조정
    dice2 = pg.transform.scale(dice2, (100, 100))
    
    screen.blit(dice1, (250, 500))
    screen.blit(dice2, (350, 500))
    
    text = font.render(str(dice_value), True, (0, 0, 0))
    screen.blit(text, (250, 600))

    if current_phase == "Throw" and move_left > 0:
        current_phase = "Move"

    elif current_phase == "Move":
        pg.time.delay(500)  # 0.5초 대기
        if move_left > 0:
            move_left -= 1
            # 토큰 위치 업데이트
            token_position = (token_position + 1) % len(board_positions)
        if move_left == 0:
            current_phase = "Throw"
    
    token_x, token_y = board_positions[token_position]
    pg.draw.circle(screen, (255, 0, 0), (token_x + 25, token_y + 25), 10) # 토큰 그리기

    pg.display.flip()
    clock.tick(60)

pg.quit()
