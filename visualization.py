import pygame as pg
import random


# 색상 및 버튼 정의
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
button_rect = pg.Rect(10, 600, 200, 50) # x, y, width, height

def draw_button(screen, rect, text, is_hovered, is_clicked):
    color = DARK_GRAY if is_clicked else GRAY
    rect = pg.Rect(rect).inflate(10, 10) if is_hovered else pg.Rect(rect)
    pg.draw.rect(screen, color, rect)
    text_surf = font.render(text, True, BLACK)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

# Pygame 초기화
pg.init()
screen = pg.display.set_mode((500, 700))
clock = pg.time.Clock()
font = pg.font.SysFont("Arial", 50)
background = pg.image.load("board.png") # 배경 이미지 로드
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

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        # 마우스 클릭 시 주사위 굴리기
        if event.type == pg.MOUSEBUTTONDOWN and is_hovered:
            is_clicked = True
        
        if event.type == pg.MOUSEBUTTONUP and is_clicked:
            is_clicked = False
            dice1_value = random.randint(1, 6)
            dice2_value = random.randint(1, 6)
            dice_value = dice1_value + dice2_value

    # 주사위 숫자 표시
    text = font.render(str(dice_value), True, (0, 0, 0))
    screen.blit(text, (250, 600))

    pg.display.flip()
    clock.tick(60)

pg.quit()
