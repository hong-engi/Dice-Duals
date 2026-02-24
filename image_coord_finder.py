import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")

        self.image_path = None
        self.original_img = None
        self.display_pil = None
        self.display_imgtk = None
        self.canvas_img = None

        self.rect = None
        self.point = None

        self.scale = 1.0

        self.start_x = None
        self.start_y = None
        self.drag_mode = None  # 'move', 'resize', 'draw', None

        self.box_orig = None  # (x1,y1,x2,y2) in original coords

        self.btn_open = tk.Button(root, text="이미지 불러오기", command=self.open_image)
        self.btn_open.pack(pady=5)

        self.canvas = tk.Canvas(root, cursor="cross", bg="gray")
        self.canvas.pack(padx=10, pady=5)

        self.info_label = tk.Label(root, text="이미지 크기: -", font=("Arial", 10))
        self.info_label.pack()

        self.coord_label = tk.Label(root, text="좌표: -", font=("Arial", 10, "bold"), fg="blue")
        self.coord_label.pack(pady=5)

        # 마우스 이벤트
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 키 이벤트 (Backspace로 삭제)
        self.root.bind("<BackSpace>", self.on_backspace)

    # ---------- 좌표 변환 ----------
    def to_orig(self, x_disp, y_disp):
        if self.scale == 0:
            return 0, 0
        return int(round(x_disp / self.scale)), int(round(y_disp / self.scale))

    def to_disp(self, x_orig, y_orig):
        return int(round(x_orig * self.scale)), int(round(y_orig * self.scale))

    def clamp_orig(self, x, y):
        w, h = self.original_img.size
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return x, y

    # ---------- 삭제 ----------
    def clear_annotations(self):
        if self.rect:
            self.canvas.delete(self.rect)
        if self.point:
            self.canvas.delete(self.point)
        self.rect = None
        self.point = None
        self.box_orig = None
        self.update_coords("좌표: -")

    def on_backspace(self, event=None):
        # 박스/점 모두 삭제
        self.clear_annotations()

    # ---------- 이미지 로딩/축소 ----------
    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if not self.image_path:
            return

        self.original_img = Image.open(self.image_path)
        ow, oh = self.original_img.size

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        max_w = int(screen_w * 0.9)
        max_h = int(screen_h * 0.75)

        self.scale = min(1.0, max_w / ow, max_h / oh)

        # ✅ 크기는 "int()"로 통일 (round 섞이면 1px 오차 생기기 쉬움)
        dw = int(ow * self.scale)
        dh = int(oh * self.scale)

        if self.scale < 1.0:
            self.display_pil = self.original_img.resize((dw, dh), Image.LANCZOS)
        else:
            self.display_pil = self.original_img

        # ✅ resize 결과의 '진짜' 크기를 다시 읽어서 캔버스에 적용 (안전장치)
        dw, dh = self.display_pil.size

        # ✅ 캔버스의 테두리/하이라이트 제거 + 정확한 크기 적용
        self.canvas.config(
            width=dw, height=dh,
            highlightthickness=0, bd=0, relief="flat"
        )

        self.canvas.delete("all")
        self.display_imgtk = ImageTk.PhotoImage(self.display_pil)

        # ✅ anchor="nw", 좌표 (0,0)에 정확히 붙이기
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.display_imgtk)

        # ✅ 스크롤영역도 딱 이미지 크기로 (패딩처럼 보이는 영역 방지)
        self.canvas.config(scrollregion=(0, 0, dw, dh))

        self.info_label.config(text=f"원본: {ow} x {oh} | 표시: {dw} x {dh} | scale={self.scale:.4f}")
        self.clear_annotations()
        self.canvas.focus_set()

    def on_button_press(self, event):
        if not self.original_img:
            return

        self.canvas.focus_set()
        self.start_x, self.start_y = event.x, event.y

        # 기존 박스가 있으면 move/resize 판정
        if self.rect and self.box_orig:
            x1o, y1o, x2o, y2o = self.box_orig
            x1d, y1d = self.to_disp(x1o, y1o)
            x2d, y2d = self.to_disp(x2o, y2o)

            left, right = min(x1d, x2d), max(x1d, x2d)
            top, bottom = min(y1d, y2d), max(y1d, y2d)

            margin = 10
            if abs(event.x - right) < margin and abs(event.y - bottom) < margin:
                self.drag_mode = "resize"
                return
            if left < event.x < right and top < event.y < bottom:
                self.drag_mode = "move"
                return

        # --- 여기부터: 점 찍기 시작(=draw 시작) ---
        self.drag_mode = "draw"

        # ✅ 점을 찍는 순간, 기존 박스는 삭제
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        self.box_orig = None

        # 기존 점은 지우고 새 점 생성
        if self.point:
            self.canvas.delete(self.point)
            self.point = None
        self.point = self.canvas.create_oval(event.x-2.5, event.y-2.5, event.x+2.5, event.y+2.5, fill="red")

        ox, oy = self.to_orig(event.x, event.y)
        ox, oy = self.clamp_orig(ox, oy)
        self.update_coords(f"점 클릭(원본): ({ox}, {oy})")
        print(f"점 클릭(원본): ({ox}, {oy})")

    def on_move_press(self, event):
        if not self.original_img:
            return

        cur_x, cur_y = event.x, event.y

        if self.drag_mode == "draw":
            # 드래그해서 박스 그리기 시작하면 시작점 점은 제거
            if self.point:
                self.canvas.delete(self.point)
                self.point = None

            x1o, y1o = self.to_orig(self.start_x, self.start_y)
            x2o, y2o = self.to_orig(cur_x, cur_y)
            x1o, y1o = self.clamp_orig(x1o, y1o)
            x2o, y2o = self.clamp_orig(x2o, y2o)

            self.box_orig = (x1o, y1o, x2o, y2o)
            self.redraw_rect_from_orig()

        elif self.drag_mode == "move" and self.box_orig:
            dx_disp = cur_x - self.start_x
            dy_disp = cur_y - self.start_y
            dx_orig = int(round(dx_disp / self.scale))
            dy_orig = int(round(dy_disp / self.scale))

            x1o, y1o, x2o, y2o = self.box_orig
            nx1, ny1 = x1o + dx_orig, y1o + dy_orig
            nx2, ny2 = x2o + dx_orig, y2o + dy_orig

            ow, oh = self.original_img.size
            minx, maxx = min(nx1, nx2), max(nx1, nx2)
            miny, maxy = min(ny1, ny2), max(ny1, ny2)

            shift_x = 0
            shift_y = 0
            if minx < 0: shift_x = -minx
            if maxx >= ow: shift_x = (ow - 1) - maxx
            if miny < 0: shift_y = -miny
            if maxy >= oh: shift_y = (oh - 1) - maxy

            self.box_orig = (nx1 + shift_x, ny1 + shift_y, nx2 + shift_x, ny2 + shift_y)
            self.redraw_rect_from_orig()

            self.start_x, self.start_y = cur_x, cur_y

        elif self.drag_mode == "resize" and self.box_orig:
            x1o, y1o, x2o, y2o = self.box_orig
            nx2, ny2 = self.to_orig(cur_x, cur_y)
            nx2, ny2 = self.clamp_orig(nx2, ny2)
            self.box_orig = (x1o, y1o, nx2, ny2)
            self.redraw_rect_from_orig()

        self.display_rect_coords()

    def on_button_release(self, event):
        self.drag_mode = None
        self.display_rect_coords()

    def redraw_rect_from_orig(self):
        if not self.box_orig:
            return
        x1o, y1o, x2o, y2o = self.box_orig
        x1d, y1d = self.to_disp(x1o, y1o)
        x2d, y2d = self.to_disp(x2o, y2o)

        if self.rect:
            self.canvas.coords(self.rect, x1d, y1d, x2d, y2d)
        else:
            self.rect = self.canvas.create_rectangle(x1d, y1d, x2d, y2d, outline="lime", width=1)

    def display_rect_coords(self):
        if self.box_orig:
            x1, y1, x2, y2 = self.box_orig
            x1n, x2n = min(x1, x2), max(x1, x2)
            y1n, y2n = min(y1, y2), max(y1, y2)
            self.update_coords(f"영역(Box, 원본): [{x1n}, {y1n}, {x2n}, {y2n}]")

    def update_coords(self, text):
        self.coord_label.config(text=text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()
