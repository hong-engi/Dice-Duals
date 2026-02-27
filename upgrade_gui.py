from __future__ import annotations

import random
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, Optional

from cards import (
    CardState,
    EnhancementEngine,
    Tier,
    TIER_LABELS_BY_NAME,
    TIERS_NAME,
    describe_card,
    diff_card,
    enhancement_counts_line,
    load_cards,
    save_cards,
)
from unit import AttackProfile, DefenseProfile, Player
from upgrade_main import EnhanceGacha, format_applied_option, format_extra_effects


class UpgradeGUI(tk.Tk):
    def __init__(self, n_choices: int = 3):
        super().__init__()
        self.title("Dice Duals - Card Upgrade GUI")
        self.geometry("1200x820")

        self.engine = EnhancementEngine.load("card_definitions.json")
        self.cards = load_cards("cards.json")
        self.gacha = EnhanceGacha()
        self.dummy_player = Player(
            unit_id="dummy_player",
            name="Dummy Player",
            max_hp=1.0,
            hp=1.0,
            attack=AttackProfile(power=100.0),
            defense=DefenseProfile(armor=0.0, shield_power=100.0),
        )
        self.n_choices = n_choices

        self.count = 0
        self.last_result = None
        self.last_before = None
        self.last_after = None
        self.last_card_label = ""
        self.preview_photo = None
        self.preview_source_image = None
        self.preview_resize_after_id = None
        self.preview_caption_text = ""

        self.status_var = tk.StringVar(value="준비 완료")
        self.forced_var = tk.StringVar(value="천장 티어: 없음")
        self.target_tier_var = tk.StringVar(value="4")
        self.preview_card_index_var = tk.StringVar(value="1")
        self.preview_caption_var = tk.StringVar(value="")

        self._build_ui()
        self.refresh_all()

    def _build_ui(self) -> None:
        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        tk.Button(top, text="강화 1회", width=12, command=self.roll_once).pack(side="left", padx=4)
        tk.Label(top, text="목표 티어(tN):").pack(side="left", padx=(16, 4))
        tk.Spinbox(top, from_=0, to=6, width=5, textvariable=self.target_tier_var).pack(side="left")
        tk.Button(top, text="목표까지 자동", width=12, command=self.roll_until_target).pack(side="left", padx=4)
        tk.Button(top, text="리셋", width=10, command=self.reset_all).pack(side="left", padx=4)
        tk.Button(top, text="저장", width=10, command=self.save_all).pack(side="left", padx=4)

        info = tk.Frame(self)
        info.pack(fill="x", padx=10)
        tk.Label(info, textvariable=self.forced_var, anchor="w").pack(side="left")
        tk.Label(info, textvariable=self.status_var, anchor="e", fg="#444").pack(side="right")

        middle = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        middle.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.Frame(middle)
        right = tk.Frame(middle)
        middle.add(left, minsize=560)
        middle.add(right, minsize=560)

        tk.Label(left, text="보유 카드").pack(anchor="w")
        self.cards_text = ScrolledText(left, height=18, wrap="word")
        self.cards_text.pack(fill="both", expand=True, pady=(4, 8))

        tk.Label(left, text="최근 결과 로그").pack(anchor="w")
        self.log_text = ScrolledText(left, height=18, wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=(4, 0))

        tk.Label(right, text="현재 강화 확률").pack(anchor="w")
        self.probs_text = ScrolledText(right, height=8, wrap="none")
        self.probs_text.pack(fill="x", pady=(4, 8))

        preview_bar = tk.Frame(right)
        preview_bar.pack(fill="x", pady=(0, 4))
        tk.Label(preview_bar, text="카드 미리보기").pack(side="left")
        tk.Label(preview_bar, text="카드 #").pack(side="left", padx=(12, 4))
        self.preview_spinbox = tk.Spinbox(
            preview_bar,
            from_=1,
            to=max(1, len(self.cards)),
            width=5,
            textvariable=self.preview_card_index_var,
        )
        self.preview_spinbox.pack(side="left")
        tk.Button(preview_bar, text="보기", width=8, command=self.show_selected_preview).pack(side="left", padx=6)

        tk.Label(right, textvariable=self.preview_caption_var, anchor="w").pack(fill="x")
        self.preview_image_frame = tk.Frame(right, bd=1, relief="solid", bg="#111111", height=360)
        self.preview_image_frame.pack(fill="both", expand=True, pady=(4, 8))
        self.preview_image_frame.pack_propagate(False)
        self.preview_image_label = tk.Label(
            self.preview_image_frame,
            text="(미리보기 없음)",
            bg="#111111",
            fg="#DDDDDD",
        )
        self.preview_image_label.pack(fill="both", expand=True)
        self.preview_image_frame.bind("<Configure>", self.on_preview_frame_configure)

        tk.Label(right, text="선택지 안내").pack(anchor="w")
        self.help_text = ScrolledText(right, height=8, wrap="word")
        self.help_text.pack(fill="x", pady=(4, 0))
        self.help_text.insert(
            "1.0",
            "강화 1회를 누르면 선택지 창이 열립니다.\n"
            "- 각 항목은 '카드 + 티어 + 옵션 + 추가효과 예고'를 보여줍니다.\n"
            "- 항목 선택 시 해당 카드에 강화가 적용됩니다.\n\n"
            "목표까지 자동은 매 회차 '가장 높은 티어'를 자동 선택합니다.\n"
            "동점이면 랜덤으로 고릅니다.\n",
        )
        self.help_text.config(state="disabled")

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def refresh_all(self) -> None:
        self.refresh_cards()
        self.refresh_probs()
        if self.last_after is not None:
            self.update_preview(self.last_after, self.last_card_label or f"{self.last_after.name} ({self.last_after.id})")
        elif self.cards:
            self.show_selected_preview()
        else:
            self.update_preview(None, "보유 카드가 없습니다.")

    def refresh_cards(self) -> None:
        lines = []
        for i, c in enumerate(self.cards, start=1):
            lines.append(f"[{i}] {c.name} ({c.id})")
            lines.append(f"  {describe_card(c, self.dummy_player)}")
            lines.append(
                "  티어 강화 횟수: "
                + enhancement_counts_line(c, [t.name for t in Tier], TIER_LABELS_BY_NAME)
            )
            lines.append("")
        self._set_text(self.cards_text, "\n".join(lines).strip() or "보유 카드가 없습니다.")
        self.preview_spinbox.config(to=max(1, len(self.cards)))
        if not self.preview_card_index_var.get().isdigit():
            self.preview_card_index_var.set("1")

    def refresh_probs(self) -> None:
        probs = self.gacha._adjusted_probs()
        forced = self.gacha._forced_min_tier()
        forced_name = TIERS_NAME[Tier(forced)] if forced >= 0 else "없음"
        self.forced_var.set(f"천장 티어: {forced_name}")

        lines = []
        for i in range(len(Tier)):
            lines.append(f"{TIERS_NAME[Tier(i)]:10s} {probs[i]*100:6.3f}%")
        self._set_text(self.probs_text, "\n".join(lines))

    def append_log(self, text: str) -> None:
        self.log_text.config(state="normal")
        if self.log_text.get("1.0", tk.END).strip():
            self.log_text.insert(tk.END, "\n" + ("-" * 60) + "\n")
        self.log_text.insert(tk.END, text.rstrip() + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _selected_preview_index(self) -> int:
        try:
            idx = int(self.preview_card_index_var.get())
        except Exception:
            idx = 1
        if idx < 1:
            idx = 1
        if idx > max(1, len(self.cards)):
            idx = max(1, len(self.cards))
        return idx

    @staticmethod
    def _resize_for_preview(img, max_w: int, max_h: int):
        from PIL import Image, ImageFilter

        src_w, src_h = img.size
        if src_w <= 0 or src_h <= 0:
            return img
        scale = min(max_w / src_w, max_h / src_h, 1.0)
        out_w = max(1, int(src_w * scale))
        out_h = max(1, int(src_h * scale))
        if (out_w, out_h) == (src_w, src_h):
            return img.copy()

        try:
            resized = img.resize((out_w, out_h), Image.Resampling.LANCZOS, reducing_gap=3.0)
        except TypeError:
            resized = img.resize((out_w, out_h), Image.Resampling.LANCZOS)

        # 다운스케일이 큰 경우 텍스트/라인 경계가 흐려지는 것을 약하게 보정한다.
        if scale < 0.8:
            resized = resized.filter(ImageFilter.UnsharpMask(radius=0.8, percent=110, threshold=2))
        return resized

    def render_preview_image(self) -> None:
        if self.preview_source_image is None:
            self.preview_photo = None
            self.preview_image_label.config(image="", text="(미리보기 없음)")
            self.preview_caption_var.set(self.preview_caption_text)
            return

        try:
            from PIL import ImageTk

            self.update_idletasks()
            max_w = max(220, self.preview_image_label.winfo_width() - 16)
            max_h = max(300, self.preview_image_label.winfo_height() - 16)
            img = self._resize_for_preview(self.preview_source_image, max_w=max_w, max_h=max_h)
            photo = ImageTk.PhotoImage(img)
            self.preview_photo = photo
            self.preview_image_label.config(image=photo, text="")
            self.preview_caption_var.set(self.preview_caption_text)
        except Exception as e:
            self.preview_photo = None
            self.preview_image_label.config(image="", text=f"미리보기 실패\n{e}")
            self.preview_caption_var.set(self.preview_caption_text)

    def on_preview_frame_configure(self, _event=None) -> None:
        if self.preview_resize_after_id is not None:
            try:
                self.after_cancel(self.preview_resize_after_id)
            except Exception:
                pass
        self.preview_resize_after_id = self.after(80, self.render_preview_image)

    def update_preview(self, card: Optional[CardState], caption: str = "") -> None:
        if card is None:
            self.preview_photo = None
            self.preview_source_image = None
            self.preview_caption_text = caption
            self.preview_image_label.config(image="", text="(미리보기 없음)")
            self.preview_caption_var.set(caption)
            return
        try:
            self.preview_source_image = card.render_visual(self.engine).copy()
            self.preview_caption_text = caption or f"{card.name} ({card.id})"
            self.render_preview_image()
        except Exception as e:
            self.preview_photo = None
            self.preview_source_image = None
            self.preview_caption_text = caption or f"{card.name} ({card.id})"
            self.preview_image_label.config(image="", text=f"미리보기 실패\n{e}")
            self.preview_caption_var.set(self.preview_caption_text)

    def show_selected_preview(self) -> None:
        if not self.cards:
            self.update_preview(None, "보유 카드가 없습니다.")
            return
        idx = self._selected_preview_index()
        self.preview_card_index_var.set(str(idx))
        card = self.cards[idx - 1]
        self.update_preview(card, f"{card.name} ({card.id})")

    def build_choice_plans(self):
        if not self.cards:
            raise ValueError("보유 카드가 없습니다.")
        slot_count = self.n_choices
        if slot_count <= 0:
            raise ValueError("보유 카드가 없습니다.")

        picks = self.gacha.preview_choice_tiers(n_choices=slot_count)
        candidate_card_indices = random.choices(range(len(self.cards)), k=slot_count)
        choice_plans = []

        for idx, cidx in zip(picks, candidate_card_indices):
            card = self.cards[cidx]
            planned_opt = self.engine.preview_option(card, Tier(idx).name, rng=random)
            if planned_opt is None:
                disp = "(옵션 없음)"
                planned_effects = []
            else:
                disp = planned_opt.get("display", planned_opt.get("id", ""))
                if planned_opt.get("max_range", False):
                    disp = f"{disp} (max range)"
                planned_effects = planned_opt.get("effects", [])

            higher_gap_for_choice = max(0, max(picks) - idx) if picks else 0
            eff_rate, lower_rate, eff2_rate, same_tier_rate = self.engine.get_proc_rates(
                Tier(idx).name, higher_tier_gap=higher_gap_for_choice
            )
            eff_eligible = bool(planned_opt) and any(
                self.engine._is_efficiency_eligible(eff) for eff in planned_effects
            )
            efficiency_proc = eff_eligible and (random.random() < eff_rate)
            efficiency_double_proc = eff_eligible and (random.random() < eff2_rate)
            lower_bonus_proc = idx > 0 and (random.random() < lower_rate)
            same_tier_bonus_proc = random.random() < same_tier_rate
            if efficiency_double_proc:
                efficiency_proc = False
            if same_tier_bonus_proc:
                lower_bonus_proc = False

            preview_parts = []
            if efficiency_proc:
                preview_parts.append("x1.5")
            if efficiency_double_proc:
                preview_parts.append("x2")
            if lower_bonus_proc:
                preview_parts.append("하위 티어 보너스!")
            if same_tier_bonus_proc:
                preview_parts.append("동일 티어 보너스!")
            preview_roll_text = " [" + "] [".join(preview_parts) + "]" if preview_parts else ""

            choice_plans.append(
                {
                    "idx": idx,
                    "card_idx": cidx,
                    "card": card,
                    "opt": planned_opt,
                    "higher_gap": higher_gap_for_choice,
                    "display": disp,
                    "efficiency_proc": efficiency_proc,
                    "efficiency_double_proc": efficiency_double_proc,
                    "lower_bonus_proc": lower_bonus_proc,
                    "same_tier_bonus_proc": same_tier_bonus_proc,
                    "preview_roll_text": preview_roll_text,
                }
            )

        return picks, choice_plans

    def choose_plan_dialog(self, choice_plans) -> Optional[int]:
        dlg = tk.Toplevel(self)
        dlg.title("강화 선택지")
        dlg.geometry("860x560")
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text=f"강화 선택지 {len(choice_plans)}개", font=("", 12, "bold")).pack(anchor="w", padx=12, pady=8)

        selected = tk.IntVar(value=0)
        body = ScrolledText(dlg, wrap="word", height=24)
        body.pack(fill="both", expand=True, padx=12, pady=4)
        body.config(state="normal")
        for i, plan in enumerate(choice_plans, start=1):
            idx = plan["idx"]
            c = plan["card"]
            preview_result = {
                "efficiency_proc": plan.get("efficiency_proc", False),
                "efficiency_double_proc": plan.get("efficiency_double_proc", False),
                "lower_bonus_proc": plan.get("lower_bonus_proc", False),
                "same_tier_bonus_proc": plan.get("same_tier_bonus_proc", False),
            }
            extra_text = format_extra_effects(preview_result)
            body.insert(
                tk.END,
                f"{i}) {c.name} ({c.id})\n"
                f"   {describe_card(c, self.dummy_player)}\n"
                f"   {TIERS_NAME[Tier(idx)]} - {plan.get('display','')}{plan.get('preview_roll_text','')}\n"
                + (f"   보너스 예고: {extra_text}\n\n" if extra_text else "\n"),
            )
        body.config(state="disabled")

        rb_frame = tk.Frame(dlg)
        rb_frame.pack(fill="x", padx=12, pady=6)
        for i in range(len(choice_plans)):
            tk.Radiobutton(rb_frame, text=f"{i+1}번 선택", variable=selected, value=i).pack(side="left", padx=8)

        result = {"idx": None}

        def on_confirm():
            result["idx"] = selected.get()
            dlg.destroy()

        def on_cancel():
            result["idx"] = None
            dlg.destroy()

        btns = tk.Frame(dlg)
        btns.pack(fill="x", padx=12, pady=10)
        tk.Button(btns, text="확인", width=10, command=on_confirm).pack(side="right", padx=6)
        tk.Button(btns, text="취소", width=10, command=on_cancel).pack(side="right")

        self.wait_window(dlg)
        return result["idx"]

    def apply_choice_plan(self, chosen_plan, picks) -> Dict[str, Any]:
        chosen_idx = chosen_plan["idx"]
        card = chosen_plan["card"]
        self.last_card_label = f"{card.name} ({card.id})"

        self.gacha.commit_choice(chosen_idx)
        tier_str = Tier(chosen_idx).name

        before = CardState.from_dict(card.to_dict())
        chosen_opt = chosen_plan.get("opt")
        higher_gap = chosen_plan.get("higher_gap", max(0, max(picks) - chosen_idx) if picks else 0)
        applied = self.engine.apply_tier(
            card,
            tier_str,
            rng=random,
            selected_option=chosen_opt,
            higher_tier_gap=higher_gap,
            forced_efficiency_proc=chosen_plan.get("efficiency_proc"),
            forced_efficiency_double_proc=chosen_plan.get("efficiency_double_proc"),
            forced_lower_bonus_proc=chosen_plan.get("lower_bonus_proc"),
            forced_same_tier_bonus_proc=chosen_plan.get("same_tier_bonus_proc"),
        )

        self.count += 1
        self.last_result = applied
        self.last_before = before
        self.last_after = CardState.from_dict(card.to_dict())
        return applied

    def log_last_result(self) -> None:
        if self.last_result is None or self.last_before is None or self.last_after is None:
            return

        rolled = self.last_result["rolled_tier"]
        used = self.last_result["tier_used"]
        rolled_name = TIERS_NAME[Tier[rolled]]
        used_name = TIERS_NAME[Tier[used]]

        lines = [f"{self.count:02d}회차 결과"]
        if self.last_card_label:
            lines.append(f"대상 카드: {self.last_card_label}")
        if rolled == used:
            lines.append(f"뽑힌 강화: {rolled_name}")
        else:
            lines.append(f"뽑힌 강화: {rolled_name} -> 적용: {used_name}")

        extra_effects = format_extra_effects(self.last_result)
        if extra_effects:
            lines.append(f"추가 효과: {extra_effects}")
        lines.append(f"적용 옵션: {format_applied_option(self.last_result)}")

        changes = diff_card(self.last_before, self.last_after)
        lines.append("[변화]")
        if changes:
            lines.extend(changes)
        else:
            lines.append("변화 없음")

        self.append_log("\n".join(lines))

    def roll_once(self) -> None:
        try:
            picks, plans = self.build_choice_plans()
        except Exception as e:
            self.status_var.set(str(e))
            return

        selected_idx = self.choose_plan_dialog(plans)
        if selected_idx is None:
            self.status_var.set("취소됨")
            return

        chosen_plan = plans[selected_idx]
        self.apply_choice_plan(chosen_plan, picks)
        self.log_last_result()
        self.status_var.set("강화 1회 적용 완료")
        self.refresh_all()

    def roll_until_target(self) -> None:
        try:
            target = int(self.target_tier_var.get())
        except Exception:
            target = 4
        target = max(0, min(6, target))

        loops = 0
        max_loops = 10000
        try:
            while loops < max_loops:
                picks, plans = self.build_choice_plans()
                best = max(p["idx"] for p in plans)
                cand = [i for i, p in enumerate(plans) if p["idx"] == best]
                selected_idx = random.choice(cand)
                chosen_plan = plans[selected_idx]
                applied = self.apply_choice_plan(chosen_plan, picks)
                loops += 1
                used = applied.get("tier_used", "COMMON")
                if Tier[used].value >= target:
                    self.log_last_result()
                    self.status_var.set(f"t{target}: {loops}번 시도 끝에 {TIERS_NAME[Tier[used]]} 획득")
                    self.refresh_all()
                    return
        except Exception as e:
            self.status_var.set(f"자동 강화 중 오류: {e}")
            self.refresh_all()
            return

        self.status_var.set(f"t{target}: {max_loops}번을 초과하여 중단")
        self.refresh_all()

    def reset_all(self) -> None:
        self.cards = load_cards("cards.json")
        self.gacha.reset_state()
        self.count = 0
        self.last_result = None
        self.last_before = None
        self.last_after = None
        self.last_card_label = ""
        self._set_text(self.log_text, "")
        self.status_var.set("리셋 완료")
        self.refresh_all()

    def save_all(self) -> None:
        path = filedialog.asksaveasfilename(
            title="cards 저장",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            self.status_var.set("저장 취소")
            return
        try:
            out = save_cards(self.cards, path)
            self.status_var.set(f"저장 완료: {out}")
        except Exception as e:
            messagebox.showerror("저장 실패", str(e))
            self.status_var.set("저장 실패")


if __name__ == "__main__":
    app = UpgradeGUI(n_choices=3)
    app.mainloop()
