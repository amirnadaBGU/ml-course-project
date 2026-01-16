import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

# ייבוא המודל שלך (וודא שהקובץ קיים באותה תיקייה)
from basic_mlp_regressor import RegressionSystem,get_prawn_data_from_row,find_image_path


class VisualRefinementTracker:
    def __init__(self, model_1, model_0, max_iters=6):
        self.model_1 = model_1
        self.model_0 = model_0
        self.max_iters = max_iters

    def run_live_visualization(self, row, base_dir, p0_initial_guess, device, save_video=False,
                               video_name="output.mp4"):
        # 1. הכנת נתונים
        data = get_prawn_data_from_row(row)
        pts = data['points']

        img_path = find_image_path(data['image_stem'], base_dir)
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path)
            W, H = image.size
        else:
            image = Image.new('RGB', (640, 360), color='white')
            W, H = 640, 360

        # המרה לטנסורים
        def to_tensor(p):
            return torch.tensor([list(p)], dtype=torch.float32).to(device)

        def to_px(t):
            return t.detach().cpu().numpy()[0][0] * W, t.detach().cpu().numpy()[0][1] * H

        p3_fixed = to_tensor(pts[3])
        p2_fixed = to_tensor(pts[2])
        p1_gt = to_tensor(pts[1])
        p0_gt = to_tensor(pts[0])

        curr_p0 = p0_initial_guess.clone().to(device)
        curr_p1 = None

        px3, py3 = to_px(p3_fixed)
        px2, py2 = to_px(p2_fixed)
        px0_gt_val, py0_gt_val = to_px(p0_gt)
        px1_gt_val, py1_gt_val = to_px(p1_gt)

        # גבולות זום
        all_x = [px0_gt_val, px1_gt_val, px2, px3]
        all_y = [py0_gt_val, py1_gt_val, py2, py3]
        pad = 80
        xlims = (max(0, min(all_x) - pad), min(W, max(all_x) + pad))
        ylims = (min(H, max(all_y) + pad), max(0, min(all_y) - pad))

        # --- אתחול הוידאו ---
        video_writer = None
        if save_video:
            # אנחנו לא יודעים את גודל הגרף בפיקסלים עדיין, אז ניצור אותו בפריים הראשון
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # קודק ל-MP4
            print(f"--- Preparing to record video: {video_name} ---")

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))  # גודל החלון

        # פונקציית ציור פנימית שמחזירה את הפריים הנוכחי כתמונה
        # פונקציית ציור פנימית שמחזירה את הפריים הנוכחי כתמונה
        def draw_and_capture(ax, p0_tensor, p1_tensor, step_desc, active_point):
            ax.clear()
            ax.imshow(image)
            ax.set_xlim(xlims);
            ax.set_ylim(ylims)
            ax.set_title(f"{step_desc}", fontsize=14, fontweight='bold')

            # חישוב נקודות
            cx0, cy0 = to_px(p0_tensor)

            # P1
            if p1_tensor is not None:
                cx1, cy1 = to_px(p1_tensor)
                ax.plot([px3, cx0, cx1, px2], [py3, cy0, cy1, py2], 'r--', linewidth=2)

                alpha_p1 = 1.0 if active_point == 'p1' else 0.4
                size_p1 = 200 if active_point == 'p1' else 80
                ax.scatter(cx1, cy1, c='red', marker='X', s=size_p1, alpha=alpha_p1, edgecolors='black')
                if active_point == 'p1': ax.text(cx1, cy1 - 15, "NEW P1", color='red', fontweight='bold')
            else:
                ax.plot([px3, cx0], [py3, cy0], 'r--', linewidth=2)

            # P0
            alpha_p0 = 1.0 if active_point == 'p0' else 0.4
            size_p0 = 200 if active_point == 'p0' else 80
            ax.scatter(cx0, cy0, c='orange', marker='X', s=size_p0, alpha=alpha_p0, edgecolors='black')
            if active_point == 'p0': ax.text(cx0, cy0 - 15, "NEW P0", color='orange', fontweight='bold')

            # GT
            ax.plot([px3, px0_gt_val, px1_gt_val, px2], [py3, py0_gt_val, py1_gt_val, py2], 'g-', alpha=0.3)
            ax.scatter([px3, px2], [py3, py2], c='blue', s=60)
            ax.scatter([px0_gt_val, px1_gt_val], [py0_gt_val, py1_gt_val], c='green', alpha=0.3)

            plt.draw()
            plt.pause(0.5)  # השהייה קצרה לויזואליזציה במסך

            # --- לכידת הפריים לוידאו (התיקון כאן) ---
            frame = None
            if save_video:
                # שימוש בשיטה החדשה המותאמת ל-Matplotlib עדכני
                try:
                    canvas = fig.canvas
                    canvas.draw()

                    # קבלת הבאפר כ-RGBA
                    rgba_buffer = canvas.buffer_rgba()

                    # המרה למערך NumPy
                    img_np = np.asarray(rgba_buffer)

                    # המרה מ-RGBA (של Matplotlib) ל-BGR (של OpenCV)
                    frame = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

                except Exception as e:
                    print(f"Frame capture failed: {e}")

            return frame
        # --- לולאת הריצה ---

        # צעד 0: התחלה
        frame = draw_and_capture(ax, curr_p0, None, "Start: Initial Guess", 'p0')

        # אתחול ה-Writer בפריים הראשון (כי רק עכשיו אנחנו יודעים את הגודל המדויק)
        if save_video and video_writer is None:
            h, w, _ = frame.shape
            video_writer = cv2.VideoWriter(video_name, fourcc, 2.0, (w, h))  # 2.0 FPS איטי כדי שיראו ברור

        if video_writer: video_writer.write(frame)

        for k in range(self.max_iters):
            # 1. חיזוי P1
            inp_forward = torch.cat([p3_fixed, curr_p0, p2_fixed], dim=1)
            curr_p1 = self.model_1(inp_forward)

            frame = draw_and_capture(ax, curr_p0, curr_p1, f"Iter {k + 1}: Found P1 (Eyes)", 'p1')
            if video_writer: video_writer.write(frame)

            # 2. תיקון P0
            inp_inverse = torch.cat([p3_fixed, curr_p1, p2_fixed], dim=1)
            curr_p0 = self.model_0(inp_inverse)

            frame = draw_and_capture(ax, curr_p0, curr_p1, f"Iter {k + 1}: Refined P0 (Carapce)", 'p0')
            if video_writer: video_writer.write(frame)

        # סגירה
        if video_writer:
            video_writer.release()
            print(f"✅ Video saved successfully as: {video_name}")

        plt.ioff()
        plt.show()


# --- Main ---
def run_interactive_demo():
    BASE_DIR = 'prawn_2025_circ_small_v1'
    EXCEL_PATH = 'final_test_data.xlsx'
    CKPT_EYES = 'weights/best_model_eyes.ckpt'
    CKPT_CARAPACE = 'weights/best_model_carapce.ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(EXCEL_PATH): return
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')

    # טעינת המודלים
    model1 = RegressionSystem.load_from_checkpoint(CKPT_EYES).to(device)
    model1.eval()
    model0 = RegressionSystem.load_from_checkpoint(CKPT_CARAPACE).to(device)
    model0.eval()

    # בחירת שורה
    row_idx = int(input(f"Row index (0-{len(df) - 1}): "))
    row = df.iloc[row_idx]

    # --- השינוי בלוגיקה כאן ---
    print("\n--- Configuration ---")
    print("Enter noise offset (e.g. 0.1).")
    print("OR press [Enter] to start from the midpoint of P2 and P3.")

    val_x = input("X offset: ")

    if val_x.strip() == "":
        # אם המשתמש לחץ אנטר ללא כלום -> חישוב ממוצע בין 2 ל-3
        print("No input detected. Using midpoint (P2+P3)/2 as initial P0 guess.")
        start_x = (row['x2'] + row['x3']) / 2
        start_y = (row['y2'] + row['y3']) / 2
    else:
        # אם המשתמש הכניס מספר -> שימוש ב-GT + רעש
        val_y = input("Y offset: ")
        nx = float(val_x)
        ny = float(val_y) if val_y.strip() != "" else 0.0  # הגנה למקרה ש-Y ריק

        start_x = row['x0'] + nx
        start_y = row['y0'] + ny
        print(f"Using Ground Truth + Noise ({nx}, {ny})")

    # יצירת הטנסור
    p0_guess = torch.tensor([[start_x, start_y]], dtype=torch.float32)

    # שאלה למשתמש האם לשמור וידאו
    save_vid = input("Save video? (y/n): ").lower() == 'y'
    vid_name = f"refinement_sample_{row_idx}.mp4" if save_vid else "temp.mp4"

    tracker = VisualRefinementTracker(model1, model0, max_iters=5)

    # הפעלה
    tracker.run_live_visualization(row, BASE_DIR, p0_guess, device, save_video=save_vid, video_name=vid_name)

if __name__ == "__main__":
    run_interactive_demo()