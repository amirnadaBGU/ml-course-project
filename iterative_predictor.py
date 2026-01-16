import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from basic_mlp_regressor import RegressionSystem  # וודא שהשם תואם לקובץ שלך


# --- 1. עדכון המחלקה כדי שתחזיר היסטוריה של שגיאות ---
class IterativeRefinementTracker:
    def __init__(self, model_predict_1, model_predict_0, max_iters=10):
        self.model_1 = model_predict_1
        self.model_0 = model_predict_0
        self.max_iters = max_iters

    def predict_and_track(self, p3_fixed, p2_fixed, p0_guess, p0_gt, p1_gt):
        """
        מבצע את הלולאה ושומר את המרחק מה-GT בכל צעד
        """
        curr_p0 = p0_guess.clone()
        curr_p1 = None

        # רשימות לשמירת ההיסטוריה של השגיאות
        errors_p0 = []  # המרחק של הצוואר מהאמת
        errors_p1 = []  # המרחק של הראש מהאמת

        # חישוב שגיאה התחלתית ל-P0 (לפני האיטרציה הראשונה)
        initial_dist_p0 = torch.dist(curr_p0, p0_gt).item()
        errors_p0.append(initial_dist_p0)
        errors_p1.append(None)  # ל-P1 עוד אין חיזוי בשלב 0

        print(f"Start: P0 Error={initial_dist_p0:.4f}")

        for k in range(self.max_iters):
            # 1. חיזוי P1 (בהינתן P3, P2, P0_curr)
            # סדר: [P3, P0, P2] (וודא שזה הסדר באימון שלך!)
            inp_forward = torch.cat([p3_fixed, curr_p0, p2_fixed], dim=1)
            curr_p1 = self.model_1(inp_forward)

            # חישוב שגיאה ל-P1
            dist_p1 = torch.dist(curr_p1, p1_gt).item()
            errors_p1.append(dist_p1)  # מעדכנים את השגיאה של P1 לאיטרציה הזו

            # 2. תיקון P0 (בהינתן P3, P2, P1_curr)
            # סדר: [P3, P1, P2]
            inp_inverse = torch.cat([p3_fixed, curr_p1, p2_fixed], dim=1)
            curr_p0 = self.model_0(inp_inverse)

            # חישוב שגיאה ל-P0
            dist_p0 = torch.dist(curr_p0, p0_gt).item()
            errors_p0.append(dist_p0)

            print(f"Iter {k + 1}: P1_Err={dist_p1:.4f}, P0_Err={dist_p0:.4f}")

        return errors_p0, errors_p1


# --- 2. פונקציה ראשית להרצה ---
def run_convergence_analysis():
    # הגדרות
    EXCEL_PATH = 'final_test_data.xlsx'  # וודא נתיב
    CKPT_EYES = 'weights/best_model_eyes.ckpt'
    CKPT_CARAPACE = 'weights/best_model_carapce.ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # טעינת מודלים
    print("Loading models...")
    model1 = RegressionSystem.load_from_checkpoint(CKPT_EYES).to(device)
    model0 = RegressionSystem.load_from_checkpoint(CKPT_CARAPACE).to(device)
    model1.eval();
    model0.eval()

    tracker = IterativeRefinementTracker(model1, model0, max_iters=20)

    # טעינת דאטה
    print("Loading Excel...")
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')

    # בחירת שורה (אקראית או ידנית)
    row_idx = int(input(f"Enter row index (0-{len(df) - 1}): "))
    row = df.iloc[row_idx]

    # חילוץ ה-Ground Truth
    # הנחה: שמות העמודות הם x0, y0, x1, y1...
    def get_tensor(c1, c2):
        return torch.tensor([[row[c1], row[c2]]], dtype=torch.float32).to(device)

    p0_gt = get_tensor('x0', 'y0')
    p1_gt = get_tensor('x1_target', 'y1_target')
    p2_fixed = get_tensor('x2', 'y2')
    p3_fixed = get_tensor('x3', 'y3')

    print(f"Selected Row {row_idx}. True P0: {p0_gt.cpu().numpy()}")

    # קבלת ניחוש התחלתי מהמשתמש (רעש)
    print("\n--- Define Initial Guess for P0 ---")
    user_offset_x = float(input("Enter X offset noise (e.g., 0.05): "))
    user_offset_y = float(input("Enter Y offset noise (e.g., 0.05): "))

    # יצירת הניחוש (האמת + הרעש)
    p0_guess = p0_gt.clone()
    p0_guess[0, 0] += user_offset_x
    p0_guess[0, 1] += user_offset_y

    print(f"Initial P0 Guess: {p0_guess.cpu().numpy()}")

    # הרצת הלולאה
    print("\nRunning Iterations...")
    err_p0, err_p1 = tracker.predict_and_track(p3_fixed, p2_fixed, p0_guess, p0_gt, p1_gt)

    # --- 3. ציור הגרף ---
    plt.figure(figsize=(10, 6))

    # ציר X: מספר האיטרציות
    iters_range = range(len(err_p0))

    plt.plot(iters_range, err_p0, marker='o', label='P0 Error (Neck)', linewidth=2, color='blue')
    # שים לב: P1 מתחיל מאיטרציה 1 ולא 0, אז נצייר אותו בהתאם
    plt.plot(range(1, len(err_p1)), err_p1[1:], marker='s', label='P1 Error (Head)', linewidth=2, color='red',
             linestyle='--')

    plt.title(f'Convergence Analysis - Sample #{row_idx}')
    plt.xlabel('Iteration Number')
    plt.ylabel('Euclidean Distance from GT')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # סימון ההתחלה והסוף
    plt.annotate('Start', (0, err_p0[0]), textcoords="offset points", xytext=(-10, 10), ha='center')
    plt.annotate(f'End: {err_p0[-1]:.4f}', (len(err_p0) - 1, err_p0[-1]), textcoords="offset points", xytext=(-10, -15),
                 ha='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_convergence_analysis()