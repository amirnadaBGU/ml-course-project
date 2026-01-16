import torch
from basic_mlp_regressor import RegressionSystem

import torch
import torch.nn as nn


class IterativeRefinementSystem:
    def __init__(self, model_predict_1, model_predict_0, max_iters=5, epsilon=1e-3):
        """
        :param model_predict_1: מקבל (P3, P2, P0) -> חוזה את P1
        :param model_predict_0: מקבל (P3, P2, P1) -> חוזה את P0 (מודל מתקן)
        :param max_iters: מספר איטרציות מקסימלי
        :param epsilon: סף עצירה
        """
        self.model_predict_1 = model_predict_1
        self.model_predict_0 = model_predict_0
        self.max_iters = max_iters
        self.epsilon = epsilon

    def predict(self, p3_fixed, p2_fixed, p0_initial_guess):
        """
        :param p3_fixed: נקודה 3 (GT) - טנסור [Batch, 2]
        :param p2_fixed: נקודה 2 (GT) - טנסור [Batch, 2]
        :param p0_initial_guess: ניחוש התחלתי לנקודה 0 - טנסור [Batch, 2]
        """
        # 1. אתחול משתנים
        curr_p0 = p0_initial_guess.clone()  # זה המשתנה שאנחנו הולכים לשפר
        curr_p1 = None

        # P3 ו-P2 הם קבועים ולא משתנים לאורך כל הלולאה!

        for k in range(self.max_iters):
            prev_p0 = curr_p0.clone()

            # --- שלב 1: חיזוי P1 (הראש) ---
            # קלט: 3, 2, 0 -> פלט: 1
            # סדר השרשור תלוי באיך שאימנת את המודל. נניח: [p3, p2, p0]
            inp_forward = torch.cat([p3_fixed, curr_p0, p2_fixed], dim=1)
            curr_p1 = self.model_predict_1(inp_forward)

            # --- שלב 2: תיקון P0 (החיבור) ---
            # קלט: 3, 2, 1 -> פלט: 0
            # עכשיו אנחנו שואלים: "בהינתן הראש שמצאנו (1) והגב הידוע (3,2), איפה אמור להיות הצוואר (0)?"
            inp_inverse = torch.cat([p3_fixed, curr_p1, p2_fixed], dim=1)
            curr_p0 = self.model_predict_0(inp_inverse)

            # --- בדיקת התכנסות ---
            # אנחנו בודקים אם התיקון של P0 היה קטן מספיק
            if k > 0:
                diff = torch.norm(curr_p0 - prev_p0, p=2)
                if diff < self.epsilon:
                    print(f"Converged at step {k}")
                    break

        return curr_p1, curr_p0


if __name__ == "__main__":
    # 1. הגדרת התקן (מומלץ לעבוד על CPU בטסטים פשוטים כדי למנוע סיבוכים)
    # אם אתה חייב ביצועים, אפשר לשנות ל-'cuda' אבל אז חובה שגם הטנסורים יעברו לשם
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    CHECKPOINT_PATH_EYES = 'weights/best_model_eyes.ckpt'
    CHECKPOINT_PATH_CARAPACE = 'weights/best_model_carapce.ckpt'

    # 2. טעינת המודלים והעברה ל-Device הנכון
    print("Loading models...")
    trained_dnn1 = RegressionSystem.load_from_checkpoint(CHECKPOINT_PATH_EYES)
    trained_dnn1.to(device)  # <--- העברה קריטית
    trained_dnn1.eval()
    trained_dnn1.freeze()    # מונע חישוב גרדיאנטים מיותר

    trained_dnn2 = RegressionSystem.load_from_checkpoint(CHECKPOINT_PATH_CARAPACE)
    trained_dnn2.to(device)  # <--- העברה קריטית
    trained_dnn2.eval()
    trained_dnn2.freeze()

    # יצירת המערכת האיטרטיבית
    system = IterativeRefinementSystem(trained_dnn1, trained_dnn2)

    # 3. יצירת הנתונים והעברה ל-Device (חובה שיהיו תואמים למודל)
    print("Preparing data...")
    # P3 ו-P2 הם קבועים (GT)
    p3_known = torch.tensor([[0.27369179, 0.55434501]], dtype=torch.float32).to(device)
    p2_known = torch.tensor([[0.28896509, 0.81054688]], dtype=torch.float32).to(device)

    # ניחוש התחלתי לנקודה P0
    p0_guess = torch.tensor([[0.28332078, 0.72180599]], dtype=torch.float32).to(device)

    # 4. הרצה
    print("Starting iterative refinement...")
    final_p1, final_p0 = system.predict(p3_known, p2_known, p0_guess)

    # 5. הדפסה (צריך להחזיר ל-CPU כדי להמיר ל-numpy)
    print(f"Result P1 (Head): {final_p1.detach().cpu().numpy()}")
    print(f"Result P0 (Neck - Refined): {final_p0.detach().cpu().numpy()}")