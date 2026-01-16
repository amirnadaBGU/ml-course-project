import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import random


# --- פונקציית עזר: הכנת דאטה לטסט ---
def prepare_test_data(input_file, output_file):
    if os.path.exists(output_file):
        # אם הקובץ קיים, לא צריך ליצור מחדש
        return

    print(f"Processing {input_file}...")
    df = pd.read_excel(input_file, engine='openpyxl')

    pivot_df = df.pivot_table(
        index=['image_stem', 'object_id'],
        columns='keypoint_index',
        values=['x_norm', 'y_norm']
    )

    final_df = pd.DataFrame()
    final_df['x3'] = pivot_df[('x_norm', 3)]
    final_df['y3'] = pivot_df[('y_norm', 3)]
    final_df['x0'] = pivot_df[('x_norm', 0)]
    final_df['y0'] = pivot_df[('y_norm', 0)]
    final_df['x2'] = pivot_df[('x_norm', 2)]
    final_df['y2'] = pivot_df[('y_norm', 2)]
    final_df['x1_target'] = pivot_df[('x_norm', 1)]
    final_df['y1_target'] = pivot_df[('y_norm', 1)]

    final_df = final_df.dropna()
    final_df.to_excel(output_file, index=False)
    print(f"Saved processed test data to {output_file}")


# --- 1. Dataset ---
class PrawnDataset(Dataset):
    def __init__(self, xlsx_file):
        data = pd.read_excel(xlsx_file, engine='openpyxl')
        self.X = torch.tensor(data[['x3', 'y3', 'x0', 'y0', 'x2', 'y2']].values, dtype=torch.float32)
        self.y = torch.tensor(data[['x1_target', 'y1_target']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- 2. DataModule ---
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch


class PrawnDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, batch_size=32):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        # שלב ה-Fit (אימון וולידציה)
        if stage == 'fit' or stage is None:
            # 1. טוענים את כל הדאטה של האימון
            full_train_dataset = PrawnDataset(self.train_file)

            # 2. מחשבים את הגדלים לחלוקה (70/30)
            train_size = int(0.30 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size

            # 3. מבצעים את החלוקה הרנדומלית
            # מומלץ להוסיף generator לקבעיות (reproducibility) אם תרצה
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        # שלב ה-Test (נפרד לחלוטין)
        if stage == 'test' or stage is None:
            self.test_dataset = PrawnDataset(self.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # כעת הולידציה מגיעה מתוך ה-30% שהפרשנו מה-Train File
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        # זה הפונקציה החדשה שתשתמש בקובץ ה-Test המקורי
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# --- 3. System (Model) ---
class RegressionSystem(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

# --- 4. פונקציית הערכה ---
import torch
import numpy as np
from torch import nn


def evaluate_model(model, datamodule, stage='test'):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # פונקציית עזר פנימית
    def run_evaluation(dataloader, name):
        all_preds = []
        all_targets = []
        running_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        final_loss = running_loss / len(dataloader.dataset)

        # --- Method 1: Pixel Error Logic ---
        W, H = 640, 360
        preds_px = all_preds.copy()
        targets_px = all_targets.copy()

        preds_px[:, 0] *= W
        preds_px[:, 1] *= H
        targets_px[:, 0] *= W
        targets_px[:, 1] *= H

        diff = preds_px - targets_px

        # חישוב MAE לכל ציר (עבור הטבלה הסופית)
        axis_mae = np.mean(np.abs(diff), axis=0)

        # החזרת מילון נתונים לשימוש חיצוני
        return {
            'loss': final_loss,
            'mae_axis_px': axis_mae,  # זה המפתח החשוב לטבלה החדשה
            'preds': all_preds,
            'targets': all_targets
        }

    results = {}
    if stage == 'fit':
        # הרצת Train ו-Val
        results['train'] = run_evaluation(datamodule.train_dataloader(), "Train")
        results['val'] = run_evaluation(datamodule.val_dataloader(), "Val")

    else:  # stage == 'test'
        # הרצת Test
        results['test'] = run_evaluation(datamodule.test_dataloader(), "Test")

    return results


def visualize_single_sample_simple(model, dataset, idx=0):
    # 1. שליפת הנתונים
    inputs, target_gt = dataset[idx]

    # inputs = [x3, y3, x0, y0, x2, y2]
    # target = [x1, y1]

    # 2. ביצוע חיזוי (Forward Pass)
    model.eval()
    with torch.no_grad():
        # הוספת מימד Batch כי המודל מצפה ל-[Batch, 6]
        pred = model(inputs.unsqueeze(0)).squeeze(0)

    # 3. המרת טנסורים למספרים רגילים
    # קלטים (ידועים)
    x3, y3 = inputs[0].item(), inputs[1].item()
    x0, y0 = inputs[2].item(), inputs[3].item()
    x2, y2 = inputs[4].item(), inputs[5].item()

    # יעד אמיתי (Ground Truth)
    x1_gt, y1_gt = target_gt[0].item(), target_gt[1].item()

    # יעד חזוי (Prediction)
    x1_pred, y1_pred = pred[0].item(), pred[1].item()

    # 4. המרה לפיקסלים (לצורך תצוגה יפה, נניח 640x360)
    W, H = 640, 360

    # פונקציית עזר להמרה
    def to_px(x, y): return x * W, y * H

    px3, py3 = to_px(x3, y3)
    px0, py0 = to_px(x0, y0)
    px2, py2 = to_px(x2, y2)
    px1_gt, py1_gt = to_px(x1_gt, y1_gt)
    px1_pred, py1_pred = to_px(x1_pred, y1_pred)

    # 5. ציור הגרף
    plt.figure(figsize=(10, 6))

    # --- ציור המבנה המקורי (Ground Truth) ---
    # קווים: 3 -> 0 -> 1 -> 2
    x_coords_gt = [px3, px0, px1_gt, px2]
    y_coords_gt = [py3, py0, py1_gt, py2]

    # קו כחול/ירוק שמחבר את הנקודות המקוריות
    plt.plot(x_coords_gt, y_coords_gt, linestyle='-', color='green', linewidth=2, label='Original Structure (GT)',
             zorder=1)

    # ציור הנקודות הידועות (3, 0, 2)
    plt.scatter([px3, px0, px2], [py3, py0, py2], color='blue', s=80, label='Input Points (3,0,2)', zorder=2)

    # ציור נקודת היעד האמיתית (1 GT)
    plt.scatter(px1_gt, py1_gt, color='green', s=100, marker='o', label='True Target (1)', zorder=3)

    # --- ציור החיזוי (Prediction) ---
    # נקודה אדומה גדולה לחיזוי
    plt.scatter(px1_pred, py1_pred, color='red', s=150, marker='X', label='Predicted (1)', zorder=4)

    # אופציונלי: קו מקווקו שמראה איך החיזוי היה נראה (0 -> חיזוי -> 2)
    plt.plot([px0, px1_pred, px2], [py0, py1_pred, py2], linestyle='--', color='red', alpha=0.5,
             label='Predicted Structure')

    # --- כיתובים (Annotations) ---
    offset = 10
    plt.annotate('3', (px3, py3), xytext=(offset, offset), textcoords='offset points', fontsize=12, fontweight='bold')
    plt.annotate('0', (px0, py0), xytext=(offset, offset), textcoords='offset points', fontsize=12, fontweight='bold')
    plt.annotate('2', (px2, py2), xytext=(offset, offset), textcoords='offset points', fontsize=12, fontweight='bold')
    plt.annotate('1 (Real)', (px1_gt, py1_gt), xytext=(offset, offset), textcoords='offset points', color='green',
                 fontweight='bold')
    plt.annotate('1 (Pred)', (px1_pred, py1_pred), xytext=(offset, -20), textcoords='offset points', color='red',
                 fontweight='bold')

    # הגדרות ציר (היפוך Y כדי שיתאים לתמונה)
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.title(f'Sample #{idx}: Keypoint Structure Visualization')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.show()


def visualize_single_sample_advanced():
    # הגדרות נתיבים - עדכן כאן אם צריך
    BASE_DIR = 'prawn_2025_circ_small_v1'
    XLSX_PATH = os.path.join(BASE_DIR, 'test_data_all.xlsx')
    CHECKPOINT_PATH = 'weights/best_model.ckpt'

    print("--- Starting Visualization Process ---")

    # א. טעינת הדאטה הגולמי
    print(f"1. Loading data from {XLSX_PATH}...")
    try:
        df = pd.read_excel(XLSX_PATH, engine='openpyxl')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # ב. בחירת סרטן וחילוץ נקודות
    print("2. Selecting a random prawn...")
    try:
        prawn_data = get_random_prawn_data(df)
    except Exception as e:
        print(e)
        return

    print(f"   Selected: Image='{prawn_data['image_stem']}', Object ID={prawn_data['object_id']}")

    pts = prawn_data['points']
    # הכנת הקלט למודל: [x3, y3, x0, y0, x2, y2]
    # נקודה 0: חיבור גוף-ראש, נקודה 1: קצה ראש (target), נקודה 2: עין, נקודה 3: גב
    input_list = [
        pts[3][0], pts[3][1],  # Point 3
        pts[0][0], pts[0][1],  # Point 0
        pts[2][0], pts[2][1]  # Point 2
    ]
    input_tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)  # הוספת מימד Batch

    # ג. טעינת המודל והרצה
    print(f"3. Loading model from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        print("   Error: Checkpoint not found! Please train the model first.")
        return

    model = RegressionSystem.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model.to('cpu')  # עבודה על CPU לויזואליזציה

    with torch.no_grad():
        pred = model(input_tensor).squeeze(0)  # תוצאה: [x1_pred, y1_pred]

    x1_pred, y1_pred = pred[0].item(), pred[1].item()
    x1_true, y1_true = pts[1][0], pts[1][1]

    print(f"   Prediction: ({x1_pred:.4f}, {y1_pred:.4f})")
    print(f"   Ground Truth: ({x1_true:.4f}, {y1_true:.4f})")

    # ד. טעינת התמונה
    print("4. Loading image...")
    img_path = find_image_path(prawn_data['image_stem'], BASE_DIR)

    if img_path:
        print(f"   Found image: {img_path}")
        image = Image.open(img_path)
        W, H = image.size
    else:
        print(f"   Warning: Image not found. Creating a blank white image.")
        W, H = 640, 360  # רזולוציית ברירת מחדל
        image = Image.new('RGB', (W, H), color='white')

    # ה. ויזואליזציה
    print("5. Displaying results...")

    # פונקציית המרה מנורמלי לפיקסלים
    def to_px(norm_x, norm_y):
        return norm_x * W, norm_y * H

    # המרת כל הנקודות
    px0, py0 = to_px(pts[0][0], pts[0][1])
    px1_gt, py1_gt = to_px(pts[1][0], pts[1][1])
    px1_pred, py1_pred = to_px(x1_pred, y1_pred)
    px2, py2 = to_px(pts[2][0], pts[2][1])
    px3, py3 = to_px(pts[3][0], pts[3][1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- צד שמאל: סכמטי ---
    ax1 = axes[0]
    ax1.set_title("Schematic View (Normalized Space)")
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)  # (0,0) בפינה שמאלית עליונה

    # חיבור קווים (GT): 3 -> 0 -> 1 -> 2
    ax1.plot([px3, px0, px1_gt, px2], [py3, py0, py1_gt, py2], 'g-', alpha=0.5, label='GT Skeleton')
    # חיבור קווים (Pred): 0 -> Pred -> 2
    ax1.plot([px0, px1_pred, px2], [py0, py1_pred, py2], 'r--', alpha=0.5, label='Pred Skeleton')

    # ציור הנקודות
    ax1.scatter([px3, px0, px2], [py3, py0, py2], c='blue', s=80, label='Input (3,0,2)')
    ax1.scatter(px1_gt, py1_gt, c='green', s=100, label='Target (1)')
    ax1.scatter(px1_pred, py1_pred, c='red', marker='X', s=150, label='Prediction')
    ax1.legend()
    ax1.grid(True)

    # --- צד ימין: על התמונה ---
    ax2 = axes[1]
    ax2.set_title("Overlay on Original Image")
    ax2.imshow(image)

    # קווים
    ax2.plot([px3, px0, px1_gt, px2], [py3, py0, py1_gt, py2], 'g-', linewidth=2, alpha=0.7)
    ax2.plot([px0, px1_pred, px2], [py0, py1_pred, py2], 'r--', linewidth=2, alpha=0.7)

    # נקודות
    ax2.scatter([px3, px0, px2], [py3, py0, py2], c='blue', s=50, edgecolors='white', zorder=5)
    ax2.scatter(px1_gt, py1_gt, c='green', s=80, edgecolors='white', zorder=5, label='True Head')
    ax2.scatter(px1_pred, py1_pred, c='red', marker='X', s=120, edgecolors='white', zorder=6, label='Pred Head')

    # טקסט
    offset = 10
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(px3 + offset, py3, "3", fontsize=8, bbox=props)
    ax2.text(px0 + offset, py0, "0", fontsize=8, bbox=props)
    ax2.text(px2 + offset, py2, "2", fontsize=8, bbox=props)

    ax2.legend()

    plt.tight_layout()
    plt.show()


def find_image_path(image_stem, base_dir):
    """
    מחפש את קובץ התמונה בתיקיות train ו-val.
    """
    # רשימת התיקיות שבהן נחפש
    search_dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        # נתיבים נוספים אפשריים אם המבנה שונה:
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'valid', 'images')
    ]

    # וידוא סיומת
    filename = image_stem if image_stem.lower().endswith(('.jpg', '.png', '.jpeg')) else f"{image_stem}.jpg"

    for folder in search_dirs:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path

    return None


def get_random_prawn_data(df):
    """
    בוחר סרטן אקראי מהדאטה, בודק תקינות ומחזיר את הנקודות שלו.
    """
    # יצירת מזהה ייחודי לכל סרטן (שם תמונה + מזהה אובייקט)
    # אנו מניחים שהעמודות הן: image_stem, object_id, keypoint_index, x_norm, y_norm
    groups = df.groupby(['image_stem', 'object_id'])
    all_keys = list(groups.groups.keys())

    if not all_keys:
        raise ValueError("No valid groups found in the dataset.")

    # מנסים למצוא סרטן תקין (עם כל 4 הנקודות)
    max_attempts = 100
    for _ in range(max_attempts):
        random.seed()
        selected_key = random.choice(all_keys)
        group = groups.get_group(selected_key)

        # בדיקה: האם יש לנו את כל הנקודות הנדרשות (0, 1, 2, 3)?
        # אנו צריכים לוודא שקיימות שורות עבור כל אינדקס
        available_kps = group['keypoint_index'].unique()
        required_kps = {0, 1, 2, 3}

        if not required_kps.issubset(set(available_kps)):
            continue  # דלג לסרטן הבא אם חסרה נקודה

        # חילוץ הנקודות למילון נוח
        points = {}
        for _, row in group.iterrows():
            k_idx = int(row['keypoint_index'])
            points[k_idx] = (row['x_norm'], row['y_norm'])

        return {
            'image_stem': selected_key[0],
            'object_id': selected_key[1],
            'points': points
        }

    raise RuntimeError("Could not find a valid prawn with all 4 keypoints after multiple attempts.")


# --- Main Logic ---
if __name__ == "__main__":

    MODE = 'eval'  # train eval or eval_visual_simple or eval_visual_advanced


    train_file = 'final_train_data.xlsx'
    test_file = 'final_test_data.xlsx'

    # נתיב קבוע למודל (כך נדע איפה לחפש אותו ב-eval)
    checkpoint_path = 'weights/best_model.ckpt'

    pl.seed_everything(42)

    # הכנת המודול של הדאטה
    data_module = PrawnDataModule(train_file, test_file)

    if MODE == 'train':
        print("--- Starting Training Mode ---")
        model = RegressionSystem()

        # שומר את המודל הכי טוב בשם 'best_model.ckpt' בתיקיית weights
        checkpoint_callback = ModelCheckpoint(
            dirpath='weights',
            filename='best_model',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )

        logger = CSVLogger("logs", name="prawn_regression")

        trainer = pl.Trainer(
            max_epochs=30,
            callbacks=[checkpoint_callback],
            logger=logger,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10
        )

        trainer.fit(model, data_module)
        print("Training finished.")

        # --- ציור גרף Matplotlib רגיל ---
        metrics = pd.read_csv(f"{logger.log_dir}/metrics.csv")
        train_loss = metrics[['epoch', 'train_loss']].dropna().set_index('epoch')
        val_loss = metrics[['epoch', 'val_loss']].dropna().set_index('epoch')

        plt.figure(figsize=(10, 5))
        plt.plot(train_loss.index, train_loss['train_loss'], label='Train Loss')
        plt.plot(val_loss.index, val_loss['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_plot_lightning.png')
        plt.show()

    elif MODE == 'eval':

        print("\n" + "=" * 40)

        print("   STARTING EVALUATION MODE")

        print("=" * 40)

        # 1. בדיקה שהמודל קיים

        if not os.path.exists(checkpoint_path):

            print(f"❌ Error: Model file not found at: {checkpoint_path}")

            print("   Please run in 'train' mode first.")


        else:

            # 2. טעינת המודל

            print(f"⬇️ Loading model from: {checkpoint_path}")

            best_model = RegressionSystem.load_from_checkpoint(checkpoint_path)

            # מילון לאיסוף כל התוצאות (Train, Val, Test)

            all_results = {}

            # ---------------------------------------------------------

            # 3. הרצת TEST

            # ---------------------------------------------------------

            print("\n>> Running Evaluation on TEST set...")

            data_module.setup(stage='test')

            test_res = evaluate_model(best_model, data_module, stage='test')

            if test_res:
                all_results.update(test_res)

            # ---------------------------------------------------------

            # 4. הרצת TRAIN + VAL

            # ---------------------------------------------------------

            print("\n>> Running Evaluation on TRAIN & VAL sets...")

            data_module.setup(stage='fit')

            fit_res = evaluate_model(best_model, data_module, stage='fit')

            if fit_res:
                all_results.update(fit_res)

            # ---------------------------------------------------------

            # 5. הדפסת טבלת סיכום

            # ---------------------------------------------------------

            print("\n" + "=" * 65)

            print("FINAL RESULTS SUMMARY (Mean Absolute Error in Pixels)")

            print("=" * 65)

            print(f"{'Dataset':<10} | {'Loss (MSE)':<12} | {'MAE X (px)':<12} | {'MAE Y (px)':<12}")

            print("-" * 65)

            # עוברים לפי סדר הגיוני ומדפיסים אם התוצאה קיימת

            for name in ['train', 'val', 'test']:

                if name in all_results:
                    metrics = all_results[name]

                    loss = metrics['loss']

                    # שליפת ה-MAE שחישבנו בפונקציה המתוקנת

                    err_x = metrics['mae_axis_px'][0]

                    err_y = metrics['mae_axis_px'][1]

                    print(f"{name.upper():<10} | {loss:.5f}      | {err_x:.2f}         | {err_y:.2f}")

            print("=" * 65)

    elif MODE == 'eval_visual_simple':

        print("--- Starting Visual Evaluation Mode ---")

        if not os.path.exists(checkpoint_path):
            print(f"Error: Model not found at {checkpoint_path}")

            print("Please run in 'train' mode first!")

            exit()

        print(f"Loading model from {checkpoint_path}...")

        # טעינת המודל

        model = RegressionSystem.load_from_checkpoint(checkpoint_path)

        # --- תיקון השגיאה: העברת המודל ל-CPU ---

        # זה מבטיח שהמודל והדאטה (שנטען ל-CPU כברירת מחדל) יהיו באותו מקום

        model.to('cpu')

        model.eval()

        print("Loading test dataset...")

        data_module.setup(stage='test')

        test_dataset = data_module.test_dataset

        random.seed()

        rand_idx = random.randint(0, len(test_dataset) - 1)

        print(f"Visualizing random sample index: {rand_idx}")

        visualize_single_sample_simple(model, test_dataset, idx=rand_idx)

    elif MODE == 'eval_visual_advanced':

        print("--- Starting Advanced Visual Evaluation Mode ---")

        visualize_single_sample_advanced()