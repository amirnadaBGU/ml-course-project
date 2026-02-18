import os
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from PIL import Image
import random
import numpy as np
import itertools


# --- 1. Dataset ---
class PrawnDataset(Dataset):
    def __init__(self, xlsx_file):
        data = pd.read_excel(xlsx_file, engine='openpyxl')
        self.X = torch.tensor(data[['x3', 'y3', 'x0', 'y0', 'x2', 'y2']].values, dtype=torch.float32)
        self.y = torch.tensor(data[['x1_target', 'y1_target']].values, dtype=torch.float32)
        self.length = torch.tensor(data['length'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.length[idx]


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
            train_size = int(0.80 * len(full_train_dataset))
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


class RegressionSystem(pl.LightningModule):
    def __init__(self, lr=1e-3, hidden_size=128, num_layers=2, optimizer_name='adam'):
        super().__init__()
        # שמירת כל ההיפר-פרמטרים
        self.save_hyperparameters()

        layers = []
        input_dim = 6  # גודל הקלט שלך
        current_dim = input_dim
        current_hidden = self.hparams.hidden_size

        # --- בניית השכבות הנסתרות באופן דינמי ---
        for i in range(self.hparams.num_layers):
            # הוספת שכבה לינארית: מהגודל הנוכחי לגודל הבא (שהוא חצי מקודמו בשכבות הבאות)
            layers.append(nn.Linear(current_dim, current_hidden))
            layers.append(nn.ReLU())  # פונקציית אקטיבציה

            # עדכון הממדים לאיטרציה הבאה
            current_dim = current_hidden
            # החלוקה ב-2 עבור השכבה הבאה (כפי שביקשת)
            # משתמשים ב-max(..., 2) כדי למנוע מצב של 0 נוירונים אם מעמיקים מדי
            current_hidden = max(current_hidden // 2, 2)

        # --- שכבת הפלט ---
        # מהשכבה הנסתרת האחרונה לגודל הפלט (2)
        layers.append(nn.Linear(current_dim, 2))

        # אריזת כל השכבות למודל אחד
        self.model = nn.Sequential(*layers)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # בחירה דינמית בין Adam ל-AdamW
        opt_name = self.hparams.optimizer_name.lower()
        if opt_name == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def evaluate_model_rmse(model, datamodule, stage='test'):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # פונקציית עזר פנימית
    def run_evaluation(dataloader, name):
        all_preds = []
        all_targets = []
        all_lengths = []
        running_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for inputs, targets,lengths in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

                # חישוב MSE Loss רגיל
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_lengths.append(lengths.cpu().numpy())

        all_lengths = np.concatenate(all_lengths)
        all_lengths = all_lengths.reshape(-1, 1)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Loss סופי (MSE)
        mse_loss = running_loss / len(dataloader.dataset)
        diff_norm = all_preds - all_targets
        diff_px = diff_norm * all_lengths
        squared_diff_px = diff_px ** 2
        mean_squared_diff_px = np.mean(squared_diff_px, axis=0)
        axis_rmse_px = np.sqrt(mean_squared_diff_px)  # [RMSE_X, RMSE_Y]
        dist_sq_per_sample = np.sum(squared_diff_px, axis=1)
        dist_rmse_px = np.sqrt(np.mean(dist_sq_per_sample))

        return {
            'loss_mse': mse_loss,
            'rmse_axis_px': axis_rmse_px,
            'rmse_dist_px': dist_rmse_px,
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


def evaluate_model(model, datamodule, stage='test'):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # פונקציית עזר פנימית
    def run_evaluation(dataloader, name):
        all_preds = []
        all_targets = []
        all_lengths = []
        running_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for inputs, targets,lengths in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_lengths.append(lengths.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_lengths = np.concatenate(all_lengths)
        all_lengths = all_lengths.reshape(-1, 1)
        final_loss = running_loss / len(dataloader.dataset)

        # --- Method 1: Pixel Error Logic ---

        diff_norm = all_preds - all_targets
        diff_px = diff_norm * all_lengths
        axis_mae = np.mean(np.abs(diff_px), axis=0)
        dist_px_lengths = np.sqrt(diff_px[:,0]**2 + diff_px[:,1]**2)
        total_mae = np.mean(dist_px_lengths, axis=0)
        # החזרת מילון נתונים לשימוש חיצוני
        return {
            'loss': final_loss,
            'mae_axis_px': axis_mae,
            'preds': all_preds,
            'targets': all_targets,
            'total_mae': total_mae
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


def do_grid_search(train_file, test_file):
    print("--- Starting Deep Grid Search (Layers, Hidden Size, Optimizers) ---")

    # 1. הגדרת מרחב החיפוש המורחב
    search_space = {
        'learning_rate': [1e-3, 5e-3],
        'batch_size': [32, 64],
        'hidden_size': [64, 128, 256],      # רוחב השכבה הראשונה
        'num_layers': [1, 2, 3],            # עומק הרשת (כמה חילוקים לבצע)
        'optimizer_name': ['adam', 'adamw']
    }

    best_val_loss = float('inf')
    best_params = None

    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total combinations to test: {len(combinations)}")

    for i, params in enumerate(combinations):
        print(f"\n[{i + 1}/{len(combinations)}] Testing: {params}")

        # --- יצירת DataModule ---
        dm = PrawnDataModule(
            train_file=train_file,
            test_file=test_file,
            batch_size=params['batch_size']
        )

        # --- יצירת המודל עם הפרמטרים החדשים ---
        model = RegressionSystem(
            lr=params['learning_rate'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],    # העברת מספר השכבות
            optimizer_name=params['optimizer_name']
        )

        # --- הגדרת Trainer ---
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            dirpath='grid_search_checkpoints',
            filename=f'trial_{i}'
        )

        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback],
        )

        # --- אימון ---
        trainer.fit(model, datamodule=dm)

        # --- בדיקת תוצאה ---
        current_val_loss = checkpoint_callback.best_model_score.item()
        print(f"Result -> Val Loss: {current_val_loss:.5f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_params = params
            print(f"*** New Best Found! ***")

        # ניקוי זיכרון
        del model
        del trainer
        torch.cuda.empty_cache()

    # 3. סיכום
    print("\n" + "=" * 40)
    print("GRID SEARCH FINISHED")
    print("=" * 40)
    print(f"Best Val Loss: {best_val_loss:.5f}")
    print(f"Best Params: {best_params}")

    return best_params


def plot_error_distribution(all_results,print_list):
    """
    מצייר היסטוגרמה של שגיאות המרחק (בפיקסלים) לכל הסטים.
    """
    plt.figure(figsize=(12, 6))

    # הגדרות רזולוציה
    W, H = 640, 360

    # צבעים לכל סט
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}

    found_data = False

    for name in print_list:
        if name not in all_results:
            continue

        found_data = True
        data = all_results[name]

        # שליפת התחזיות והאמת
        preds = data['preds']
        targets = data['targets']

        # המרה לפיקסלים (עותק כדי לא לשנות את המקור)
        p_px = preds.copy()
        t_px = targets.copy()

        p_px[:, 0] *= W
        p_px[:, 1] *= H
        t_px[:, 0] *= W
        t_px[:, 1] *= H

        # חישוב מרחק לכל נקודה בנפרד (לא ממוצע!)
        diff = p_px - t_px
        errors = np.linalg.norm(diff, axis=1)  # וקטור של שגיאות בגודל N

        # חישוב מדדים ללגנד
        mean_err = np.mean(errors)
        max_err = np.max(errors)

        # ציור ההיסטוגרמה
        # alpha=0.5 נותן שקיפות כדי שנוכל לראות חפיפות
        # bins=50 מחלק את הטווח ל-50 עמודות
        plt.hist(errors, bins=50, alpha=0.5, label=f"{name.upper()}: Mean={mean_err:.1f}px, Max={max_err:.1f}px",
                 color=colors.get(name, 'gray'))

    if not found_data:
        print("No data to plot.")
        return

    plt.title("Error Distribution (Euclidean Distance in Pixels)")
    plt.xlabel("Error (Pixels)")
    plt.ylabel("Count (Number of Samples)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # אופציונלי: שמירת הגרף לקובץ
    plt.savefig('error_histogram.png')
    print("Histogram saved to 'error_histogram.png'")

    plt.show()


def get_prawn_data_from_row(row):
    """
    פונקציית עזר שממירה שורה מהאקסל למבנה הנתונים שהויזואליזציה מצפה לו
    """
    # המרה ל-str ליתר ביטחון, למקרה שזה מספר
    stem = str(row['image_stem']) if 'image_stem' in row else str(row.name)

    points = {
        0: (row['x0'], row['y0']),
        1: (row['x1_target'], row['y1_target']),  # שים לב: בקובץ המעובד זה x1_target
        2: (row['x2'], row['y2']),
        3: (row['x3'], row['y3'])
    }
    return {
        'image_stem': stem,
        'object_id': row['object_id'] if 'object_id' in row else row.name,
        'points': points
    }


def find_and_visualize_worst_samples(n_worst=5):
    # === תיקון הסתירה: שימוש בקובץ המעובד ולא במקורי ===
    # אנחנו מניחים שהקובץ המעובד נמצא באותה תיקייה שבה רץ הסקריפט
    PROCESSED_TEST_PATH = 'final_test_data.xlsx'
    CHECKPOINT_PATH = 'weights/best_model.ckpt'
    BASE_DIR = 'prawn_2025_circ_small_v1'  # נתיב לתמונות

    print("--- 1. Loading Data & Model ---")

    if not os.path.exists(PROCESSED_TEST_PATH):
        print(f"Error: {PROCESSED_TEST_PATH} not found. Run in 'train' mode first to generate it.")
        return

    # טעינת נתונים מהקובץ המעובד (שיש בו x0, y0 וכו')
    df = pd.read_excel(PROCESSED_TEST_PATH, engine='openpyxl')

    # טעינת מודל
    if not os.path.exists(CHECKPOINT_PATH):
        print("Error: Checkpoint not found.")
        return

    model = RegressionSystem.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model.to('cpu')

    errors = []

    print(f"--- 2. Scanning {len(df)} samples for errors ---")

    # קבועי רזולוציה לחישוב פיקסלים
    W, H = 640, 360

    for idx, row in df.iterrows():
        # חילוץ נתונים
        # הערה: get_prawn_data_from_row הותאמה לקרוא x1_target
        pts = get_prawn_data_from_row(row)['points']

        # הכנת קלט: [x3, y3, x0, y0, x2, y2]
        input_list = [
            pts[3][0], pts[3][1],
            pts[0][0], pts[0][1],
            pts[2][0], pts[2][1]
        ]
        input_tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(input_tensor).squeeze(0)

        # המרה לפיקסלים לצורך חישוב השגיאה האמיתית
        px_pred_x = pred[0].item() * W
        px_pred_y = pred[1].item() * H

        px_true_x = pts[1][0] * W
        px_true_y = pts[1][1] * H

        # חישוב מרחק אוקלידי בפיקסלים
        distance_px = np.sqrt((px_pred_x - px_true_x) ** 2 + (px_pred_y - px_true_y) ** 2)

        errors.append({
            'error': distance_px,  # שומרים את השגיאה בפיקסלים
            'row_data': row,
            'prediction': (pred[0].item(), pred[1].item())  # שומרים מנורמל לציור
        })

    # מיון לפי השגיאה מהגדול לקטן
    errors.sort(key=lambda x: x['error'], reverse=True)

    print(f"--- 3. Visualizing Top {n_worst} Worst Predictions ---")

    for i in range(min(n_worst, len(errors))):
        item = errors[i]
        print(f"\n#{i + 1} Worst Error: {item['error']:.2f} px")
        visualize_specific_sample(item, BASE_DIR)


def visualize_specific_sample(error_item, base_dir):
    row = error_item['row_data']
    prawn_data = get_prawn_data_from_row(row)
    pts = prawn_data['points']

    # חיזוי (מנורמל)
    x1_pred_norm, y1_pred_norm = error_item['prediction']

    # טעינת התמונה
    img_path = find_image_path(prawn_data['image_stem'], base_dir)
    if img_path and os.path.exists(img_path):
        image = Image.open(img_path)
        W, H = image.size
    else:
        print(f"   [WARN] Image missing. Using blank.")
        W, H = 640, 360
        image = Image.new('RGB', (W, H), color='white')

    # פונקציית המרה לפיקסלים
    def to_px(norm_x, norm_y):
        return norm_x * W, norm_y * H

    # המרת כל הנקודות לפיקסלים
    px0, py0 = to_px(pts[0][0], pts[0][1])
    px1_gt, py1_gt = to_px(pts[1][0], pts[1][1])
    px1_pred, py1_pred = to_px(x1_pred_norm, y1_pred_norm)
    px2, py2 = to_px(pts[2][0], pts[2][1])
    px3, py3 = to_px(pts[3][0], pts[3][1])

    # --- חישוב גבולות לחיתוך (Zoom) ---
    all_x = [px0, px1_gt, px1_pred, px2, px3]
    all_y = [py0, py1_gt, py1_pred, py2, py3]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    padding = 60  # קצת מרווח מסביב
    crop_x1 = max(0, min_x - padding)
    crop_y1 = max(0, min_y - padding)
    crop_x2 = min(W, max_x + padding)
    crop_y2 = min(H, max_y + padding)

    # --- ציור ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"ID: {prawn_data['object_id']} | Error: {error_item['error']:.2f} px", fontsize=16)

    # פונקציית עזר פנימית לציור האלמנטים (כדי לא לשכפל קוד לשני הצדדים)
    def draw_elements(ax):
        # קווים
        ax.plot([px3, px0, px1_gt, px2], [py3, py0, py1_gt, py2], 'g-', linewidth=3, alpha=0.6, label='GT Skeleton')
        ax.plot([px0, px1_pred, px2], [py0, py1_pred, py2], 'r--', linewidth=2, alpha=0.8, label='Pred Skeleton')

        # נקודות
        ax.scatter(px1_gt, py1_gt, c='green', s=120, edgecolors='white', zorder=5, label='Target (GT)')
        ax.scatter(px1_pred, py1_pred, c='red', marker='X', s=150, edgecolors='white', zorder=6, label='Prediction')

        # טקסט לנקודות העוגן
        props = dict(boxstyle='circle', facecolor='blue', alpha=0.3)
        ax.text(px0, py0, "0", color='black', fontsize=9, fontweight='bold', bbox=props)
        ax.text(px2, py2, "2", color='black', fontsize=9, fontweight='bold', bbox=props)
        ax.text(px3, py3, "3", color='black', fontsize=9, fontweight='bold', bbox=props)

        # הגדרת הזום (אותו זום לשני הצדדים)
        ax.set_xlim(crop_x1, crop_x2)
        ax.set_ylim(crop_y2, crop_y1)  # היפוך ציר Y (0 למעלה)

    # --- צד שמאל: תמונה אמיתית (זום) ---
    ax_img = axes[0]
    ax_img.set_title("Real Image (Cropped)")
    ax_img.imshow(image)
    draw_elements(ax_img)
    ax_img.legend(loc='upper right', fontsize='small')

    # --- צד ימין: סכמטי (זום) ---
    ax_sch = axes[1]
    ax_sch.set_title("Schematic / Pixel View (Cropped)")
    ax_sch.grid(True, linestyle='--', alpha=0.5)
    ax_sch.set_facecolor('#f8f9fa')  # צבע רקע אפרפר בהיר מאוד
    ax_sch.set_aspect('equal')  # לשמור על פרופורציות
    draw_elements(ax_sch)

    plt.tight_layout()
    plt.show()


def find_image_path(stem, base_dir):
    """
    מחפש את התמונה בתיקיות משנה נפוצות.
    """
    stem_str = str(stem)  # המרה למחרוזת למקרה שזה int
    filename = f"{stem_str}.jpg"  # הנחה שהסיומת היא jpg

    # רשימת מקומות לחפש בהם
    possible_folders = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
    ]

    for folder in possible_folders:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path

    return None


def regenerate_test_data_with_names():
    BASE_DIR = 'prawn_2025_circ_small_v1'
    RAW_INPUT_FILE = os.path.join(BASE_DIR, 'test_data_all.xlsx')  # הקובץ המקורי הגולמי
    OUTPUT_FILE = 'final_test_data.xlsx'  # הקובץ שאנחנו רוצים לתקן

    print(f"--- Fixing Data File ---")
    print(f"Reading raw data from: {RAW_INPUT_FILE}")

    if not os.path.exists(RAW_INPUT_FILE):
        print(f"❌ Error: Could not find raw file at {RAW_INPUT_FILE}")
        return

    # 2. קריאת הקובץ הגולמי
    df = pd.read_excel(RAW_INPUT_FILE, engine='openpyxl')

    # 3. המרה לפורמט רחב (Pivot)
    # שים לב: אנחנו מגדירים את image_stem ו-object_id כאינדקס
    pivot_df = df.pivot_table(
        index=['image_stem', 'object_id'],
        columns='keypoint_index',
        values=['x_norm', 'y_norm']
    )

    # 4. בניית הדאטה-פריים החדש
    final_df = pd.DataFrame()

    # חילוץ הקואורדינטות
    # (מניחים שהעמודות קיימות, אם חסר משהו זה יפול כאן)
    final_df['x3'] = pivot_df[('x_norm', 3)]
    final_df['y3'] = pivot_df[('y_norm', 3)]
    final_df['x0'] = pivot_df[('x_norm', 0)]
    final_df['y0'] = pivot_df[('y_norm', 0)]
    final_df['x2'] = pivot_df[('x_norm', 2)]
    final_df['y2'] = pivot_df[('y_norm', 2)]
    final_df['x1_target'] = pivot_df[('x_norm', 1)]
    final_df['y1_target'] = pivot_df[('y_norm', 1)]

    # שליפת השמות המקוריים מתוך האינדקס
    final_df['image_stem'] = pivot_df.index.get_level_values('image_stem')
    final_df['object_id'] = pivot_df.index.get_level_values('object_id')

    # 5. ניקוי ושמירה
    final_df = final_df.dropna()

    # מחיקת הקובץ הישן אם קיים כדי למנוע בלבול
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    final_df.to_excel(OUTPUT_FILE, index=False)

    print(f"✅ Success! Created '{OUTPUT_FILE}' with correct image names.")
    print(f"   First row example: Name={final_df.iloc[0]['image_stem']}, ID={final_df.iloc[0]['object_id']}")


import seaborn as sns


def plot_spatial_error_heatmap(n_excluded=10, grid_size=(20, 10)):
    """
    יוצר מפת חום המראה באילו אזורים בתמונה השגיאה גבוהה יותר.
    מסיר את ה-N דגימות הגרועות ביותר כדי לא לעוות את הגרף.

    Args:
        n_excluded: כמה דגימות קיצון להסיר.
        grid_size: רזולוציית הרשת (X bins, Y bins).
    """
    # הגדרות נתיבים
    BASE_DIR = 'prawn_2025_circ_small_v1'
    PROCESSED_TEST_PATH = 'final_test_data.xlsx'
    CHECKPOINT_PATH = 'weights/best_model.ckpt'

    print("--- Generating Spatial Error Heatmap ---")

    # 1. טעינת נתונים ומודל
    if not os.path.exists(PROCESSED_TEST_PATH):
        print("Data file not found.")
        return
    df = pd.read_excel(PROCESSED_TEST_PATH, engine='openpyxl')

    model = RegressionSystem.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model.to('cpu')

    # 2. חישוב שגיאות לכל הדאטה
    results = []
    W, H = 640, 360  # רזולוציית התמונה

    print(f"Scanning {len(df)} samples...")

    with torch.no_grad():
        for idx, row in df.iterrows():
            # הכנת קלט
            pts = get_prawn_data_from_row(row)['points']
            input_tensor = torch.tensor([
                pts[3][0], pts[3][1],
                pts[0][0], pts[0][1],
                pts[2][0], pts[2][1]
            ], dtype=torch.float32).unsqueeze(0)

            # חיזוי
            pred = model(input_tensor).squeeze(0)

            # המרה לפיקסלים
            pred_x, pred_y = pred[0].item() * W, pred[1].item() * H
            gt_x, gt_y = pts[1][0] * W, pts[1][1] * H

            # חישוב שגיאה
            error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            results.append({
                'gt_x': gt_x,
                'gt_y': gt_y,
                'error': error
            })

    # המרה ל-DataFrame זמני לניתוח
    res_df = pd.DataFrame(results)

    # 3. סינון ה-N הגרועים ביותר
    print(f"Filtering top {n_excluded} worst samples...")
    res_df_sorted = res_df.sort_values('error', ascending=False)
    # לוקחים את כל הנתונים החל מהמקום ה-N והלאה (מסירים את ההתחלה)
    filtered_df = res_df_sorted.iloc[n_excluded:].copy()

    max_err = filtered_df['error'].max()
    mean_err = filtered_df['error'].mean()
    print(f"Stats after filtering: Max Error={max_err:.2f}px, Mean={mean_err:.2f}px")

    # 4. יצירת מפת החום (Binning)
    # חלוקת התמונה למשבצות
    x_bins = np.linspace(0, W, grid_size[0] + 1)
    y_bins = np.linspace(0, H, grid_size[1] + 1)

    # חישוב שגיאה ממוצעת לכל משבצת
    # הפונקציה מחזירה 3 ערכים, אנחנו צריכים את הראשון (הסטטיסטיקה)
    from scipy.stats import binned_statistic_2d

    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
        filtered_df['gt_x'],
        filtered_df['gt_y'],
        filtered_df['error'],
        statistic='mean',
        bins=[x_bins, y_bins]
    )

    # 5. ציור
    plt.figure(figsize=(12, 7))

    # שימוש ב-Seaborn לציור יפה
    # statistic.T נדרש כי המטריצה יוצאת הפוכה (X מול Y)

    # החלפת NaN באפס או בערך נייטרלי כדי לא לשבור את הגרף (אופציונלי)
    # statistic = np.nan_to_num(statistic)

    ax = sns.heatmap(statistic.T, cmap='YlOrRd', cbar_kws={'label': 'Mean Error (px)'},
                     xticklabels=False, yticklabels=False)

    ax.invert_yaxis()  # היפוך ציר Y שיתאים לתמונה

    plt.title(f"Spatial Error Heatmap (Excluding top {n_excluded} outliers)\nGrid: {grid_size[0]}x{grid_size[1]}",
              fontsize=15)
    plt.xlabel(f"Image Width (0-{W})")
    plt.ylabel(f"Image Height (0-{H})")

    # הוספת רשת עדינה
    plt.grid(True, which='both', color='white', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    MODE = 'eval_rmse'  # train eval or eval_visual_simple or eval_visual_advanced


    train_file = 'final_train_data.xlsx'
    test_file = 'final_test_data.xlsx'

    # נתיב קבוע למודל (כך נדע איפה לחפש אותו ב-eval)
    checkpoint_path = 'weights/best_model.ckpt'

    pl.seed_everything(42)

    # הכנת המודול של הדאטה
    data_module = PrawnDataModule(train_file, test_file)


    if MODE == 'train':
        print("--- Starting Training Mode ---")
        model = RegressionSystem(lr=5e-3, hidden_size=256, num_layers=1, optimizer_name='adam')

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
            max_epochs=50,
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

            print(f"{'Dataset':<10} | {'Loss (MSE)':<12} | {'MAE X (px)':<12} | {'MAE Y (px)':<12} | {'total MAE (px)':<12}")

            print("-" * 65)

            # עוברים לפי סדר הגיוני ומדפיסים אם התוצאה קיימת

            for name in ['train', 'val', 'test']:

                if name in all_results:
                    metrics = all_results[name]

                    loss = metrics['loss']

                    # שליפת ה-MAE שחישבנו בפונקציה המתוקנת

                    err_x = metrics['mae_axis_px'][0]

                    err_y = metrics['mae_axis_px'][1]

                    total_mae = metrics['total_mae']

                    print(f"{name.upper():<10} | {loss:.5f}      | {err_x:.2f}         | {err_y:.2f} | {total_mae:.2f}")

            print("=" * 65)

            # === כאן מוסיפים את הציור ===
            print("Creating Error Histogram...")
            plot_error_distribution(all_results,['test'])

    elif MODE == 'eval_rmse':

        print("\n" + "=" * 40)

        print("   STARTING EVALUATION MODE (RMSE)")

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

            test_res = evaluate_model_rmse(best_model, data_module, stage='test')

            if test_res:
                all_results.update(test_res)

            # ---------------------------------------------------------

            # 4. הרצת TRAIN + VAL

            # ---------------------------------------------------------

            print("\n>> Running Evaluation on TRAIN & VAL sets...")

            data_module.setup(stage='fit')

            fit_res = evaluate_model_rmse(best_model, data_module, stage='fit')

            if fit_res:
                all_results.update(fit_res)

            # ---------------------------------------------------------

            # 5. הדפסת טבלת סיכום

            # ---------------------------------------------------------


            print("\n" + "=" * 85)

            print("FINAL RESULTS SUMMARY (RMSE in Pixels)")

            print("=" * 85)

            # הוספת עמודת RMSE Dist

            print(f"{'Dataset':<10} | {'Loss (RMSE)':<12} | {'RMSE X (px)':<12} | {'RMSE Y (px)':<12} | {'RMSE Dist':<12}")

            print("-" * 85)

            # עוברים לפי סדר הגיוני ומדפיסים אם התוצאה קיימת

            for name in ['train', 'val', 'test']:

                if name in all_results:
                    metrics = all_results[name]

                    loss = metrics['rmse_dist_px']

                    err_x = metrics['rmse_axis_px'][0]

                    err_y = metrics['rmse_axis_px'][1]

                    # שליפת הנתון החדש שחושב (המרחק הכולל)

                    err_dist = metrics['rmse_dist_px']

                    print(
                        f"{name.upper():<10} | {loss:.5f}      | {err_x:.2f}         | {err_y:.2f}         | {err_dist:.2f}")

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

    elif MODE == 'grid_search':
        do_grid_search(train_file, test_file)

    elif MODE == 'find_worst':
        print("--- Starting Worst Samples Visualization Mode ---")
        find_and_visualize_worst_samples(n_worst=10)

    elif MODE == 'generate_file_with_names':
        print("--- Regenerating Test Data File with Image Names ---")
        regenerate_test_data_with_names()

    elif MODE == 'heatmap':
         plot_spatial_error_heatmap(n_excluded=15, grid_size=(32, 18))
