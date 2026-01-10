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
import matplotlib
matplotlib.use('TkAgg')

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
class PrawnDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, batch_size=32):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        # נטען את הדאטה-סטים כאן כדי שיהיו זמינים גם ב-Train וגם ב-Eval
        if stage == 'fit' or stage is None:
            self.train_dataset = PrawnDataset(self.train_file)
            self.test_dataset = PrawnDataset(self.test_file)

        if stage == 'test' or stage is None:
            self.test_dataset = PrawnDataset(self.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
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
def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # חישוב הפרשים אבסולוטיים (ללא מיצוע בינתיים)
    abs_errors = np.abs(all_preds - all_targets)

    # חישוב MAE לכל ציר בנפרד
    mae_x = np.mean(abs_errors[:, 0])  # עמודה 0 היא X
    mae_y = np.mean(abs_errors[:, 1])  # עמודה 1 היא Y

    # חישובים כלליים
    mse = np.mean((all_preds - all_targets) ** 2)
    mae_total = np.mean(abs_errors)

    print(f"\n--- תוצאות הערכה (Evaluation) ---")
    print(f"Total MSE Loss: {mse:.6f}")
    print(f"Total MAE:      {mae_total:.6f}")

    print("-" * 40)
    # המרה לפיקסלים לפי הרזולוציה המקורית (640x360)
    print(f"X-Axis Error: {mae_x:.6f} (norm) -> ~{mae_x * 640:.2f} px")
    print(f"Y-Axis Error: {mae_y:.6f} (norm) -> ~{mae_y * 360:.2f} px")


def visualize_single_sample(model, dataset, idx=0):
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

# --- Main Logic ---
if __name__ == "__main__":

    MODE = 'eval_visual'  # train eval or eval_visual


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
        print("--- Starting Evaluation Mode ---")

        if not os.path.exists(checkpoint_path):
            print(f"Error: Could not find model at {checkpoint_path}")
            print("Please run in 'train' mode first to create the model.")
        else:
            print(f"Loading model from: {checkpoint_path}")

            # טעינת המודל מהקובץ
            best_model = RegressionSystem.load_from_checkpoint(checkpoint_path)

            # הרצת setup כדי לטעון את הנתונים
            data_module.setup(stage='test')

            # הרצת ההערכה
            evaluate_model(best_model, data_module.val_dataloader())

    elif MODE == 'eval_visual':

        print("--- Starting Visual Evaluation Mode ---")

        import random

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

        visualize_single_sample(model, test_dataset, idx=rand_idx)