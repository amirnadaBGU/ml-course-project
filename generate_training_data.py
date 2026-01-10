import pandas as pd

# 1. טעינת הדאטה
df = pd.read_excel('unique_test_data.xlsx')

# 2. סידור מחדש (Pivot) - הפיכת כל אובייקט לשורה אחת
# המפתח הייחודי הוא שילוב של שם התמונה ומזהה האובייקט
pivot_df = df.pivot_table(
    index=['image_stem', 'object_id'],
    columns='keypoint_index',
    values=['x_norm', 'y_norm']
)

# 3. יצירת הדאטה-פריים הסופי לפי הסדר שביקשת
final_df = pd.DataFrame()

# קלטים (Features) - נקודות 3, 0, 2
final_df['x3'] = pivot_df[('x_norm', 3)]
final_df['y3'] = pivot_df[('y_norm', 3)]
final_df['x0'] = pivot_df[('x_norm', 0)]
final_df['y0'] = pivot_df[('y_norm', 0)]
final_df['x2'] = pivot_df[('x_norm', 2)]
final_df['y2'] = pivot_df[('y_norm', 2)]

# יעד (Target) - נקודה 1
final_df['x1_target'] = pivot_df[('x_norm', 1)]
final_df['y1_target'] = pivot_df[('y_norm', 1)]

# הצגת התוצאה ושמירה
print(final_df.head())
final_df.to_excel('final_train_data.xlsx', index=False)