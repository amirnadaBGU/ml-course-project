import pandas as pd
import hashlib


def filter_unique_annotations(input_excel, output_excel):
    # טעינת הנתונים מהקובץ
    df = pd.read_excel(input_excel)
    print(f"סה\"כ שורות מקוריות: {len(df)}")

    # פונקציית עזר: מקבלת את כל השורות של תמונה אחת ומחזירה "חתימה" (טקסט ייחודי)
    def get_image_fingerprint(group):
        # מיון הנתונים כדי להבטיח שהסדר לא ישפיע על החתימה
        group = group.sort_values(by=['object_id', 'keypoint_index'])

        # בחירת העמודות שמעניינות אותנו (בלי שם התמונה, כי הוא משתנה)
        relevant_data = group[['object_id', 'class_id', 'keypoint_index', 'x_norm', 'y_norm', 'visibility']]

        # המרת הנתונים למחרוזת טקסט אחת ואז לקוד הצפנה קצר (MD5)
        data_str = relevant_data.to_string(index=False, header=False)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()

    # --- תחילת החלק הערוך והברור ---

    # שלב 1: יצירת אובייקט של קבוצות
    # מחלקים את הטבלה הגדולה לערימות קטנות, כל ערימה מכילה את כל השורות של תמונה אחת
    grouped_images = df.groupby('image_stem')

    # שלב 2: מעבר על כל תמונה וחישוב החתימה שלה
    results_list = []  # רשימה ריקה שתכיל את התוצאות

    # לולאה שעוברת תמונה-אחר-תמונה
    # המשתנה name מקבל את שם התמונה, והמשתנה group מקבל את הטבלה שלה
    for name, group in grouped_images:
        # חישוב החתימה הייחודית עבור התמונה הנוכחית
        fingerprint = get_image_fingerprint(group)

        # הוספת התוצאה (שם התמונה והחתימה) לרשימה שלנו
        results_list.append({'image_stem': name, 'fingerprint': fingerprint})

    # שלב 3: המרת הרשימה לטבלה סופית (DataFrame)
    # כעת יש לנו טבלה מסודרת עם עמודות: 'image_stem' ו-'fingerprint'
    image_fingerprints = pd.DataFrame(results_list)

    # --- סוף החלק הערוך ---

    # שלב 4: זיהוי וסינון כפילויות
    # משאירים רק את השורה הראשונה עבור כל חתימה ייחודית
    unique_fingerprints = image_fingerprints.drop_duplicates(subset='fingerprint', keep='first')

    # יוצרים רשימה של שמות התמונות ה"מנצחות" (הייחודיות)
    unique_image_names = unique_fingerprints['image_stem'].tolist()

    # שלב 5: סינון הדאטה המקורי ושמירה
    # משאירים בטבלה המקורית רק את התמונות שנמצאו ברשימה הנקייה
    clean_df = df[df['image_stem'].isin(unique_image_names)]

    # הדפסת סיכום
    print(f"מספר תמונות מקורי: {df['image_stem'].nunique()}")
    print(f"מספר תמונות ייחודיות (לאחר סינון): {len(unique_image_names)}")

    # שמירה לקובץ אקסל
    clean_df.to_excel(output_excel, index=False)
    print(f"הקובץ נשמר בהצלחה: {output_excel}")


# --- הרצת הפונקציה ---
input_file = 'train_data_all.xlsx'
output_file = 'unique_train_data.xlsx'

try:
    filter_unique_annotations(input_file, output_file)
except Exception as e:
    print(f"התרחשה שגיאה: {e}")