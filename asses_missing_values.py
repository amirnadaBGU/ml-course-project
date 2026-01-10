import pandas as pd

# טעינת הקובץ
df = pd.read_excel('final_test_data.xlsx')

# חישוב מספר הערכים החסרים בכל עמודה
missing_counts = df.isnull().sum()

print("מספר הערכים החסרים בכל עמודה:")
print(missing_counts)