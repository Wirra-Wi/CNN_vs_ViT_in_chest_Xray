import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE=67+1

# 讀取
df = pd.read_csv('clip.csv')

# 自動偵測所有類別並轉換為 One-Hot 標籤
# 假設標籤間以 '|' 隔開，此步會自動生成所有疾病欄位
labels_df = df['Finding Labels'].str.get_dummies('|')
CLASS_NAMES = labels_df.columns.tolist() # 取得所有自動偵測到的類別清單
df = pd.concat([df, labels_df], axis=1)

# 以 Patient ID 進行資料集切分 (Train:Val:Test = 7:1:2)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))

df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['Patient ID']))

df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

# 標註 Split
df_train['Split'] = 'train'
df_val['Split'] = 'val'
df_test['Split'] = 'test'

# 合併
df_final = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

# 動態保留必要欄位與所有自動偵測到的疾病欄位
keep_columns = ['Image Index', 'Patient ID', 'Split'] + CLASS_NAMES
df_final = df_final[keep_columns]

# 輸出
df_final.to_csv('label.csv', index=False)

print("CSV 處理完成！")
print(f"類別總數: {len(CLASS_NAMES)}")
print(f"類別清單: {CLASS_NAMES}")
print(f"訓練集照片數: {len(df_final[df_final['Split']=='train'])}")
print(f"驗證集照片數: {len(df_final[df_final['Split']=='val'])}")
print(f"測試集照片數: {len(df_final[df_final['Split']=='test'])}")

# 檢查各個資料集中的疾病分佈
print("\n--- 各資料集的疾病數量分佈 ---")
distribution = df_final.groupby('Split')[CLASS_NAMES].sum().T
distribution['Total'] = distribution.sum(axis=1)

# 計算比例，看看有沒有哪一個 Split 的疾病直接歸零
print(distribution)

# 檢查是否有任何疾病在 val 或 test 中為 0
for split in ['val', 'test']:
    zero_classes = distribution[distribution[split] == 0].index.tolist()
    if zero_classes:
        print(f"⚠️ 警告: 以下疾病在 {split} 集中的數量為 0，請考慮更換 random_state 或使用分層切分: {zero_classes}")