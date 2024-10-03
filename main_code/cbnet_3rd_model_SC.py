#このモデルでは、銅酸化物系および鉄系超伝導体を取り除いてBCS系のみで学習をさせる
#crabnet_train_data.csvから情報を削る


# %%
#ライブラリ
from os.path import join
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from crabnet.crabnet_ import CrabNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)
pd.set_option('display.max_rows', 1000)


# %%
#全データを回していく
#codespaceで編集したデータが追加されている学習用データセットを用いて行う

X = pd.read_csv("/workspaces/SC_model_1st/main_code/CrabNet_train_data.csv")
X = X.sample(frac=1, random_state=42).reset_index(drop=True)#データシャッフル


#sorted_keys_comp_atomsにある元素リストを参照して銅酸化物系および鉄系超伝導体を取り除く
#29, 8 銅酸化物　26, 33 鉄系超伝導体 26, 34 鉄系超伝導体
Cu = "29"
O = "8"
Fe = "26"
As = "33"
Se = "34"
drop_idx = []
for i in range(len(X)):
    if (Cu in X.loc[i, "sorted_keys_comp_atoms"]) and (O in X.loc[i, "sorted_keys_comp_atoms"]):
        drop_idx.append(i)
    elif (Fe in X.loc[i, "sorted_keys_comp_atoms"]) and (As in X.loc[i, "sorted_keys_comp_atoms"]):
        drop_idx.append(i)
    elif (Fe in X.loc[i, "sorted_keys_comp_atoms"]) and (Se in X.loc[i, "sorted_keys_comp_atoms"]):
        drop_idx.append(i)
        
X = X.drop(index=drop_idx)
X = X.reset_index(drop=True)
y = X["Tc"]
X = X.drop(columns=["Tc"])




# %% K-fold cross-validation
ss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
cv = GroupKFold()
cvtype = "gcv"

groups = X["formula"]


trainval_idx, test_idx = list(ss.split(X, y, groups=groups))[0]



# %%
X_test, y_test = X.iloc[test_idx, :], y[test_idx]
X, y = X.iloc[trainval_idx, :], y.iloc[trainval_idx]

subgroups = X["formula"]

crabnet_dfs = []





# %%
for train_index, test_index in cv.split(X, y, subgroups):#上記のcvをしたそれぞれのデータセットのペアごとに処理を実行
    X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    train_df = pd.DataFrame(
        {
            "formula": X_train["formula"], 
            "efermi": X_train["efermi"],
            "band_gap": X_train["band_gap"],
            "target": y_train
        }
    )
    val_df = pd.DataFrame(
        {
            "formula": X_val["formula"], 
            "efermi": X_val["efermi"],
            "band_gap": X_val["band_gap"],
            "target": y_val
        }
    )
    
    cb = CrabNet(#ここでCrabNetのモデルを作成 学習時に表示されている学習曲線の表示や学習時のbatch数などについてはここで設定を行う
                #今回のモデルだとデフォルトのbatch数は512
        model_name = "SC_model_3rd_4head",
        #checkin = 10,
        extend_features=["efermi", "band_gap"],
        #epochs = 100,
        verbose=True,
        random_state = 42,
        learningcurve=True,
        losscurve = True,
        heads=4,
    )
    
    cb.fit(train_df)#モデル訓練

    y_pred, y_std, y_true,  = cb.predict(test_df = val_df, return_uncertainty = True, return_true = True)
    
    crabnet_dfs.append(#答えのファイルの作成
        pd.DataFrame(
            {
                "actual_Tc": y_true,
                "predicted_Tc": y_pred,
                "y_std": y_std,
                "formula": val_df["formula"],
            }
        )
    )



# %%
crabnet_df = pd.concat(crabnet_dfs)
display(crabnet_df)


# %%
import matplotlib.pyplot as plt
X = crabnet_df["actual_Tc"]
Y = crabnet_df["predicted_Tc"]
plt.xlim(-2, 60)
plt.scatter(X, Y)
plt.show()

# %%
import os
candidate_df_list = []
for csv in os.listdir("/workspaces/SC_model_1st/main_code/candidate_material_list"):
    sub_df = pd.read_csv(f"/workspaces/SC_model_1st/main_code/candidate_material_list/{csv}")
    candidate_df_list.append(sub_df)
    
candidate_df = pd.concat(candidate_df_list)
candidate_df["target"] = None
input_df = pd.DataFrame(
    {
        "formula": candidate_df["new_formula"],
        "efermi": candidate_df["efermi"],
        "band_gap": candidate_df["band_gap"],
        "target": candidate_df["target"],
    }
)
display(input_df)
    
# %%
y_pred = cb.predict