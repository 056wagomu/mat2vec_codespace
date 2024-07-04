# %%
#ライブラリ
from os.path import join
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from crabnet.crabnet_ import CrabNet
from crabnet.kingcrab import SubCrab
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)
pd.set_option('display.max_rows', 1000)
import torch

from crabnet.utils.utils import (
    Scaler
)



# %%
X = []

for csv in os.listdir("/workspaces/mat2vec_codespace/main_code/candidate_stable_material_list"):
    X.append(pd.read_csv("/workspaces/mat2vec_codespace/main_code/candidate_stable_material_list" + "/" + csv))
    
X = pd.concat(X, axis=0)
X = X.reset_index(drop=True)

drop_idx = []
#指定の有毒であったりする元素や銅酸化物系鉄系の化合物を削除
H = "1"
Cl = "17"
As = "33"
Se = "34"
Br = "35"
Tc = "43"
Cd = "48"
Sb = "51"
Pb = "82"
U = "92"
Ra = "88"
N = "7"
Hg = "80"
I = "53"
remove_list = [H, Cl, As, Se, Br, Tc, Cd, Sb, Pb, U, Ra, N, Hg, I]

for i in range(len(X)):
    for remove in remove_list:
        if remove in X.loc[i, "sorted_keys_comp_atoms"]:
            drop_idx.append(i)
            break
    
    if "O" in X.loc[i, "sorted_keys_comp_atoms"] and "Cu" in X.loc[i, "sorted_keys_comp_atoms"]:
        drop_idx.append(i)
        
    if X.loc[i, "database_IDs"] == "False":
        drop_idx.append(i)
        
    if "O" not in X.loc[i, "sorted_keys_comp_atoms"] and "Li" in X.loc[i, "sorted_keys_comp_atoms"]:
        drop_idx.append(i)
        
drop_idx = list(set(drop_idx))
        
X = X.drop(drop_idx, axis=0)
X = X.reset_index(drop=True)

input_df = pd.DataFrame(
    {
        "formula": X["new_formula"],
        "efermi" : X["efermi"],
        "band_gap": X["band_gap"],
        "database_IDs": X["database_IDs"],
        "target": None
    }
)

display(input_df)
print(len(drop_idx))
print(input_df.isnull().sum())



# %%
#ドープ後の結果を予測するためにデータフレームを自作する
#Zn(Cu, Ga), Sn(In)の2つでドープ可能性サイトがある
doped_info_1 = {"formula": "Li2Zn0.9Cu0.1Sn3O8", "efermi": 5.36169647, "band_gap": 1.9407, "target": None}
doped_info_2 = {"formula": "Li2Zn0.9Ga0.1Sn3O8", "efermi": 5.36169647, "band_gap": 1.9407, "target": None}
doped_info_3 = {"formula": "Li2Zn1Sn2.9In0.1O8", "efermi": 5.36169647, "band_gap": 1.9407, "target": None}
doped_info_4 = {"formula": "Li2Cu0.1Ni0.9O3", "efermi": 2.36381262, "band_gap": 1.1037, "target": None}
doped_info_5 = {"formula": "Li2Ni0.9Co0.1O3", "efermi": 2.36381262, "band_gap": 1.1037, "target": None}
doped_info_6 = {"formula": "Li1V2.9Ti0.1Te2O12", "efermi": 2.06195658, "band_gap": 1.8778, "target": None}
doped_info_7 = {"formula": "Li1V2.9Cr0.1Te2O12", "efermi": 2.06195658, "band_gap": 1.8778, "target": None}

input_df_doped = pd.DataFrame([doped_info_1, doped_info_2, doped_info_3, doped_info_4, doped_info_5, doped_info_6, doped_info_7])
display(input_df_doped)


# %%
model_state_dict = torch.load("/workspaces/mat2vec_codespace/main_code/models/trained_models/SC_model_3rd_4head.pth")
#print(model_state_dict.keys())


cbnet = CrabNet()
cbnet.model = SubCrab()
cbnet.scaler = Scaler(torch.zeros(3))



# %%
cbnet.model.load_state_dict(model_state_dict["weights"])

cbnet.scaler.load_state_dict(model_state_dict["scaler_state"])
cbnet.model_name = model_state_dict["model_name"]
print(cbnet.model_name)


# %%
y_pred= cbnet.predict(test_df = input_df_doped)


# %%

pred_df = pd.DataFrame(
    {
        "formula": X["new_formula"],
        "predicted_Tc": y_pred,
        "efermi" : X["efermi"],
        "band_gap": X["band_gap"],
        "database_IDs": X["database_IDs"]
    }
)

pred_df = pred_df.sort_values("predicted_Tc", ascending=False)

pred_df = pred_df.drop_duplicates()

display(pred_df)

pred_df.to_csv("/workspaces/mat2vec_codespace/main_code/predicted_Tc_by3rd_final.csv", index=False)


# %%
pred_df_filterd = pred_df.loc[(pred_df["band_gap"] < 3) & (pred_df["band_gap"] > 0), :]

display(pred_df_filterd)
pred_df_filterd.to_csv("/workspaces/mat2vec_codespace/main_code/predicted_Tc_filterd_by3rd_final.csv", index=False)


# %%
print(y_pred)