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


#X = pd.read_csv("/workspaces/mat2vec_codespace/main_code/candidate_material_list/B_candidate_material_list.csv")

input_df = pd.DataFrame(
    {
        "formula": X["new_formula"],
        "efermi" : X["efermi"],
        "band_gap": X["band_gap"],
        "target": None
    }
)

input_df = input_df.reset_index(drop=True)

display(input_df)


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


# %%
y_pred= cbnet.predict(test_df = input_df)


# %%

pred_df = pd.DataFrame(
    {
        "formula": X["new_formula"],
        "predicted_Tc": y_pred,
        "efermi" : X["efermi"],
        "band_gap": X["band_gap"]
    }
)

pred_df = pred_df.sort_values("predicted_Tc", ascending=False)

display(pred_df)

# %%
#pred_df.to_csv("/workspaces/mat2vec_codespace/main_code/predicted_Tc.csv", index=False)


# %%
#pred_df_filterd = pred_df.loc[pred_df["formula"].str.len() < 10, :].head(100)
pred_df_filterd = pred_df.loc[(pred_df["band_gap"] < 3) & (pred_df["band_gap"] > 0), :].head(100)

display(pred_df_filterd)
#pred_df_filterd.to_csv("/workspaces/mat2vec_codespace/main_code/predicted_Tc_filterd.csv", index=False)

# %%
