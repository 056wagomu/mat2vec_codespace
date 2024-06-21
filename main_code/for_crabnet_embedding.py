import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

file_path = "/workspaces/mat2vec_codespace/main_code/20240322_Supercon_data_allstr_handedited.csv"
check_words = ["pressure", "gpa", "mbar"]
export_file_path = "/workspaces/mat2vec_codespace/main_code/final_SC_X_data_ver2.csv"

def main():
    SC_data = pd.read_csv(file_path, encoding='utf-8')
    
    crab_input_data = SC_data.rename(columns={"chemical formula": "formula", "Tc (of this sample) recommended": "Tc"})
    crab_input_data = crab_input_data[crab_input_data["Tc"].notna()]
    crab_input_data = crab_input_data.iloc[1:, :]
    NonOxide_SC_data = crab_input_data[crab_input_data["oxygen"].isna()].drop(columns=["oxygen", "common formula of oxygen", "measured value of Oxygen content"])
    Oxide_SC_data = crab_input_data[crab_input_data["oxygen"].notna()]

    #measured valueがある場合
    measured_value_of_Oxygen_index = Oxide_SC_data.loc[:, "measured value of Oxygen content"].notna()
    Oxide_SC_data.loc[measured_value_of_Oxygen_index, "value of oxygen"] = Oxide_SC_data.loc[measured_value_of_Oxygen_index, "measured value of Oxygen content"]

    #measured valueがないがcommon formulaに数字がある場合
    common_value_of_Oxygen_index = Oxide_SC_data.loc[:, "measured value of Oxygen content"].isna() & Oxide_SC_data.loc[:, "common formula of oxygen"].str.isdigit()
    Oxide_SC_data.loc[common_value_of_Oxygen_index, "value of oxygen"] = Oxide_SC_data.loc[common_value_of_Oxygen_index, "common formula of oxygen"]

    final_Oxide_SC_data = Oxide_SC_data.drop(columns=["oxygen", "common formula of oxygen", "measured value of Oxygen content"])
    final_Oxide_SC_data = final_Oxide_SC_data[final_Oxide_SC_data["value of oxygen"].notna()]

    new_SC_data = pd.concat([final_Oxide_SC_data, NonOxide_SC_data], axis = 0).reset_index(drop = True)

    new_SC_data["Tc"] = new_SC_data["Tc"].astype(float)

    #new_SC_data = new_SC_data.sort_values("Tc", ascending = False)

    high_pressure_data = new_SC_data["title of reference"].astype(str).apply(lambda x: any(word.lower() in x.lower() for word in check_words))
    not_high_pressure_data = ~high_pressure_data

    final_SC_data = new_SC_data[not_high_pressure_data].iloc[5:, :].drop(columns = ["title of reference"])

    final_SC_data.to_csv(export_file_path, index = False)
    
    pass

if __name__ == "__main__":
    main()