import pandas as pd


def main():
    data = pd.read_csv("20240322_Supercon_data.txt",delimiter='\t')
    data.to_csv("20240322_Supercon_data.csv",index=False)
    pass

if __name__ == "__main__":
    main()