import pandas as pd

df = pd.read_csv("result_context.csv")

while True:
    id = int(input("ID: "))
    print(df[df['id'] == id]['content'].iloc[0])
    print("=========================================================================================================")