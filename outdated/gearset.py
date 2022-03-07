
import pandas as pd

nameList = [
            "Health_20_0.csv",
            "Chipped_20_0.csv",
            "Miss_20_0.csv",
            "Root_20_0.csv",
            "Surface_20_0.csv",
            "Health_30_2.csv",
            "Chipped_30_2.csv",
            "Miss_30_2.csv",
            "Root_30_2.csv",
            "Surface_30_2.csv"
           ]

for fileName in nameList:
  originalPath = f"gearset/{fileName}"
  newPath = f"raw_gearset/raw_{fileName}" 
  df = pd.read_csv(originalPath)

  df = df.iloc[11:] #get the data part only

  #split the data, convert to float and split the dataframe into columns
  df = df[df.columns[0]].apply(lambda x: [float(item) for item in x.split("\t")[0:-1]])
  df = pd.DataFrame(df.values.tolist())
  df.to_csv(newPath, index=False)