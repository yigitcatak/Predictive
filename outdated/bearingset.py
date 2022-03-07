import pandas as pd

nameList = [
            "health_20_0.csv",
            "comb_20_0.csv",
            "inner_20_0.csv",
            "outer_20_0.csv",
            "health_30_2.csv",
            "comb_30_2.csv",
            "inner_30_2.csv",
            "outer_30_2.csv",
            "ball_30_2.csv"
           ]

for fileName in nameList:
  originalPath = f"bearingset/{fileName}"
  newPath = f"raw_bearingset/raw_{fileName}" 
  df = pd.read_csv(originalPath)

  df = df.iloc[11:] #get the data part only

  #split the data, convert to float and split the dataframe into columns
  df = df[df.columns[0]].apply(lambda x: [float(item) for item in x.split("\t")[0:-1]])
  df = pd.DataFrame(df.values.tolist())
  df.to_csv(newPath, index=False)

df = pd.read_csv("bearingset/ball_20_0.csv")
df = df.iloc[17:]
df = df.iloc[17:]
df = df.drop(columns=df.columns[-1])
df.columns = [0,1,2,3,4,5,6,7]
df = df.astype(float)
df.to_csv("raw_bearingset/raw_ball_20_0.csv", index=False)