import pandas as pd
import numpy as np
import zipfile

data_root = './data'

# load data
df_train = pd.read_csv("%s/train.csv"%data_root)
df_test = pd.read_csv("%s/test.csv"%data_root)
df_ad = pd.read_csv("%s/ad.csv"%data_root)

# process data
# merge the train data and test data with ad data
df_train = pd.merge(df_train, df_ad, on="creativeID")
df_test = pd.merge(df_test, df_ad, on="creativeID")

# to change to numpy type
# label means if user click the advertisement. 0 -> not click 1 -> clicked
y_train = df_train["label"].values

# model building
key = "appID"
# group by appid
# cvr = Click Value Rate 后验转化率
df_cvr = df_train.groupby(key).apply(lambda df : np.mean(df["label"])).reset_index()
df_cvr.columns = [key, "arg_cvr"]
df_test = pd.merge(df_test, df_cvr, how="left", on=key)
# If df_test's avg_cvr value is NONE, fill the NONE by mean of df_train
df_test["avg_cvr"].fillna(np.mean(df_train["label"]), inplace = True)
proba_test = df_test["arg_cvr"].values

# submission
df = pd.DataFrame({"instanceID": df_test["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
