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
df_train = pd.merge(df_train, df_ad, on="createiveID")
df_test = pd.merge(df_test, df_ad, on="createiveID")

# to change to numpy type
y_train = df_train["label"].value

# model building
key = "appID"
# group by appid
# cvr = Click Value Rate 后验转化率
df_cvr = df_train.groupby(key).apply(lambda df : np.mean(df["label"])).reset_index()