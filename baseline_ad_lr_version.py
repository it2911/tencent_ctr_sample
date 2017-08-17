import zipfile

import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

data_root = './data'

# load data
df_train = pd.read_csv("%s/train.csv"%data_root)
df_test = pd.read_csv("%s/test.csv"%data_root)
df_ad = pd.read_csv("%s/ad.csv"%data_root)

df_train = pd.merge(df_train, df_ad, on="creativeID")
df_test = pd.merge(df_test, df_ad, on="creativeID")

y_train = df_train["label"].values

# feature engineering / encoding
enc = OneHotEncoder()
feats = ["creativeID", "adID", "camgaignID", "advertiserID", "appID", "appPlatform"]
for i, feat in enumerate(feats):
    # fit() : 渡されたデータの最大値、最小値、平均、標準偏差、傾き...などの統計を取得して、内部メモリに保存する。
    # transform() : fit()で取得した統計情報を使って、渡されたデータを実際に書き換える。
    # fit_transform() : fit()を実施した後に、同じデータに対してtransform()を実施する。

    # The fit method to get the train data set means, variance etc.
    # The transform method to get convert the feature info.
    x_train = enc.fit_transform(df_train[feat].values.reshape(-1, 1))
    # Second times to call transform method, the enc variable had remember the train data set info when first call fit method.
    x_test = enc.transform(df_test[feat].values.reshape(-1, 1))

    if(i == 0):
        # Not handle the creativeID data
        X_train, X_test = x_train, x_test
    else:
        # sparse.hstack -> let data to be sparse (稀疏) -> reduce the dimension for save memory
        # sparse.hstack -> let data to be dense (稠密) -> increase the dimension for assure feature
        X_train, X_test = sparse.hstack((x_train, x_train)), sparse.hstack((x_test, x_test))

# model training
lr = LogisticRegression()
# train the model
lr.fit(X_train, y_train)
# predict the test data set
proba_test = lr.predict_proba(X_test)[:, 1]

# submission
df = pd.DataFrame({"instanceID": df_test["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)