
import numpy as np
import DataPreprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# return BOW vector
def BOWprocess(target_df, i):
    tar_list = target_df["Content"].tolist()
    count_vect = CountVectorizer(stop_words="english")
    X_train_counts = count_vect.fit_transform([tar_list[i]])
    return X_train_counts.toarray()[:, 0:25]


# return TFIDF matrix
def TFIDFprocess(BOWmatrix):
    tf_transformer = TfidfTransformer()
    X_train_tf = tf_transformer.fit_transform(BOWmatrix)
    return X_train_tf.toarray()


target_df = DataPreprocess.target_df
bow_mat = BOWprocess(target_df, 0)
# get BOW matrix by combining vectors
for i in range(1, target_df.shape[0]):
    tran_arr = BOWprocess(target_df, i)
    if len(tran_arr[0]) == 25:
        bow_mat = np.concatenate((bow_mat, tran_arr), axis=0)
tfidf_mat = TFIDFprocess(bow_mat)
label_mat = target_df["Index"].tolist()

print(bow_mat)