
import DataPreprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# return BOW vector
def BOWprocess(target_df):
    tar_list = target_df["Content"].tolist()
    count_vect = CountVectorizer(stop_words="english")
    X_train_counts = count_vect.fit_transform(tar_list)
    return X_train_counts.toarray()


# return TFIDF matrix
def TFIDFprocess(BOWmatrix):
    tf_transformer = TfidfTransformer()
    X_train_tf = tf_transformer.fit_transform(BOWmatrix)
    return X_train_tf.toarray()


target_df = DataPreprocess.target_df
bow_mat = BOWprocess(target_df)
tfidf_mat = TFIDFprocess(bow_mat)
label_mat = target_df["Index"].tolist()

print(tfidf_mat)