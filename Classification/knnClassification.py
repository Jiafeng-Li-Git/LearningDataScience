
import numpy as np
import DataBowTfidf
from sklearn.neighbors import KNeighborsClassifier


# use knn classifier, return accuracy of predictions
def knnclassfication(matrix_arr, label, test_arr, test_label):
    classifier = KNeighborsClassifier()
    classifier.fit(matrix_arr, label)
    score = classifier.score(test_arr, test_label)
    print(score)


# create train set and test set using BOW
tar_mat = DataBowTfidf.bow_mat
lable_list = DataBowTfidf.label_mat
train_BOW = np.vstack((tar_mat[0:160], tar_mat[200:360], tar_mat[400:560], tar_mat[600:760], tar_mat[800:960]))
test_BOW = np.vstack((tar_mat[160:200], tar_mat[360:400], tar_mat[560:600], tar_mat[760:800], tar_mat[960:1000]))
train_lable = lable_list[:160]+lable_list[200:360]+lable_list[400:560]+lable_list[600:760]+lable_list[800:960]
test_lable = lable_list[160:200]+lable_list[360:400]+lable_list[560:600]+lable_list[760:800]+lable_list[960:1000]

knnclassfication(train_BOW, train_lable, test_BOW, test_lable)

# create train set and test set using Tfidf
tar_mat = DataBowTfidf.tfidf_mat
lable_list = DataBowTfidf.label_mat
train_BOW = np.vstack((tar_mat[0:160], tar_mat[200:360], tar_mat[400:560], tar_mat[600:760], tar_mat[800:960]))
test_BOW = np.vstack((tar_mat[160:200], tar_mat[360:400], tar_mat[560:600], tar_mat[760:800], tar_mat[960:1000]))
train_lable = lable_list[:160]+lable_list[200:360]+lable_list[400:560]+lable_list[600:760]+lable_list[800:960]
test_lable = lable_list[160:200]+lable_list[360:400]+lable_list[560:600]+lable_list[760:800]+lable_list[960:1000]

knnclassfication(train_BOW, train_lable, test_BOW, test_lable)