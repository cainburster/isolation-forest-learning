# test the abnormal figure in a list
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_curve


def create_random_list_with_abnormal(length=1000, ratio_abnormal=0.05):
    normal_numbers = int(length * (1 - ratio_abnormal))
    abnormal_numbers = length - normal_numbers

    # normal figure lie in the distribution of O(100, 5)
    normal_set = np.random.normal(100, 5, (normal_numbers, 1))
    abnormal_set = np.random.normal(1000, 5, (abnormal_numbers, 1))
    whole_set = np.concatenate([normal_set, abnormal_set])
    raw_labels = np.concatenate([np.zeros((normal_numbers,), dtype=int), np.ones((abnormal_numbers,), dtype=int)])

    whole_set, raw_labels = shuffle(whole_set, raw_labels)
    return whole_set, raw_labels

def show_list_hist(nums):
    count, bins, ignored = plt.hist(nums)
    # Plot the distribution curve
    plt.show(block=False)


# unsupervised learning
def run_model(data):
    model=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.05), max_features=1.0)
    model.fit(data)
    return model

def infer_model(model, data):
    scores = model.decision_function(data)
    predicts = model.predict(data)
    return scores, predicts


if __name__ == '__main__':
    data, labels = create_random_list_with_abnormal(ratio_abnormal=0.05)
    model = run_model(data)
    scores, predicts = infer_model(model, data)
    predicts = (-predicts + 1) // 2
    tn, fp, fn, tp = confusion_matrix(labels, predicts).ravel()
    print("tn: {}\nfp: {}\nfn: {}\ntp: {}".format(tn, fp, fn, tp))
    print("recall: {}".format(tp/(tp+fn)))

    print(data[np.where(predicts != labels)])


