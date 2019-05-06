
import numpy as np
from sklearn import datasets
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics


class NaiveBayes():

    def fit(self, X, y):
       self.X = X
       self.y = y
       self.labels = np.unique(self.y)
       self.info = {}

       for i, j in enumerate(self.labels):
           tmp_X = X[np.where(y == j)]
           tmp_mean = np.mean(tmp_X, axis=0, keepdims=True)
           tmp_std = np.std(tmp_X, axis=0, keepdims=True)
           prior = tmp_X.shape[0]/self.X.shape[0]
           tmp = { "mean": tmp_mean, "std": tmp_std, "prior": prior}
           self.info["class"+str(j)] = tmp

    def _gussian(self, X, label):
        eps = 1e-4
        mean = self.info["class" + str(label)]["mean"]
        std = self.info["class" + str(label)]["std"]
        con = (np.exp(-1*((X-mean)**2)/((2*std*std)+eps)))/(np.sqrt((2*np.pi*std*std)+eps))
        result = np.sum(np.log(con), axis=1, keepdims=True)

        return result.T

    def _pred(self, X):
        output = []
        for y in self.labels:
            prior = self.info["class" + str(y)]["prior"]
            posterior = self._gussian(X, y)
            con = np.log(prior) + posterior
            output.append(con)
        return output

    def predict(self, X):
        output = self._pred(X)
        output = np.reshape(output, (self.labels.shape[0], X.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction


if __name__ == "__main__":
    data = datasets.load_iris()
    X = preprocessing.minmax_scale(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print("X_train", X_train.shape)
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    # clf2 = GaussianNB()
    # clf2.fit(X_train, y_train)
    # y_pred2 = clf2.predict(X_test)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # accuracy2 = accuracy_score(y_test, y_pred2)

    print("Accuracy:", accuracy)
    # print("Accuracy:", accuracy2)



















