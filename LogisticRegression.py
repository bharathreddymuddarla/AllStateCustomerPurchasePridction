from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from splitdf2 import split_train_test

def logisticreg_fitmodel(df,testdf):
    print("entered model")
    X, Xtest, y, ytest = split_train_test(df)
    clf = LogisticRegression()
    clf.fit(X, y)
    acc = (accuracy_score(clf.predict(Xtest), ytest))
    ypred = clf.predict(testdf)
    return acc, ypred
