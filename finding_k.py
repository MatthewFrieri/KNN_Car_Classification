import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("cars.data")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

ks = {}

k = 1
while k <= 11:
    ks.setdefault(k, [])
    for _ in range(100):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        ks[k].append(accuracy)

        predicted = model.predict(x_test)

        names = ["unacc", "acc", "good", "vgood"]
        correct = 0
        for i in range(len(predicted)):
            if predicted[i]==y_test[i]:
                correct += 1
                print("\033[92m",f"Predicted: {names[predicted[i]]} Actual: {names[y_test[i]]} Data: {x_test[i]}")
            else:
                print("\033[91m",f"Predicted: {names[predicted[i]]} Actual: {names[y_test[i]]} Data: {x_test[i]}")

        print("\033[93m", correct, "/", len(predicted), "=", accuracy, "\033[0m")

    k += 1

best = 0
bestK = 0
for i in range(len(ks)):
    avg, k = sum(ks[i+1])/100, i+1
    if avg > best:
        best = avg
        bestK = k

print(f"Most effective k value is {bestK}")
