import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


data = pd.read_csv("car.data")

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


k = 9

model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]
correct = 0
for i in range(len(predicted)): 
    if predicted[i]==y_test[i]:
        correct += 1
        print("\033[92m",f"Predicted: {names[predicted[i]]} Actual: {names[y_test[i]]} Data: {x_test[i]}")
    else:
        pass
        print("\033[91m",f"Predicted: {names[predicted[i]]} Actual: {names[y_test[i]]} Data: {x_test[i]}")
    #n = model.kneighbors([x_test[i]], k, True)
    #print(n)
print("\033[93m", "ACCURACY:", correct, "/", len(predicted), "=", accuracy, "\033[0m")