# Adrian Girone
# 20 March 2021
# Machine Learning Project 1

from sklearn import tree

# Let the first parameter be color where 0 denotes "Red" and 1 denotes "Orange"
# Let the second parameter be roughness where 0 denotes "Smooth" and 1 denotes "Rough"
# Let the third parameter represent the weight of the fruit in grams

data = [[0, 0, 100],  # apple
        [0, 0, 95],  # apple
        [1, 1, 50],  # orange
        [0, 0, 105],  # apple
        [1, 1, 45],  # orange
        [1, 1, 55],  # orange
        [0, 0, 102],  # apple
        [1, 1, 60],  # orange
        [0, 0, 97],  # apple
        [1, 1, 52]]  # orange

labels = ['apple', 'apple', 'orange', 'apple', 'orange', 'orange', 'apple', 'orange', 'apple',
                                                                                      'orange']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(data, labels)

prediction = clf.predict([[0, 0, 99]])

print(prediction)

