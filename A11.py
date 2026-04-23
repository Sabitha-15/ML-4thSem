from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(2,),max_iter=1000)
clf.fit([[0,0],[0,1],[1,0],[1,1]],[0,0,0,1])
print("AND:",clf.predict([[1,1]]))
clf.fit([[0,0],[0,1],[1,0],[1,1]],[0,1,1,0])
print("XOR:",clf.predict([[1,1]]))