from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

#SVM Parameters
C = [2 ** i for i in range(-5, 16, 2)]
gamma = [2 ** i for i in range(-15, 4, 2)]

MODELS = {'Bayes' : {'clf': GaussianNB(), 
					'parameters' : {}},
	'KNN' : {'clf' : KNeighborsClassifier(),
			'parameters' : {
			'n_neighbors' : [3, 5, 7, 9],}
	},
	'DecisionTree' : { 'clf' : DecisionTreeClassifier(random_state=42),
						'parameters' : {
		'criterion' : ['gini', 'entropy'], #log_loss
		'max_depth': [1, 2, 3],}
	},
	'RF' : { 'clf' : RandomForestClassifier(random_state=42),
			'parameters' : {
				'n_estimators' : [50, 75, 100, 200],
				'max_depth': [1, 2, 3],}
	},
	'SVM' : {'clf' : SVC(random_state=42),
			'parameters' : {
		'C':C,
		'kernel': ['linear', 'poly', 'rbf'],
		'gamma':gamma,}
	},
	# 'MLP1Camada' : { 'clf' : MLPClassifier(random_state=42), #, max_iter=100
	# 				'parameters' : {
	# 	'hidden_layer_sizes': [(50,), (100,), (200,)],
	# 	'activation' : ['identity', 'logistic', 'tanh', 'relu']}
	# },
	# 'MLP2Camadas' : { 'clf' : MLPClassifier(random_state=42), #, max_iter=100
	# 				'parameters' : {
	# 	'hidden_layer_sizes': [(50,70), (75, 100), (100, 200)],
	# 	'activation' : ['identity', 'logistic', 'tanh', 'relu']}
	# },
	# 'MLP3Camadas' : { 'clf' : MLPClassifier(random_state=42), #, max_iter=100
	# 				'parameters' : {
	# 	'hidden_layer_sizes': [(25, 50, 75), (75, 100, 200), (150, 200, 300)],
	# 	'activation' : ['identity', 'logistic', 'tanh', 'relu']}
	# }
}

SELECTED_MODEL = {'Bayes' : {'clf': GaussianNB(), 
					'parameters' : {}},
}