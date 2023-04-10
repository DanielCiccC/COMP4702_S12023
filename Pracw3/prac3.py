import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
n_neighbors=3

'''
Q1) import data and create scatterplots
'''
w3_classif = pd.read_csv('w3classif.csv', header=None)
w3_classif = w3_classif.rename(columns={0: "x1", 1: "x2", 2:'y'})
w3_regr = pd.read_csv('w3regr.csv', header=None)
w3_regr = w3_regr.rename(columns={0: "x1", 1: "y"})


def create_scatter_classif():
    print(w3_classif.head())
    plt.scatter(w3_classif['x1'], w3_classif['x2'], c=w3_classif['y'], alpha=0.5)
    plt.show()

def create_scatter_regr():
    print(w3_regr.head())
    plt.scatter(w3_regr['x1'], w3_regr['y'], alpha=0.5)
    plt.show()

'''
Q2) Shuffle the datasets and split them into training and testing
'''
X = w3_classif[['x1', 'x2']]
y = w3_classif['y']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)

X2 = w3_regr['x1']
y2 = w3_regr['y']
X_train2, X_test2, y_train2, y_test2 = train_test_split( X2, y2, test_size=0.30, random_state=42)

'''
Q3) 
a) Build a K-nn classifier for the classif and find the training and test loss
i.e. misclassification rate
''' 
def misclassification():
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    missclassification = classification_report(y_test, y_predict)
    print(missclassification)
    print(f"Accuracy score: {accuracy_score(y_test, y_predict)}")

'''
b) plot the decision regions for the classifier together with the training points
'''
def build_classif():
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(X, y)
    cmap_light = ListedColormap(["orange", "cornflowerblue"])
    cmap_bold = ["darkorange", "darkblue"]

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=cmap_light,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    xlabel='x1',
    ylabel='x2',
    shading="auto",
    )

    # Plot also the training points
    sns.scatterplot(
        x=X['x1'],
        y=X['x2'],
        hue=y,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )

    plt.title(
        "2-Class classification (k = %i)" % (n_neighbors)
        )

    plt.show()

'''
c) Experiment with different k values and see how it affects the loss values and the 
decisions regions
'''
def variation_k():
    classification_rates = []

    for k in range(1,25):
        clf = KNeighborsClassifier(k)
        clf.fit(X_train, y_train)
        
        y_predict = clf.predict(X_test)
        classification_rates.append(accuracy_score(y_test, y_predict))

        classification = pd.DataFrame(classification_rates, columns=['accuracy score'])
        classification = classification.reset_index() #add index
        classification['index'] = classification['index'] + 1

    print(classification.head())
    sns.lineplot(data=classification, x='index', y='accuracy score')
    plt.show()

'''
Q4) 

a) build a k-NN regression models with k=3 for dataset w3regr.csv and find the 
training and test loss
'''
def find_sse():
    #training
    clf = KNeighborsRegressor(n_neighbors=3)
    clf.fit(X_train2.to_numpy().reshape(-1,1), y_train2)

    y_predict = clf.predict(X_test2.to_numpy().reshape(-1,1))

    print(mean_squared_error(y_test2, y_predict))

'''
b) plot the training and/or test data together, with the predicted 'function' of 
the model
'''
def regression():
    #training
    clf = KNeighborsRegressor(n_neighbors=3)
    clf.fit(X_train2.to_numpy().reshape(-1,1), y_train2)

    # for x and y testing. Wee need to sort them first, if we want to plot them
    #incrementally
    test_sort = np.sort(X_test2.to_numpy())
    y_predict = clf.predict(test_sort.reshape(-1,1))

    print(X_test2.to_numpy().reshape(-1,1).shape)
    print(y_predict.reshape(-1,1).shape) # make sure these are the same size

    #plot them
    plt.subplot()
    plt.scatter(X2.to_numpy().reshape(-1,1), y2.to_numpy().reshape(-1,1), color="darkorange", label="data")
    plt.plot(test_sort.reshape(-1,1), y_predict.reshape(-1,1), color="navy", label="prediction")
    plt.axis("tight")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i)" % (n_neighbors))

    plt.tight_layout()
    plt.show()  


'''
c) plot the training and/or test data together, with the predicted 'function' of 
the model
'''
def vary_k():
    

    # print(X_test2.to_numpy().reshape(-1,1).shape)
    # print(y_predict.reshape(-1,1).shape) # make sure these are the same size

    #plot them
    fig, axs = plt.subplots(4, 4)

    for i in range(1, 17):

        #training
        clf = KNeighborsRegressor(n_neighbors=i)
        clf.fit(X_train2.to_numpy().reshape(-1,1), y_train2)

        # for x and y testing. Wee need to sort them first, if we want to plot them
        #incrementally
        test_sort = np.sort(X_test2.to_numpy())
        y_predict = clf.predict(test_sort.reshape(-1,1))

        axs[(i-1) % 4, (i-1) // 4].plot(test_sort.reshape(-1,1), y_predict.reshape(-1,1), color="navy", label="prediction")
        axs[(i-1) % 4, (i-1) // 4].scatter(X2.to_numpy().reshape(-1,1), y2.to_numpy().reshape(-1,1), color="darkorange", label="data")
        axs[(i-1) % 4, (i-1) // 4].set_title(f"KNeighborsRegressor (k = {i})")

    plt.tight_layout()
    plt.show()  


'''
Q 5
a) Build a decision tree classifier for for dataset w3classif.csv and find training and 
test loss
'''
def decision_tree():
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    missclassification = classification_report(y_test, y_predict)
    print(missclassification)
    print(f"Accuracy score: {accuracy_score(y_test, y_predict)}")

'''
b) Plot the decision regions for your classifier together with the training/test points
'''
def build_decision_tree():
    # Train
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    cmap_bold = ["darkorange", "darkblue"]

    # Plot the decision boundary
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        xlabel='x',
        ylabel='y',
    )

    # Plot also the training points
    sns.scatterplot(
        x=X['x1'],
        y=X['x2'],
        hue=y,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )

    plt.title(
        'Decision tree classifier'
        )

    plt.show()


'''
c) Experiment with different depth values
'''
def vary_depth():
    #plot them
    plot_colors = "ryb"
    cmap_bold = ["darkorange", "darkblue"]
    plot_step = 0.02
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):

        # Train
        clf = DecisionTreeClassifier(max_depth=(pairidx + 1)).fit(X, y)
     

        # Plot the decision boundary
        ax = plt.subplot(2, 3, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            xlabel='x',
            ylabel='y'
        )

        # sns.scatterplot(
        # x=X['x1'],
        # y=X['x2'],
        # hue=y,
        # palette=cmap_bold,
        # alpha=1.0,
        # edgecolor="black",
        # )
        

        sns.scatterplot(x=X['x1'], y=X['x2'], hue=y_train, alpha=1)
        
        sns.scatterplot(x=X_test['x1'], y=X_test['x2'], hue=y_test, style=y_test, palette='bright', s=20, marker='X')

    # plt.suptitle("Decision surface of decision trees trained on pairs of features")
    # plt.legend(loc="lower right", borderpad=0, handletextpad=0)
    _ = plt.axis("tight")
    plt.show()

'''
Q 6
a) Build a decision tree regressor for for dataset w3regr.csv and find SSE
'''
def decision_tree_regr():
    clf = DecisionTreeRegressor()
    clf.fit(X_train2.to_numpy().reshape(-1,1), y_train2)
    y_predict2 = clf.predict(X_test2.to_numpy().reshape(-1,1))

    # missclassification = classification_report(y_test2, y_predict2)
    # print(missclassification)
    print(f"MSE: {mean_squared_error(y_test2, y_predict2)}")
    print(f"SSE: {mean_squared_error(y_test2, y_predict2) * len(y_test2)}")
    

'''
b) Plot the decision tree regression for your classifier together with the training/test points
'''
def build_decision_tree():
    # Train
    clf = DecisionTreeRegressor()
    clf.fit(X_train2.to_numpy().reshape(-1,1), y_train2)

    # for x and y testing. Wee need to sort them first, if we want to plot them
    #incrementally
    test_sort = np.sort(X_test2.to_numpy())
    y_predict = clf.predict(test_sort.reshape(-1,1))

    plt.subplot()
    plt.scatter(X2.to_numpy().reshape(-1,1), y2.to_numpy().reshape(-1,1), color="darkorange", label="data")
    plt.plot(test_sort.reshape(-1,1), y_predict.reshape(-1,1), color="navy", label="prediction")
    plt.axis("tight")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Decision tree (k = %i)" % (n_neighbors))

    plt.tight_layout()
    plt.show()  

'''
c) test with different k values
'''
create_scatter_classif() #Q1, classifier
create_scatter_regr() #Q1, regression
misclassification() #Q3) misclassification
build_classif() #decision regions for the classfier and training points
variation_k() #different k values


vary_depth()