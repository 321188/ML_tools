import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.datasets as datasets

def main():

    # import iris data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    use_features = [0, 2]
    X = X[:,use_features]
    y = np.where(y==0, -1, 1)

    eta = 0.01
    epoch_num = 10
    ptn = Perceptron(eta, epoch_num)
    ptn.fit(X, y)

    # train data
    plt.clf()
    plt.scatter(x=X[y==-1,0], y=X[y==-1,1], alpha=0.5, label="setosa", color="red", marker="o")
    plt.scatter(x=X[y==1,0], y=X[y==1,1], alpha=0.5, label="others", color="blue", marker="x")
    plt.title("Sepal length and Petal length")
    plt.xlabel("Sepal length[cm]")
    plt.ylabel("Petal length[cm]")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

    # Train times and weight
    train_index = range(0, len(ptn.weight_history))
    W = np.array(ptn.weight_history)
    plt.clf()
    plt.plot(train_index, W[:,0], color="red", label="w0")
    plt.plot(train_index, W[:,1], color="blue", label="w1")
    plt.plot(train_index, W[:,2], color="green", label="w2")
    plt.title("Train times and weight")
    plt.xlabel("train times")
    plt.ylabel("weight")
    plt.legend(loc="upper left")
    plt.xlim(0, 1500)
    plt.xticks(range(0,1600,100))
    plt.grid()
    plt.show()

    # epoch number and accuracy
    epoch_index = range(1, epoch_num+1)
    plt.clf()
    plt.plot(epoch_index, np.array(ptn.accuracy_each_epoch_result)*100, marker="o", label="Accuracy")
    plt.title("Epoch and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy[%]")
    plt.grid()
    plt.xlim(1,10)
    plt.xticks(range(1,11,1))
    plt.show()

    # decision regions
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max() +1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max() +1
    interval = 0.02
    X1 = np.arange(x1_min, x1_max, interval)
    X2 = np.arange(x2_min, x2_max, interval)
    X1, X2 = np.meshgrid(X1, X2)
    test_data = np.array([X1.ravel(), X2.ravel()]).T
    print(test_data.shape)
    print(np.matrix(test_data).shape)
    Z = ptn.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    markers = ('s', 'o', 'x', '^', 'v')
    colors = ('green', 'yellow','red', 'blue', 'lightgreen', 'gray', 'cyan')
    c_map = ListedColormap(colors[:len(np.unique(y))])
    labels = ('setosa', 'others')
    plt.clf()
    plt.contourf(X1, X2, Z, alpha=0.5, cmap=c_map)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=1.0, c=c_map(idx),
                    marker=markers[idx], label=labels[idx])
    plt.title("Decision regions")
    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


class Perceptron(object):

    def __init__(self, eta=0.01, epoch_num=10):
        self.eta = eta #study speed
        self.epoch_num = epoch_num # how many study times

        self.threshold_array = []
        self.weight = None
        self.weight_history = []
        self.accuracy_each_epoch_result = []

    def __act_function(self, z):
        return np.where(np.array(z) >= 0.0, 1, -1)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0],1)),X]
        print(X.shape)
        print(self.weight)
        return self.__act_function(np.dot(X,self.weight))

    def fit(self, X, y):
        #  add constant value for w0
        X = np.c_[np.ones((X.shape[0],1)),X]
        data_num, feature_num = X.shape

        self.weight = np.zeros(feature_num) # include w0
        self.weight_history.append(self.weight.tolist())

        for epoch in range(1, self.epoch_num + 1):
            correct_answer = []

            for xi,yi in zip(X,y):
                #z = w0*1 + w1*x1 + w2*x2 + w3*x3 ...
                zi = np.dot(xi,self.weight)
                pred_y = self.__act_function(zi)
                correct_answer.append(int(yi == pred_y))

                #update weights
                delta_w = self.eta* (yi - pred_y) * xi
                self.weight = self.weight + delta_w
                self.weight_history.append(self.weight.tolist())

            # calculate accuracy each epoch
            self.accuracy_each_epoch_result.append(float(sum(correct_answer))/len(correct_answer))

if __name__ == '__main__':
    main()

