import math
import os
import random

def column_means(X):
    # zip(*X) -> transposes the data set so that each col contains all values of one feature
    # sum(col)/len(X) -> get the means for each feature
    return [sum(col)/len(X) for col in zip(*X)]

def column_stds(X, means):
    # get the standard deviations (sqrt(sum of the squared deviations / number of the samples)) for each feature
    return [
        math.sqrt(sum((x[i] - means[i]) ** 2 for x in X)/len(X))
        for i in range(len(means))
    ]

def z_score_norm(X, means, stds):
    # apply z-score formula (z = (x - mean)/sqrt(std)) to each feature
    Xn = []
    for row in X:
        Xn.append([
            (row[i] - means[i])/stds[i] if stds[i] != 0 else 0.0
            for i in range(len(row))
        ])
    
    return Xn

class KDNode:
    # kd-tree data structure
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point # feature vector
        self.label = label # class label
        self.axis = axis # dimension used to split the data
        self.left = left # child node
        self.right = right # child node

def build_kdtree(points, depth =0):
    # build kd-tree
    # points = [(features, label)]

    # base = empty tree
    if not points:
        return None
    
    k = len(points[0][0]) # num of features
    axis = depth % k # axis cyles through the dimension

    points.sort(key=lambda x: x[0][axis]) # sort points by the curr axis
    median = len(points) // 2 # choose median point as the node

    return KDNode( # create a node, recursively build left and right subtrees
        point=points[median][0],
        label=points[median][1],
        axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median+1:], depth+1)
    )

def euclidean(a, b):
    # compute euclidean distance between two points
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))

def kd_search(node, target, k, neighbors):
    # searches the KD-tree for the k nearest neighbors of target

    # base = empty node
    if node is None:
        return 
    
    dist = euclidean(target, node.point)
    neighbors.append((dist, node.label))
    neighbors.sort(key=lambda x: x[0])

    # keep only the k closest neighbors
    if len(neighbors) > k:
        neighbors.pop()

    # compute distance along the splitting axis
    axis = node.axis
    diff = target[axis] - node.point[axis]

    # choose where to search first
    close_branch = node.left if diff < 0 else node.right
    away_branch = node.right if diff < 0 else node.left

    kd_search(close_branch, target, k, neighbors)

    if len(neighbors) < k or abs(diff) < neighbors[-1][0]:
        kd_search(away_branch, target, k, neighbors)

def kd_tree_predict(tree, x, k):
    # find k nearest neghbors of x
    neighbors = []
    kd_search(tree, x, k, neighbors)

    # count how many times each class appears
    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    # return the class with the most votes
    return max(votes, key=votes.get)

def evaluate_accuracy(X, y, k_neighbors):
    # build kd-tree from the dataset
    train_points = list(zip(X, y))
    tree = build_kdtree(train_points)

    # loop through the samples
    correct = 0
    for i in range(len(X)):
        # predict the label and count the correct predictions
        pred = kd_tree_predict(tree, X[i], k_neighbors)
        if pred == y[i]:
            correct+=1

    # return accuracy as a percentage
    return correct / len(y) * 100

def cross_validate_kdtree(X, y, k_neighbors, k_folds=10):
    # combine features and labels and randomly shuffle data
    data = list(zip(X, y))
    random.shuffle(data)
    fold_size = len(data)//k_folds # compute fild size 
    fold_accuracies = []

    # iterate over each fold
    for i in range(k_folds):
        # split data into training and test folds
        test = data[i*fold_size:(i+1)*fold_size]
        train = data[:i*fold_size] + data[(i+1)*fold_size:]

        # extract training and test features and labels
        X_train = [x for x, _ in train]
        y_train = [label for _, label in train] 
        X_test = [x for x, _ in test]
        y_test = [label for _, label in test]

        # compte normalization parameters and normalize
        means = column_means(X_train)
        stds = column_stds(X_train, means)
        X_train = z_score_norm(X_train, means, stds)
        X_test = z_score_norm(X_test, means, stds)

        # build kd-tree from the training data
        train_points = list(zip(X_train, y_train))
        tree = build_kdtree(train_points)

        # count correct predictions on the test fold
        correct = sum(
            1 for i in range(len(X_test))
            if kd_tree_predict(tree, X_test[i], k_neighbors) == y_test[i] 
        )

        fold_accuracies.append(correct/len(y_test)*100)

    # compute mean accuracy and standard deviation
    avg = sum(fold_accuracies) / k_folds
    std = math.sqrt(sum((a-avg)**2 for a in fold_accuracies)/k_folds)

    return fold_accuracies, avg, std

def load_data(filename="iris.data"):
    # function that loads the data set
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, filename)

    X, y = [], []

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(",")
            # split the parameters into features and labels
            features = list(map(float, parts[:4]))
            label = parts[4]
            X.append(features)
            y.append(label)
    
    return X, y

def stratified_split(X, y, test_ratio=0.2):
    data_by_class = {}

    # group samples by their labels
    for xi, yi in zip(X, y):
        data_by_class.setdefault(yi, []).append(xi)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for label, samples in data_by_class.items():
        # randomly shuffle samples and compute split point
        random.shuffle(samples)
        split = int(len(samples)*(1 - test_ratio))

        # add samples
        for x in samples[:split]:
            X_train.append(x)
            y_train.append(label)

        for x in samples[split:]:
            X_test.append(x)
            y_test.append(label)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # get input
    neighbors = int(input().strip())
    X, y = load_data() # X - data, y - labels
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # train set accuracy
    means = column_means(X_train)
    stds = column_stds(X_train, means)

    X_train = z_score_norm(X_train, means, stds)
    X_test = z_score_norm(X_test, means, stds)

    train_acc = evaluate_accuracy(X_train, y_train, neighbors)
    print("1. Train Set Accuracy:")
    print(f"\tAccuracy: {train_acc:.2f}%")

    # cross validation
    print("2. 10-Fold Cross-Validation Results:")
    fold_accs, avg_acc, std_acc = cross_validate_kdtree(X_train, y_train, neighbors)

    for i, acc in enumerate(fold_accs):
        print(f"\tAccuracy Fold {i+1}: {acc:.2f}%")
    
    print(f"\n\tAverage Accuracy: {avg_acc:.2f}%")
    print(f"\tStandard Deviation: {std_acc:.2f}%")

    # test set accuracy
    test_acc = evaluate_accuracy(X_test, y_test, neighbors)
    print("3. Test Set Accuracy:")
    print(f"\tAccuracy: {test_acc:.2f}%")