import math
import os
import random

def column_means(X):
    return [sum(col)/len(X) for col in zip(*X)]

def column_stds(X, means):
    return [
        math.sqrt(sum((x[i] - means[i]) ** 2 for x in X)/len(X))
        for i in range(len(means))
    ]

def z_score_norm(X, means, stds):
    Xn = []
    for row in X:
        Xn.append([
            (row[i] - means[i])/stds[i] if stds[i] != 0 else 0.0
            for i in range(len(row))
        ])
    
    return Xn

# kd-tree data structure
class KDNode:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

# build kd-tree
# each el in points is ([features], label)
def build_kdtree(points, depth =0):
    if not points:
        return None
    
    k = len(points[0][0])
    axis = depth % k

    points.sort(key=lambda x: x[0][axis])
    median = len(points) // 2

    return KDNode(
        point=points[median][0],
        label=points[median][1],
        axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median+1:], depth+1)
    )

def euclidean(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))

def kd_search(node, target, k, neighbors):
    if node is None:
        return 
    
    dist = euclidean(target, node.point)
    neighbors.append((dist, node.label))
    neighbors.sort(key=lambda x: x[0])

    if len(neighbors) > k:
        neighbors.pop()

    axis = node.axis
    diff = target[axis] - node.point[axis]

    close_branch = node.left if diff < 0 else node.right
    away_branch = node.right if diff < 0 else node.left

    kd_search(close_branch, target, k, neighbors)

    if len(neighbors) < k or abs(diff) < neighbors[-1][0]:
        kd_search(away_branch, target, k, neighbors)

def kd_tree_predict(tree, x, k):
    neighbors = []
    kd_search(tree, x, k, neighbors)

    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)

def evaluate_accuracy(X, y, k_neighbors):
    train_points = list(zip(X, y))
    tree = build_kdtree(train_points)

    correct = 0
    for i in range(len(X)):
        pred = kd_tree_predict(tree, X[i], k_neighbors)
        if pred == y[i]:
            correct+=1

    return correct / len(y) * 100

def cross_validate_kdtree(X, y, k_neighbors, k_folds=10):
    data = list(zip(X, y))
    random.shuffle(data)
    fold_size = len(data)//k_folds
    fold_accuracies = []

    for i in range(k_folds):
        test = data[i*fold_size:(i+1)*fold_size]
        train = data[:i*fold_size] + data[(i+1)*fold_size:]

        X_train = [x for x, _ in train]
        y_train = [label for _, label in train]

        X_test = [x for x, _ in test]
        y_test = [label for _, label in test]

        means = column_means(X_train)
        stds = column_stds(X_train, means)

        X_train = z_score_norm(X_train, means, stds)
        X_test = z_score_norm(X_test, means, stds)

        train_points = list(zip(X_train, y_train))
        tree = build_kdtree(train_points)

        correct = sum(
            1 for i in range(len(X_test))
            if kd_tree_predict(tree, X_test[i], k_neighbors) == y_test[i] 
        )

        fold_accuracies.append(correct/len(y_test)*100)

    avg = sum(fold_accuracies) / k_folds
    std = math.sqrt(sum((a-avg)**2 for a in fold_accuracies)/k_folds)

    return fold_accuracies, avg, std

def load_data(filename="iris.data"):
    """
    function that loads the data set
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, filename)

    X, y = [], []

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(",")
            features = list(map(float, parts[:4]))
            label = parts[4]
            X.append(features)
            y.append(label)
    
    return X, y

def stratified_split(X, y, test_ratio=0.2):
    data_by_class = {}

    for xi, yi in zip(X, y):
        data_by_class.setdefault(yi, []).append(xi)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for label, samples in data_by_class.items():
        random.shuffle(samples)
        split = int(len(samples)*(1 - test_ratio))

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