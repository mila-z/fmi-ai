import math
import os

# kd-tree data structure
class KDNode:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

def column_means(X):
    """
    function that gets the means
    """
    n_samples = len(X)
    n_features = len(X[0])

    means = [0.0]*n_features

    for row in X:
        for j in range(n_features):
            means[j] += row[j] # get the sum by columns

    for j in range(n_features):
        means[j] /= n_samples # divide each sum by the count

    return means

def column_stds(X, means):
    """
    function that gets the standardization
    """
    n_samples = len(X)
    n_features = len(X[0])

    stds = [0.0]*n_features

    for row in X:
        for j in range(n_features):
            stds[j] += (row[j] - means[j]) ** 2 # сумата на отклоненията

    for j in range(n_features):
        stds[j] = math.sqrt(stds[j] / n_samples)

    return stds

def z_score_norm(X, means, stds):
    """
    function that returns the normalized data
    """
    X_norm = []

    for row in X:
        new_row = []
        for j in range(len(row)):
            if stds[j] == 0:
                new_row.append(0.0) # avoid 0 division
            else:
                new_row.append((row[j] - means[j]) / stds[j])
        X_norm.append(new_row) # new_row is the standardization for each column

    return X_norm

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
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5

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
    folds = [[] for _ in range(k_folds)]
    for i in range(len(X)):
        folds[i%k_folds].append((X[i], y[i]))

    fold_accuracies = []

    for i in range(k_folds):
        test_fold = folds[i]
        train_folds = folds[:i] + folds[i+1:]
        train_data = [item for fold in train_folds for item in fold]

        X_train = [x for x, _ in train_data]
        y_train = [label for _, label in train_data]

        X_test = [x for x, _ in test_fold]
        y_test = [label for _, label in test_fold]

        means = column_means(X_train)
        stds = column_stds(X_train, means)

        X_train_norm = z_score_norm(X_train, means, stds)
        X_test_norm = z_score_norm(X_test, means, stds)

        train_points = list(zip(X_train_norm, y_train))
        tree = build_kdtree(train_points)

        correct = 0
        for j in range(len(X_test_norm)):
            pred = kd_tree_predict(tree, X_test_norm[j], k_neighbors)
            if pred == y_test[j]:
                correct += 1

        fold_accuracies.append(correct / len(X_test_norm) * 100)

    avg = sum(fold_accuracies) / k_folds
    std = (sum((a-avg) ** 2 for a in fold_accuracies) / k_folds) ** 0.5

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


if __name__ == "__main__":
    # get input
    neighbors = int(input().strip())
    X, y = load_data() # X - data, y - labels

    # train set accuracy
    means = column_means(X)
    stds = column_stds(X, means)
    X_norm = z_score_norm(X, means, stds)

    train_points = list(zip(X_norm, y))
    tree = build_kdtree(train_points)
    
    correct = 0
    for i in range(len(X_norm)):
        pred = kd_tree_predict(tree, X_norm[i], neighbors)
        if pred == y[i]:
            correct += 1
    
    train_acc = correct / len(y) * 100

    print("1. Train Set Accuracy:")
    print(f"\tAccuracy: {train_acc:.2f}%")

    # cross validation
    print("2. 10-Fold Cross-Validation Results:")
    fold_accs, avg_acc, std_acc = cross_validate_kdtree(X, y, neighbors)

    for i, acc in enumerate(fold_accs):
        print(f"\tAccuracy Fold {i+1}: {acc:.2f}%")
    
    print(f"\n\tAverage Accuracy: {avg_acc:.2f}%")
    print(f"\tStandard Deviation: {std_acc:.2f}%")

    # test set accuracy
    print("3. Test Set Accuracy:")
    print(f"\tAccuracy: {fold_accs[-1]:.2f}%")