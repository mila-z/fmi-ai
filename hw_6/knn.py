# 5. Number of Instances: 150 (50 in each of three classes)

# 7. Attribute Information:
#    1. sepal length in cm
#    2. sepal width in cm
#    3. petal length in cm
#    4. petal width in cm
#    5. class: 
#       -- Iris Setosa
#       -- Iris Versicolour
#       -- Iris Virginica



# needs normalization because the 4 features have different numeric ranges
# (~4,3-7.9, ~2.0-4.4, ~1.0-6.9, ~0.1-2.5) -> larger ranges will have bigger influence
# rule of thumb: if the alg uses distances, similarities, or dot products -> normalize

# Z-score normalization (standardizarion) rescales each feature so it has mean=0 & standard deciation=1

import math
import os

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

def euclidean_dist(a, b):
    """ 
    find the euclidean distanse
    """
    total = 0.0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return math.sqrt(total)

def knn_predict(X_train, y_train, x_test, k):
    distances = []

    for i in range(len(X_train)):
        d = euclidean_dist(X_train[i], x_test)
        distances.append((d, y_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1
    
    return max(votes, key=votes.get)

def build_folds(X, y, k_folds=10):
    data = list(zip(X, y))
    folds = [[] for _ in range(k_folds)]

    for i, item in enumerate(data):
        folds[i % k_folds].append(item)
    
    return folds

def cross_validate_knn(X, y, k_neighbors, k_folds=10):
    folds = build_folds(X, y, k_folds)
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

        correct = 0
        for j in range(len(X_test_norm)):
            pred = knn_predict(X_train_norm, y_train, X_test_norm[j], k_neighbors)
            if pred == y_test[j]:
                correct += 1
        
        accuracy = (correct / len(X_test_norm)) * 100
        fold_accuracies.append(accuracy)

    avg = sum(fold_accuracies) / k_folds
    std = math.sqrt(sum((a - avg) ** 2 for a in fold_accuracies)/k_folds)

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

    train_preds = [knn_predict(X_norm, y, X_norm[i], neighbors) for i in range(len(X))]
    train_acc = sum(1 for i in range(len(y)) if train_preds[i] == y[i])/len(y) * 100

    print("1. Train Set Accuracy:")
    print(f"\tAccuracy: {train_acc:.2f}%")

    # cross validation
    print("2. 10-Fold Cross-Validation Results:")
    fold_accs, avg_acc, std_acc = cross_validate_knn(X, y, neighbors)

    for i, acc in enumerate(fold_accs):
        print(f"\tAccuracy Fold {i+1}: {acc:.2f}%")
    
    print(f"\n\tAverage Accuracy: {avg_acc:.2f}%")
    print(f"\tStandard Deviation: {std_acc:.2f}%")

    # test set accuracy
    print("3. Test Set Accuracy:")
    print(f"\tAccuracy: {fold_accs[-1]:.2f}%")
