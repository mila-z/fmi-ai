import math
import random
import os

def load_data(filename="house-votes-84.data"):
    """
    function that loads data from file
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, filename)
    data = []

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(",")
            label = parts[0]
            features = parts[1:]
            data.append((label, features))

    return data

def fill_missing_by_class(data):
    """
    Fills '?'
    """

    # counts[class][feature_index][value]
    counts = {}
    for label, feats in data:
        counts.setdefault(label, [{} for _ in range(len(feats))])
        for i, v in enumerate(feats):
            if v != "?":
                counts[label][i][v] = counts[label][i].get(v, 0) + 1

    new_data = []
    for label, feats in data:
        new_feats = []
        for i, v in enumerate(feats):
            if v == "?":
                mode = max(counts[label][i], key=counts[label][i].get)
                new_feats.append(mode)
            else:
                new_feats.append(v)
        new_data.append((label, new_feats))

    return new_data

def stratified_split(data, test_ratio=0.2):
    by_class = {}
    for item in data:
        by_class.setdefault(item[0], []).append(item)

    train, test = [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        split = int(len(items) * (1 - test_ratio))
        train.extend(items[:split])
        test.extend(items[split:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test

def train_naive_bayes(data, lambda_value):
    class_counts = {}
    feature_counts = {}
    feature_values = set()

    for label, feats in data:
        class_counts[label] = class_counts.get(label, 0) + 1
        feature_counts.setdefault(label, [{} for _ in range(len(feats))])

        for i, v in enumerate(feats):
            feature_values.add(v)
            feature_counts[label][i][v] = feature_counts[label][i].get(v, 0) + 1

    model = {
        "class_counts": class_counts,
        "feature_counts": feature_counts,
        "total": len(data),
        "values": list(feature_values),
        "lambda": lambda_value
    }
    return model

def predict(model, features):
    best_class = None
    best_log_prob = -float("inf")

    for cls in model["class_counts"]:
        log_prob = math.log(model["class_counts"][cls] / model["total"])
        for i, v in enumerate(features):
            count = model["feature_counts"][cls][i].get(v, 0)
            denom = model["class_counts"][cls] + model["lambda"]*len(model["values"])
            prob = (count + model["lambda"]) / denom
            log_prob += math.log(prob)

        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_class = cls

    return best_class

def accuracy(model, data):
    correct = 0
    for label, feats in data:
        if predict(model, feats) == label:
            correct += 1
    
    return correct / len(data)

def cross_validation(data, k, lambda_value):
    random.shuffle(data)
    fold_size = len(data) // k
    accuracies = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        val = data[start:end]
        train = data[:start] + data[end:]

        model = train_naive_bayes(train, lambda_value)
        acc = accuracy(model, val)
        accuracies.append(acc)

    mean = sum(accuracies)/k
    std = math.sqrt(sum((a - mean) ** 2 for a in accuracies) / k)
    return accuracies, mean, std

if __name__ == "__main__":
    mode = int(input().strip())
    lambda_value = 1.0

    data = load_data()

    if mode == 1:
        data = fill_missing_by_class(data)

    train, test = stratified_split(data)

    model = train_naive_bayes(train, lambda_value)

    train_acc = accuracy(model, train)

    print("1. Train Set Accuracy:")
    print(f"\tAccuracy: {train_acc * 100:.2f}%\n")
    print("10-Fold Cross-Validation Results:\n")
    folds, mean, std = cross_validation(train, 10, lambda_value)
    for i, acc in enumerate(folds, 1):
        print(f"\tAccuracy Fold {i}: {acc*100:.2f}%")
    print(f"\n\tAverage Accuracy: {mean*100:.2f}%")
    print(f"\tStandard Deviation: {std*100:.2f}%\n")

    test_acc = accuracy(model, test)
    print("2. Test Set Accuracy:")
    print(f"\tAccuracy: {test_acc * 100:.2f}%")
