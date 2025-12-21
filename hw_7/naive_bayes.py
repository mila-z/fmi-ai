import math
import random
import os

def load_data(filename="house-votes-84.data"):
    """
    function that loads data from file
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, filename)
    # stores the data as [(label, [features])], where label is democrat or republican and features are voting attributes
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
    Function that handles mode '1'
    Replaces '?' using the most frequent value per class and per feature (class-conditional mode imputation)
    For example Democrat + feature #3 -> use the most common Democrat vote for feature #3
    """
    # counts[class][feature_index][value] = frequency, for example counts[Democrat][3]['y'] = some_frequency
    counts = {}
    for label, feats in data:
        counts.setdefault(label, [{} for _ in range(len(feats))])
        for i, v in enumerate(feats):
            if v != "?": # ignore missing values
                counts[label][i][v] = counts[label][i].get(v, 0) + 1 # increment count for this feature value and class

    # new_data = new cleaned data
    new_data = []
    for label, feats in data:
        new_feats = [] # new list of features for this instance
        for i, v in enumerate(feats):
            if v == "?":
                mode = max(counts[label][i], key=counts[label][i].get) # find the most frequent value for this class and this feature index
                new_feats.append(mode) # replace '?' with that value
            else: # keep original
                new_feats.append(v)
        new_data.append((label, new_feats)) # add to cleaned instance

    return new_data

def stratified_split(data, test_ratio=0.2):
    """
    Split data into train and test
    Preserve class proportions
    """
    # group all democrats together, republicans together
    by_class = {}
    for item in data:
        by_class.setdefault(item[0], []).append(item)

    train, test = [], []
    for _, items in by_class.items():
        random.shuffle(items) # randomize order inside each class
        split = int(len(items) * (1 - test_ratio)) # get 80% index
        # organize into sets
        train.extend(items[:split])
        test.extend(items[split:])

    random.shuffle(train)
    random.shuffle(test)

    return train, test

def train_naive_bayes(data, lambda_value):
    """
    Learns probabilities from training data
    """
    class_counts = {} # how many times each class appears
    feature_counts = {} # counts feature values per class
    feature_values = set() # stores all possible feature values

    for label, feats in data:
        class_counts[label] = class_counts.get(label, 0) + 1 # count class occurances 
        feature_counts.setdefault(label, [{} for _ in range(len(feats))]) # preprate feature dictionaries per class

        for i, v in enumerate(feats):
            feature_values.add(v)
            feature_counts[label][i][v] = feature_counts[label][i].get(v, 0) + 1 # count feature value per class

    # store learned parameters in dictionary
    model = {
        "class_counts": class_counts,
        "feature_counts": feature_counts,
        "total": len(data),
        "values": list(feature_values),
        "lambda": lambda_value
    }
    return model

def predict(model, features):
    """
    Predicts class for one instance
    """
    # initialize maximum log-probability
    best_class = None
    best_log_prob = -float("inf")

    for cls in model["class_counts"]:
        # for each class compute log prior probability 
        log_prob = math.log(model["class_counts"][cls] / model["total"])
        for i, v in enumerate(features):
            # how often a feature appears in a class
            count = model["feature_counts"][cls][i].get(v, 0)
            # calculate denominator with laplace smoothing
            denom = model["class_counts"][cls] + model["lambda"]*len(model["values"])
            # calculate smoothed conditional probability
            prob = (count + model["lambda"]) / denom
            # add log probability
            log_prob += math.log(prob)

        # keep class with hughest probability
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_class = cls

    return best_class

def accuracy(model, data):
    """
    Computes classification accuracy
    """
    correct = 0
    for label, feats in data:
        if predict(model, feats) == label:
            correct += 1
    
    return correct / len(data)

def cross_validation(data, k, lambda_value):
    """
    Implement k-fold cross-validation
    """
    random.shuffle(data)
    fold_size = len(data) // k
    accuracies = []

    for i in range(k):
        # fold boundaries
        start = i * fold_size
        end = start + fold_size
        # validation fold
        val = data[start:end]
        # remaining data = training set
        train = data[:start] + data[end:]

        # training model
        model = train_naive_bayes(train, lambda_value)
        # evaluate on validation fold
        acc = accuracy(model, val)
        accuracies.append(acc)

    # average accuracy and standard deviation
    mean = sum(accuracies)/k
    std = math.sqrt(sum((a - mean) ** 2 for a in accuracies) / k)

    return accuracies, mean, std

if __name__ == "__main__":
    mode = int(input().strip())
    lambda_value = 1.0 # laplace smoothing parameter

    data = load_data()

    if mode == 1:
        data = fill_missing_by_class(data)

    # create sets
    train, test = stratified_split(data)
    # train model
    model = train_naive_bayes(train, lambda_value)
    # training accuracy
    train_acc = accuracy(model, train)

    print("1. Train Set Accuracy:")
    print(f"\tAccuracy: {train_acc * 100:.2f}%\n")
    print("10-Fold Cross-Validation Results:\n")

    # 10-fold cross-validation
    folds, mean, std = cross_validation(train, 10, lambda_value)
    for i, acc in enumerate(folds, 1):
        print(f"\tAccuracy Fold {i}: {acc*100:.2f}%")
    print(f"\n\tAverage Accuracy: {mean*100:.2f}%")
    print(f"\tStandard Deviation: {std*100:.2f}%\n")

    test_acc = accuracy(model, test)
    print("2. Test Set Accuracy:")
    print(f"\tAccuracy: {test_acc * 100:.2f}%")
