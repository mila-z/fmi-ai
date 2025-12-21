# Naive Bayes Classifier implementation for the Congressional Voting Records data set
You can find the data set at https://archive.ics.uci.edu/dataset/105/congressional+voting+records

## Implemented Approach
- **Classifier:** Naive Bayes (categorial)
- **Probability computation:** uses logarithms to avoid numerical underflow
- **Zero-probability handling:** uses Laplace smoothing with parameter lambda (default lambda = 1.0)
- **Data splitting:** 80% training / 20% test; stratified, preserving class proportions
- **Evaluation:** training accuracy; 10-fold cross-validation (average accuracy + standart deviation); test set accuracy

## Handling of '?' values
The program supports two modes, selected via user input:
- **Input `0`:** treats `?` as a third valid value representing "abstained"
- **Input `1`:** replaces `?` with the most frequent value of the corresponding attrivute within the same class. This preserved class-conditional distributions

## How to run
### 1. Clone or Download
Save the code in a file named **naive_bayes.py** or clone/download the repo.

### 2. Download the dataset 
Download from https://archive.ics.uci.edu/dataset/105/congressional+voting+records  
Place the house-votes-84.data file in the same directory as the code.

### 3. Run in a Terminal or IDE
```bash
python naive_bayes.py
```

### 4. Enter input
Enter either `0` or `1`.

### 5. The results will be printed in the terminal
Once you enter either 0 or 1, you will see the following in the terminal:
- The accuracy of the trained set
- The accuracy of each fold of the 10-Fold Cross Validation, the average accuracy and the standard deviation
- The accuracy of the test set