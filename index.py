import numpy as np

X = np.array([
    [180, 80, 27], 
    [175, 78, 26],
    [170, 72, 25],
    [178, 75, 27],
    [160, 55, 23],  
    [165, 60, 24],
    [155, 50, 22],
    [162, 58, 23]
])

y = np.array(['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female'])


def separate_by_class(X, y):
    classes = np.unique(y)
    separated = {}
    for c in classes:
        separated[c] = X[y == c]
    return separated

data_by_class = separate_by_class(X, y)

def mean_variance(data):
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    return mean, var

summaries = {}
for c, rows in data_by_class.items():
    summaries[c] = mean_variance(rows)


def gaussian_probability(x, mean, var):
    eps = 1e-6  
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponent = np.exp(- ((x - mean) ** 2) / (2 * var + eps))
    return coeff * exponent

classes, counts = np.unique(y, return_counts=True)
class_priors = {c: count/len(y) for c, count in zip(classes, counts)}

def predict_single(x, summaries, class_priors):
    posteriors = {}
    for c, (mean, var) in summaries.items():
        prior = class_priors[c]
        likelihood = np.prod(gaussian_probability(x, mean, var))
        posteriors[c] = prior * likelihood
    return max(posteriors, key=posteriors.get)

print("======================================")
print("   Gender Prediction using")
print("   Gaussian Naive Bayes ")
print("======================================")
print("Please enter the following details.\n")

height = float(input("Height (in cm): "))
weight = float(input("Weight (in kg): "))
footsize = float(input("Foot size (in cm): "))

new_sample = np.array([height, weight, footsize])
prediction = predict_single(new_sample, summaries, class_priors)

print("\n--------------------------------------")
print("Prediction Result")
print("--------------------------------------")
print("Predicted Gender:", prediction)



# Group Members--------------------------------------------------------------ID
# 1. Abel Getachew  ......................................................UGR/6211/15
# 2. Dagmawi Heywot   ....................................................UGR/4392/15
# 3. Geleta Tamiru   .....................................................UGR/2035/15
# 4. Nathnael Lule   .....................................................UGR/1003/15