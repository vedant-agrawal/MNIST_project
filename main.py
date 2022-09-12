# Importing the dependencies
import mnist
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import cv2
import random


def find_accuracy(matrix):
    """
    Takes in a confusion matrix and returns its accuracy
    :param
        matrix (2-d array): a matrix
    :return:
        acc : accuracy given by trace/sum
    """
    acc = matrix.trace() / matrix.sum()
    return acc


# Train-test splitting
X_train = mnist.train_images()
y_train = mnist.train_labels()
X_test = mnist.test_images()
y_test = mnist.test_labels()

# Showing random image from database for prediction
a = random.randint(0, 9999)
cv2.imshow("Image", X_test[a])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Reshaping nd array so it can be fed to MLPClassifier
X_train = X_train.reshape((-1, 28 * 28))
X_test = X_test.reshape((-1, 28 * 28))

# Model
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64)))
])
pipe.fit(X_train, y_train)

# Predicting the random image from before and checking accuracy
prediction = pipe.predict(X_test)
print(prediction[a])
accuracy_matrix = confusion_matrix(y_test, prediction)
accuracy = find_accuracy(accuracy_matrix)
print("\nAccuracy of model:", accuracy)
