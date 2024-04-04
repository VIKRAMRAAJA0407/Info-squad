import streamlit as st
import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature=None, value=None, leaf_class=None):
        self.feature = feature        # Feature index
        self.value = value            # Value to split on (for numerical features)
        self.leaf_class = leaf_class  # Class label if node is a leaf
        self.children = {}            # Dictionary to store children nodes

class DecisionTreeID3:
    def __init__(self):
        self.root = None

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        entropy_val = 0
        total_samples = len(y)
        for count in counts:
            p = count / total_samples
            entropy_val -= p * np.log2(p)
        return entropy_val

    def information_gain(self, X, y, feature, split_value=None):
        parent_entropy = self.entropy(y)
        if split_value is not None:
            left_indices = X[:, feature] <= split_value
            right_indices = ~left_indices
            left_child_entropy = self.entropy(y[left_indices])
            right_child_entropy = self.entropy(y[right_indices])
            n_left = np.sum(left_indices)
            n_right = np.sum(right_indices)
            n_total = len(y)
            child_entropy = (n_left / n_total) * left_child_entropy + (n_right / n_total) * right_child_entropy
        else:
            values = np.unique(X[:, feature])
            child_entropy = 0
            for value in values:
                indices = X[:, feature] == value
                child_entropy += (np.sum(indices) / len(y)) * self.entropy(y[indices])
        return parent_entropy - child_entropy

    def find_best_split(self, X, y):
        max_info_gain = -np.inf
        best_feature = None
        best_split_value = None
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for value in values:
                info_gain = self.information_gain(X, y, feature, split_value=value)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature = feature
                    best_split_value = value
        return best_feature, best_split_value

    def fit(self, X, y):
        self.root = self._fit(X, y)

    def _fit(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(leaf_class=y[0])
        elif X.shape[1] == 0:
            return Node(leaf_class=np.argmax(np.bincount(y)))
        else:
            best_feature, best_split_value = self.find_best_split(X, y)
            if best_feature is None:
                return Node(leaf_class=np.argmax(np.bincount(y)))
            node = Node(feature=best_feature, value=best_split_value)
            if best_split_value is not None:
                left_indices = X[:, best_feature] <= best_split_value
                right_indices = ~left_indices
                node.children['left'] = self._fit(X[left_indices], y[left_indices])
                node.children['right'] = self._fit(X[right_indices], y[right_indices])
            else:
                values = np.unique(X[:, best_feature])
                for value in values:
                    indices = X[:, best_feature] == value
                    node.children[value] = self._fit(X[indices], y[indices])
            return node

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if node.leaf_class is not None:
            return node.leaf_class
        else:
            if node.value is not None:
                if x[node.feature] <= node.value:
                    return self._predict(x, node.children['left'])
                else:
                    return self._predict(x, node.children['right'])
            else:
                return self._predict(x, node.children[x[node.feature]])

@st.cache
def load_data():
    return pd.read_csv("trainingdata.csv")  # Replace "your_dataset.csv" with your dataset file name

def preprocess_data(df):
    label_encoders = [LabelEncoder() for _ in range(df.shape[1])]
    for i, encoder in enumerate(label_encoders):
        df.iloc[:, i] = encoder.fit_transform(df.iloc[:, i])
    return df

def main():
    st.title("Decision Tree Classifier")

    st.sidebar.header("Upload your dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write(df.head())

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X = preprocess_data(X)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the decision tree model
        tree = DecisionTreeID3()
        tree.fit(X_train.values, y_train.values)

        st.header("Decision Tree Visualization")
        st.write("Visualize the decision tree here.")

        st.header("Make Predictions")
        st.write("Enter values for the features to predict the class:")
        new_sample = []
        for i in range(X.shape[1]):
            new_sample.append(st.number_input(f"Feature {i+1}", value=X.iloc[0, i]))
        new_sample = np.array(new_sample)

        if st.button("Predict"):
            predicted_class = tree.predict(new_sample.reshape(1, -1))
            st.write(f"The predicted class is: {predicted_class[0]}")

            # Evaluate the model's accuracy
            y_pred = tree.predict(X_test.values)
            accuracy = accuracy_score(y_test.values, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
