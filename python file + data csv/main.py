import pandas as pd
from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

basedir = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(basedir, 'mushrooms.csv')
mooshroom_og = pd.read_csv(data_file)

# LabelEncoder pandas method to get numerical values, and then fit_transform our data
label_encoder = LabelEncoder()
mooshroom = mooshroom_og.apply(label_encoder.fit_transform)


# There are 8124 rows and 23 columns in our dataset, so 8124 shroomies with 23 examined features
# There are a shit ton of feature, so it's optimal to check which ones correlate best to our label
def roundto3digits(z):
    return round(z, 3)


mooshroom_correlation_data = mooshroom.corr().apply(roundto3digits)
# the shape of this is 23x23, it contains the correlation between all of them, we are interested in the correlation
# between each feature and the class feature. iloc[0,:] retrieves all the info related to the first column (class
# feature corerlations)
mooshroom_correlation_classfeature = mooshroom_correlation_data.iloc[0, :]
correlations_dic = dict(zip(mooshroom_correlation_classfeature.axes[0], mooshroom_correlation_classfeature.array))

# now we want to remove the features that are let's say lower than 0.2 correlation (in absolute value of course)
x = mooshroom.copy()
for feature in mooshroom_correlation_classfeature.axes[0]:
    if math.isnan(correlations_dic[feature]):
        x = x.drop(feature, axis=1)
    elif abs(correlations_dic[feature]) < 0.2:
        x = x.drop(feature, axis=1)

# let's now drop the class feature and assign it to a variable of it's own
# 70/30 data split in favor of training
y = mooshroom['class']
x = x.drop('class', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

svm_clf = SVC()
svm_clf.fit(x_train.values, y_train.values)

y_pred_train = svm_clf.predict(x_train.values)
y_pred_test = svm_clf.predict(x_test.values)

accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
confusion_matrix = confusion_matrix(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test)
# print("accuracy on training set is :" + str(accuracy_train))
# print("accuracy on testing set is :" + str(accuracy_test))

encodemenet_dic = {'bruises': {'true': 1, 'false': 0}, 'gill-spacing': {'close': 0, 'crowded': 1},
                   'gill-size': {'broad': 0, 'narrow': 1},
                   'gill-color': {'buff': 0, 'red': 1, 'gray': 2, 'chocolate': 3, 'black': 4, 'brown': 5, 'orange': 6,
                                  'pink': 7, 'green': 8, 'purple': 9, 'white': 10,
                                  'yellow': 11},
                   'stalk-root': {'?': 0, 'bulbous': 1, 'club': 2, 'equal': 3, 'rooted': 4, },
                   'stalk-surface-above-ring': {'fibrous': 0,
                                                'silky': 1, 'smooth': 2, 'scaly': 3},
                   'stalk-surface-below-ring': {'fibrous': 0, 'silky': 1, 'smooth': 2, 'scaly': 3},
                   'ring-number': {'none': 0, 'one': 1, 'two': 2},
                   'ring-type': {'evanescent': 0, 'flaring': 1, 'large': 2, 'none': 3, 'pendant': 4},
                   'population': {'abundant': 0, 'clustered': 1, 'numerous': 2, 'scattered': 3, 'several': 4,
                                  'solitary': 5},
                   'habitat': {'woods': 0, 'grasses': 1, 'leaves': 2, 'meadows': 3, 'paths': 4, 'urban': 5, 'waste': 6}}


def encodement_getter(name, number):
    dic = {}
    for z in range(number):
        for i in range(8124):
            if x[name][i] == z:
                dic[z] = mooshroom_og[name][i]
    return dic


# requires a dictionary, values are the attributes and here are the keys : bruises gill-spacing  gill-size    gill-color   stalk-root  stalk-surface-above-ring
# stalk-surface-below-ring ring-number  ring-type populationhabitat

def data_converter(dic_attributes):
    encoded_data = []
    for key in dic_attributes:
        encoded_data.append(encodemenet_dic[key][dic_attributes[key]])
    return encoded_data


@app.route("/")
def hello():
    return "hello"


@app.route("/predict", methods=["POST"])
def data_predicter():
    data = request.get_json(force=True)

    dic_attributes = data
    x_to_predict = data_converter(dic_attributes)
    prediction = svm_clf.predict(np.array(x_to_predict).reshape(1, -1))
    if prediction == [0]:
        return " edible"
    if prediction == [1]:
        return " poisonous"


if __name__ == '__main__':
    app.run(debug=True)
