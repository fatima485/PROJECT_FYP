import pandas as pd

legitimate_urls = pd.read_csv("extracted_legitimate_urls.csv")
phishing_urls = pd.read_csv("extracted_phishing_urls.csv")

print(legitimate_urls.head(10))
print(phishing_urls.head(10))

#data preprocessing

#merging
urls = legitimate_urls.append(phishing_urls)
print(urls)


print(urls.head(5))

print(urls.shape[1])

print(urls.columns)

#Removing Unnecessary columns
urls = urls.drop(urls.columns[[0,1,2]],axis=1)

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
urls = urls.sample(frac=1).reset_index(drop=True)

#Removing class variable from the datase
urls_without_labels = urls.drop('label',axis=1)
print(urls_without_labels.columns)
labels = urls['label']

#splitting the data into train data and test data
#Dividing the data in the ratio of 70:30 [train_data:test_data]
from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(urls_without_labels, labels, test_size=0.30, random_state=110)

print(len(data_train),len(data_test),len(labels_train),len(labels_test))

'''checking the split of labels in train and test data
The split should be in equal proportion for both classes

Phishing - 1

Legitimate - 0'''

#initially checking the split of labels_train data 
print(labels_train.value_counts())

#checking the split for labels_test data
labels_test.value_counts()

#Creating the model and fitting the data into the model
#creating the model with default parameters
from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier()

print(random_forest_classifier.fit(data_train,labels_train))

#Predicting the result for test data
prediction_label = random_forest_classifier.predict(data_test)

#Creating confusion matrix and checking the accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cpnfusionMatrix = confusion_matrix(labels_test,prediction_label)
print(cpnfusionMatrix)
print(accuracy_score(labels_test,prediction_label))

import matplotlib.pyplot as plt
import numpy as np

#feature_importances_ : array of shape = [n_features] ------ The feature importances (the higher, the more important the feature).

#feature_importances_  -- This method returns the quantified relative importance in the order the features were fed to the algorithm

importances = random_forest_classifier.feature_importances_

#std = np.std([tree.feature_importances_ for tree in random_forest_classifier.estimators_],axis=0)   #[[[estimators_ :explaination ---  list of DecisionTreeClassifier ----- (The collection of fitted sub-estimators.)]]]

#To make the plot pretty, we’ll instead sort the features from most to least important.
indices = np.argsort(importances)[::-1] 
print(f"indices of columns : {indices}")

# Print the feature ranking
print("\n ***Feature ranking: *** \n")
print("Feature name : Importance")

for f in range(data_train.shape[1]):
    print(f"{f+1} {data_train.columns[indices[f]]}   :  {importances[indices[f]]} \n")
    
print("**** The red bars are the feature importances of the randomforest classifier, along with their inter-trees variability*****")

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(data_train.shape[1]), importances[indices],
       color="r", align="center")   
#yerr=std[indices] -- this is another parameter that can be included if std is calculated above
#and also it gives error bar that's the reason we calculate std above. but here we are not making it plot.

plt.xticks(range(data_train.shape[1]), data_train.columns[indices])
plt.xlim([-1, data_train.shape[1]])

plt.rcParams['figure.figsize'] = (35,15)  #this will increase the size of the plot
plt.show()







