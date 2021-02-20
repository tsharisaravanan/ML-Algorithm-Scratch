# Library 
# Note: The following library is only used for fetching data from local machine 
# to Google Colab 
import io
import math
import pandas as pd
from google.colab import files

#******************************************************************************#
# Functionality: Read the data from local machine to google colab and convert  #
#                dataframe into array data type                                #
# Input : File_name(String)                                                    #
# Output: Header(list), Dataset(list), Target(List)                            #
#******************************************************************************#
def reading_data(file_name):
  # Uploading file from local machine
  uploaded = files.upload()
  # Assigning the uploaded dataset into dataframe using Pandas Library
  dataset = pd.read_csv(io.BytesIO(uploaded[file_name]))
  # Collecting column names into list
  header = dataset.columns
  # For convenient converting dataset from dataframe to array
  dataset = dataset.values.tolist()
  # Fetching target column 
  target_idx = len(dataset[0])-1
  target = [dataset[row][target_idx] for row in range(len(dataset))]
  # Removing target column from dataset using pop() method
  for row in dataset:
    row.pop()

  return header, dataset, target

#******************************************************************************#
# Functionality: Create lookup table to store unique clasification label and   #
#                encoding value in order to do encoding to target class label  #
# Input : Target(List)                                                         #
# Output: Lookup table(Dictonary), New_target(List)                            #
#******************************************************************************#
def target_encoding(target):
  # Creating Dictonary to store lookup table values
  lookup_table = dict()
  # Finding unique values from the target feature class
  unique_class = set(target)
  # Encoding technique using enumerate
  for encoding, key in enumerate(unique_class):
    lookup_table[key] = encoding
  # Assigning encoded values to target column
  new_target = [lookup_table[row] for row in target]
  
  return lookup_table, new_target 

#******************************************************************************#
# Functionality:Convert all feature values into same scale using normalization # 
#               technique, where all values between 0 to 1                     #
# Input : Dataset(List)                                                        # 
# Output: Dataset(List)                                                        #
#******************************************************************************#
def normalization(dataset):
  # Finding Min-Max values on each feature
  min_max = list()
  for col in range(len(dataset[0])):
    _column = [row[col] for row in dataset]
    min_max.append([min(_column), max(_column)])
  # Column length
  col_len = len(dataset[0])
  # Rescalling the dataset using normalization formula
  for row in dataset:
    for col in range(len(row)):
      row[col] = ((row[col] - min_max[col][0]) / 
                    (min_max[col][1] - min_max[col][0]))
                    
  return dataset

#******************************************************************************#
# Functionality: Get training from given dataset to find best K-Parameter      # 
# Input : Dataset(List)                                                        # 
# Output: best_k_param(Int), Accuracy_list(List)                               #
#******************************************************************************#
def train(features, target, unique_class):
    # Initializing output parameters
    accuracy_list= list()
    best_accuracy = 0.0
    best_k_param  = 0
    # Exprementing the dataset with different K parameter value using KNN
    for k in range(1,15):
        prediction, correctness = nearest_neighbours(features, target, 
                                                               unique_class, k)
        # Accuracy with respect to K parameter
        _accuracy = (correctness/len(features))*100
        # Appending accuracy and K parameter value in list
        accuracy_list.append([k, _accuracy])
        # Updating the best K parameter value if accuracy is best compare to previous
        if best_accuracy < _accuracy:
            best_k_param = k
            best_accuracy = _accuracy

    return best_k_param, accuracy_list
    
#******************************************************************************#
# Functionality: Finding K nearest neighbour using distance metrics            # 
# Input : Features(List), Target(List), Unique_class(Int), K(Int)              # 
# Output: Prediction(List), Correctness(Int)                                   #
#******************************************************************************#  
def nearest_neighbours(features, target, unique_class, k):
    # Initializing the temporary & output variables
    neighbours   = list()
    prediction   = list()
    correctness  = 0
    # Iterating each datapoint one by one
    for idx, row in enumerate(features):
        # Temporary distance list to store the neighbours distance
        distance_list = list()
        # Iterating main datapoint vs remaining datapoint
        for idy, _row in enumerate(features):
            if idx != idy:
                # Distance metrics calculation formula
                #distance = euclidean_distance(row, _row)
                distance = chi2_distance(row, _row)
                distance_list.append([distance, target[idy]])
        # Sorting the neighbours distance and selecting first K datapoint
        distance_list.sort()
        neighbours = distance_list[:k]
        # Creating lookup table to store the distance for each class label
        lookup_table = dict()
        for record in neighbours:
            value = record[1]
            # If the class label is already existing then add +1 to its value
            if value in lookup_table:
                lookup_table[value] += record[0]
            # Else create new entry in lookuptable with value as 1
            else:
                lookup_table[value] = record[0]
        # Weight is inversely propotional to distance
        # Finding Majority of class label based on distance
        max_class = max(lookup_table, key=lookup_table.get)
        # Correctness of the prediction
        if max_class == target[idx]:
            correctness += 1
        # Confidence about the prediction
        confidence = max_class / unique_class
        prediction.append([max_class, confidence])  

    return prediction, correctness
    
#******************************************************************************#
# Functionality: Finding distance between two vectors and return 1 / Distance  #
#                Distance can be calculated using chi2 distance matrics        #
# Input : Row1(List), Row2(List)                                               # 
# Output: Distance(Int)                                                        #
#******************************************************************************#  
def chi2_distance(row1, row2):
    distance = 0.0
    for index in range(len(row1)-1):
        distance += ((row2[index] - row1[index])**2) / (row2[index] + 
                                                                    row1[index])
    # Weighted KNN = 1 / distance
    return 1 / (0.5 * distance)

#******************************************************************************#
# Functionality: Finding distance between two vectors and return 1 / Distance  #
#                Distance can be calculated using Euclidean distance matrics   #
# Input : Row1(List), Row2(List)                                               # 
# Output: Distance(Int)                                                        #
#******************************************************************************# 
def euclidean_distance(row1, row2):
    distance = 0.0
    for index in range(len(row1)-1):
        distance += ((row2[index] - row1[index])**2)
    # Weighted KNN = 1 / distance
    
    return 1 / (math.sqrt(distance))

#******************************************************************************#
# Functionality: Predicting class label for new query point using best K-Param #
# Input : features(List), target(List), new_point(List), k(Int), class_table   # 
# Output:class_label(String)                                                   #
#******************************************************************************# 
def predict(features, target, new_point, k, class_table):
    # Creating lookup table to store the distance for each class label
    lookup_table = dict()
    # Final prediction class label
    class_label = None
    # Distance calculation between all datapoint 
    distance_list = [[euclidean_distance(new_point, row), 
                              target[idx]] for idx, row in enumerate(features)]
    # Sorting the distance list and selecting first K neighbours
    distance_list.sort()
    distance_list = distance_list[:k]
    # lookup table to store the distance for each class label
    for record in distance_list:
        value = record[1]
        # If the class label is already existing then add +1 to its value
        if value in lookup_table:
            lookup_table[value] += record[0]
        # Else create new entry in lookuptable with value as 1
        else:
            lookup_table[value] = record[0]
    # Weight is inversely propotional to distance so finding max distance
    # Finding Majority of class label based on distance
    max_vote = max(lookup_table, key=lookup_table.get)
    # Idenfing the result class label
    for key, value in class_table.items():
        if value == max_vote:
            class_label = key

    return class_label

def main():
  # 1. Reading input data from local machine
  header, dataset, target = reading_data("Iris_Data.csv")
  # 2. Lookup table for target feature
  lookup_table, target = target_encoding(target)
  # 3. Normalization of the dataset
  #dataset = normalization(dataset)
  # 4. Training the dataset in order to find K-Parameter value
  unique_class = len(lookup_table)
  k_param, accuracy_list = train(dataset, target, unique_class)
  # 5. Prediction for new data points
  new_point = [5.1, 2.2, 1.4, 0.2]
  predicted_class = predict(dataset, target, new_point, k_param, lookup_table)
  print(predicted_class)
  
if __name__ == '__main__':
  main() 
