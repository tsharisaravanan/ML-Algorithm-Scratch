from csv import reader
import math 

""" Method Name: read_csv 
     -->  Import: File name
    <--   Export: Header[] (column name) & Dataset[] (data values) 
    f():  Functionality: Reading the CSV file from the loacal system using reader() method under CSV library"""
def read_csv(file_name):
    header = dataset = list()
    header_flag = True
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            elif header_flag == True:
                header = row
                header_flag = False
            else:
                dataset.append(row)
    return header, dataset

""" Method Name: string_to_float
     -->  Import: dataset [[]]
    <--   Export: dataset [[]]
    f():  Functionality: Typecast the string datatype to float for input features """
def string_to_float(dataset):
    for row in range(len(dataset)):
        for column in range(len(dataset[row])-1):
            dataset[row][column] = float(dataset[row][column].strip())
    return dataset

""" Method Name: Lookup_functionality
     -->  Import: dataset [[]]
    <--   Export: lookup_table {Target : encode_value}
    f():  Functionality: This method will create dictonary with unique target variables and encode value """
def lookup_functionality(dataset):
    lookup_table = dict()
    # Extracting unique target values
    target_idx = len(dataset[0])-1
    unique_val = set([dataset[row][target_idx] for row in range(len(dataset))])
    for encode, key in enumerate(unique_val):
        lookup_table[key] = encode
    return lookup_table

""" Method Name: Lookup_functionality()
     -->  Import: lookup_table {} & dataset [[]]
    <--   Export: dataset [[]]
    f():  Functionality: This method will return the list with numerical encoded target variable """
def target_encoding(lookup_table, dataset):
    target_idx = len(dataset[0])-1
    for row in range(len(dataset)):
        dataset[row][target_idx] = lookup_table[dataset[row][target_idx]]
    return dataset

""" Method Name: min_max_functionality()
     --> Import: dataset [[]]
    <--  Export: min_max [[]]
    f(): Functionality: This method will return the minimum & maximum value of each column in the form of list """
def min_max_function(dataset):
    min_max = list()
    for col in range(len(dataset[0])):
        _column = [row[col] for row in dataset]
        min_max.append([min(_column), max(_column)])
    return min_max

""" Method Name: normalization()
     --> Import: dataset [[]]
    <--  Export: dataset [[]]
    f(): Functionality: This method will return the normalized dataset """
def normalization(dataset):
    min_max = min_max_function(dataset)
    for row in dataset:
        for col in range(len(row)-1):
            row[col] = ((row[col] - min_max[col][0]) / (min_max[col][1] - min_max[col][0]))
    return dataset

""" Method Name: dataset_split()
     --> Import: dataset [[]]
    <--  Export: dataset [[]] & target []
    f(): Functionality: This method will split the input features & target features in seprate list """
def dataset_split(dataset):
    features = list()
    target = list()
    target_idx = len(dataset[0])-1
    for row in dataset:
        val = row[target_idx]
        target.append(val)
        row.remove(row[target_idx])
    return dataset, target

""" Method Name: euclidean_distance()
     --> Import: row1 [] & row2 []
    <--  Export: distance
    f(): Functionality: This method will return the distance between two vectors using euclidean distance formula """
def euclidean_distance(row1, row2):
    distance = 0.0
    for index in range(len(row1)-1):
        distance += ((row2[index] - row1[index])**2)
    return math.sqrt(distance)

""" Method Name: chi2_distance()
     --> Import: row1 [] & row2 []
    <--  Export: distance
    f(): Functionality: This method will return the distance between two vectors using chi square distance formula """
def chi2_distance(row1, row2):
    distance = 0.0
    for index in range(len(row1)-1):
        distance += ((row2[index] - row1[index])**2) / (row2[index] + row1[index])
    return 0.5 * distance

""" Method Name: nearest_neighbours()
     --> Import: features [[]], target [], unique_class, k
    <--  Export: prediction [](list of prediction accuracy) & Correctness (correctness value of target class)
    f(): Functionality: This method will return the prediction accuracy & correctness value WRT K-parameter """
def nearest_neighbours(features, target, unique_class, k):
    neighbours   = list()
    prediction   = list()
    correctness  = 0
    for idx, row in enumerate(features):
        distance_list = list()
        for idy, _row in enumerate(features):
            if idx != idy:
                distance = euclidean_distance(row, _row)
                #distance = chi2_distance(row, _row)
                distance_list.append([distance, target[idy]])
        distance_list.sort()
        neighbours = distance_list[:k]
        lookup_table = dict()
        for record in neighbours:
            value = record[1]
            if value in lookup_table:
                lookup_table[value] += 1
            else:
                lookup_table[value] = 1
        max_class = max(lookup_table, key=lookup_table.get)
        # Correctness of the prediction
        if max_class == target[idx]:
            correctness += 1
        # Confidence about the prediction
        confidence = max_class / unique_class
        prediction.append([max_class, confidence])    
    return prediction, correctness

""" Method Name: finding_k_parameter()
     --> Import: features [[]], target [], unique_class
    <--  Export: best_k_value & accuracy_list
    f(): Functionality: This method will find the best K parameter to find the neighbours with respective accuracy"""
def finding_k_parameter(features, target, unique_class):
    accuracy_list= list()
    best_accuracy = 0.0
    best_k_param  = 0
    for k in range(1,15):
        prediction, correctness = nearest_neighbours(features, target, unique_class, k)
        # Accuracy with respect to K parameter
        _accuracy = (correctness/len(features))*100
        accuracy_list.append([k, _accuracy])
        # Updating the best K parameter value 
        if best_accuracy < _accuracy:
            best_k_param = k
            best_accuracy = _accuracy
    return best_k_param, accuracy_list

""" Method Name: predict()
     --> Import: features [[]], target [], new_points, class_table {}
    <--  Export: _class
    f(): Functionality: This method will predict the output class for new point with respect to best K parameter value"""
def predict(features, target, new_point, k, class_table):
    lookup_table = dict()
    _class = None
    distance_list = [[euclidean_distance(new_point, row), target[idx]] for idx, row in enumerate(features)]
    distance_list.sort()
    distance_list = distance_list[:k]
    for record in distance_list:
        value = record[1]
        if value in lookup_table:
            lookup_table[value] += 1
        else:
            lookup_table[value] = 1
    max_vote = max(lookup_table, key=lookup_table.get)
    for key, value in class_table.items():
        if value == max_vote:
            _class= key
    return _class
    
def main():
    # 1. Reading the input file from the local machine
    header, dataset = read_csv("Iris dataset.csv")
    # 2. Data preprocessing
    dataset = string_to_float(dataset)
    # 3. Encoding to target variable
    lookup_table = lookup_functionality(dataset)
    dataset = target_encoding(lookup_table, dataset)
    # 4. Normalizing the dataset
    #dataset = normalization(dataset)
    # 5. Data sepration 
    features, target = dataset_split(dataset)
    # 6. Finding best K-Parameter value
    unique_class = len(lookup_table)
    k_parameter, accuracy_list = finding_k_parameter(features, target, unique_class)
    # 7. Prediction for new data points
    new_point = [5.1, 2.2, 1.4, 0.2]
    predicted_class = predict(features, target, new_point, k_parameter, lookup_table)
    print(predicted_class)
    
if __name__ == "__main__":
    main()
