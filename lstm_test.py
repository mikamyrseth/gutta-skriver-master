X = [18.80, 18.92, 18.21, 18.06, 18.19, 18.38, 18.22, 17.96, 18.09, 17.82, 18.39, 18.40, 18.69, 18.12, 18.09, 18.22, 18.39, 18.44, 18.53, 18.41]
Y = [102.99, 102.64, 102.86, 102.54, 102.71, 102.75, 103.03, 103.02, 103.23, 103.43, 103.71, 103.46, 103.17, 102.67, 102.69, 102.74, 102.89, 103.01, 103.35, 103.31]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.01, random_state=101)

def create_lstm_prediction