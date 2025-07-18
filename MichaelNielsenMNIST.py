import pickle 
import gzip
import numpy as np
import MichaelNielsenNetwork as nt

def load_data():
    with gzip.open("neural-networks-and-deep-learning-master\data\mnist.pkl.gz") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return(training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

training_data, validation_data, test_data = \
load_data_wrapper()
net = nt.Network([784,30,10])
net.SGD(training_data,30, 10, 3, test_data=test_data)
