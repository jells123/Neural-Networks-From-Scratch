import h5py
import numpy as np

def get_labels(idx):
    W_label = 'W{}'.format(idx)
    b_label = 'b{}'.format(idx)

    A_label = 'A{}'.format(idx)
    A_prev_label = 'A{}'.format(idx-1)
    
    return W_label, b_label, A_label, A_prev_label
    
def get_d_labels(idx):
    dA_label = 'dA{}'.format(idx)
    dZ_label = 'dZ{}'.format(idx)
    dW_label = 'dW{}'.format(idx)
    db_label = 'db{}'.format(idx)
    
    return dA_label, dZ_label, dW_label, db_label

def load_hdf5(db_file):
    with h5py.File(db_file, 'r') as f:
        train_x, train_y = f['train_x'][...], f['train_y'][...]
        test_x, test_y = f['test_x'][...], f['test_y'][...]
        
        return train_x, train_y, test_x, test_y

def init_weights(input_size, hidden_layers, output_size, method="random"):
    weights, bias = {}, {}
    
    if method == 'random':
        
        weights['W1'] = np.random.randn(hidden_layers[0], input_size) * 0.1
        weights['W{}'.format(len(hidden_layers)+1)] = np.random.randn(output_size, hidden_layers[-1]) * 0.1

        bias['b1'] = np.zeros(shape=(hidden_layers[0], 1), dtype=np.float32)
        bias['b{}'.format(len(hidden_layers)+1)] = np.zeros(shape=(output_size, 1), dtype=np.float32)

        for idx, units in enumerate(hidden_layers[:-1]):

            label = 'W{}'.format(idx+2)
            weights[label] = np.random.randn(hidden_layers[idx+1], hidden_layers[idx]) * 0.1

            label = 'b{}'.format(idx+2)
            bias[label] = np.zeros(shape=(hidden_layers[idx+1], 1), dtype=np.float32)
    
    elif method == 'xavier' or method == 'he':
        
        mean = 0
        
        if method == 'xavier':
            std_dev = np.sqrt(2 / (input_size+hidden_layers[0]))
            weights['W1'] = np.random.normal(mean, std_dev, (hidden_layers[0], input_size))
            
            std_dev = np.sqrt(2 / (output_size+hidden_layers[-1]))
            weights['W{}'.format(len(hidden_layers)+1)] = np.random.normal(mean, std_dev, (output_size, hidden_layers[-1]))
    
        elif method == 'he':
            std_dev = np.sqrt(2 / input_size)
            weights['W1'] = np.random.normal(mean, std_dev, (hidden_layers[0], input_size))
            
            std_dev = np.sqrt(2 / hidden_layers[-1])
            weights['W{}'.format(len(hidden_layers)+1)] = np.random.normal(mean, std_dev, (output_size, hidden_layers[-1]))

        bias['b1'] = np.zeros(shape=(hidden_layers[0], 1), dtype=np.float32)
        bias['b{}'.format(len(hidden_layers)+1)] = np.zeros(shape=(output_size, 1), dtype=np.float32)

        for idx, units in enumerate(hidden_layers[:-1]):

            if method == 'xavier':
                std_dev = np.sqrt(2 / (hidden_layers[idx]+hidden_layers[idx+1]))
            elif method == 'he':
                std_dev = np.sqrt(2 / (hidden_layers[idx]))
            
            label = 'W{}'.format(idx+2)
            weights[label] = np.random.normal(mean, std_dev, (hidden_layers[idx+1], hidden_layers[idx]))

            label = 'b{}'.format(idx+2)
            bias[label] = np.zeros(shape=(hidden_layers[idx+1], 1), dtype=np.float32)
                
    return weights, bias

