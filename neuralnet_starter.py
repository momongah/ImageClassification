import numpy as np
import pickle


config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  num = np.exp(x)
  den = np.sum(np.exp(x), axis=1)
  output = (num.T / den).T
  return output


def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  pathname = "data/" + fname
  data = pickle.load(open(pathname, 'rb'), encoding='latin1')
  images = np.array([img[:-1] for img in data])
  ys = [int(img[-1]) for img in data]
  length = len(ys)
  labels = np.zeros((length, 10))

  for i in range(length):
    labels[i, ys[i]] = 1

  return images, labels


class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.relu(a)
  
  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    
    return grad * delta
      
  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = 1 / (1 + np.exp(-x))
    return output

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = np.tanh(x)
    return output

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = np.maximum(0, x)
    return output

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    grad = self.sigmoid(self.x) * (1 - self.sigmoid(self.x))
    return grad

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    grad = 1  - self.tanh(self.x) ^ 2
    return grad

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    grad = np.where(self.x > 0, 1, 0)
    return grad


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this || prod of delta & x ???
    self.d_w = None  # Save the gradient w.r.t w in this || prod of delta & w ???
    self.d_b = None  # Save the gradient w.r.t b in this || prod of delta & b ???

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    # add bias
    x_w_ones = np.append(np.ones([x.shape[0], 1]), x, 1)
    w_b = np.concatenate((self.b, self.w))
    self.a = x_w_ones @ w_b

    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    #print(delta.shape)
    self.d_x = delta.T @ self.x
    #print(self.x.shape)
    self.d_w = delta @ self.w.T
    #print(self.w.shape)
    return self.d_w
      

class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))  
    
  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    if targets is None:
      loss = None
    else:
      self.targets = targets
      loss = self.loss_func(self.y, self.targets)


    result = x
    for layer in self.layers:
      result = layer.forward_pass(result)

    # softamax activation on input
    self.y = softmax(result)

    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    return 1
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    #delta of t - y
    delta = self.targets - self.y
    for layer in reversed(self.layers):
      delta = layer.backward_pass(delta)

    #update weights
    for i in np.arange(0, 3, 2):
      layer = self.layers[i]
      layer.w = layer.w + config['learning_rate'] * layer.d_x.T #add learning rate later


      

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  # loop for number of epochs
  # shuffle inputs based off seed
  # need to shuffle validation based off same seed
  # forward prop and get xenloss
  # backprop and update weights
  for i in range(1):#config["epochs"]):
    np.random.seed(i)
    np.random.shuffle(X_train)

    np.random.seed(i)
    np.random.shuffle(y_train)

    num_batches = int(X_train.shape[0] / config["batch_size"])
    k=0
    for j in range(num_batches):
      # choose minibatch
      x = X_train[j * config["batch_size"] : (j+1) * config["batch_size"]]
      targets = y_train[j * config["batch_size"] : (j+1) * config["batch_size"]]
      loss, y_pred = model.forward_pass(x, targets)
      model.backward_pass()
      k +=1
      if k < 5 or k > 44:
        print(targets[0, :])
        print(y_pred[0, :])
        print(y_pred[0, :].sum())
        print(k, '=============')


  
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  return accuracy
      

if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  # test_acc = test(model, X_test, y_test, config)
