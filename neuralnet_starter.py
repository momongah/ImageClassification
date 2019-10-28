import numpy as np
import pickle
from matplotlib import pyplot as plt


config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 3  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm (0.0001 original value)

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
      return self.ReLU(a)
  
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
    grad = 1  - self.tanh(self.x) ** 2
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
    # unccomment for 2b, numerical approximation of gradient
    # if out_units == 10:
    #   e = -0.1
    # else:
    #   e = 0
    self.w = np.random.randn(in_units, out_units)# Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    # self.w[0, 0] += e
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this || prod of delta & x ???
    self.d_w = None  # Save the gradient w.r.t w in this || prod of delta & w ???
    self.d_b = None  # Save the gradient w.r.t b in this || prod of delta & b ???
    self.v = None # save the 'prev' weight change (for momentum)

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    # add bias
    x_w_ones = np.append(np.ones([x.shape[0], 1]), x, 1)
    w_b = np.concatenate((self.b, self.w))
    self.a = x_w_ones @ w_b
    
    # print("shape of x: ", x.shape, "shape of x_w_ones: ", x_w_ones.shape)
    # print("shape of w: ", self.w.shape, "shape of w_b: ", w_b.shape, "shape of bias: ", self.b.shape)
    # print("shape of a: ", self.a.shape)
    
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """

    # don't think i accounted for bias, maybe it has a large effect

    # print("shape of delta incoming: ", delta.shape, "shape of x: ", self.x.shape)
    self.d_x = delta.T @ self.x
    # print("SHAPE OF GRADIENT: ", self.d_x.shape)
    
    # saving 
    #self.v = delta.T @ self.x

    # backprop for bias weights
    x_0 = np.ones([len(delta), 1])
    self.d_b = delta.T @ x_0

    # print("shape of BIAS GRAD: ", self.d_b.shape)

    self.d_w = delta @ self.w.T
    # print("shape of w.T: ", self.w.T.shape, "shape of RETURN delta: ", self.d_w.shape)
    #print(self.w.shape)
    return self.d_w
      

class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    #self.v = None # 
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

    result = x
    for layer in self.layers:
      result = layer.forward_pass(result)

    # softamax activation on input
    self.y = softmax(result)

    if targets is not None:
       loss = self.loss_func(self.y, self.targets)

    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    prod = targets * np.log(logits)
    return -1 * np.sum(prod)
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    a = config['learning_rate']
    y = config['momentum_gamma']
    m = config['momentum']

    



    #delta of t - y
    delta = self.targets - self.y
    for layer in reversed(self.layers):
      delta = layer.backward_pass(delta)
    
    #update weights
    for i in np.arange(0, 3, 2):
      layer = self.layers[i]
      #first iteration so no 'prev weight change'
      # if iteration == 0:
      #   prev_w_chng = np.zeros(layer.d_x.T.shape)
      layer.w = layer.w + a * layer.d_x.T #+ (y * prev_w_chng) #updating non-bias weights
      # prev_w_chng = a * layer.d_x.T + (y * prev_w_chng)
      layer.b = layer.b + a * layer.d_b.T #updating bias weights

      '''(a * layer.d_x.T) + (y * prev_weight_change)''' # is the 'last
      '''prev_weight_change = (a * layer.d_x.T) + (y * prev_weight_change)'''

def numerical_approximation(model, X_train, y_train, X_valid, y_valid, config):
  # choose minibatch
  x = X_train[0 : config["batch_size"]]
  targets = y_train[0 : config["batch_size"]]
  loss, y_pred = model.forward_pass(x, targets)
  model.backward_pass()
  loss = loss / (config['batch_size'] * 10)  # 10 classes

  print(loss)
  # 1.1031559403913496
  # 1.1043907379043258
  a = 1.077788134752671
  b = 1.0779071056951173
  approx = (a - b) / (2 * 0.1) * -1
  # out_bias = model.layers[0].d_b[0]
  # in2hid_1 = model.layers[0].d_w[0, 0]
  # in2hid_2 = model.layers[0].d_w[0, 1]
  hid2out_1 = model.layers[2].d_w[0, 0]
  hid2out_2 = model.layers[2].d_w[0, 1]
  grad = hid2out_1 / 10000
  print(approx, grad)
  print(approx - grad)

  """
  output_bias:
  numerical - 1.1031559403913496
  backprop - 1.1043907379043258
  difference - 0.00019159

  hidden bias:
  numerical - 0.0004665521402780204
  backprop - 0.00046105
  difference - 5.5028226e-06

  input -> hidden #1:
  numerical - -0.0004665521332924971
  backprop - -0.00015547085702011495
  difference - 0.0003110812762723822

  input -> hidden #2:
  numerical - 0.0006169833390679003
  backprop - -0.00021182621937263678
  difference - 0.0008288095584405371

  hidden -> output #1:
  numerical - 0.0005948547122314185
  backprop - 6.990824645101484e-06
  difference - 0.000587863887586317z

  hidden -> output #2:
  numerical - -0.010276905589781116
  backprop - 9.518257371033264e-06
  difference - 0.010286423847152148
  """
  

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

  stop_count = config['early_stop_epoch']

  xnloss = []
  val_loss = [float('inf')]
  test_scores = []


  #validation loss increase per epoch counter
  c = 0
  
  for i in range(config["epochs"]):
    np.random.seed(i)
    np.random.shuffle(X_train)

    np.random.seed(i)
    np.random.shuffle(y_train)

    '''You should average the loss across all mini batches'''
    #means sum up loss from all mini-batches and divide by num_batches
    sums = 0

    num_batches = int(X_train.shape[0] / config["batch_size"])
    k=0
    for j in range(num_batches):
      # choose minibatch
      x = X_train[j * config["batch_size"] : (j+1) * config["batch_size"]]
      targets = y_train[j * config["batch_size"] : (j+1) * config["batch_size"]]
      loss, y_pred = model.forward_pass(x, targets)
      loss = loss / (config['batch_size'] * 10)  # 10 classes
      sums += loss
      #xnloss.append(loss)
      model.backward_pass()
      k +=1
      # if k < 5 or k > 44:
      #   print(targets[0, :])
      #   print(y_pred[0, :])
      #   print(y_pred[0, :].sum())
      #   print(k, '=============')

    # mini-batch done here, take avg of loss
    avg_loss = sums / num_batches
    xnloss.append(avg_loss)
    
    ''' epochs loop continues here
     0) perform validation and compute its (val) loss

     1) calculate test accuracy for every epoch where the
     validation loss is better than the previous validation loss.
     
     2) Save this result (test score OR loss?) and choose the best 
     one when you hit the early stopping criteria.

     3) early stopping - stop training (epochs loop) after 5th consecutive 
    increase in validation loss. (Experiment with diff values).
    '''

    '''VALIDATION PERFORMACE'''
    v_loss, v_pred = model.forward_pass(X_valid, y_valid)
    v_loss_norm = v_loss / (len(X_valid) * 10)


    '''TEST ACCURACY''' 
    #if val loss better (less) than prev: calculate test scores
    if v_loss_norm < val_loss[i]:
      '''insert code for test accu here'''
      val_loss.append(v_loss_norm)
    else: #else val loss increased, so increment counter
      print("val loss greater than last time")
      c += 1

    
    '''EARLY STOPPING'''
    if c == stop_count:
      print("early stopped at epoch =", i+1)
      break

  #outside of epochs loop
  plt.plot(xnloss, label='training loss')
  plt.plot(val_loss[1:], label='validation loss')
  plt.title("losses across all epochs")
  plt.xlabel("epochs")
  plt.ylabel("avg loss for the epoch")
  plt.legend()
  # plt.savefig('trainVSval_loss.png')
  plt.show()
  #firstplot.png is training loss against # of batches, in 1 epoch
  #avgacrossepochs.png is avg training loss of all batches, across 50 epochs
  # both_losses = []
  # for i in range(len(xnloss)):
  #   both_losses.append((val_loss[i], xnloss[i]))
  # print("validation errors: ", [(val_loss[i], xnloss[i]) for i in range(len(xnloss))])
  
  
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  accuracy = 0
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
  # uncomment for numerical approximation
  # numerical_approximation(model, X_train, y_train, X_valid, y_valid, config)
  # test_acc = test(model, X_test, y_test, config)

  # print(X_valid.shape, X_train.shape, X_test.shape, y_test.shape)
