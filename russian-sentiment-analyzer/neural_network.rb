require 'nmatrix'
require 'pp'
include Math

def layer_sizes(x, y)
  #####
  #      Arguments:
  #      X -- input dataset of shape (input size, number of examples)
  #      Y -- labels of shape (output size, number of examples)
  #
  #      Returns:
  #      n_x -- the size of the input layer
  #      n_h -- the size of the hidden layer
  #      n_y -- the size of the output layer
  #####
  n_x = x.shape[0] # size of input layer
  n_h = 4
  n_y = y.shape[0] # size of output layer

  return n_x, n_h, n_y
end

def initialize_parameters(n_x, n_h, n_y)
  #####
  #    Argument:
  #    n_x -- size of the input layer
  #    n_h -- size of the hidden layer
  #    n_y -- size of the output layer
  #
  #    Returns:
  #    params -- hash containing parameters:
  #      W1 -- weight matrix of shape (n_h, n_x)
  #      b1 -- bias vector of shape (n_h, 1)
  #      W2 -- weight matrix of shape (n_y, n_h)
  #      b2 -- bias vector of shape (n_y, 1)
  #####

  w1 = NMatrix.random([n_h, n_x]) * 0.01
  b1 = NMatrix.zeroes([n_h, 1])
  w2 = NMatrix.random([n_y, n_h]) * 0.01
  b2 = NMatrix.zeroes([n_y, 1])

  parameters = {W1: w1,
                b1: b1,
                W2: w2,
                b2: b2}
end

def forward_propagation(x, parameters)
  ######
  #    Argument:
  #    X -- input data of size (n_x, m)
  #    parameters -- a hash containing parameters (output of initialization function)
  #
  #    Returns:
  #    A2 -- The sigmoid output of the second activation
  #    cache -- a hash containing "Z1", "A1", "Z2" and "A2"
  ######

  w1 = parameters[:W1]
  b1 = parameters[:b1]
  w2 = parameters[:W2]
  b2 = parameters[:b2]

  z1 = w1.dot(x)

  #z1_with_bias = z1.each_column do |col|
	#col + b1
  #end
  #pp z12
  (0 .. z1.shape[1] - 1).each do |j|
    (0 .. z1.shape[0] - 1).each do |i|
      # pp z1[i,j]
      z1[i,j] += b1[i]
    end
  end
  a1 = activation(z1, 'tanh')

  z2 = w2.dot(a1)
  #z2_with_bias = z2.each_column do |col|
  #  col + b2
  #end
  (0 .. z2.shape[1] - 1).each do |j|
    (0 .. z2.shape[0] - 1).each do |i|
      # pp z1[i,j]
      z2[i,j] += b2[i]
    end
  end
  a2 = activation(z2, 'sigmoid')

  #pp "w1: #{w1.shape}"
  #pp "x: #{x.shape}"
  #pp "b: #{b1.shape}"
  #pp "z1: #{z12.shape}"
  #pp "a1: #{a1.shape}"
  #pp "z1: #{z22.shape}"
  #pp "a2: #{a2.shape}"
  cache = {Z1: z1,
           A1: a1,
           Z2: z2,
           A2: a2}
end

def compute_cost(a2, y, parameters)
  #####
  #    Computes the cross-entropy cost given in equation (13)
  #
  #    Arguments:
  #    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
  #    Y -- "true" labels vector of shape (1, number of examples)
  #    parameters -- a hash containing your parameters W1, b1, W2 and b2
  #
  #    Returns:
  #    cost -- cross-entropy cost given equation (13)
  #####

  m = y.shape[1] # number of example

  # Compute the cross-entropy cost
  a21 = a2.log
  a22 = m_matrix(a2.shape, 1) - a2
  a22 = a22.log
  y2 = m_matrix(y.shape, 1) - y
  # puts y2.class
  #pp "y: #{y.shape}"
  #pp "a21: #{a21.shape}"
  #pp "y2: #{y2.shape}"
  #pp "a22: #{a22.shape}"
  logprobs = y * a21 + y2 * a22
  ones_vector = NMatrix.ones([m, 1])
  #pp logprobs
  #pp ones_vector
  logprobs_sum = logprobs.dot(ones_vector)
  cost = - logprobs_sum / m
end

def backward_propagation(parameters, cache, x, y)
  ######
  #    Implement the backward propagation using the instructions above.
  #
  #    Arguments:
  #    parameters -- a hash containing our parameters
  #    cache -- a hash containing "Z1", "A1", "Z2" and "A2".
  #    X -- input data of shape (2, number of examples)
  #    Y -- "true" labels vector of shape (1, number of examples)
  #
  #    Returns:
  #    grads -- a hash containing your gradients with respect to different parameters
  ######
  m = x.shape[1]

  # First, retrieve W1 and W2 from the hash "parameters".
  w1 = parameters[:W1]
  w2 = parameters[:W2]

  # Retrieve also A1 and A2 from hash "cache".
  a1 = cache[:A1]
  a2 = cache[:A2]

  # Backward propagation: calculate dW1, db1, dW2, db2.
  dz2 = a2 - y
  #pp dz2.shape
  #pp a1.shape
  dw2 = dz2.dot(a1.transpose)
  dw2 = dw2 / m_matrix(dw2.shape, m)
  ones_vector_1 = NMatrix.ones([dz2.shape[1], 1])
  db2 = dz2.dot(ones_vector_1)
  db2 = db2 / m_matrix(db2.shape, m)
  w2 = w2.transpose
  dz1_1 = w2.dot(dz2)
  dz1_2 = derivative(a1, 'tanh')
  #pp dz1_1.shape
  #pp dz1_2.shape
  dz1 = dz1_1 * dz1_2
  dw1 = dz1.dot(x.transpose)
  dw1 = dw1 / m_matrix(dw1.shape, m)
  #db1 = NMatrix.zeroes([dw1.shape[0], 1])
  #dz1.each_column {|col| db1 + col}
  ones_vector_2 = NMatrix.ones([dz1.shape[1], 1])
  db1 = dz1.dot(ones_vector_2)
  db1 = db1 / m_matrix(db1.shape, m)

  #pp "dw1: #{dw1.shape}"
  #pp "db1: #{db1.shape}"
  #pp "dw2: #{dw2.shape}"
  #pp "db2: #{db2.shape}"

  grads = {dW1: dw1,
           db1: db1,
           dW2: dw2,
           db2: db2}
end

def update_parameters(parameters, grads, learning_rate = 1.2)
  #####
  #    Updates parameters using the gradient descent update rule given above
  #
  #    Arguments:
  #    parameters -- a hash containing your parameters
  #    grads -- a hash containing your gradients
  #
  #    Returns:
  #    parameters -- a hash containing your updated parameters
  #####

  # Retrieve each parameter from the hash "parameters"
  w1 = parameters[:W1]
  b1 = parameters[:b1]
  w2 = parameters[:W2]
  b2 = parameters[:b2]

  #pp "w1: #{w1.shape}"
  #pp "b1: #{b1.shape}"
  #pp "w2: #{w2.shape}"
  #pp "b2: #{b2.shape}"

  # Retrieve each gradient from the dictionary "grads"
  dw1 = grads[:dW1]
  db1 = grads[:db1]
  dw2 = grads[:dW2]
  db2 = grads[:db2]

  dw1 = dw1 * m_matrix(dw1.shape, learning_rate)
  db1 = db1 * m_matrix(db1.shape, learning_rate)
  dw2 = dw2 * m_matrix(dw2.shape, learning_rate)
  db2 = db2 * m_matrix(db2.shape, learning_rate)

  # Update rule for each parameter
  w1 = w1 - dw1
  b1 = b1 - db1
  w2 = w2 - dw2
  b2 = b2 - db2

  parameters = {W1: w1,
                b1: b1,
                W2: w2,
                b2: b2}

end

def nn_model(x, y, n_h, lr, num_iterations = 10000, print_cost=false)
  #####
  #    Arguments:
  #    X -- dataset of shape (2, number of examples)
  #    Y -- labels of shape (1, number of examples)
  #    n_h -- size of the hidden layer
  #    num_iterations -- Number of iterations in gradient descent loop
  #    print_cost -- if True, print the cost every 1000 iterations
  #
  #    Returns:
  #    parameters -- parameters learnt by the model. They can then be used to predict.
  #####

  n_x = layer_sizes(x, y)[0]
  n_y = layer_sizes(x, y)[2]
  #print(n_x)
  #print(n_y)

  # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
  parameters = initialize_parameters(n_x, n_h, n_y)
  w1 = parameters["W1"]
  b1 = parameters["b1"]
  w2 = parameters["W2"]
  b2 = parameters["b2"]


  # Loop (gradient descent)

  (0..num_iterations).each do |i|

    # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
    cache = forward_propagation(x, parameters)
    a2 = cache[:A2]

    # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
    grads = backward_propagation(parameters, cache, x, y)

    # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
    parameters = update_parameters(parameters, grads, lr)

    # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
    cost = compute_cost(a2, y, parameters)

    # Print the cost every 1000 iterations
    if print_cost and i % 1000 == 0
        puts "Cost after iteration #{i}: #{cost}"
    end
  end

  parameters
end

def predict(parameters, x)
  #####
  #    Using the learned parameters, predicts a class for each example in X
  #
  #    Arguments:
  #    parameters -- a hash containing parameters
  #    X -- input data of size (n_x, m)
  #
  #    Returns
  #    predictions -- vector of predictions of our model (true / false)
  #####

  # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
  cache = forward_propagation(x, parameters)
  a2 = cache[:A2]
  predictions = (a2 > 0.5)
end

def sigmoid(matrix)
    NMatrix.ones(matrix.shape)/(NMatrix.ones(matrix.shape)+(-matrix).exp)
end 

def tanh(matrix)
    ( (matrix).exp - ( - matrix).exp )/( (matrix).exp + ( - matrix).exp )
end

def activation(matrix, type)
  if type == 'sigmoid'
    sigmoid(matrix)
  elsif type == 'tanh'
    tanh(matrix)
  end
end

def derivative(matrix, type)
  if type == 'sigmoid'
    return (sigmoid(matrix) * (NMatrix.ones(matrix.shape) - sigmoid(matrix)))
  elsif type == 'tanh'
    return (NMatrix.ones(matrix.shape) - tanh(matrix)) * (NMatrix.ones(matrix.shape) + tanh(matrix))
  end
end

def m_matrix (shape, m)
  NMatrix.new(shape, m)
end