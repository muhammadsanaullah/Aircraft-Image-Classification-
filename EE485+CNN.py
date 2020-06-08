
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# In[2]:

input_path = "D:\\Aircraft Dataset\\data\\64ptx\\"
drone_data = np.load(input_path + "drone_image_data.npy")
fighter_data = np.load(input_path + "fighter_image_data.npy")
helicopter_data = np.load(input_path + "helicopter_image_data.npy")
missile_data = np.load(input_path + "missile_image_data.npy")
plane_data = np.load(input_path + "plane_image_data.npy")
rocket_data = np.load(input_path + "rocket_image_data.npy")


# In[4]:

data_without_label = np.concatenate((drone_data,fighter_data,helicopter_data, missile_data, plane_data, rocket_data), axis = 0).astype(np.float32)
data_without_label /= 255.0
print("Shape of the unlabeled data matrix: " + str(data_without_label.shape))


# In[6]:

drone_data_flat = drone_data.reshape(drone_data.shape[0], -1).T
fighter_data_flat = fighter_data.reshape(fighter_data.shape[0], -1).T
helicopter_data_flat = helicopter_data.reshape(helicopter_data.shape[0], -1).T
missile_data_flat = missile_data.reshape(missile_data.shape[0], -1).T
plane_data_flat = plane_data.reshape(plane_data.shape[0], -1).T
rocket_data_flat = rocket_data.reshape(rocket_data.shape[0], -1).T


# In[7]:

#one-hot encoding of the labels
labels = np.array([1,0,0,0,0,0]).reshape(6,1)
f_ix = 0
l_ix = drone_data_flat.shape[1]
for i in range(f_ix +1, l_ix):
    labels = np.concatenate((labels, np.array([1,0,0,0,0,0]).reshape(6,1)), axis = 1)
f_ix += drone_data_flat.shape[1]
l_ix += fighter_data_flat.shape[1]
for i in range(f_ix, l_ix):
    labels = np.concatenate((labels, np.array([0,1,0,0,0,0]).reshape(6,1)), axis = 1)
f_ix += fighter_data_flat.shape[1]
l_ix += helicopter_data_flat.shape[1]
for i in range(f_ix, l_ix):
    labels = np.concatenate((labels, np.array([0,0,1,0,0,0]).reshape(6,1)), axis = 1)
f_ix += helicopter_data_flat.shape[1]
l_ix += missile_data_flat.shape[1]
for i in range(f_ix, l_ix):
    labels = np.concatenate((labels, np.array([0,0,0,1,0,0]).reshape(6,1)), axis = 1)
f_ix += missile_data_flat.shape[1]
l_ix += plane_data_flat.shape[1]
for i in range(f_ix, l_ix):
    labels = np.concatenate((labels, np.array([0,0,0,0,1,0]).reshape(6,1)), axis = 1)
f_ix += plane_data_flat.shape[1]
l_ix += rocket_data_flat.shape[1]
for i in range(f_ix, l_ix):
    labels = np.concatenate((labels, np.array([0,0,0,0,0,1]).reshape(6,1)), axis = 1)
labels = labels.T
print("Shape of the one-hot encoded label matrix: " + str(labels.shape))


# In[ ]:

data = np.concatenate((data_without_label, labels.T), axis = 0)
print("Shape of the data matrix is:" + str(data.shape))


# In[8]:

def zero_pad(X, pad):
    return np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode = 'constant', constant_values = (0,0))


# In[9]:

def conv_single_step(a_slice_prev, W, b):
    # Applying one step of convolution to the sliced part of A with the filter W, and bias b.
    s = a_slice_prev * W
    Z = np.sum(s) + b
    return Z


# In[10]:

def conv_forward(A_prev, W, b, hparameters):
    #Forward propagation for a convolution function in a single layer.
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    n_H = int((n_H_prev + 2*pad - f)/2 + 1)
    n_W = int((n_W_prev + 2*pad - f)/2 + 1)
    
    #initialize output matrix Z
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            vert_start = h*stride
            vert_end = vert_start + f
            
            for w in range(n_H):
                horiz_start = w*stride
                horiz_end = horiz_start + f
                
                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                    #print(a_slice_prev.shape)
                    # Convolving the slice with the filter W and bias b
                    weights = W[:,:,:,c]
                    #print("W: " + str(weights.shape))
                    biases = b[:,:,:,c]
                    #print("b: " + str(biases.shape))
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                                        
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Saving information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache


# In[11]:

def pool_forward(A_prev, hparameters, mode = "max"):
    #The forward pass of the pooling layer 

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         
        for h in range(n_H):
            vert_start = h*stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w*stride
                horiz_end = horiz_start + f
                for c in range (n_C):
                    a_prev_slice = A_prev[i,vert_start:vert_end, horiz_start:horiz_end,:]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
                        
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache


# In[12]:

def conv_backward(dZ, cache):
    #Back-propagation for a convolution function in a single layer
    #retrieve info from cache
    (A_prev, W, b, hparameters) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]
  
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = np.zeros(A_prev.shape)                           
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):    
            for w in range(n_W):              
                for c in range(n_C):          
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
            
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


# In[13]:

def create_mask_from_window(x):
    return np.max(x) == x


# In[14]:

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz/ (n_H*n_W)
    a = np.ones(shape) * average
    return a


# In[76]:

def pool_backward(dA, cache, mode = "max"):
    #back-propagation of the pooling layer

    (A_prev, hparameters) = cache

    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):                 
            for w in range(n_W):               
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f
                    
                    # Compute the back-propagation in both modes.
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        #print(mask.shape)
                        #print(dA[i,h,:,:].shape)
                        #print(dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c].shape)
                        #print([vert_start, vert_end, horiz_start, horiz_end])
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask,dA[i,h,w,c])
                        
                    elif mode == "average":
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev


# In[16]:

def initialize_parameters_FCLayers(layer_dims):
    #initializing parameter for the fully connected layer
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1])) * 0.01
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# In[17]:

def sigmoid(x): 
    return 1 / (1 + np.exp(x*-1)), x

def relu(x):
    return x * (x > 0), x


# In[18]:

def linear_forward(A, W, b):
    #calculation of the weights * input + bias for one layer
    Z = np.matmul(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


# In[51]:

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# In[52]:

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    #Linear forward pass with ReLU for L-1 layers
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    
    #Sigmoid activation for the last layer 
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    #assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


# In[66]:

def compute_cost(AL, Y):
    #cross-entropy cost function
    
    m = AL.shape[0]
    costs = []
    #print(AL.shape, Y.shape)
    for i in range(m):
        costs.append( np.sum(Y.T[i]*np.log(AL[i]) + (1-Y.T[i])*np.log(1-AL[i]))*(-1/m) )

    cost = np.squeeze(sum(costs))
    assert(cost.shape == ())
    
    return cost


# In[72]:

def sigmoid_backward(dA, cache):
    Z = cache
    s, Z = sigmoid(Z)
    #print(dA.shape, s.shape)
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    
    return dZ
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    
    return dZ


# In[46]:

def linear_backward(dZ, cache):
    #back_propagation for linear portion of the layer
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.matmul(dZ, A_prev.T) /m
    db = np.sum(dZ, axis = 1, keepdims = True) /m
    dA_prev = np.matmul(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# In[47]:

def linear_activation_backward(dA, cache, activation):
    #Back-propagation for the Linear activation
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# In[48]:

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
   
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Loop from l=L-2 to l=0 for relu
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads


# In[49]:

def update_parameters(parameters, grads, learning_rate):
    
    
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        
    return parameters


# # Overview of our CNN model

# In[74]:

def CNNModel(X, Y, layers_dims, hparameters1, hparameters2, learning_rate = 0.0075, num_iterations = 3000, print_cost = True):
    #initializing parameters for the first convolution filter (f = 3, # of filters = 10)
    Wc1 = np.random.randn(hparameters1["f"], hparameters1["f"], X.shape[3],10)
    bc1 = np.zeros((1,1,1,10))
    #initializing parameters for the second convolution filter (f = 5, # of filters = 10)
    Wc2 = np.random.randn(hparameters2["f"], hparameters2["f"], Wc1.shape[3],10)
    bc2 = np.zeros((1,1,1,10))
    
    #number of fully connected layers
    L = len(layers_dims)
    costs = []
    fcl_parameters = initialize_parameters_FCLayers(layers_dims)
    """
    hparameters1 = {"stride": 1, "f": 3, "pad": 1}
    hparameters2 = {"stride": 2, "f": 5, "pad": 0}
    """
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: Conv1 -> MaxPool -> Conv2 -> MaxPool 
        # -> [Linear with ReLU activation]*(L-1) -> Linear with Sigmoid activation.
        
        out_conv1, cache1 = conv_forward(X, Wc1, bc1, hparameters1)
        out_pool1, cache2 = pool_forward(out_conv1, hparameters1, mode = "max")
        
        out_conv2, cache3 = conv_forward(out_pool1, Wc2, bc2, hparameters2)
        out_pool2, cache4 = pool_forward(out_conv2, hparameters2, mode = "max")
        input_forFCL = out_pool2.reshape(out_pool2.shape[0], -1).T
        AL, caches = L_model_forward(input_forFCL, fcl_parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
        
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters of the FCL
        fcl_parameters = update_parameters(fcl_parameters, grads, learning_rate)
        
        dInput_forFCL = grads["dA0"].reshape(out_pool2.shape)
        dPool2 = pool_backward(dInput_forFCL, cache4, mode = "max")
        dConv2, dWc2, dbc2 = conv_backward(dPool2, cache3)
        dPool1 = pool_backward(dConv2, cache2, mode = "max")
        dConv1, dWc1, dbc1 = conv_backward(dPool1, cache1)
        
        #update the convolution parameters
        Wc1 -= learning_rate * dWc1
        bc1 -= learning_rate * dbc1
        Wc2 -= learning_rate * dWc2
        bc2 -= learning_rate * dbc2
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return Wc1, bc1, Wc2, bc2, fcl_parameters



# In[40]:

def predictCNN(X, Y, Wc1, bc1, Wc2, bc2, hparameters1, hparameters2, fcl_parameters):
    out_conv1, cache1 = conv_forward(X, Wc1, bc1, hparameters1)
    out_pool1, cache2 = pool_forward(out_conv1, hparameters1, mode = "max")
        
    out_conv2, cache3 = conv_forward(out_pool1, Wc2, bc2, hparameters2)
    out_pool2, cache4 = pool_forward(out_conv2, hparameters2, mode = "max")
    input_forFCL = out_pool2.reshape(out_pool2.shape[0], -1).T
    AL, caches = L_model_forward(input_forFCL, fcl_parameters)
    m = AL.shape[1]
    true_pred = 0
    preds = []
    for i in range(AL.shape[1]):
        pred_ix = np.argmax(AL.T[i])
        if(labels.T[i][pred_ix] == 1):
            true_pred += 1
        preds.append(pred_ix)
    return true_pred / m, preds


# In[77]:

hparameters1 = {"stride": 1, "f": 3, "pad": 1}
hparameters2 = {"stride": 2, "f": 5, "pad": 0}
layers_dims = [250, 120, 64, 6]
Wc1f, bc1f, Wc2f, bc2f, fcl_parameters = CNNModel(data_without_label[0:200,:,:,:], labels[0:200,:], layers_dims, hparameters1, hparameters2, learning_rate = 0.0075, num_iterations = 3000, print_cost = True)


# In[ ]:

accuracy, predictions = predictCNN(data_without_label, labels, Wc1, bc1, Wc2, bc2, hparameters1, hparameters2, fcl_parameters)


# In[ ]:



