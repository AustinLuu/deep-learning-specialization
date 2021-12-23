import numpy as np
# import time

# a = np.array([1,2,3,4])
# print(a)

# a = np.random.rand(1000000)
# b = np.random.rand(1000000)
# tic = time.time()
# c = np.dot(a,b) 
# toc = time.time()
# print(c)
# print("Vectorized version: " + str(1000*(toc-tic))+"ms")

# c = 0
# tic = time.time()
# for i in range (1000000):
#     c+= a[i]*b[i]
# toc = time.time()
# print(c)
# print ("For loop:" +str(1000*(toc-tic))+"ms")

# a=np.random.randn(4,3)

# b = np.random.randn(3, 2)
# c = a*b
# print(c)
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    #(≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    # x_exp = ...

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    # x_sum = ...
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    # s = ...
    
    # YOUR CODE STARTS HERE
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    s=x_exp/x_sum
    # YOUR CODE ENDS HERE
    
    return s
# t_x = np.array([[9, 2, 5, 0, 0],
#                 [7, 5, 0, 0 ,0]])
# print("softmax(x) = " + str(softmax(t_x)))
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True) 
print(B.shape)