import numpy as np
from activation import ActivationFunction
import matplotlib.pyplot as plt

class SignActivation(ActivationFunction):
   """ 
         Sign activation: `f(x) = 1 if x > 0, 0 if x <= 0`
   """
   """
      using a simple boolean statement we created the forward to return 0 if false (below 0) and 1 if true (above 0)
   """
   def forward(self, x):
      """
         This is the output function.
         TODO: Define the correct return function, given input `x`
      """
      f = int(x>0) 
      return f
      
   """
      TODO: write explanation for why no gradient to start with
   """
   def gradient(self, x):
      """
         Function derivative.
         TODO: Define the correct return value (derivative), given input `x`
      """
      
      return None

class Perceptron:
   """ 
      Perceptron neuron model
      Parameters
      ----------
      n_inputs : int
         Number of inputs
      act_f : Subclass of `ActivationFunction`
         Activation function
   """
   """
      put act_f as the intialized version of the action function class
      put np.random with a variable amount of inputs based on class initialization
   """
   def __init__(self, n_inputs, act_f, w0):
      """
         Perceptron class initialization
         TODO: Write the code to initialize weights and save the given activation function
      """
      if not isinstance(act_f, type) or not issubclass(act_f, ActivationFunction):
         raise TypeError('act_f has to be a subclass of ActivationFunction (not a class instance).')
      # weights
      self.w = np.random.normal(0,0.5,n_inputs) #np.random.normal(mean, standard deviation, size)
      # self.w = np.append([w0], self.w)    # add in bias weight
      self.w0 = w0
      # print(f"{n_inputs} WHYYYYYYY {self.w}")
      # activation function
      self.f = act_f()

      if self.f is not None and not isinstance(self.f, ActivationFunction):
         raise TypeError("self.f should be a class instance.")

   def activation(self, x):
      """
         It computes the activation `a` given an input `x`
         TODO: Fill in the function to provide the correct output
         NB: Remember the bias
      """
      w_total = np.append([self.w0], self.w)
      a = 0
      index = 0
      for val in x:
         a = a + (val*w_total[index])
         index = index + 1
      # a = sum(x * y for x, y in zip(x, self.w))# 1/(1 + np.exp(-x))
      return a

   def output(self, a):
      """
         It computes the neuron output `y`, given the activation `a`
         TODO: Fill in the function to provide the correct output
      """
      y = self.f.forward(a)
      return y

   def predict(self, x):
      """
         It computes the neuron output `y`, given the input `x`
         TODO: Fill in the function to provide the correct output
      """
      x = np.append([1], x)     # adding x0
      a = self.activation(x)
      # print("activation results")
      # print(a)
      y = self.output(a)
      return y

   def gradient(self, a):
      """
         It computes the gradient of the activation function, given the activation `a`
      """
      return self.f.gradient(a)

if __name__ == '__main__':
   data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   xdata = data[:,:2]
   ydata = data[:,2]
   print(xdata)
   print(ydata)

   # 1.1 - results are as expected TODO: write more!
   actF = SignActivation()
   print(actF.forward(1))  # Returns 1
   print(actF.forward(0))  # Returns 0
   print(actF.forward(-1))  # Returns 0

   # act_f = ActivationFunction()
   per = Perceptron(2, SignActivation, 0.1)
   print(per.w)
   
   ## TODO Test your activation function 1.1
   print("test 1.1")
   a = SignActivation()
   print(a.forward(2))
   "print(a.forward(0))"

   ## TODO Test perceptron initialization 1.2 - 1.3
   print("test 1.2")
   x = [0,1]
   print(f"xdata: {xdata}")
   print(f"ydata: {ydata}")
   w0 = 0.5
   p = Perceptron(len(xdata[0,:]), SignActivation, w0)
   print(p.predict(xdata[0,:]) )

## TODO Learn the weights
r = 0.1 # learning rate
## calculate the error and update the weights
num_epochs = 70
misclassified_arr = []
for epoch in range(num_epochs):
   print(f"EPOCH: {epoch}")
   misclassified = 0
   for x, label in zip(xdata, ydata):     # xdata = features, ydata = labels
      target = label
      # print(f" TARGET {type(target)}")
      # print(f"x val : {x}")
      output = p.predict(x)
      print(f"OUTPUT: {output}")
      delta_w = (target - output)
      print(f"ERROR: {delta_w}")

      if np.absolute(delta_w) > 0.1:
         print("misclassified")
         misclassified = misclassified + 1
         # x = np.append([1], x)
         # print("AAAAAAAAAAAA")
         # print(r*(delta_w)*x)
         # print(p.w)
         print(f"p_w before: {p.w}")
         p.w = p.w + r*(delta_w)*x
         print(f"p_w after: {p.w}")

   misclassified_arr.append(misclassified)

print(f"RESULTS: {p.w}")
print(misclassified_arr)


## TODO plot points and linear decision boundary
# data points
xp = xdata[:,0]
# print(xp)
yp = xdata[:,1]
# print(yp)

# create line for weights
x = np.linspace(-0.5, 2, 100)
y = np.absolute(p.w[0]+p.w[1])*x+w0    # equation to plot a linear fit

plt.scatter(xp,yp, color=['red','red','red', 'blue', 'blue', 'blue'])
plt.plot(x,y,'k--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# h = .02
# x_min, x_max = xdata[:, 0].min() - 1, xdata[:, 0].max() + 1
# y_min, y_max = xdata[:, 1].min() - 1, xdata[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, m_max]x[y_min, y_max].
# fig, ax = plt.subplots()
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
# ax.axis('off')

# # Plot also the training points
# ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

# ax.set_title('Perceptron')