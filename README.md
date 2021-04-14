
# Winter of Code 3.0

## ML Bootcamp

The objective of this project is to get familiar with basic ML algorithms - Linear regression, Logistic regression, KNN and Neural Network.
The following libraries are used: NumPy, Pandas, and Matplotlib, to code these algorithms from scratch.

### Linear Regression

Linear regression attempts to model the relationship between two variables by fitting a linear equation to the observed data points.
It is used to find the line of best fit, by minimising the sum of mean square errors by recursively performing gradient descent.

The code consists of 3 parts: 

1. Visualisation - Used to visualise the features by converting them into a 28 x 28 matrix.
2. Linear regression - The input dataset is normalised, and gradient descent is run to find the best parameters. 
The cost function is plotted against number of iterations to check if gradient descent is running properly.
3. Testing - The testing set is normalised and the obtained parameters are used to predict labels and calculate its accuracy.


**Accuracy: 62.03%**<br/>
**Hyperparameters:**<br/>
Learning Rate: 0.001<br/>
Number of Epochs: 6000



### Logistic Regression

Logistic regression models the data using the sigmoid function. It is used as a classification algorithm. 
Here, it is used for multinomial classification (10 classes).

The code consists of 2 methods - log_reg() and multi_call().

log_reg() is used for binary classification - by one hot encoding Y values.
multi_call() is used for classifying the data into 10 classes by using the one vs. all strategy.

The cost function is plotted against number of iterations for each class to check if gradient descent is running properly.
The obtained parameters are then used to predict labels and calculate its accuracy.


**Accuracy: 97.739%**<br/>
**Hyperparameters:**<br/>
Learning Rate: 0.000009<br/>
Number of Epochs: 3500


### K Nearest Neighbours Algorithm

KNN is a model that classifies data points based on the points that are most similar to it.  
The algorithm works by finding the Euclidean distance between the mathematical values of these point.

The code consists of 3 parts: 

1) Data preprocessing - The training and testing datasets are read and stored in their respective arrays.
2) KNN() - This function calculates the K nearest neighbours of one training example.
3) call() - This function calculates the K nearest neighbours of all the training examples using a for loop, and calculates accuracy.

**Accuracy: 95.92%**<br/>
**Hyperparameters:**<br/>
K: 5


### Neural Network
 
A neural network functions when some input data is fed to it. 
This data is then processed via layers of perceptrons to produce the desired output.

The code consists of the following methods: 
- sigmoid_func(): Defines the sigmoid function.
- forward_prop(): Used for forward propagation.
- back_prop(): Used for calculating error using back propagation.
- update_params(): Used to update the weights and biases using the obtained error.
- grad_desc(): Used for iterating through all the training examples.
- accuracy(): Used for calculating accuracy.


**Accuracy: 10.10%**<br/>
**Hyperparameters:**<br/>
Number of hidden layers: 2<br/>
Number of nodes per layer: 16<br/>
Learning rate: 0.0009<br/>
Number of Epochs: 500

