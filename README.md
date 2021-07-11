# Image_Analysis
The dataset can be found here:  https://www.kaggle.com/tongpython/cat-and-dog

In this project we classify input image into dog or cat, using CNN.
A convolutional neural network (CNN) is a specific type of artificial neural network that uses perceptrons, a machine learning unit algorithm, for supervised learning, to analyze data. CNNs apply to image processing, natural language processing and other kinds of cognitive tasks.Like other kinds of artificial neural networks, a convolutional neural network has an input layer, an output layer and various hidden layers. Some of these layers are convolutional, using a mathematical model to pass on results to successive layers. This simulates some of the actions in the human visual cortex.


Below photo shows the working of convolutional neural netwok
![image](https://user-images.githubusercontent.com/85051683/125204936-53497d00-e29d-11eb-819b-707aa0cf71b2.png)









CNN contains maily four layers i.e Convolution Layer , Max Pooling , Flattening , Full Connections.


1st layer is Convoltional layer

![image](https://user-images.githubusercontent.com/85051683/125204983-9277ce00-e29d-11eb-9120-224c09f633d9.png)


Convolutional layers are the layers where filters are applied to the original image, or to other feature maps in a deep CNN. This is where most of the user-specified parameters are in the network. The most important parameters are the number of kernels and the size of the kernels.





2nd layer is Pooling Layer
![image](https://user-images.githubusercontent.com/85051683/125204999-ac191580-e29d-11eb-9fba-203bb92bf170.png)


Pooling layers are similar to convolutional layers, but they perform a specific function such as max pooling, which takes the maximum value in a certain filter region, or average pooling, which takes the average value in a filter region. These are typically used to reduce the dimensionality of the network.





3rd layer is used to flatten the pooled feature map so that it can be processed in Artificial Neural Network(ANN) for further processing.





Last layer is Fuuly connected layer
![image](https://user-images.githubusercontent.com/85051683/125205020-cd7a0180-e29d-11eb-9fc3-a95d16451afb.png)



Fully connected layers are placed before the classification output of a CNN and are used to flatten the results before classification. 
