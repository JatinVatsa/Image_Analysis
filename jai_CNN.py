# Convolutional Neural Networks (CNN)

# Part 1 Data preprocessing
# since in this data like numbers r not present so we do not preprocess the data
# importing data  in this we have images so we have to preprcessing images  


# Part 1 - Building the CNN 
#importing Keras libraries & packages (it deals with images)

from keras.models import Sequential     #to inalize neural network using sequence of layers or graph
from keras.layers import Convolution2D  # for step 1 in CNN 
from keras.layers import MaxPooling2D   # for step 2 in CNN
from keras.layers import Flatten        # for step 3 in CNN
from keras.layers import Dense          # to add fully connected layers in classic neural n/w 

#initialsing the CNN
classifier= Sequential() # create object of sequential class

# step 1 - convolution 
classifier.add(Convolution2D(32,3,3, border_mode='same',input_shape=(64,64,3), activation='relu'))
  

# step 2 - pooling
# if odd rows(col, square matrix)then no of row in pooling layer is (row+1)/2 if even row then row/2
classifier.add(MaxPooling2D(pool_size=(2,2)))




# adding second covolution layer for incrasing accuracy 
# we does not include input shape in this becoz it understand automatically, but in 1st layer we include input shape becoz it does not know what is input image size
classifier.add(Convolution2D(32,3,3, border_mode='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))





# step 3 -flattening
classifier.add(Flatten()) 

# step 4_ full connection 
classifier.add(Dense(output_dim= 128 , activation='relu'))
classifier.add(Dense(output_dim= 1, activation='sigmoid')) # for o/p layer we use sigmoid fun for finding probability
# since we have binary outcome we use sigmoid, if we have more than 2 than we can use softmax function


# Compiling the CNN(ANN) i.e applying stochastic gradient descent method
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics= ['accuracy']) 
#if we have more than 2 output then we choose loss fun as categorical_crossentropy 


#Part 2 - fitting the CNN to the images 
# we have to oranginesed folder into 2 parts in machine laearning folder
# we can use trick (from keras documentation website) for image argumentation to neglect overfitting 
# use method 2 -->flow_from_directory
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set  = train_datagen.flow_from_directory( 'folder/training_set',
                                                    target_size=(64, 64), #same as input shape in step 1
                                                    batch_size=32,         
                                                    class_mode='binary')  

test_set = test_datagen.flow_from_directory( 'folder/test_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary') # binary outcome

classifier.fit_generator( training_set,
                         samples_per_epoch=8000,   
                         nb_epoch=25, 
                         validation_data=test_set,
                         nb_val_samples=2000) 


#for increasing aacuracy of test set we can increase one more convolution layer or add another fully connecyed layer ,best is adding convolution layer 
