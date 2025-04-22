CancerNet Readme
This zipfile contains six files (not including this one:
To run the files you will have the best results with the following dependencies:
python version 3.6.10
tensorflow version 1.12.0
keras version 2.0.8
pandas version 1.0.5
numpy version 1.18.5
sklearn version 0.23.1
pickle version 4.0

1. VAE.py: Python file that contains the custom class for creating CancerNet or other networks similar to it (VAE's with a classifier branching from the latent layer and a combined loss metric)
creates a VAE model of the depth and dimensions defined by user
    Inputs:
        encoder_dims: list of ints with the length of the encoder depth. First entry is size of input
        latent_dim: int, defines the size of the probablistic layer
        decoder_dims: list of ints with the length of the decoder depth
        classifier_dims: list of ints with the length of the classifier depth including the output class 	layer
	output_weights: a list of ints of length 2 where the first entry is the weight of the 	generative loss and the second is the weight of the classification loss
Call build function to build the vae.
2.  cancernet_hyperparams.pickle
	contains a dictionary with the hyperparameters neccessarry to create CancerNet using the VAE class contained in VAE.py
3. 34_class_CancerNet_weights.h5
	trained weights of CancerNet 34 class model
4. sample_x.pickle
	input data for CancerNet. Will produce the same results in the publication. open using the open_file method in vae.py to open or use pickle to open. use the keras method "predict" to get CancerNet output

5. sample_y.pickle
	labels for x_test.pickle. open using the open_file method in vae.py to open or use pickle to open. provided to allow users that run sample_x.pickle to get performance statistics.

6. sample_info.pickle
	metadata describing the samples included here. THe columns are listed and described below
'Index': TCGA barcode of each sample
'tissue': the source of the sample as the TCGA abbreviation
'sample type': sample type code found inthe barcode and described here https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
'label': the number label used in the training of cancernet

Example of use:
Below is a short tutorial on the use of VAE.py to build and load weights of CancerNet as well as how to create and train your own model. You must first load the VAE.py file into your python environment so that the class is available.
Now we create the hyperparameters. Here we load the cancernet_hyperparameters.pickle file using the fuction found in VAE.py called 'open_file' which takes the file address as the only argument
	cancernet_hp = open_file('cancernet_hyperparams.pickle')
this opens a dictionary with all of the hyperparameters in their appropriate form to pass to the VAE class

next we create an instance of the VAE class using these hyperparameters
	cancernet = VAE(cancernet_hp['encoder'], cancernet_hp['latent'],  cancernet_hp	['decoder'], cancernet_hp['latent'], cancernet_hp['lclassifier'], cancernet_hp	['loss_weights'])
then we build the model
	cancernet.build()
now we  load the CancerNet weights using the Keras method load_weights which is inherited by the model we have built already
	cancernet.full_model.load_weights('34_class_CancerNet_weights.h5')
now we can use the keras predict function to predict on cancer data that has the appropriate format (sample_x.pickle has been provided as an example)
	cancernet_output = cancernet.full_model.predict(sample_x)
this will give us an output that is a list of two numpy arrays. The first being the output from the generator the second being the classifier output. 

To create a model from scratch load the VAE.py file and create your own set of hyperparameters. Create a new vae class instance with these hyperparameters and use the build function as previously done. Instead of loading pretrained weights the model will now be trained. For this a train, validation and test set must be provided with labels (in one hot). The keras  .fit() function may be used to trian the model and will look like this:
vae.full_model.fit(x_train, [x_train,y_train] , batch_size =<batch size>, nb_epoch=<number of desired epochs>, validation_data=(x_val, [x_val,y_val]))
this may seem odd due to the list of outputs for the train and validation sets however they must be provided this way due to the model having two outputs. x_train is provided to compute the VAE loss while y_train is provided to calculate the classification loss, the same is true for validation sets.