# Keras-train-your-own-data
## Importance
This repository is for people who are new for keras but don't know how to train their dataset. If you are already familiar with keras, you do not need to watch this code. 

## Introduction
Data pipeline has always been the first difficult problem for newcomers facing their own model. In particular, there were not many convenient functions to use before, so it was very frustrating to train models at the beginning when we first start to learn how to train our model. But with Keras' evolution it is now very easy to train how to input data into a model. This repository will provide one of the data pipeline methods. Besides, I used VGG16 as the training model (there is a an off-the-shelf VGG model so that you can construct VGG by pre-defined function rather than code from scratch). Of course you can replace the model you want to try.

## Explanation
The method is simple. Please find the following two codes in the py file

    # TODO : Add your own full path training data folder (training_path)
    train_generator = train_datagen.flow_from_directory(...)
    
    AND
    
    # TODO : Add your own full path validation data folder (validation_path)
    validation_generator = val_datagen.flow_from_directory(...)
Please replace the first arguement to your path of train and validation data folders. In addition, please classify the photos in each folder (train an val folders) and put them in different folders. For example, if you have the dataset of cat and dog, for train and val folder construction need to be like this.

    ./train/cat/
    ./train/dog/
    ./val/cat/
    ./val/dog/
Also, if you want to test picture, fnd the code bellow and change the folder path.

    # TODO : Add your own full path testing data folder (testing_path)
    test_path = r'test folder'
Finally, have fun to play with your data and models
