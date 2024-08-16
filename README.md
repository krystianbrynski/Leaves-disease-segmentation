# Leaves disease segmentation project

The aim of the project is to perform binary segmentation, specifically to train a model that will segment the diseased parts of a leaf

I used Jupyter Notebook to analyze the data (photos) and also I chose the Unet model.

Data: https://www.kaggle.com/datasets/sovitrath/leaf-disease-segmentation-with-trainvalid-split

Model architecture: 
![Screenshot](https://github.com/krystianbrynski/Leaves-disease-segmentation/blob/main/photos)

## Description of files:
- **main.py -**
  Main file where function calls from other files are located.

- **dataset.py -**
  This file returns 3 datasets for train, test and valid data

- **dataloader.py -**
  This file returns 3 dataloaders for train, test and valid data

- **split_and_copy.py -**
   This file create new dir for train, test and valid data and also split and copy this data to new dir
- **model.py -**
   It contains the implemented U-Net network
- **earlystopping.py -** 
   It contains the implemented classes which I used in train.py for better model training

- **train_model.py** - 
  The file where model training takes place

- **test_model.py** - The file where model is testing

- **scores.txt** - It keeps model scores

## Metrics used to record results:

In folder scores you can follow scores.txt which contains the result of models:

- **Dice Score:** 2 * (number of common elements) / (number of elements in set A + number of elements in set B)
- **Accuracy:** Overall correctness of the model's predictions.
- **Precision:** Proportion of true positive predictions out of all positive predictions.
- **Recall:** Proportion of true positive predictions out of all actual positive instances.

## To improve the model's performance, I used

- I used a pre-trained model, which give better results than my implemented one
- I increased the training data size five times and applied augmentation to each image.
- I added an EarlyStopping class that saves the best model based on validation loss.
- I added a scheduler that monitored the validation loss and reduced the learning rate if it did not improve for 5 epochs.


## Scores
After these improvements, I was able to achieve the following results:
- Accuracy: 95.64%
- Dice score: 85.71%
- Precision: 89.33%
- Recall: 83.04%








