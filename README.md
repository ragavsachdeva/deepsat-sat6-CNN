# deepsat-sat6-CNN

A classifier for [DeepSat SAT-6](https://www.kaggle.com/crawford/deepsat-sat6) dataset hosted at Kaggle.

![alt text](/extra/sat_img.png)

### Prerequisites

* Python3

## Getting Started

* Clone/Download this repository and cd into the directory.
* Download the dataset from [Kaggle](https://www.kaggle.com/crawford/deepsat-sat6) and save the csv files in ```data/CSVs/``` directory.
* Install all the requirements using $```pip install -r requirements.txt```

## Data preprocessing

Step 1: Dump all the rows in the csv file(s) into actual images.

* Convert training data to images using $```python csv2images.py data/CSVs/X_train_sat6.csv -o data/training/training_images```.
* Convert testing data to images using $```python csv2images.py data/CSVs/X_test_sat6.csv -o data/testing/testing_images```.

Step 2: Generate annotations using labels' csv files.

* Generate annotations for training data using $```python annotations.py data/CSVs/y_train_sat6.csv -o data/training```.
* Generate annotations for testing data using $```python annotations.py data/CSVs/y_test_sat6.csv -o data/testing```.

## Training the model

* Use $```python main.py``` to train the model. The trained model will be saved as ```deepsat-cnn.pkl``` file.

* Use $```tensorboard --logdir runs``` to visualise the loss rate.

![alt text](/extra/loss_graph.png)

## Author

* **Ragav Sachdeva**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
