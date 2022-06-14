## **Descriptions**

Install the requirements \
`$ pip install --user -r requirements.txt` 

Open the folder which contains all the codes.
`$ cd codes` 

If you want to retrain the model, you can download the datasets from the links given inside the train-sets folder.

Firstly, execute the python code below to extract the dataset into pieces. This is done due to the limited use of resources in the NVIDIA TX2. This code is built to be able to train in the TX2 device. \
`$ python splitdatasets.py`

If you want to normally train the model, you can use the below command \
`$ python train.py`

Using the command below to do the lightweight training. \
`$ python lightweighttrain.py`

You don't need to adjust the new saved model name in the `model.py` file. The code would detect it automatically. 

Run the app \
`$ python app.py`