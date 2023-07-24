# Toxic-Comment-Classification-using-Deep-Learning

Deep Learning based Toxic comment classification model using LSTM. It classify the given comment into six types : toxic, severe_toxic,
insult, obscene, threat, identity_hate.Web app for this model was developed using flask and deployed on local machine.




## Dataset
Dowload the dataset from the following link:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

## Result
train_acccuracy : 99.21%   
test_accuracy : 98.41%
## Training
To train the model run the following command
```bash
  python model.py
```
It will also save text_ vectorization as tv_layer.pkl and model as cmodel.h5, which is required in deployment.
## Deployment

To deploy this project run the following command

```bash
  python app.py
```


