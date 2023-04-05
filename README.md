# How to code a Transformer autoencoder model for time series forecasting in PyTorch
## PyTorch implementation of Transformer model, refered to the implementation of the paper: "Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case"

This is the repo of the Transformer autoencoder model for time series forecasting

The sandbox.py file shows how to use the Transformer to make a training prediction on the data from the .csv file in "/data".

The inference_sandbox.py file contains the function that takes care of inference, and the inference_example.py file shows a pseudo-ish code example of how to use the function during model validation and testing. 

care about the learning rate to be smaller than 1e-5 if use the transformer decoder, otherwise may lead to overfitting.

## train
python sandbox.py

## test
python inference_sandbox.py