# Context

This repository is a copy of the working repository in my undergraduate Neural Networks Course. This repository only shows the final submitted version of the best performing model. The task description, data collection and cleaning, and data loading starter code were all provided ahead of time. 

# The Final Model

The final model is an extended "bert-base-uncased" pre-trained model, extended with a bidirectional LSTM (to provide a trainable layer and capture sequence ordering) and a dense sigmoid layer (to output predictions for n different classification categories). The model was trained on scraped and labelled twitter data classifying text samples based on 7 non-exclusive emotion classifications. The final model performance on the unseen test data was an f1-score of 0.83 resulting in 12 place out of 50 within my class.
