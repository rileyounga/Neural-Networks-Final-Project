import pandas
import numpy
import datasets
import transformers
import tensorflow as tf

# use the tokenizer from BERT for consistency with the pre-trained model
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

def bert_model(input_dim, output_dim, initial_learning_rate=0.005):
    """Creates a BERT model with a bidirectional LSTM layer and a dense layer
    with sigmoid activation"""

    # Specify the input shapes
    input_ids = tf.keras.layers.Input(shape=(input_dim,), dtype=tf.int64, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(input_dim,), dtype=tf.int64, name="attention_mask")
    token_type_ids = tf.keras.layers.Input(shape=(input_dim,), dtype=tf.int64, name="token_type_ids")

    # Load the pre-trained BERT model
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    bert_model.trainable = False
    bert_output = bert_model.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # Get the last hidden state from the BERT output to feed into the rest of the model
    sequence_output = bert_output.last_hidden_state
    # Create a bidirectional LSTM layer to process the BERT output
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(sequence_output)
    # Create a pooling layer to merge the LSTM outputs
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    # Dropout layer then dense layer with sigmoid activation to get the final outputs
    output = tf.keras.layers.Dense(output_dim, activation="sigmoid")(dropout)

    # Create the model and compile it
    model = tf.keras.models.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(initial_learning_rate),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])
    return model

def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")

def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):
    """Trains the model and saves it to a file. The model is saved after each
    epoch if it has the best F1 score on the dev data."""
    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # prepare the datasets for use by the model by using the bert tokenizer
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    
    hf_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

    # convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "token_type_ids"],
        label_cols="labels",
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "token_type_ids"],
        label_cols="labels",
        batch_size=16)
    
    #model = bert_model(64, len(labels))
    model = tf.keras.models.load_model(model_path)

    model.summary()    

    model.fit(
        train_dataset, 
        epochs=20,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_score",
                mode="max",
                patience=5,
                restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True)
        ])


def predict(model_path="model", input_path="dev.csv", zip=True):
    """Predicts labels for the test data and writes them to a CSV file"""
    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)
    
    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "token_type_ids"])

    # convert Huggingface datasets to Tensorflow datasets
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "token_type_ids"],
        batch_size=16)
    
    # predict labels for the test data
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # write the predictions to a CSV file
    df.iloc[:, 1:] = predictions

    if zip:
        # zip the CSV file to reduce its size
        df.to_csv("submission.zip", index=False,
                    compression=dict(method="zip", archive_name=f"submission.csv"))
    else:
        df.to_csv("dev_predictions.csv", index=False)

def test_predictions():
    """Tests the predictions by comparing them to the validation data
    and computing the F1 score for each label and overall."""
    predictions = pandas.read_csv("dev_predictions.csv", header=None)
    validation = pandas.read_csv("dev.csv", header=None)
    # compare f1 scores for each label and overall
    def f1_score(labels, predictions):
        """Computes the F1 score given a 2d lists of labels and predictions."""
        tp = numpy.sum(labels * predictions)
        fp = numpy.sum((1 - labels) * predictions)
        fn = numpy.sum(labels * (1 - predictions))
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
    
    pred_labels = []
    val_labels = []
    for i in range(1, len(predictions)):
        pred_labels.append(predictions.iloc[i, 1:].values.tolist())
        val_labels.append(validation.iloc[i, 1:].values.tolist())
    pred_labels = numpy.array(pred_labels, dtype=numpy.int32)
    val_labels = numpy.array(val_labels, dtype=numpy.int32)

    print("Total F1 Score:", f1_score(val_labels, pred_labels))
    print("Admiration F1 Score:", f1_score(val_labels[:, 0], pred_labels[:, 0]))
    print("Amusement F1 Score:", f1_score(val_labels[:, 1], pred_labels[:, 1]))
    print("Gratitude F1 Score:", f1_score(val_labels[:, 2], pred_labels[:, 2]))
    print("Love F1 Score:", f1_score(val_labels[:, 3], pred_labels[:, 3]))
    print("Pride F1 Score:", f1_score(val_labels[:, 4], pred_labels[:, 4]))
    print("Relief F1 Score:", f1_score(val_labels[:, 5], pred_labels[:, 5]))
    print("Remorse F1 Score:", f1_score(val_labels[:, 6], pred_labels[:, 6]))


if __name__ == "__main__":
    predict(input_path="test-in.csv")