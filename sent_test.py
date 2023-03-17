from sent_functions import *


def main():
    """ Driver function."""

    # Initializing the constant variables
    VOCAB_SIZE = 88584
    MAXLEN = 250
    BATCH_SIZE = 64

    # Downloading and splitting the data into training-testing sets
    (training_data, training_labels), (testing_data, testing_labels) = imdb.load_data(num_words=VOCAB_SIZE)
    training_data = utils.pad_sequences(training_data, maxlen=MAXLEN)
    testing_data = utils.pad_sequences(testing_data, maxlen=MAXLEN)

    # Building the hyperparameter tuning object
    tuner = RandomSearch(
        build_model,
        objective='val_acc',
        max_trials=3,
        executions_per_trial=3,
        directory='my_dir',
        project_name='my_project')

    # Obtaining the best model to use with the Random Search
    tuner.search(training_data, training_labels, epochs=3, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]

    # Training and evaluating the model
    history = best_model.fit(training_data, training_labels, epochs=3, validation_split=0.2)
    results = best_model.evaluate(testing_data, testing_labels)

    # Geting word-to-index mapping for IMDB dataset
    word_index = imdb.get_word_index()

    # Predicting a positive review
    positive_review = "That movie was so awesome! I really loved it"
    print(predict(positive_review, best_model, word_index, MAXLEN))

    # Predicting a negative review
    negative_review = "That movie sucked. I hated it and wouldn't watch it again."
    print(predict(negative_review, best_model, word_index, MAXLEN))


if __name__ == '__main__':
    main()
