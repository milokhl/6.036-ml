import project1 as p1
import utils

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
# val_data = utils.load_data('reviews_val.tsv')
# test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
# val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
# test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

# dictionary = p1.bag_of_words(train_texts)
# dictionary_no_stopwords = p1.bag_of_words_removed_stopwords(train_texts)

# print("Length of Normal Dictionary:", len(dictionary), "\nLength of Dictionary Without Stopwords and Punc:", len(dictionary_no_stopwords))

# train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
# val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
# test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

# # get the feature vectors with stopwords removed, punctuation removed, and words counted with frequency
# train_bow_features_no_stopwords = p1.extract_bow_feature_vectors_with_frequency(train_texts, dictionary_no_stopwords)
# val_bow_features_no_stopwords = p1.extract_bow_feature_vectors_with_frequency(val_texts, dictionary_no_stopwords)
# test_bow_features_no_stopwords = p1.extract_bow_feature_vectors_with_frequency(test_texts, dictionary_no_stopwords)


# # get the final features
# train_final_features = p1.extract_final_features(train_texts, dictionary_no_stopwords)
# val_final_features   = p1.extract_final_features(val_texts, dictionary_no_stopwords)
# test_final_features  = p1.extract_final_features(test_texts, dictionary_no_stopwords)


# # get the feature vectors with NO STOPWORDS
# train_bow_features_no_stopwords = p1.extract_bow_feature_vectors(train_texts, dictionary_no_stopwords)
# val_bow_features_no_stopwords = p1.extract_bow_feature_vectors(val_texts, dictionary_no_stopwords)
# test_bow_features_no_stopwords = p1.extract_bow_feature_vectors(test_texts, dictionary_no_stopwords)


#
#-------------------------------------------------------------------------------
# Section 1.7
#-------------------------------------------------------------------------------
# toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

# T = 10
# L = 0.2

# thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
# thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
# thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)

# def plot_toy_results(algo_name, thetas):
#     utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

# plot_toy_results('Perceptron', thetas_perceptron)
# plot_toy_results('Average Perceptron', thetas_avg_perceptron)
# plot_toy_results('Pegasos', thetas_pegasos)
#-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.9.b
#-------------------------------------------------------------------------------
# T = 10
# L = 0.01

# print("ACCURACY OF EACH METHOD WITH STOPWORDS")
# # Training accuracy with stopwords
# pct_train_accuracy, pct_val_accuracy = \
#    p1.perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
# print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

# avg_pct_train_accuracy, avg_pct_val_accuracy = \
#    p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
# print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

# avg_peg_train_accuracy, avg_peg_val_accuracy = \
#    p1.pegasos_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))


# print("ACCURACY OF EACH METHOD WITH STOPWORDS REMOVED AND FEATURE VECTORS NORMALIZED")
# # Training accuracy with stopwords removed
# pct_train_accuracy, pct_val_accuracy = \
#    p1.perceptron_accuracy(train_bow_features_no_stopwords_normalized,val_bow_features_no_stopwords_normalized,train_labels,val_labels,T=T)
# print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
# print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

# avg_pct_train_accuracy, avg_pct_val_accuracy = \
#    p1.average_perceptron_accuracy(train_bow_features_no_stopwords_normalized,val_bow_features_no_stopwords_normalized,train_labels,val_labels,T=T)
# print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
# print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

# avg_peg_train_accuracy, avg_peg_val_accuracy = \
#    p1.pegasos_accuracy(train_bow_features_no_stopwords_normalized,val_bow_features_no_stopwords_normalized,train_labels,val_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

##-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.10
#-------------------------------------------------------------------------------
# data = (train_bow_features, train_labels, val_bow_features, val_labels)

# # values of T and lambda to try
# Ts = [1, 5, 10, 15, 25, 50, 100]
# Ls = [0.001, 0.005, 0.01, 0.1, 0.2, 0.5, 1.0]

# # print("Tuning perceptron...")
# # pct_tune_results = utils.tune_perceptron(Ts, *data)
# # print("Tuning average perceptron...")
# # avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)

# # train_accs_pct, val_accs_pct = pct_tune_results
# # train_accs_avgpct, val_accs_avgpct = avg_pct_tune_results

# # fix values for L and T while tuning Pegasos T and L, respective
# best_L = 0.01
# best_T = 50

# # print("Tuning pegasos T value...")
# # avg_peg_tune_results_T = utils.tune_pegasos_T(best_L, Ts, *data)
# print("Tuning pegasos lambda...")
# avg_peg_tune_results_L = utils.tune_pegasos_L(best_T, Ls, *data)

# # utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# # utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# # print("Plotting pegasos accuracy vs. T...")
# # utils.plot_tune_results('Pegasos', 'T', Ts, *avg_peg_tune_results_T)

# print("Plotting pegasos accuracy vs. L...")
# utils.plot_tune_results('Pegasos', 'L', Ls, *avg_peg_tune_results_L)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11a
#
# Call one of the accuracy functions that you wrote in part 2.9.a and report
# the hyperparameter and accuracy of your best classifier on the test data.
# The test data has been provided as test_bow_features and test_labels.
#-------------------------------------------------------------------------------

# test_bow_features
# test_labels

# print("TRAINING AVG. PERCEPTRON ON ORIGINAL FEATURES")
# best_T = 14
# print("Training average perceptron on original features...")
# avg_pct_theta, avg_pct_theta0 = p1.average_perceptron(train_bow_features, train_labels, best_T)

# # classify the test samples WITH STOPWORDS
# print("Predicting labels for test set...")
# predicted_labels = p1.classify(test_bow_features, avg_pct_theta, avg_pct_theta0)

# # compute accuracy
# print("Computing accuracy of avg. pct on test set...")
# test_accuracy = p1.accuracy(predicted_labels, test_labels)
# print("Test set accuracy:", test_accuracy)


# print("TRAINING AVG. PERCEPTRON ON FEATURES WITH STOPWORDS REMOVED")
# best_T = 14
# print("Training average perceptron with stopwords removed...")
# avg_pct_theta, avg_pct_theta0 = p1.average_perceptron(train_bow_features_no_stopwords, train_labels, best_T)

# # classify the test samples WITH STOPWORDS
# print("Predicting labels for test set...")
# predicted_labels = p1.classify(test_bow_features_no_stopwords, avg_pct_theta, avg_pct_theta0)

# # compute accuracy
# print("Computing accuracy of avg. pct on test set...")
# test_accuracy = p1.accuracy(predicted_labels, test_labels)
# print("Test set accuracy:", test_accuracy)


# print("TRAINING AVG. PERCEPTRON ON FEATURES WITH STOPWORDS REMOVED AND LENGTH ADDED")
# best_T = 14
# print("Training average perceptron with stopwords removed and length added...")
# avg_pct_theta, avg_pct_theta0 = p1.average_perceptron(train_final_features, train_labels, best_T)

# print("Predicting labels for test set...")
# predicted_labels = p1.classify(test_final_features, avg_pct_theta, avg_pct_theta0)

# # compute accuracy
# print("Computing accuracy of avg. pct on test set...")
# test_accuracy = p1.accuracy(predicted_labels, test_labels)
# print("Test set accuracy:", test_accuracy)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11b
#
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
# best_T = 14
# avg_pct_theta, avg_pct_theta0 = p1.average_perceptron(train_bow_features, train_labels, best_T)
# best_theta = avg_pct_theta
# wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
# sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
# print("Most Explanatory Word Features")
# print(sorted_word_features[:10])
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.12
#
# After constructing a final feature representation, use code similar to that in
# sections 2.9b and 2.10 to assess its performance on the validation set.
# You may use your best classifier from before as a baseline.
# When you are satisfied with your features, evaluate their accuracy on the test
# set using the same procedure as in section 2.11a.
#-------------------------------------------------------------------------------
# dictionary = p1.bag_of_words(train_texts)
#
# train_final_features = p1.extract_final_features(train_texts, dictionary)
# val_final_features   = p1.extract_final_features(val_texts, dictionary)
# test_final_features  = p1.extract_final_features(test_texts, dictionary)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.13
#
# Modify the code below to extract your best features from the submission data
# and then classify it using your most accurate classifier.
#-------------------------------------------------------------------------------
submit_texts = [sample['text'] for sample in utils.load_data('reviews_submit.tsv')]

# 1. Extract your preferred features from the train and submit data
dictionary_no_stopwords = p1.bag_of_words_removed_stopwords(submit_texts)
train_final_features = p1.extract_bow_feature_vectors_with_frequency(train_texts, dictionary_no_stopwords)
submit_final_features = p1.extract_bow_feature_vectors_with_frequency(submit_texts, dictionary_no_stopwords)

# 2. Train your most accurate classifier
final_thetas = p1.average_perceptron(train_final_features, train_labels, T=14)
print("Trained average perceptron")

# 3. Classify and write out the submit predictions.
submit_predictions = p1.classify(submit_final_features, *final_thetas)
print("Classified submit texts")
utils.write_predictions('reviews_submit.tsv', submit_predictions)
print("Wrote results to file")
#-------------------------------------------------------------------------------
