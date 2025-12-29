import numpy as np
import pandas as pd
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from model1 import ourmodel_1mer, ourmodel_4mer, build_combined_model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
my_seed = 42
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_prob = y_pred[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_labels, y_pred_labels).ravel()
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    mcc = matthews_corrcoef(y_true_labels, y_pred_labels)
    auc_score = roc_auc_score(y_true_labels, y_pred_prob)
    precision = precision_score(y_true_labels, y_pred_labels)
    recall = recall_score(y_true_labels, y_pred_labels)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return accuracy, mcc, auc_score, precision, recall, specificity, sensitivity

# Load data
data1 = np.load('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/100_4mer.npz')
x_train_4mer = data1['x_train']
x_test_4mer = data1['x_test']

data2 = np.load('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/100_1mer.npz')
x_train_1mer = data2['x_train']
x_test_1mer = data2['x_test']

# Load and convert labels to categorical
y_train = pd.read_csv('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/train/y_train.csv').to_numpy()
y_train = to_categorical(y_train, dtype='int')
y_test = pd.read_csv('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/test/y_test.csv').to_numpy()
y_test = to_categorical(y_test, dtype='int')

# Prepare for cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=my_seed)

# Double loop to iterate over batch sizes and epochs
accuracies = []
mccs = []
aucs = []
precisions = []
recalls = []
specificities = []
sensitivities = []

fold_no = 1
for train, test in kfold.split(x_train_1mer, np.argmax(y_train, axis=1)):
#    print(f'Training on fold {fold_no} with batch size {batch_size} and epochs {epochs}...')

     # Train the model
    model_1mer = ourmodel_1mer(input_shape=(100, 150))
    model_4mer = ourmodel_4mer(input_shape=(97, 150))
    combined_model = build_combined_model(model_1mer, model_4mer)
    combined_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = combined_model.fit(
           [x_train_1mer[train], x_train_4mer[train]], y_train[train],
           batch_size=32,
           epochs=16,
           verbose=1,
           validation_data=([x_train_1mer[test], x_train_4mer[test]], y_train[test]),
            )

            # Generate predictions for the validation set
    y_pred_val = combined_model.predict([x_train_1mer[test], x_train_4mer[test]])

            # Calculate metrics for the validation set
    accuracy, mcc, auc, precision, recall, specificity, sensitivity = calculate_metrics(y_train[test], y_pred_val)

            # Append metrics to the lists
    accuracies.append(accuracy)
    mccs.append(mcc)
    aucs.append(auc)
    precisions.append(precision)
    recalls.append(recall)
    specificities.append(specificity)
    sensitivities.append(sensitivity)
    print('AUC:', aucs)
    print('Accuracy:', accuracies)
    print('MCC: ', mccs )
    print('Specificity: ', specificities)
 #   print('MCC:', MCC)
    print('Sensitivity: ', sensitivities)
                                                       
    fold_no += 1

        # Print average metrics for current batch size and epochs
#print(f'Average metrics for batch size {batch_size} and epochs {epochs}:')
print(f'Average AUC: {np.mean(aucs):.4f}')
print(f'Average Accuracy: {np.mean(accuracies):.4f}')
print(f'Average MCC: {np.mean(mccs):.4f}')
print(f'Average Precision: {np.mean(precisions):.4f}')
print(f'Average Recall: {np.mean(recalls):.4f}')
print(f'Average Specificity: {np.mean(specificities):.4f}')
print(f'Average Sensitivity: {np.mean(sensitivities):.4f}')

# Final evaluation on the test set
final_scores = combined_model.evaluate([x_test_1mer, x_test_4mer], y_test, verbose=1)
print(f'Final Test Score: {combined_model.metrics_names[0]} of {final_scores[0]}; {combined_model.metrics_names[1]} of {final_scores[1] * 100}%')
