# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import Utils.MidiParser as MidiParser
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def draw_confusion_matrix(y_true, y_pred, hand, path, labels):
    # create confusion matrix
    all_labels = range(5) # 5 fingerings in total
    matrix = confusion_matrix(y_true, y_pred, labels = all_labels)
    # plot confusion matrix
    plt.figure(figsize = (10, 8))
    ax = sns.heatmap(matrix, annot = True, cmap = "Blues", fmt = "g", xticklabels = labels, yticklabels = labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {hand} hand")
    plt.savefig(path)
    plt.show()
    plt.close()

def df_to_X(df, window_size = 5):
    df_as_np = df.to_numpy()
    X = []

    for i in range(len(df_as_np) - window_size + 1):
        row = [[a] for a in df_as_np[i:i+window_size,:9]] # get n data (ex. 0 ~ 19)
        X.append(row) # input 9 features for 20 data
    
    return tf.convert_to_tensor(X)

fingerings_mapping = {-1.0: 0, -2.0: 1, -3.0: 2, -4.0: 3, -5.0: 4,
                        1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
time_step = 5

def exec():
    """Midi file preprocessing"""
    # list all midi files
    midiFiles = os.listdir("./MidiData./MidiFiles")
    for i, midiFile in enumerate(midiFiles):
        print(f'{i}. {midiFile}')

    # choose the midi file
    midiFileName = input("\nEnter the name of the above midi files (ex. \"midiFileName.mid\" or \"midiFileName\"): ")
    print("\n> Initializing...")
    outputPath = f"./MidiData./Preprocessed./" + (midiFileName[:-4] if midiFileName.endswith(".mid") else midiFileName)
    os.makedirs(outputPath, exist_ok = True)
    
    # preprocess features of the midi file
    MidiParser.main(midiFileName, outputPath)

    """features preprocessing"""
    # read processed midi features
    lp_path = f"./{outputPath}/left_features.csv"
    left_pred_df = pd.read_csv(lp_path)
    left_pred_df.to_csv(lp_path, index = False)
    left_pred_df = left_pred_df.drop("Duration", axis = 1)
    rp_path = f"./{outputPath}/right_features.csv"
    right_pred_df = pd.read_csv(rp_path)
    right_pred_df.to_csv(rp_path, index = False)
    right_pred_df = right_pred_df.drop("Duration", axis = 1)

    # impute missing values with -1 to fit LSTM
    dummy_row = [-1] * left_pred_df.shape[1]
    for i in range(time_step - 1):
        left_pred_df.loc[-1] = dummy_row
        left_pred_df.index = left_pred_df.index + 1
        left_pred_df = left_pred_df.sort_index()
        right_pred_df.loc[-1] = dummy_row
        right_pred_df.index = right_pred_df.index + 1
        right_pred_df = right_pred_df.sort_index()

    # to fit time steps
    left_pred_x = df_to_X(left_pred_df, time_step)
    right_pred_x = df_to_X(right_pred_df, time_step)

    # reshape (one-hot)
    left_pred_x = np.reshape(left_pred_x, (left_pred_x.shape[0], left_pred_x.shape[1], left_pred_x.shape[3]))
    right_pred_x = np.reshape(right_pred_x, (right_pred_x.shape[0], right_pred_x.shape[1], right_pred_x.shape[3]))

    """Predict single midi file"""
    # load models
    modelName = "RNN_model" # input('Enter the model name: ')
    left_model = tf.keras.models.load_model(f"./Models./{modelName}./left./model.h5")
    right_model = tf.keras.models.load_model(f"./Models./{modelName}./right./model.h5")
    
    # predict
    print("\n> Predicting...")
    left_pred_probs_Begin = left_model.predict(left_pred_x)
    right_pred_probs_Begin = right_model.predict(right_pred_x)

    # write predicted probabilities
    predicted_path = f"./midiData/Predicts/{midiFileName}./{modelName}"
    os.makedirs(predicted_path, exist_ok = True)
    left_pred_prob_Begin_df = pd.DataFrame(left_pred_probs_Begin, columns = ["L-1", "L-2", "L-3", "L-4", "L-5"])
    left_pred_prob_Begin_df.to_csv(f"{predicted_path}./left_pred_probs.csv", index = False)
    right_pred_prob_Begin_df = pd.DataFrame(right_pred_probs_Begin, columns = ["R1", "R2", "R3", "R4", "R5"])
    right_pred_prob_Begin_df.to_csv(f"{predicted_path}./right_pred_probs.csv", index = False)

    ## transform into actual fingerings and write
    left_pred_Begin_fingerings = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0][np.argmax(row)] for row in left_pred_probs_Begin])
    left_pred_Begin_df = pd.DataFrame({"Begin_fingering": left_pred_Begin_fingerings})
    left_pred_Begin_df.to_csv(f"{predicted_path}./left_pred.csv", index = False)
    right_pred_Begin_fingerings = np.array([[1.0, 2.0, 3.0, 4.0, 5.0][np.argmax(row)] for row in right_pred_probs_Begin])
    right_pred_Begin_df = pd.DataFrame({"Begin_fingering": right_pred_Begin_fingerings})
    right_pred_Begin_df.to_csv(f"{predicted_path}./right_pred.csv", index = False)
    print("> Finished.\n")

    """Estimate single midi file"""
    if input("If you want ot estimate the results, enter \"Y\": ") != "Y":
        print()
        return
    try:
        print("> Estimating...")
        # load human-labeled true fingerings of the piece

        left_true_Begin_df = pd.read_csv(f"./MidiData./TrueFingerings./{midiFileName.replace('.mid', '')}./left_true.csv")
        right_true_Begin_df = pd.read_csv(f"./MidiData./TrueFingerings./{midiFileName.replace('.mid', '')}./right_true.csv")

        # map fingerings
        left_true_Begin_mapped = left_true_Begin_df["Begin_fingering"].map(fingerings_mapping)
        left_pred_Begin_mapped = left_pred_Begin_df["Begin_fingering"].map(fingerings_mapping)
        right_true_Begin_mapped = right_true_Begin_df["Begin_fingering"].map(fingerings_mapping)
        right_pred_Begin_mapped = right_pred_Begin_df["Begin_fingering"].map(fingerings_mapping)

        # confusion matrix
        draw_confusion_matrix(left_true_Begin_mapped, left_pred_Begin_mapped, "left", f'{predicted_path}./left_confusion_matrix.png', [-1, -2, -3, -4, -5])
        draw_confusion_matrix(right_true_Begin_mapped, right_pred_Begin_mapped, "right", f'{predicted_path}./right_confusion_matrix.png', [1, 2, 3, 4, 5])

        # reset dataframe
        left_pred_Begin_df.reset_index(drop = True, inplace = True)
        left_true_Begin_df.reset_index(drop = True, inplace = True)
        right_pred_Begin_df.reset_index(drop = True, inplace = True)
        right_true_Begin_df.reset_index(drop = True, inplace = True)

        # calculate accuracy
        left_accuracy = np.mean(left_pred_Begin_df["Begin_fingering"] == left_true_Begin_df["Begin_fingering"])
        right_accuracy = np.mean(right_pred_Begin_df["Begin_fingering"] == right_true_Begin_df["Begin_fingering"])

        # transform mapped fingerings into one-hot
        left_true_Begin_oneHot = tf.keras.utils.to_categorical(left_true_Begin_mapped)
        right_true_Begin_oneHot = tf.keras.utils.to_categorical(right_true_Begin_mapped)

        # calculate loss
        left_loss = tf.keras.losses.categorical_crossentropy(left_true_Begin_oneHot, left_pred_prob_Begin_df).numpy().mean()
        right_loss = tf.keras.losses.categorical_crossentropy(right_true_Begin_oneHot, right_pred_prob_Begin_df).numpy().mean()

        # precision score
        left_precision = precision_score(left_true_Begin_mapped, left_pred_Begin_mapped, average = "weighted", zero_division = 0)
        right_precision = precision_score(right_true_Begin_mapped, right_pred_Begin_mapped, average = "weighted", zero_division = 0)
        
        # recall score
        left_recall = recall_score(left_true_Begin_mapped, left_pred_Begin_mapped, average = "weighted", zero_division = 0)
        right_recall = recall_score(right_true_Begin_mapped, right_pred_Begin_mapped, average = "weighted", zero_division = 0)
        
        # F1 score
        left_f1 = f1_score(left_true_Begin_mapped, left_pred_Begin_mapped, average = "weighted", zero_division = 0)
        right_f1 = f1_score(right_true_Begin_mapped, right_pred_Begin_mapped, average = "weighted", zero_division = 0)

        # record
        with open(f"{predicted_path}./predict_evaluates.txt", "w") as rr: # results-record
            rr.write("Left hand: \n")
            rr.write(f"\taccuracy: {round(left_accuracy, 3)}\n")
            rr.write(f"\tloss: {round(left_loss, 3)}\n")
            rr.write(f"\tPrecision score: {round(left_precision, 3)}\n")
            rr.write(f"\tRecall score: {round(left_recall, 3)}\n")
            rr.write(f"\tF1 score: {round(left_f1, 3)}\n")
            rr.write("Right hand: \n")
            rr.write(f"\taccuracy: {round(right_accuracy, 3)}\n")
            rr.write(f"\tloss: {round(right_loss, 3)}\n")
            rr.write(f"\tPrecision: {round(right_precision, 3)}\n")
            rr.write(f"\tRecall score: {round(right_recall, 3)}\n")
            rr.write(f"\tF1 score: {round(right_f1, 3)}\n")
        print("> Finished.\n")
    except FileNotFoundError as e:
        print(f"Human-labeled fingerings not found in \"./MidiData./TrueFingerings./{midiFileName.replace('.mid', '')}\"")
        print(e)
    print()

# main
print("Current model: RNN_model")
while True:
    exec()
    if input("If continue, enter \"Y\": ") != "Y":
        break