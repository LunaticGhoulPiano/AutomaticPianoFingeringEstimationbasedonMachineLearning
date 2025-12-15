# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import Utils.MidiParser as MidiParser
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def draw_confusion_matrix(y_true, y_pred, path, labels):
    # create confusion matrix
    all_labels = range(10) # 10 fingerings in total
    matrix = confusion_matrix(y_true, y_pred, labels = all_labels)
    # plot confusion matrix
    plt.figure(figsize = (10, 8))
    ax = sns.heatmap(matrix, annot = True, cmap = 'Blues', fmt = 'g', xticklabels = labels, yticklabels = labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - merged hands')
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

fingerings_mapping = {-1.0: 4, -2.0: 3, -3.0: 2, -4.0: 1, -5.0: 0,
                      1.0: 5, 2.0: 6, 3.0: 7, 4.0: 8, 5.0: 9}
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
    generateMergedFingerings = True
    MidiParser.main(midiFileName, outputPath, generateMergedFingerings)

    """features preprocessing"""
    # read processed midi features
    mp_path = f"./{outputPath}/merged_features.csv"
    merged_pred_df = pd.read_csv(mp_path)
    merged_pred_df.to_csv(mp_path, index = False)
    merged_pred_df = merged_pred_df.drop("Duration", axis = 1)

    # impute missing values with -1 to fit LSTM
    dummy_row = [-1] * merged_pred_df.shape[1]
    for i in range(time_step - 1):
        merged_pred_df.loc[-1] = dummy_row
        merged_pred_df.index = merged_pred_df.index + 1
        merged_pred_df = merged_pred_df.sort_index()

    # to fit time steps
    merged_pred_x = df_to_X(merged_pred_df, time_step)

    # reshape (one-hot)
    merged_pred_x = np.reshape(merged_pred_x, (merged_pred_x.shape[0], merged_pred_x.shape[1], merged_pred_x.shape[3]))

    """Predict single midi file"""
    # load models
    modelName = "RNN_merged_model" # input('Enter the model name: ')
    merged_model = tf.keras.models.load_model(f"./Models./{modelName}./model.h5")
    
    # predict
    print("\n> Predicting...")
    merged_pred_probs_Begin = merged_model.predict(merged_pred_x)

    # write predicted probabilities
    predicted_path = f"./midiData/Predicts/{midiFileName}./{modelName}"
    os.makedirs(predicted_path, exist_ok = True)
    merged_pred_prob_Begin_df = pd.DataFrame(merged_pred_probs_Begin, columns = ["L-1", "L-2", "L-3", "L-4", "L-5", "R-1", "R-2", "R-3", "R-4", "R-5"])
    merged_pred_prob_Begin_df.to_csv(f"{predicted_path}./merged_pred_probs.csv", index = False)

    ## transform into actual fingerings and write
    merged_pred_Begin_fingerings = np.array([[-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0][np.argmax(row)] for row in merged_pred_probs_Begin])
    merged_pred_Begin_df = pd.DataFrame({"Begin_fingering": merged_pred_Begin_fingerings})
    merged_pred_Begin_df.to_csv(f"{predicted_path}./merged_pred.csv", index = False)
    print("> Finished.\n")

    """Estimate single midi file"""
    if input("If you want ot estimate the results, enter \"Y\": ") != "Y":
        print()
        return
    try:
        print("> Estimating...")
        # load human-labeled true fingerings of the piece
        merged_true_Begin_df = pd.read_csv(f"./MidiData./TrueFingerings./{midiFileName.replace('.mid', '')}./merged_true.csv")

        # map fingerings
        merged_true_Begin_mapped = merged_true_Begin_df["Begin_fingering"].map(fingerings_mapping)
        merged_pred_Begin_mapped = merged_pred_Begin_df["Begin_fingering"].map(fingerings_mapping)

        # confusion matrix
        draw_confusion_matrix(merged_true_Begin_mapped, merged_pred_Begin_mapped, f'{predicted_path}./merged_confusion_matrix.png', [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])

        # reset dataframe
        merged_pred_Begin_df.reset_index(drop = True, inplace = True)
        merged_true_Begin_df.reset_index(drop = True, inplace = True)

        # calculate accuracy
        merged_accuracy = np.mean(merged_pred_Begin_df["Begin_fingering"] == merged_true_Begin_df["Begin_fingering"])

        # transform mapped fingerings into one-hot
        merged_true_Begin_oneHot = tf.keras.utils.to_categorical(merged_true_Begin_mapped)

        # calculate loss
        merged_loss = tf.keras.losses.categorical_crossentropy(merged_true_Begin_oneHot, merged_pred_prob_Begin_df).numpy().mean()

        # precision score
        merged_precision = precision_score(merged_true_Begin_mapped, merged_pred_Begin_mapped, average = "weighted", zero_division = 0)
        
        # recall score
        merged_recall = recall_score(merged_true_Begin_mapped, merged_pred_Begin_mapped, average = "weighted", zero_division = 0)
        
        # F1 score
        merged_f1 = f1_score(merged_true_Begin_mapped, merged_pred_Begin_mapped, average = "weighted", zero_division = 0)

        # record
        with open(f"{predicted_path}./predict_evaluates.txt", "w") as rr: # results-record
            rr.write("Merged hands: \n")
            rr.write(f"\taccuracy: {round(merged_accuracy, 3)}\n")
            rr.write(f"\tloss: {round(merged_loss, 3)}\n")
            rr.write(f"\tPrecision score: {round(merged_precision, 3)}\n")
            rr.write(f"\tRecall score: {round(merged_recall, 3)}\n")
            rr.write(f"\tF1 score: {round(merged_f1, 3)}\n")
        print("> Finished.\n")
    except FileNotFoundError as e:
        print(f"Human-labeled fingerings not found in \"./MidiData./TrueFingerings./{midiFileName.replace('.mid', '')}\"")
        print(e)
    print()

# main
print("Current model: RNN_merged_model")
exec()