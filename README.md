# Automatic Piano Fingering Estimation based on Machine Learning

- This project use the simple machine learning techniques to verify the feasibility on estimating piano fingering by supervised learning model with MIDI (Musical Instrument Digital Interface) data.
- [Demo video on YouTube](https://youtu.be/b91wM3cI2VE)

## Data preprocessing
- [PIG Dataset](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/) from [Statistical Learning and Estimation of Piano Fingering](https://arxiv.org/abs/1904.10237)
- Note that every single MIDI file in this PIG dataset is **a part of piano piece**, not a piece with complete movements.

We define a ```note event``` is a bounch of information on a single piano note.
In the following tables, each table refers to a single note event.
The original data formats (using 001-1_fingering.txt as example):

| Item | Explaination | Example |
| - | - | - |
| Note ID | ID of each note, every piece start from 0 | 7 |
| Onset time | Start time of the note in second | 1.25032 |
| Offset time | End time of the note in second | 1.54397 |
| Pitch | Pitch in SPN (Scientific Pitch Notation) | G4 |
| Onset velocity | Start MIDI velocity of the note | 64 |
| Offset velocity | End MIDI velocity of the note | 80 |
| Channel | left hand: 1, right nahd: 0 | 0 |
| Finger number | left hand: -1 ~ -5, right hand: +1 ~ +5, us ```_``` to seperate when encountered fingering subtitution on the same note | 4_1 |

The improved data formats:

| Item | Type | Explaination | Example |
| - | - | - | - |
| Note ID | Feature | ID of each note, every piece start from 0 | 7.0 |
| Onset time | Feature | Start time of the note in second | 1.25032 |
| Offset time | Feature | End time of the note in second | 1.54397 |
| Pitch | Feature | Pitch in MIDI | 67.0 |
| Key Position x | Feature | the x-axis after transforming the note by its pitch | 4.0 |
| Key Position y | Feature | the y-axis after transforming the note by its pitch | 0.0 |
| Onset velocity | Feature | Start MIDI velocity of the note | 64 |
| Offset velocity | Feature | End MIDI velocity of the note | 80 |
| Channel | Feature | left hand: 1, right nahd: 0 | 0 |
| Begin fingering | Label | The initial fingering, left hand: -1 ~ -5, right hand: +1 ~ +5 | 4.0 |

Note that we ignore the fingering subtitution and only use the first fingering due to its low occurrence.

Then We randomly shuffled the 309 files, split them into training, validation, and test sets with an 8:8:1 ratio, and concatenated the files in each subset into ordered note sequences.

Finally we use ```-1``` to pad the missing value and use one-hot encoding to format the fingering numbers.

## Model Design
- We chose to apply a Bi-LSTM model because the notes in a piece exhibit sequential dependencies.
- There are two types of model: seperated hands and merged hands. The former one output 5 labels (for a single hand), and the latter output 10 labels (for both hands).
- The reason we chose the "seperated" and "merged" design is the 3 types of hand symmetry mentiond in [Statistical Learning and Estimation of Piano Fingering](https://arxiv.org/abs/1904.10237).
- Activation function using softmax, Loss function using categorical crossentropy.

## Experiments
- Both seperated and merged models processed three experiments:
    1. Deciding the basic models for the latter two experiments, with hidden layers that stacked by different number of BiLSTM layer.
    2. Adding a single-head attention-layer before and after the last BiLSTM layer of models.
    3. Simply changed the single-head into multi-head attention-layer in second experiment.
- And we got three model with the best performance:
    - Seperated models:
        - Left-hand model: 4 BiLSTM layers, named as ```LSModel```
        - Rightt-hand model: 5 BiLSTM layers, named as ```RSModel```
    - Merged models: 3 BiLSTM layers + 1 Mul-Head Attention layer + 1 BiLSTM layer, named as ```MulAtt-MSModel```
- The activation function and loss function in output layer are ```softmax``` and ```categorical crossentropy```.
- Experiment environment: Windows 10/11, Jupyter Notebool + Python 3.11.7 + Keras + cuDNN.

## Evaluation on the models with PIG dataset
- We used the perviously split test set to evaluate ```LSModel```, ```RSModel```, and ```MulAtt-MSModel```.

| Nrom | ```LSModel``` | ```RSModel``` | ```MulAtt-MSModel``` |
| - | - | - | - |
| Accuracy | 0.624 | 0.626 | **0.638** |
| Loss | **1.024** | 1.057 | 1.037 |
| F1 score | 0.615 | 0.623 | **0.635** |

## Evaluation on the models with self-composed piano pieces
- To find out the generalization ability of the models, I composed three piano piece with **full phrases**, differ from the PIG dataset by comparison.
- I use MuseScore 3 to compose and exported as MIDI file, then converted to the format of imporved PIG dataset (second table in [Data Processing](#data-preprocessing)).
- These three pieces had different number of note events with ostinato.

| Piece | Note Events of Left Hand | Note Events of Right Hand | Number of Note Events in Total |
| - | - | - | - |
| Piece A | 11 | 30 | 41 |
| Piece B | 94 | 42 | 136 |
| Piece C | 557 | 569 | 1126 |

- Performances in three pieces:
    1. Piece A
    
    | Norm | **LSModel** | RSModel | MulAtt-MSModel |
    | - | - | - | - |
    | Accuracy | **0.364** | 0.200 | 0.171 |
    | Loss | **2.069** | 3.513 | 5.190 |
    | F1 score | **0.379** | 0.075 | 0.102 |

    2. Piece B

    | Norm | **LSModel** | RSModel | MulAtt-MSModel |
    | - | - | - | - |
    | Accuracy | **0.617** | 0.214 | 0.243 |
    | Loss | **1.069** | 3.680 | 2.860 |
    | F1 score | **0.580** | 0.167 | 0.277 |

    3. Piece C

    | Norm | **LSModel** | RSModel | MulAtt-MSModel |
    | - | - | - | - |
    | Accuracy | **0.499** | 0.466 | 0.289 |
    | Loss | **1.225** | 1.526 | 2.782 |
    | F1 score | 0.486 | **0.499** | 0.253 |

- Conclusion
    - The probability of correctly predicting reasonable piano fingerings increases as the number of note events grows.
    - When fingering prediction errors occur, the number of errors for a single finger decreases from inner to outer fingers; non-dominant fingers (e.g., the ring finger) tend to have lower misprediction rates.
    - In cases of incorrect predictions, fingerings that are closer to the ground-truth finger are more likely to be acceptable alternatives.
    - We established piano fingering annotation models for both single-hand and two-hand scenarios, and identified correlations between musical pieces and fingering patterns. Among all factors, the number of note events had the most significant impact on performance when evaluating complete musical pieces.
    - Due to the incompleteness and inaccuracy of the dataset, the model performance was limited. Applying other model architectures would introduce additional challenges, such as missing data and inconsistent formats (e.g., irregular temporal units).
