import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from transformers import (AutoModelForMaskedLM, AutoTokenizer, BertConfig,
                          BertTokenizer, TFBertModel)

os.chdir('Your_Path')

tags_all = [
    'DRUG ', 'DRUG_INGREDIENT ', 'DISEASE ', 'SYMPTOM ', 'SYNDROME ',
    'DISEASE_GROUP ', 'FOOD ', 'FOOD_GROUP ', 'PERSON_GROUP ', 'DRUG_GROUP ',
    'DRUG_DOSAGE ', 'DRUG_TASTE ', 'DRUG_EFFICACY '
]
num_tags = len(tags_all) + 2

max_length = 60
batch_size = 128
epochs = 30


def read_data():
    anns = []  # annotation
    text = []  # original text
    for i in range(1000):  # 1000 files in total
        with open('%s.ann' % i, encoding='utf-8') as f:
            anns.append(f.read())

        with open('%s.txt' % i, encoding='utf-8') as f:
            text.append(f.read())
    return anns, text


def tokenize_data(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    encoded = tokenizer.batch_encode_plus(
        data,
        add_special_tokens=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_tensors="tf",
    )

    return np.array(encoded["input_ids"], dtype="int32")


def label_signle_passage(i):
    # cleaning annotation
    anns_temp = anns[i].replace('\t', ' ')
    anns_temp = anns_temp.split('\n')
    anns_temp = np.array([x.split(' ') for x in anns_temp])

    # initialize tags
    tags = np.array(['none' for _ in range(len(X_all[i]))])

    # tagging all annotation
    for ann_temp in anns_temp:
        start = int(ann_temp[2])
        end = int(ann_temp[3])
        tags[start:end] = ann_temp[1]
    return list(tags)


anns, X_all = read_data()   # get annotations and original text
X_all = tokenize_data(X_all)


# apply to all 1000 files
y_all = list(map(lambda x: label_signle_passage(x), range(1000)))
y_all = list(map(lambda x: LabelEncoder().fit_transform(x), y_all))
y_all = tf.keras.preprocessing.sequence.pad_sequences(
    y_all, padding="post", maxlen=max_length)
y_all = list(map(lambda x: tf.keras.utils.to_categorical(
    x, num_classes=num_tags), y_all))

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42)


def bert():
    input_ids = tf.keras.layers.Input(
        shape=(max_length), dtype=tf.int32, name="input_ids")

    embedding = transformers.TFBertModel.from_pretrained("bert-base-chinese")
    embedding.trainable = False    # we dont want to train embedding

    sequence_output, pooled_output = embedding(input_ids)

    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    dropout = tf.keras.layers.Dropout(0.3)(bi_lstm)
    output = tf.keras.layers.Dense(num_tags, activation="softmax")(dropout)

    model = tf.keras.models.Model(
        inputs=[input_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )
    return model


model = bert()
model.summary()

history = model.fit(X_train,
                    np.array(y_train),
                    validation_data=[X_test, np.array(y_test)],
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1)

y_pre = model.predict(X_test)
