import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from configs.config import Config
from modules.model import CNNModel
from modules.dataset import get_train_data


def prepare_training_data(Config):
    train_meta_df = pd.read_csv(
        str(Config.ROOT_TRAIN_DIR / "public_train_metadata.csv"))
    train_meta_df['path'] = train_meta_df['uuid'].apply(lambda x: str(
        Config.ROOT_TRAIN_DIR / f"public_train_audio_files/{x}.wav"))

    train_extra_df = pd.read_csv(
        str(Config.ROOT_EXTRA_TRAIN_DIR / "extra_public_train_1235samples.csv"))
    train_extra_df = train_extra_df.loc[train_extra_df["uuid"]
                                        != "c096b45b-fc28-4ba1-86ce-799cf31e1f48"]
    train_extra_df['path'] = train_extra_df['uuid'].apply(lambda x: str(
        Config.ROOT_EXTRA_TRAIN_DIR / f"new_1235_audio_files/{x}.wav"))

    return get_train_data(train_meta_df, train_extra_df)


def train():
    print("====== PREPARE TRAINING DATA ======")
    train_data = prepare_training_data(Config)
    print("====== TRAINING ======")
    y = train_data['label'].values
    train_data.drop(['label'], axis=1, inplace=True)
    train_data = train_data.iloc[:, :].values.reshape(
        train_data.shape[0], 13, -1)
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold = 1
    for train_index, val_index in skf.split(train_data, y):
        X_train = train_data[train_index]
        X_validation = train_data[val_index]
        y_train = y[train_index]
        y_validation = y[val_index]
        X_train = X_train[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        cnn = CNNModel(input_shape)
        model = cnn.define()
        optimizer = keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model_name = str(Config.WEIGHT_PATH/f"model-kfold-{fold}.h5")
        model.fit(X_train, y_train, validation_data=(
            X_validation, y_validation), batch_size=50, epochs=50)
        model.save_weights(model_name)
        fold += 1


if __name__ == "__main__":
    train()
