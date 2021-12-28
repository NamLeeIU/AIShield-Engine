from pathlib import Path


class Config:
    DATASET_PATH = Path("./data")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    ROOT_TRAIN_DIR = DATASET_PATH / "aicv115m_final_public_train"
    ROOT_EXTRA_TRAIN_DIR = DATASET_PATH / "aicv115m_extra_public_1235samples"
    ROOT_TEST_DIR = DATASET_PATH / "aicv115m_final_private_test"

    WEIGHT_PATH = Path("./weights")
    WEIGHT_PATH.mkdir(parents=True, exist_ok=True)

    SUBMISSION_PATH = Path("./submissions")
