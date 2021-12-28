#!/bin/bash

# Download train data
pip install gdown
mkdir -p data
cd data
gdown https://drive.google.com/uc?id=1Oq9UgA9cEGMNRGvF7oNKkFOg6udsDprl
gdown https://drive.google.com/uc?id=1UcRDOkq9cHfRrNOlFSaqJPNfCZPoFth_

# Unzip files
unzip -q aicv115m_final_public_train.zip -d ./
unzip -q aicv115m_extra_public_1235samples.zip -d ./

# Remove zip file
rm -f aicv115m_final_public_train.zip
rm -f aicv115m_extra_public_1235samples.zip



cd ..
