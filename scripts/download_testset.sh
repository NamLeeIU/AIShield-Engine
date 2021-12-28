#!/bin/bash

pip install gdown
mkdir -p data
cd data
# Download private test data
gdown https://drive.google.com/uc?id=1KJmH12giVJl8mIP-sTP2RK7Y5fKABFx_

# Unzip files
unzip -q aicv115m_final_private_test.zip -d ./

# Remove zip file
rm -f aicv115m_final_private_test.zip



cd ..
