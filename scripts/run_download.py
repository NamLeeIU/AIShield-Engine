import gdown 
url = 'https://drive.google.com/drive/folders/1Q-3FqIV8ilJnFPM4RXAPNfX-ZugX8u5M?usp=sharing'
gdown.download_folder(url, quiet=True, no_cookies=True)

