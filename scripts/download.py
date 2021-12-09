import gdown

urls = {
    "model-kfold-1.h5": "1REzAtMyw7YE1g80JUUaEv92i4RVr1Mx4",
    "model-kfold-2.h5":"1JSZGuTbogZsMIF0Q9jQ9vY7Jo16fNQjS",
    "model-kfold-3.h5":"14tLtyF1OL5bDpwLw0I15bQjG_kT2K8bH",
    "model-kfold-4.h5":"1q7Ju7vCKb0d2GdgDw2_5xlGuvWTz_GXa",
    "model-kfold-5.h5":"1AlFNzlr-XSmyu20ozsDrX-2D4LkV8-y7",
}

for name, id in urls.items():
    url = f"https://drive.google.com/uc?id={id}"
    output = f"weights/{name}"
    gdown.download(url, output, quiet=False)
    print(f"Loaded {name}")
