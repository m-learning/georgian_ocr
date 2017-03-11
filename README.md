# Georgian OCR

### Requierments

    pip install tensorflow
    pip install keras
    pip install cairocffi
    pip install editdistance
    pip install nose2
    cp fonts/* /usr/share/fonts/truetype



### Recognition command

`python -m geocr/recognizer --weights results/weights20.h5 --image data/test256_1.jpeg --width 256 --model results/model256.yaml`




