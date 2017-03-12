# Georgian OCR

## Requierments

    pip install tensorflow
    pip install keras
    pip install cairocffi
    pip install editdistance
    pip install nose2
    cp fonts/* /usr/share/fonts/truetype
    
## Train model

```
python -m src/trainer

```


## Generating test images

#### for single word
```
python -m src/image_generator ტექსტი \
        -w 256 \
        -s data/chunks \

```

#### for many words

apply words separated by space to generate multiple images 

```
python -m src/image_generator ერთი ორი სამი ოთხი \
        -w 256 \
        -s data/chunks \

```


## Recognition

#### from file

```
python -m src/recognizer \
        -W results/weights20.h5 \
        -i data/chunks/ერთი.jpg \
        -w 256 \
        -m results/model256.yaml
```


#### from folder

```
python -m src/recognizer \
        -W results/weights20.h5 \
        -i data/chunks \
        -w 256 \
        -m results/model256.yaml
```

#### output in english letters
```
python -m src/recognizer \
        -W results/weights20.h5 \
        -i data/chunks \
        -w 256 \
        -m results/model256.yaml \
        -e
```



