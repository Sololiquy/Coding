from flask import Flask, render_template, request
from keras_preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
import json
import numpy as np
import tensorflow as tf

app = Flask(__name__)

def load_char_resources():
    with open('resource/tokenizer_char.json') as t:
        data = json.load(t)
        tokenizer = tokenizer_from_json(data)

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='resource/model_char.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details, tokenizer

def load_word_resources():
    with open('resource/tokenizer_word.json') as t:
        data = json.load(t)
        tokenizer = tokenizer_from_json(data)

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='resource/model_word.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details, tokenizer


def predict_text(input_data, predict_type, loop_times, interpreter, input_details, output_details, tokenizer):
    for _ in range(loop_times):
        # Initialize data
        if predict_type == "word":
            token_list = tokenizer.texts_to_sequences([input_data])[0]
            token_list = pad_sequences([token_list], maxlen=15-1, padding='pre') #maxlen 14 because every word is split into 15

        else:
            to_characters = []
            for i in range(0, len(input_data)):
                to_characters.append(input_data[i])
            token_list = tokenizer.texts_to_sequences([to_characters])[0]
            token_list = pad_sequences([token_list], maxlen=50-1, padding='pre') #maxlen 49 because every char is split into 50

        # Do prediction
        token_list = np.array(token_list).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], token_list)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output_data, axis=-1).item()
        output_word = tokenizer.index_word[predicted]
        
        if predict_type == "word":
            input_data += " " + output_word
        else:
            input_data += output_word
        
    output_data = input_data

    return output_data
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    predict_type = request.form['radio_input']
    loop_times = int(request.form['loop_amount'])

    if predict_type == "word":
        interpreter, input_details, output_details, tokenizer = load_word_resources()
    else:
        interpreter, input_details, output_details, tokenizer = load_char_resources()

    output_data = predict_text(input_data, predict_type, loop_times, interpreter, input_details, output_details, tokenizer)

    return render_template('output.html', output=output_data)

if __name__ == '__main__':
    app.run()
