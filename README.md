# Chatbot

A simple AI chatbot using TensorFlow and NLTK that learns from conversation patterns and responds to user messages.

## Features

- Natural language understanding
- Intent-based responses
- Support for English and Hindi greetings
- Neural network powered classification
- Easy to customize and extend

## Requirements

- Python 3.11
- TensorFlow 2.x
- NLTK
- NumPy

## Setup

1. Clone or download this project
2. Install requirements:
```bash
pip install tensorflow nltk numpy
```

3. Train the model:
```bash
python first.py
```

4. Run the chatbot:
```bash
python chatbot.py
```

## Project Structure

- `intents.json` - Training data with patterns and responses
- `first.py` - Neural network training script
- `chatbot.py` - Main chatbot interface
- `test_tf.py` - Testing utilities
- `chatbot_model.h5` - Trained neural network model
- `words.pkl` - Processed vocabulary data
- `classes.pkl` - Intent classification data

## How It Works

1. **Training**: The bot learns from patterns in `intents.json`
2. **Processing**: User input is tokenized and lemmatized
3. **Classification**: Neural network predicts the intent
4. **Response**: Bot selects appropriate response for the intent

## Chat Examples

```
You: hello
Bot: Hello! Good to see you again!

You: tell me a joke
Bot: I ate a clock yesterday, it was very time-consuming.

You: namaste
Bot: Namaste! 🙏 How are you today?

You: who are you
Bot: I'm a friendly chatbot built to answer your questions!
```

## Supported Intents

- **Greetings**: hi, hello, hey
- **Farewells**: bye, goodbye, see you later
- **Thanks**: thank you, thanks
- **Jokes**: tell me a joke, make me laugh
- **Small talk**: what's up, how are you
- **About bot**: who are you, what can you do
- **Help requests**: help, can you help me
- **Hindi greetings**: namaste, pranam

## Customization

### Adding New Intents

Edit `intents.json` and add new conversation patterns:

```json
{
    "tag": "weather",
    "patterns": [
        "how's the weather",
        "is it raining",
        "weather today"
    ],
    "responses": [
        "I can't check weather, but I hope it's nice!",
        "Try checking a weather app for current conditions."
    ]
}
```

After editing, retrain the model:
```bash
python first.py
```

### Model Configuration

Key settings in `first.py`:
- Learning rate: 0.01
- Training epochs: 200
- Hidden layers: 128 and 64 neurons
- Dropout rate: 0.5 for regularization

## Troubleshooting

**Bot gives wrong answers:**
```bash
# Delete old model files and retrain
rm chatbot_model.h5 words.pkl classes.pkl
python first.py
```

**Low confidence responses:**
- Add more training patterns to intents.json
- Make sure patterns are varied and realistic

**Import errors:**
```bash
# Make sure all packages are installed
pip install --upgrade tensorflow nltk numpy
```

## Technical Details

- **Architecture**: Sequential neural network with dropout layers
- **Text Processing**: NLTK tokenization and lemmatization
- **Classification**: Bag-of-words with softmax output
- **Training**: Categorical crossentropy loss with SGD optimizer

## License

MIT License - free to use and modify.
