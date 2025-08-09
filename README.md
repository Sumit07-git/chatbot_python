# Simple Neural Network Chatbot

A conversational AI chatbot built with TensorFlow and NLTK that can understand and respond to user queries based on predefined intents.

## Features

- Natural language processing using NLTK
- Neural network classification with TensorFlow
- Support for multiple languages (English, Hindi greetings)
- Intent-based conversation handling
- Extensible through JSON configuration

## Prerequisites

```bash
pip install tensorflow nltk numpy
```

## Files Structure

```
chatbot/
├── intents.json          # Training data and responses
├── first.py             # Model training script
├── chatbot.py           # Main chatbot interface
├── test_tf.py           # Testing script
├── chatbot_model.h5     # Trained model (generated)
├── words.pkl            # Vocabulary (generated)
├── classes.pkl          # Intent classes (generated)
├── chatbot_env/         # Virtual environment (ignored)
└── .gitignore           # Git ignore file
```

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd chatbot
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv chatbot_env
   source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow nltk numpy
   ```

4. **Train the model:**
   ```bash
   python first.py
   ```

5. **Run the chatbot:**
   ```bash
   python chatbot.py
   ```

3. **Start chatting:**
   ```
   You: hello
   Bot: Hello!
   
   You: tell me a joke
   Bot: I ate a clock yesterday, it was very time-consuming.
   ```

## Supported Intents

- **greeting**: hi, hello, hey
- **namaste**: namaste, pranam (Hindi greetings)
- **goodbye**: bye, see you later
- **thanks**: thank you, thanks
- **jokes**: tell me a joke, make me laugh
- **whatsup**: what's up, how you doing
- **aboutbot**: who are you, what are you
- **help**: help, can you help me

## Customization

### Adding New Intents

Edit `intents.json` to add new conversation patterns:

```json
{
    "tag": "weather",
    "patterns": [
        "how's the weather",
        "is it raining",
        "weather forecast"
    ],
    "responses": [
        "I can't check weather, but I hope it's nice!",
        "I don't have access to weather data."
    ]
}
```

After editing, retrain the model:
```bash
python train.py
```

### Model Parameters

Key training parameters in `first.py`:
- **Learning rate**: 0.01
- **Epochs**: 200
- **Batch size**: 5
- **Hidden layers**: 128, 64 neurons

## Troubleshooting

### Bot gives wrong responses
- Delete model files and retrain:
  ```bash
  rm chatbot_model.h5 words.pkl classes.pkl
  python first.py
  ```

### Low confidence predictions
- Add more training patterns to `intents.json`
- Increase training epochs
- Check for typos in patterns

### Import errors
- Install required packages:
  ```bash
  pip install tensorflow nltk numpy
  ```

## Technical Details

### Architecture
- Input layer: Bag-of-words representation
- Hidden layers: 128 → Dropout(0.5) → 64 → Dropout(0.5)
- Output layer: Softmax classification
- Optimizer: SGD with Nesterov momentum

### Text Processing
1. Tokenization using NLTK
2. Lemmatization for word normalization
3. Bag-of-words feature extraction
4. Neural network classification

### Training Process
1. Load intents from JSON
2. Create vocabulary and intent classes
3. Generate training data (input-output pairs)
4. Train neural network
5. Save model and preprocessed data

## License

MIT License
