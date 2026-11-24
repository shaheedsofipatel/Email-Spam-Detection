"""
================================================================================
Email Spam Detection - Command Line Prediction Tool
================================================================================
Use this script to test any email or SMS message against the trained model.

Usage Examples:
    1. Interactive mode (prompts you to type/paste messages):
       python predict.py

    2. Direct argument mode:
       python predict.py "Congratulations! You won a $1,000 cash prize!"
================================================================================
"""

import sys
import os
import joblib

MODEL_PATH = 'model.pkl'

# Common trigger words often associated with spam
SPAM_TRIGGER_WORDS = {
    'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 'urgent!', 
    'selected', 'reward', 'call', 'txt', 'text', 'guaranteed', 'credit', 
    'bonus', 'lottery', 'congratulations', 'congrats', 'apply', 'offer', 
    '100%', 'risk-free', 'buy', 'order', 'deal', 'discount', 'expire', 
    'password', 'verify', 'account', 'suspended', 'click', 'link'
}

def load_trained_model():
    """Loads the serialized model pipeline."""
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model file '{MODEL_PATH}' not found!")
        print("👉 Please train the model first by running: python train.py\n")
        sys.exit(1)
        
    return joblib.load(MODEL_PATH)

def analyze_message(model, text):
    """Predicts spam/ham and analyzes message keywords."""
    clean_text = text.strip()
    if not clean_text:
        return None
        
    # Predict label & probabilities
    prediction = model.predict([clean_text])[0]
    probabilities = model.predict_proba([clean_text])[0]
    
    # 0 = Ham, 1 = Spam
    is_spam = (prediction == 1)
    spam_prob = probabilities[1] * 100
    ham_prob = probabilities[0] * 100
    
    # Identify spam indicator keywords present in the text
    words_in_text = set(clean_text.lower().split())
    found_triggers = [w for w in words_in_text if any(trig in w for trig in SPAM_TRIGGER_WORDS)]
    
    return {
        'is_spam': is_spam,
        'label': '🚨 SPAM' if is_spam else '✅ HAM (LEGITIMATE)',
        'confidence': spam_prob if is_spam else ham_prob,
        'spam_probability': spam_prob,
        'ham_probability': ham_prob,
        'trigger_words': found_triggers
    }

def print_result(text, result):
    """Displays formatted prediction results."""
    print("\n" + "=" * 60)
    print(f"📧 Message: \"{text}\"")
    print("-" * 60)
    print(f"Prediction:     {result['label']}")
    print(f"Confidence:     {result['confidence']:.2f}%")
    print(f"Probabilities:  Spam: {result['spam_probability']:.1f}% | Ham: {result['ham_probability']:.1f}%")
    
    if result['trigger_words']:
        print(f"Keywords Found: {', '.join(result['trigger_words'])}")
    else:
        print("Keywords Found: None of the common spam keywords detected.")
    print("=" * 60 + "\n")

def main():
    print("============================================================")
    print("          📩 EMAIL SPAM DETECTOR - PREDICTOR 📩")
    print("============================================================")
    
    model = load_trained_model()
    
    # Check if text was passed as command-line argument
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result = analyze_message(model, input_text)
        if result:
            print_result(input_text, result)
        return

    # Interactive console loop
    print("Type or paste an email message below to test.")
    print("Type 'exit' or 'quit' to close the program.\n")
    
    while True:
        try:
            user_input = input("Enter email text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit', 'q'):
                print("👋 Exiting. Have a great day!")
                break
                
            result = analyze_message(model, user_input)
            if result:
                print_result(user_input, result)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting. Goodbye!")
            break

if __name__ == '__main__':
    main()
