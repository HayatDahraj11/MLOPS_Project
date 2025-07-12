import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Preprocess a single text"""
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        
        return ' '.join(tokens)

    def process_dataset(self, df):
        """Process the entire dataset"""
        print("Processing text data...")
        df['processed_text'] = df['headline'].apply(self.preprocess_text)
        return df