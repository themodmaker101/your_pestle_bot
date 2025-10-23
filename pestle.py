"""
PESTLE MODEL - COMPLETE USAGE GUIDE WITH MODEL PERSISTENCE
==========================================================

This script demonstrates:
1. Training and saving the model
2. Loading a saved model
3. Making predictions with prompts
4. Batch predictions
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class PESTLEModel:
    """Production-ready PESTLE classifier with save/load functionality"""
    
    def __init__(self):
        self.model = None
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        self.best_model_name = None
        self.pestle_keywords = {
            'Political': ['government', 'election', 'policy', 'congress', 'senate', 
                         'president', 'legislation', 'vote', 'parliament', 'diplomacy'],
            'Economic': ['economy', 'market', 'stock', 'trade', 'gdp', 'inflation',
                        'interest rate', 'unemployment', 'fed', 'revenue', 'profit'],
            'Social': ['healthcare', 'education', 'social', 'community', 'demographic',
                      'population', 'immigration', 'diversity', 'equality', 'housing'],
            'Technological': ['technology', 'ai', 'artificial intelligence', 'innovation',
                            'digital', 'cyber', 'data', 'software', 'internet', 'automation'],
            'Legal': ['law', 'court', 'legal', 'lawsuit', 'judge', 'attorney',
                     'regulation', 'compliance', 'contract', 'patent', 'trial'],
            'Environmental': ['climate', 'environment', 'carbon', 'emission', 'pollution',
                            'renewable', 'energy', 'sustainability', 'green', 'conservation']
        }
        self.metadata = {}
    
    def train(self, csv_path='pestle_news_samples_6000_rows.csv'):
        """Train the model from scratch"""
        print("="*80)
        print("TRAINING PESTLE MODEL".center(80))
        print("="*80)
        
        # Load data
        print("\n1. Loading data...")
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} records")
        
        # Prepare text features
        print("\n2. Preparing features...")
        df['text_features'] = (
            df['Headline'].fillna('') + ' ' +
            df['Description'].fillna('') + ' ' +
            df['Topic_Tags'].fillna('').str.replace(',', ' ')
        ).str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        # Create keyword features
        keyword_features = []
        for _, row in df.iterrows():
            text = row['text_features']
            features = []
            for category, keywords in self.pestle_keywords.items():
                score = sum(1 for kw in keywords if kw in text) / len(keywords)
                features.append(score)
            keyword_features.append(features)
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X_tfidf = tfidf.fit_transform(df['text_features'])
        self.vectorizers['tfidf'] = tfidf
        print(f"   ✅ TF-IDF features: {X_tfidf.shape}")
        
        # Combine features
        X_combined = hstack([X_tfidf, csr_matrix(keyword_features)])
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['PESTLE_Category'])
        
        # Train-test split
        print("\n3. Training models...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=30, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, C=1.0, class_weight='balanced', random_state=42
            )
        }
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"   {name}: {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        self.model = best_model
        self.best_model_name = best_name
        
        # Store metadata
        self.metadata = {
            'model_type': best_name,
            'accuracy': best_score,
            'trained_date': datetime.now().isoformat(),
            'n_samples': len(df),
            'categories': self.label_encoder.classes_.tolist()
        }
        
        print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_score:.4f})")
        print("\n   Category Performance:")
        report = classification_report(y_test, self.model.predict(X_test),
                                      target_names=self.label_encoder.classes_,
                                      output_dict=True)
        for cat in self.label_encoder.classes_:
            f1 = report[cat]['f1-score']
            print(f"   - {cat}: F1={f1:.3f}")
        
        return True
    
    def save(self, model_name="pestle_model"):
        """Save model to disk"""
        print(f"\n{'='*80}")
        print(f"SAVING MODEL: {model_name}".center(80))
        print("="*80)
        
        model_dir = Path("pestle_models") / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved")
        
        # Save vectorizers
        with open(model_dir / "vectorizers.pkl", 'wb') as f:
            pickle.dump(self.vectorizers, f)
        print(f"✅ Vectorizers saved")
        
        # Save label encoder
        with open(model_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✅ Label encoder saved")
        
        # Save keywords
        with open(model_dir / "keywords.pkl", 'wb') as f:
            pickle.dump(self.pestle_keywords, f)
        print(f"✅ Keywords saved")
        
        # Save metadata
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✅ Metadata saved")
        
        print(f"\n📁 Model saved to: {model_dir.absolute()}")
        return str(model_dir)
    
    def load(self, model_name="pestle_model"):
        """Load model from disk"""
        print(f"\n{'='*80}")
        print(f"LOADING MODEL: {model_name}".center(80))
        print("="*80)
        
        model_dir = Path("pestle_models") / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load components
        with open(model_dir / "model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        print("✅ Model loaded")
        
        with open(model_dir / "vectorizers.pkl", 'rb') as f:
            self.vectorizers = pickle.load(f)
        print("✅ Vectorizers loaded")
        
        with open(model_dir / "label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✅ Label encoder loaded")
        
        with open(model_dir / "keywords.pkl", 'rb') as f:
            self.pestle_keywords = pickle.load(f)
        print("✅ Keywords loaded")
        
        with open(model_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        print("✅ Metadata loaded")
        
        print(f"\n📊 Model Info:")
        print(f"   Type: {self.metadata.get('model_type', 'Unknown')}")
        print(f"   Accuracy: {self.metadata.get('accuracy', 0):.4f}")
        print(f"   Trained: {self.metadata.get('trained_date', 'Unknown')}")
        print(f"   Categories: {', '.join(self.metadata.get('categories', []))}")
        
        return True
    
    def predict(self, text, show_probabilities=True):
        """Predict PESTLE category for text"""
        if self.model is None:
            raise ValueError("Model not loaded. Call train() or load() first.")
        
        # Preprocess text
        text_processed = text.lower()
        text_processed = ''.join(c for c in text_processed if c.isalnum() or c.isspace())
        
        # Extract TF-IDF features
        X_tfidf = self.vectorizers['tfidf'].transform([text_processed])
        
        # Extract keyword features
        keyword_features = []
        for category, keywords in self.pestle_keywords.items():
            score = sum(1 for kw in keywords if kw in text_processed) / len(keywords)
            keyword_features.append(score)
        
        # Combine features
        X_combined = hstack([X_tfidf, csr_matrix([keyword_features])])
        
        # Predict
        prediction = self.model.predict(X_combined)[0]
        predicted_category = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {'category': predicted_category}
        
        if show_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_combined)[0]
            prob_dict = {
                cat: float(prob)
                for cat, prob in zip(self.label_encoder.classes_, probabilities)
            }
            result['probabilities'] = prob_dict
            result['confidence'] = float(max(probabilities))
        
        return result
    
    def predict_batch(self, texts):
        """Predict categories for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text, show_probabilities=True))
        return results


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_1_train_and_save():
    """Example 1: Train a new model and save it"""
    print("\n" + "="*80)
    print("EXAMPLE 1: TRAIN AND SAVE MODEL".center(80))
    print("="*80)
    
    model = PESTLEModel()
    model.train('pestle_news_samples_6000_rows.csv')
    model.save("pestle_model")
    
    print("\n✅ Model trained and saved successfully!")


def example_2_load_and_predict():
    """Example 2: Load saved model and make predictions"""
    print("\n" + "="*80)
    print("EXAMPLE 2: LOAD MODEL AND PREDICT".center(80))
    print("="*80)
    
    # Load model
    model = PESTLEModel()
    model.load("pestle_model")
    
    # Test prompts
    test_prompts = [
        "Congress passes new healthcare reform bill",
        "Stock market reaches all-time high amid economic growth",
        "New AI technology revolutionizes manufacturing",
        "Supreme Court ruling on environmental regulations",
        "Rising sea levels threaten coastal communities",
        "Social media platforms face data privacy concerns"
    ]
    
    print("\n" + "="*80)
    print("PREDICTIONS".center(80))
    print("="*80)
    
    for i, prompt in enumerate(test_prompts, 1):
        result = model.predict(prompt)
        print(f"\n{i}. Text: {prompt}")
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Top 3 Probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in sorted_probs:
            print(f"      - {cat}: {prob:.2%}")


def example_3_interactive_mode():
    """Example 3: Interactive prediction mode"""
    print("\n" + "="*80)
    print("EXAMPLE 3: INTERACTIVE MODE".center(80))
    print("="*80)
    
    model = PESTLEModel()
    
    # Try to load existing model, otherwise train new one
    try:
        model.load("pestle_model")
    except FileNotFoundError:
        print("\n⚠️  No saved model found. Training new model...")
        model.train('pestle_news_samples_6000_rows.csv')
        model.save("pestle_model")
    
    print("\n" + "="*80)
    print("Enter text to classify (or 'quit' to exit)".center(80))
    print("="*80)
    
    while True:
        text = input("\n📝 Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text")
            continue
        
        result = model.predict(text)
        print(f"\n🎯 Predicted Category: {result['category']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")


def example_4_batch_prediction():
    """Example 4: Batch prediction with export"""
    print("\n" + "="*80)
    print("EXAMPLE 4: BATCH PREDICTION".center(80))
    print("="*80)
    
    model = PESTLEModel()
    model.load("pestle_model")
    
    # Sample batch data
    batch_texts = [
        "Federal Reserve raises interest rates",
        "Climate change summit reaches agreement",
        "Tech giant faces antitrust lawsuit",
        "New immigration policy announced",
        "Breakthrough in quantum computing",
        "Healthcare costs continue to rise"
    ]
    
    print(f"\nProcessing {len(batch_texts)} texts...")
    results = model.predict_batch(batch_texts)
    
    # Create DataFrame
    df_results = pd.DataFrame({
        'Text': batch_texts,
        'Category': [r['category'] for r in results],
        'Confidence': [r['confidence'] for r in results]
    })
    
    print("\n" + "="*80)
    print("BATCH RESULTS".center(80))
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Save to CSV
    output_file = "batch_predictions.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PESTLE MODEL - USAGE GUIDE".center(80))
    print("="*80)
    print("\nChoose an example to run:")
    print("1. Train and save a new model")
    print("2. Load model and make predictions")
    print("3. Interactive prediction mode")
    print("4. Batch prediction with export")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        example_1_train_and_save()
    elif choice == '2':
        example_2_load_and_predict()
    elif choice == '3':
        example_3_interactive_mode()
    elif choice == '4':
        example_4_batch_prediction()
    else:
        print("Invalid choice. Running example 1...")
        example_1_train_and_save()
