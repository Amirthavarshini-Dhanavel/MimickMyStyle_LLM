from collections import Counter
from typing import List, Dict, Set
import numpy as np
from nltk import pos_tag, word_tokenize, sent_tokenize, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import nltk

class StyleEvaluator:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
    def calculate_vocabulary_similarity(self, original_text: str, generated_text: str) -> Dict[str, float]:
       
        original_words = set(word_tokenize(original_text.lower()))
        generated_words = set(word_tokenize(generated_text.lower()))
        
        jaccard = len(original_words.intersection(generated_words)) / len(original_words.union(generated_words))
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([original_text, generated_text])
        cosine_sim = 1 - cosine(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1])
        
        return {
            "jaccard_similarity": jaccard,
            "tfidf_cosine_similarity": cosine_sim
        }
    
    def calculate_pos_similarity(self, original_text: str, generated_text: str) -> Dict[str, float]:
        
        def get_pos_distribution(text: str) -> np.ndarray:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            pos_counts = Counter(tag for _, tag in pos_tags)
            total = sum(pos_counts.values())
            
            
            pos_list = ['NN', 'VB', 'JJ', 'RB', 'IN', 'DT', 'CC']
            distribution = np.array([pos_counts.get(tag, 0)/total for tag in pos_list])
            return distribution
        
        orig_dist = get_pos_distribution(original_text)
        gen_dist = get_pos_distribution(generated_text)
        
        cosine_sim = 1 - cosine(orig_dist, gen_dist)
        
        return {
            "pos_distribution_similarity": cosine_sim
        }
    
    def calculate_sentence_length_similarity(self, original_text: str, generated_text: str) -> Dict[str, float]:
        
        def get_sentence_lengths(text: str) -> List[int]:
            sentences = sent_tokenize(text)
            return [len(word_tokenize(sent)) for sent in sentences]
        
        orig_lengths = get_sentence_lengths(original_text)
        gen_lengths = get_sentence_lengths(generated_text)
        
        
        orig_dist = np.array(orig_lengths) / np.sum(orig_lengths)
        gen_dist = np.array(gen_lengths) / np.sum(gen_lengths)
        
        
        max_len = max(len(orig_dist), len(gen_dist))
        orig_dist = np.pad(orig_dist, (0, max_len - len(orig_dist)))
        gen_dist = np.pad(gen_dist, (0, max_len - len(gen_dist)))
        
        emd = wasserstein_distance(orig_dist, gen_dist)
        
        return {
            "sentence_length_emd": emd
        }
    
    def calculate_ngram_similarity(self, original_text: str, generated_text: str, n: int = 2) -> Dict[str, float]:
        
        def get_ngrams(text: str, n: int) -> Set[tuple]:
            tokens = word_tokenize(text.lower())
            return set(ngrams(tokens, n))
        
        orig_ngrams = get_ngrams(original_text, n)
        gen_ngrams = get_ngrams(generated_text, n)
        
        # N-gram overlap
        overlap = len(orig_ngrams.intersection(gen_ngrams)) / len(orig_ngrams.union(gen_ngrams))
        
        return {
            f"{n}gram_overlap": overlap
        }
    
    def evaluate_style_similarity(self, original_text: str, generated_text: str) -> Dict[str, float]:
        
        metrics = {}
        
        metrics.update(self.calculate_vocabulary_similarity(original_text, generated_text))
        metrics.update(self.calculate_pos_similarity(original_text, generated_text))
        metrics.update(self.calculate_sentence_length_similarity(original_text, generated_text))
        metrics.update(self.calculate_ngram_similarity(original_text, generated_text, n=2))
        metrics.update(self.calculate_ngram_similarity(original_text, generated_text, n=3))
        
        return metrics 