import torch
from datasets import Dataset
import pandas as pd
import ast
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
import textstat
from collections import Counter
import docx
import PyPDF2
import io
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
    def extract_text_from_file(self, file_content, filename):
        
        if filename.endswith('.txt'):
            return file_content.decode('utf-8')
        elif filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_content))
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        elif filename.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(io.BytesIO(file_content))
            return ' '.join([page.extract_text() for page in pdf.pages])
        else:
            raise ValueError("Unsupported file format")

    def analyze_writing_style(self, text):
        

        def count_syllables(word):
            vowels = 'aeiou'
            word = word.lower()
            count = 0
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count += 1
            return count

        sentences = sent_tokenize(text)
        words = text.split()
        pos_tags = pos_tag(words)

        avg_sentence_length = len(words) / len(sentences)
        complex_sentences = sum(1 for s in sentences if len(s.split()) > 20)
        complex_ratio = complex_sentences / len(sentences)
        complex_sentence_ratio = complex_sentences/ len(sentences)
        total_syllables = sum(count_syllables(word) for word in words)
        flesch_kincaid = 0.39 * (len(words)/ len(sentences)) + 11.8 * (total_syllables / len(words)) - 15.59
      
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_tokens = len(pos_tags)
       
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(30)]
        
        
        summary = (
            f"Follow this Writing Style Summary while writing 1. Vocabulary: - {complex_ratio:.2f} - commonly use these adjectives, verbs, nouns, or adverbs in the sentences: {', '.join(common_words)}"
            f"2. Sentence Structure:- Follow the sentence length:- Average length: {avg_sentence_length:.2f} words "
            f"- Complex sentence ratio:{complex_ratio:.2f},Follow the Complex sentence ratio:{complex_sentence_ratio:.2f} "
            f"3. Parts of Speech:"
            f"- Nouns: {pos_counts.get('NN', 0)} ({pos_counts.get('NN', 0)/total_tokens*100:.2f}%) "
            f"- Verbs: {pos_counts.get('VB', 0)} ({pos_counts.get('VB', 0)/total_tokens*100:.2f}%) "
            f"- Adjectives: {pos_counts.get('JJ', 0)} ({pos_counts.get('JJ', 0)/total_tokens*100:.2f}%) "
            f"- Adverbs: {pos_counts.get('RB', 0)} ({pos_counts.get('RB', 0)/total_tokens*100:.2f}%), "
            f"4. Readability:  follow this Readability:- Flesch-Kincaid Grade Level: {flesch_kincaid:.2f}, 5. Follow this Style Characteristics: - {'Uses complex sentences frequently' if complex_sentence_ratio > 0.3 else 'Prefers simpler sentence structures'}- {'Rich and diverse vocabulary' if complex_ratio > 0.2 else 'More focused vocabulary'} - {'Formal tone' if flesch_kincaid > 10 else 'Casual tone'}"
            f"apart from that Follow the writing style,capture context and the nuanced flow from this text:{text}"      
        )
        return summary

    def create_csv_from_text(self, text, output_path):
       
       
        sentences = nltk.sent_tokenize(text)       
        
        if len(sentences) % 2 != 0:
            sentences = sentences[:-1]        
        
        conversations = []
        for i in range(0, len(sentences), 2):
            conv = {
                "conversations": [
                    {"role": "system", "content": "Mimick the writing style"},
                    {"role": "user", "content": f"Follow the writing style and predict the next sentence for the given line: {sentences[i]}"},
                    {"role": "assistant", "content": sentences[i+1]}
                ]
            }
            conversations.append(conv)        
        
        df = pd.DataFrame(conversations)
        df.to_csv(output_path, index=False)

    def prepare_dataset(self, input_file_path, output_dir):
        
        try:
            with open(input_file_path, 'rb') as file:
                file_content = file.read()
                filename = input_file_path.split('/')[-1]
                
                
                extracted_text = self.extract_text_from_file(file_content, filename)
                if len(extracted_text.split()) < 1000:
                    raise ValueError("Text should have at least 1000 words")                
                
                style_summary = self.analyze_writing_style(extracted_text)                
                
                csv_path = f"{output_dir}/output.csv"
                self.create_csv_from_text(extracted_text, csv_path)                
                
                csv_data = pd.read_csv(csv_path)
                csv_data['conversations'] = csv_data['conversations'].apply(ast.literal_eval)
                dataset = Dataset.from_pandas(csv_data)                
               
                sentences = sent_tokenize(extracted_text)
                train_sentences, eval_sentences = train_test_split(
                    sentences, 
                    test_size=0.2, 
                    random_state=42
                )

                extracted_text_train = " ".join(train_sentences)

                with open(f"{output_dir}/eval_set.txt", "w") as f:
                    f.write(" ".join(eval_sentences))
                
                dataset.save_to_disk(f"{output_dir}/preprocessed_data")
                
                with open(f"{output_dir}/style_summary.txt", 'w') as f:
                    f.write(style_summary)
                
                return style_summary
                
        except Exception as e:
            raise Exception(f"Error in data preprocessing: {str(e)}")