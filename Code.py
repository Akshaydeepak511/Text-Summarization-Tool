import nltk
import heapq
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def summarize_article(text, num_sentences=3):
    nltk.download("punkt")
    nltk.download("stopwords")
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Tokenize words and remove stopwords
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Calculate word frequencies
    word_frequencies = {}
    for word in words:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    # Normalize frequencies
    max_frequency = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency
    
    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]
    
    # Select top sentences
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    
    return summary

# Example usage
if __name__ == "__main__":
    article_text = """Natural language processing (NLP) is a sub-field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages in a way that is both meaningful and useful. Various techniques are used in NLP, including machine learning algorithms and linguistic rules, to process and analyze large amounts of natural language data. Applications of NLP include language translation, sentiment analysis, chatbots, and more. NLP has advanced significantly in recent years, with deep learning models achieving state-of-the-art performance in many tasks."""
    
    summary = summarize_article(article_text, num_sentences=2)
    print("Summary:", summary)
