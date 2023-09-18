import unicodedata
import re
import string
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize, TweetTokenizer, sent_tokenize
from nltk.corpus import stopwords
import nltk

class WordPrep:
    """
    Funciones de preprocesado
    """
    def __init__(self):
        pass
    # Vocabulario para eliminar apóstrofes
    APPO = {
        "aren't" : "are not", "can't" : "cannot", "couldn't" : "could not",
        "didn't" : "did not", "doesn't" : "does not", "don't" : "do not",
        "hadn't" : "had not", "hasn't" : "has not", "haven't" : "have not",
        "he'd" : "he would", "he'll" : "he will", "he's" : "he is",
        "i'd" : "I would", "i'd" : "I had", "i'll" : "I will", "i'm" : "I am",
        "isn't" : "is not", "it's" : "it is", "it'll":"it will", "i've" : "I have",
        "let's" : "let us", "mightn't" : "might not", "mustn't" : "must not",
        "shan't" : "shall not", "she'd" : "she would", "she'll" : "she will",
        "she's" : "she is", "shouldn't" : "should not", "that's" : "that is",
        "there's" : "there is", "they'd" : "they would", "they'll" : "they will",
        "they're" : "they are", "they've" : "they have", "we'd" : "we would",
        "we're" : "we are", "weren't" : "were not", "we've" : "we have",
        "what'll" : "what will", "what're" : "what are", "what's" : "what is",
        "what've" : "what have", "where's" : "where is", "who'd" : "who would",
        "who'll" : "who will", "who're" : "who are", "who's" : "who is",
        "who've" : "who have", "won't" : "will not", "wouldn't" : "would not",
        "you'd" : "you would", "you'll" : "you will", "you're" : "you are",
        "you've" : "you have", "'re": " are", "wasn't": "was not", "we'll":" will",
        "didn't": "did not", "tryin'":"trying"
    }
    # Funciones pequeñas separadas
    def remove_non_ascii(self,text):
        words = text.split()
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word) \
                                .encode('ascii', 'ignore') \
                                .decode('utf-8' ,'ignore')
            new_words.append(new_word)
        text = ' '.join(new_words)
        return text

    def remove_http_links(self,text):
        text = re.sub(r'http\S+', ' ', text)
        return text

    def remove_emails(self,text):
        text = re.sub(r'www\S+', ' ', text)
        return text

    def remove_punctuation(self,text):
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        return text

    def remove_one_char_words(self,text):
        words = text.split()
        words = [word for word in words if len(word.strip()) > 1]
        text = ' '.join(words)
        return text

    def remove_numbers(self,text):
        text = re.sub(r'\w*\d\w*', '', text)
        return text

    def lemmatize_with_postag(self,text, _tag_='n'):
        sentence = TextBlob(text)
        tag_dict = {'J': 'a', 'N': 'n', 'V': 'v', 'R': 'r'}
        words_and_tags = [(w, tag_dict.get(pos[0], _tag_)) for w, pos in sentence.tags]
        lemmatized_words = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        text = ' '.join(lemmatized_words)
        return text
    

    # Funcion general de preprocesado de texto
    def update_stopwords(self):
        nltk.download('stopwords')

    def corpus_text_preprocessing(self,text):
        lemmatizer = WordNetLemmatizer()
        tweetTokenizer = TweetTokenizer()
        
        nltk_eng_stopwords = stopwords.words("english")
       
        text = text.lower()
        text = self.remove_non_ascii(text)
        text = self.remove_emails(text)
        text = self.remove_http_links(text)
        
        # Expresiones regulares
        text = re.sub('\\n', '', text)
        text = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)

        text = self.remove_punctuation(text)
        text = self.remove_one_char_words(text) 
        text = self.remove_numbers(text)

        # Stopwords
        
        words = tweetTokenizer.tokenize(text)
        words = [self.APPO[word] if word in self.APPO else word for word in words]
        words = [w for w in words if not w in nltk_eng_stopwords]
        text = ' '.join(words)

        # Lemmatización
        text = self.lemmatize_with_postag(text, 'v')
        text = self.lemmatize_with_postag(text, 'n')

        # Más stopwords
        words = tweetTokenizer.tokenize(text)
        
        words = [w for w in words if not w in nltk_eng_stopwords]
        cleaned_text = ' '.join(words)

        # Eliminar puntuación
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        cleaned_text = cleaned_text.translate(table)
        cleaned_text = ' '.join([w for w in cleaned_text.split()])

        return cleaned_text



