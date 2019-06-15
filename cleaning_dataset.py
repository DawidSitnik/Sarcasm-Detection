#file used to clear dataset and save new one to the file

#reading file
file = 'headlines.json'
data = pd.read_json(file, lines=True)

#cleansing
import string
from string import digits, punctuation

hl_cleansed = []
for hl in data['headline']:
    #     Remove punctuations
    clean = hl.translate(str.maketrans('', '', punctuation))
    #     Remove digits/numbers
    clean = clean.translate(str.maketrans('', '', digits))
    hl_cleansed.append(clean)

# Tokenization process
hl_tokens = []
for hl in hl_cleansed:
    hl_tokens.append(hl.split())

# Lemmatize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Init Lemmatizer
lemmatizer = WordNetLemmatizer()

hl_lemmatized = []
for tokens in hl_tokens:
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    hl_lemmatized.append(lemm)

data['headline'] = hl_lemmatized

#saving a new csv file
classification = pd.DataFrame(data, columns=['headline', 'is_sarcastic']).to_csv('./headlines_clean.csv')
