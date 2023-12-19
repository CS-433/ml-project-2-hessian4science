import torch
import torch.utils.data
from nltk.corpus import stopwords  # to remove the stopwords
import random
import re
import pandas as pd

import nltk  # to use word tokenize (split the sentence into words)
from nltk.corpus import wordnet, stopwords

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Set up stopwords and add additional terms
stop_words = set(stopwords.words('english'))
stop_words.add("rt")  # add 'rt' to remove retweet in dataset (noise)








########################################################################
# ================ Augmentation/Cleaning Functions  ================== #
########################################################################

# ========================== Synonym Replacement ========================== #
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(sentence, n):
    
    words = sentence.split()
    
    random_word_list  = [word for word in words if word not in stopwords.words('english')]
    random_replaces = min(len(words), min(len(random_word_list), n))
    random_word_list = [ [word,get_synonyms(word)] for word in random_word_list][:random_replaces]
    new_sentence = words
    for i in range(random_replaces):
        initial_word = random_word_list[i][0]
        synonum = random_word_list[i][1]
        if len(synonum) ==0:
          continue
        # obtain the position of the word to be replaced
        position = words.index(initial_word)
        # replace the word
        new_sentence[position] = synonum[0]
    new_sentence = ' '.join(new_sentence)
            
    return new_sentence


# ========================== Random Deletion ========================== #
def random_deletion(sentence, p, max_deletion_n):

    words = sentence.split()
    max_deletion_n = min(max_deletion_n, len(words)-1)
    
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return " ".join(words)


    new_sentence = []
    deletions = 0
    for word in words:
      if random.random() < p and deletions < max_deletion_n:
        deletions += 1
        continue
      
      new_sentence.append(word)    
    new_sentence = " ".join(new_sentence)
    
    return new_sentence


# ========================== Random Swap ========================== #
def swap_word(sentence):
    
    words = sentence.split()
    if len(words) <= 1:
      return sentence
    random_idx_1, random_idx_2 = random.sample(range(len(words)), 2)
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    new_sentence = words
    new_sentence = " ".join(new_sentence)
    

    return new_sentence

# ========================== Random Insertion ========================== #
def random_insertion(sentence, n):
    
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    new_sentence = ' '.join(new_words)
    return new_sentence

def add_word(new_words):
    
    synonyms = []
    try:
      random_synonym = random.choice(get_synonyms(random.choice(new_words)))
      new_words.insert(random.randint(0, len(new_words)), random_synonym)
    except:
      pass

# ========================== Choose and apply random augmentations ========================== #
def random_augmentation(tweet, largo, number_augmenteations=3):
  # List of augmentation functions
  augmentation_functions = [synonym_replacement, random_deletion, random_insertion, swap_word]

  # Select three unique augmentation functions randomly
  selected_functions = random.sample(augmentation_functions, number_augmenteations)

  # Apply the selected augmentation functions
  for func in selected_functions:
    
      if func in [synonym_replacement, random_insertion]:
          tweet = func(tweet, n=largo)
      elif func == random_deletion:
          tweet = func(tweet, p=0.35, max_deletion_n=largo)
      elif func == swap_word:
          for _ in range(largo):
              tweet = func(tweet)

  return tweet




# ========================== Apply augmentation ========================== #
def apply_augmentation(clean_tweets, labels):
    
  # Get the tweets with label 0
  tweets_with_label_zero = [clean_tweet for label, clean_tweet in zip(labels, clean_tweets) if label == 0]

  new_tweets = []
  for tweet in tweets_with_label_zero:
      intensity = len(tweet.split(" ")) // 3

      if intensity < 2:
          intensity = random.randint(2, 3)

      # RoundS of augmentation
      for _ in range(2):
          augmented_tweet = random_augmentation(tweet, intensity)
          new_tweets.append(augmented_tweet)


  # Update labels and extend the original dataset
  new_labels = [0] * len(new_tweets)
  clean_tweets.extend(new_tweets)
  labels.extend(new_labels)

  return clean_tweets, labels



# ========================== Data Cleaning ========================== #

# remove html entity:
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text


# change the user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "user", raw_text)

    return text

# remove urls
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)

    return text

# remove unnecessary symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')

    return text

# remove stopwords
def remove_stopwords(raw_text):
    tokenize = nltk.word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = " ".join(text)

    return text

## this function in to clean all the dataset by utilizing all the function above
def preprocess(datas):

    clean = []
    # change the @xxx into "user"
    clean = [change_user(text) for text in datas]
    # remove emojis (specifically unicode emojis)
    clean = [remove_entity(text) for text in clean]
    # remove urls
    clean = [remove_url(text) for text in clean]
    # remove trailing stuff
    clean = [remove_noise_symbols(text) for text in clean]
    # remove stopwords
    clean = [remove_stopwords(text) for text in clean]

    return clean




def apply_augmentation(dataframe):

    train_tweets = list(dataframe['tweet'])
    train_labels = list(dataframe['class'])


    print("Applying augmentation...")
    zero_pairs = [clean_tweet for label, clean_tweet in zip(train_labels, train_tweets) if label == 0]
    new_tweets = []
    for tweet in zero_pairs:
        largo = len(tweet.split(" ")) // 3
        if largo < 2:
            largo = random.randint(2, 3)

        # First round of augmentation
            
        for _ in range(2):
            augmented_tweet = random_augmentation(tweet, largo)
            new_tweets.append(augmented_tweet)

    # Update labels and extend the original dataset
    new_labels = [0] * len(new_tweets)
    train_tweets.extend(new_tweets)
    train_labels.extend(new_labels)

    return train_tweets, train_labels






# ========================== Load Dataset ========================== #

def process_dataset(path, seed):
    '''
    Load the dataset from a CSV file and apply preprocessing and augmentation if specified.
    :param path: Path to the CSV file.
    :param augmentation: Whether to apply augmentation.
    '''
    # data loading
    print("Loading dataset...")
    data = pd.read_csv(path)
    tweets = list(data['tweet'])
    labels = list(data['class'])

    #Data cleaning
    print("Cleaning dataset...")
    clean_tweets = preprocess(tweets)
    data['tweet'] = clean_tweets

    #Split data
    train_data = data[:int(len(data)*0.8)]
    val_data   = data [int(len(data)*0.8):int(len(data)*0.9)]
    test_data  = data[int(len(data)*0.9):]

    
    #Augmentation of training data
    train_tweets = list(train_data['tweet'])
    train_labels = list(train_data['class'])


    print("Applying augmentation...")
    zero_pairs = [clean_tweet for label, clean_tweet in zip(train_labels, train_tweets) if label == 0]
    new_tweets = []

    for tweet in zero_pairs:
        word_count = max(len(tweet.split(" ")) // 3, 2)

        for _ in range(2):
            augmented_tweet = random_augmentation(tweet, word_count)
            new_tweets.append(augmented_tweet)


    # Update labels and extend the original dataset
    new_labels = [0] * len(new_tweets)
    train_tweets.extend(new_tweets)
    train_labels.extend(new_labels)

    #Shuffle data
    print("spliting dataset...")
    train_data       = pd.DataFrame(list(zip(train_labels, train_tweets)), columns = ['class', 'tweet']).sample(frac=1, random_state=seed)

    #Downsampling for class 1
    semi       = train_data[train_data['class']==1]
    train_data = train_data[(train_data['class'] == 2) | (train_data['class'] == 0)]
    semi       = semi[:int(len(semi)*0.4)]

    train_data = pd.concat([train_data, semi]).sample(frac=1)

    #Create new CSV
    train_data.to_csv("train.csv", index=False)
    val_data.to_csv("val.csv", index=False)
    test_data.to_csv("test.csv", index=False)


    #Data imbalance
    series = train_data['class'].value_counts().sort_index() / len(train_data)
    train_count = torch.tensor(series).float()
    print("Everything is ready!")

    return train_count



