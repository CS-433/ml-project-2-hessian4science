import random
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
  
from nltk.corpus import wordnet, stopwords

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