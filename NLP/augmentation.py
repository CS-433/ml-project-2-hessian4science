import random
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
  
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
    
    ############################################################################
    # TODO: Replace up to n random words in the sentence with their synonyms.  #
    #   You should                                                             #
    #   - (i)   replace random words with one of its synonyms, until           #
    #           the number of replacement gets to n or all the words           #
    #           have been replaced;                                            #
    #   - (ii)  NO stopwords should be replaced!                               #
    #   - (iii) return a new sentence after all the replacement.               #
    ############################################################################
    # Replace "..." with your code
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
            
        
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return new_sentence


# ========================== Random Deletion ========================== #
def random_deletion(sentence, p, max_deletion_n):

    words = sentence.split()
    max_deletion_n = min(max_deletion_n, len(words)-1)
    
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return " ".join(words)

    ############################################################################
    # TODO: Randomly delete words with probability p. You should               #
    # - (i)   iterate through all the words and determine whether each of them #
    #         should be deleted;                                               #
    # - (ii)  you can delete at most `max_deletion_n` words;                   #
    # - (iii) return the new sentence after deletion.                          #
    ############################################################################
    # Replace "..." with your code

    new_sentence = []
    deletions = 0
    for word in words:
      if random.random() < p and deletions < max_deletion_n:
        deletions += 1
        continue
      
      new_sentence.append(word)    
    new_sentence = " ".join(new_sentence)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    
    return new_sentence


# ========================== Random Swap ========================== #
def swap_word(sentence):
    
    words = sentence.split()
    if len(words) <= 1:
      return sentence
    ############################################################################
    # TODO: Randomly swap two words in the sentence. You should                #
    # - (i)   randomly get two indices;                                        #
    # - (ii)  swap two tokens in these positions.                              #
    ############################################################################
    # Replace "..." with your code
    random_idx_1, random_idx_2 = random.sample(range(len(words)), 2)
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    new_sentence = words
    new_sentence = " ".join(new_sentence)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

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
    ############################################################################
    # TODO: Randomly choose one synonym and insert it into the word list.      #
    # - (i)  Get a synonym word of one random word from the word list;         #
    # - (ii) Insert the selected synonym into a random place in the word list. #
    ############################################################################
    # Replace "..." with your code
    try:
      random_synonym = random.choice(get_synonyms(random.choice(new_words)))
      new_words.insert(random.randint(0, len(new_words)), random_synonym)
    except:
      pass
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
