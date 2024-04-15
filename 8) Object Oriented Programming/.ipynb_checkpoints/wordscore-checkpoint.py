
def word_dataset(rack):
    '''Reads all possible words from text file (sowpods.txt) in same folder, and reduces it to a 
    set of words that are <= length of the provided rack.''' 
    with open("sowpods.txt","r") as infile:
        raw_input = infile.readlines()
        data = set([datum.strip('\n') for datum in raw_input]) #converting the data into a data set for faster look-up

    #Reduction of the data by removing all values with length longer than scrabble rack
    reduced_dataset = [word for word in data if len(word) <= len(rack)]
    return (reduced_dataset)

def words_possible(rack, wildcard_count, word_dataset): 
    '''Function returns all possible words for a given rack with wildcards. 
    Iterating through all the words in a dataset, and confirming if each rack letter exists in the scrabble rack.
    If a letter is not in the rack, any wildcards availabe are used until depleted. All wildcard letters are stored in return.''' 
    words = [] #list that will hold all possible words, and the letters covered by wildcards in that word
    for word in word_dataset:
        letter_list = rack.copy() #temp variable to reset rack values for each iteration
        wildcards = wildcard_count #temp variable to reset wildcard_count value for each iteration
        word_in = True #boolean for whether a word works. default is true unless proven false. 
        wild_letters = [] #temp holder for letters used by wildcard
        for letter in word:
            if letter in letter_list: #confirming a letter match and removing it from available letters for future iteration
                letter_list.remove(letter)
            elif wildcards > 0: #using and depleting wildcards available, storing letter used by wildcard
                wildcards -= 1
                wild_letters.append(letter)
            else: #setting word match as false and breaking iteration for given word
                word_in = False
                break
        if word_in: #for all word matches, appending nested list with the word and wildcard letters
            words.append([word, ''.join(wild_letters)])
    return words   

def score_word(words_to_score):
    '''This fuction recieves a nested list of words and wildcard letters,
    and returns what that translates into scrabble scores.
    Format of argument: words_to_score = [[word, wildcard letters],]'''

    #dictionary provided for letters and their scrabble values
    scores = {"a": 1, "c": 3, "b": 3, "e": 1, "d": 2, "g": 2,
            "f": 4, "i": 1, "h": 4, "k": 5, "j": 8, "m": 3,
            "l": 1, "o": 1, "n": 1, "q": 10, "p": 3, "s": 1,
            "r": 1, "u": 1, "t": 1, "w": 4, "v": 4, "y": 4,
            "x": 8, "z": 10}

    #identify the letters from each word that are scored when considering the use of wildcards. 
    word_scores = [] #new list that will index the scores for each word.
    for i in range(len(words_to_score)):
        word_scores.insert(i,0) #adding new index to list and setting score to 0
        for letter in words_to_score[i][0]: #checking each letter in the solution words to see if a wildcard was used. 
            if letter in words_to_score[i][1]: #if the letter was a wildcard, remove it from wildcard letters, so it isn't reused.
                words_to_score[i][1] = words_to_score[i][1].replace(letter,'',1)
            else: #if a scoreable letter, get letter score value and add to indexed score.
                word_scores[i] += scores.get(letter.lower())
    return word_scores   