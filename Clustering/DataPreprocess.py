
import re
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


# create 200 partitions of the book. Each partition has 100 words.
# add these partitions to the data frame with the index(a, b, c etc according to the book). Return data frame
def readbook(book, index, target_df):
    content = book
    segmets = re.split("[^a-zA-Z]", content)

    contentSeries = pd.Series(dtype = str)
    indexSeries = pd.Series(dtype = str)
    ii = target_df.shape[0]

    t = int(len(segmets)/200)
    tar = 0
    result = ""
    i, x = 0, 0
    while (i < 200) & (i < t):
        x = 0
        if tar >= len(segmets):
            break
        while x <= 200:
            if tar >= len(segmets):
                break
            if re.search("[a-zA-Z*]", segmets[tar]):
                result = result + segmets[tar] + " "
                x = x + 1
            tar = tar + 1
        if x >= 99:
            indexSeries = indexSeries.append(pd.Series({ii + i: index}))
            contentSeries = contentSeries.append(pd.Series({ii + i: result}))
        result = ""
        i = i + 1
    contents = pd.DataFrame({'Index': indexSeries, 'Content': contentSeries})
    target_df = target_df.append(contents)
    return target_df


# select certain partition in the data frame
# remove stopwords in the selected content and return frequency of every word appeared in the partition
def tokenizeFilter(target_df, iloc):
    target_txt = target_df.iloc[iloc].iat[1]
    tok_word = nltk.word_tokenize(target_txt)
    stop_words = set(stopwords.words("english"))
    filtered_words = []
    for word in tok_word:
        if word not in stop_words:
            filtered_words.append(word)
    fdist = FreqDist(filtered_words)
    return fdist, filtered_words


target_df = pd.DataFrame()
nltk.corpus.gutenberg.fileids()
target_df = readbook(nltk.corpus.gutenberg.raw("burgess-busterbrown.txt"), "a", target_df)
target_df = readbook(nltk.corpus.gutenberg.raw("austen-persuasion.txt"), "b", target_df)
target_df = readbook(nltk.corpus.gutenberg.raw("milton-paradise.txt"), "c", target_df)
target_df = readbook(nltk.corpus.gutenberg.raw("bryant-stories.txt"), "d", target_df)
target_df = readbook(nltk.corpus.gutenberg.raw("shakespeare-macbeth.txt"), "e", target_df)

# now we have the data frame full of book's content with its index
print(target_df)

nltk.download()
target_df.insert(target_df.shape[1], "fdist", 0)
fdist, filtered_words = tokenizeFilter(target_df, 0)
print(fdist.most_common(50))
fdist.plot(30, cumulative=False)
plt.show()
