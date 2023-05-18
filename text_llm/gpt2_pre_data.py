# ### Prepare data
import pandas as pd

VALIDATION_SIZE = 0.2


def load_doc_data(max_data_loaded):
    # Doc,Date
    df = pd.read_csv('data/ns-data.csv')

    # Drop the docs with content too long (after more than 1024 tokens, does not work)
    df = df[df['Doc'].apply(lambda x: len(x.split(' ')) < 5000)]

    # Create a very small test set to compare generated text with the reality
    test_set = df.sample(n=max_data_loaded)
    df = df.loc[~df.index.isin(test_set.index)]
    print(f"{len(df)} examples loaded")

    # Reset the indexes
    test_set = test_set.reset_index()
    df = df.reset_index()

    test_set.head()

    return df, test_set


def load_lyric_data(max_data_loaded):
    # ALink,SName,SLink,Lyric,language
    lyrics = pd.read_csv('data/lyrics-data.csv')
    lyrics = lyrics[lyrics['language'] == 'en']

    # Only keep popular artists, with genre Rock/Pop and popularity high enough
    # Artist,Genres,Songs,Popularity,Link
    artists = pd.read_csv('data/artists-data.csv')
    artists = artists[(artists['Genres'].isin(['Rock'])) & (artists['Popularity'] > 5)]

    df = lyrics.merge(artists[['Artist', 'Genres', 'Link']], left_on='ALink', right_on='Link', how='inner')
    df = df.drop(columns=['ALink', 'SLink', 'language', 'Link'])
    # Drop the songs with lyrics too long (after more than 1024 tokens, does not work)
    df = df[df['Lyric'].apply(lambda x: len(x.split(' ')) < 5000)]

    # Create a very small test set to compare generated text with the reality
    test_set = df.sample(n=max_data_loaded)
    df = df.loc[~df.index.isin(test_set.index)]
    print(f"{len(df)} examples loaded")

    # Reset the indexes
    test_set = test_set.reset_index()
    df = df.reset_index()

    # For the test set only, keep last 20 words in a new column, then remove them from original column
    test_set['True_end_lyrics'] = test_set['Lyric'].str.split().str[-20:].apply(' '.join)
    test_set['Lyric'] = test_set['Lyric'].str.split().str[:-20].apply(' '.join)

    test_set.head()

    return df, test_set

