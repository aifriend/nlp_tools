## CONFIGURATION WIDGET

    https://spacy.io/usage/training#quickstart

### CPU and EFFICIENCY

    To use it with 'spacy train' you can run spacy init fill-config to auto-fill all default settings

    [paths]
    train = null
    dev = null
    
    [system]
    gpu_allocator = null
    
    [nlp]
    lang = "es"
    pipeline = ["tok2vec","ner"]
    batch_size = 1000
    
    [components]
    
    [components.tok2vec]
    factory = "tok2vec"
    
    [components.tok2vec.model]
    @architectures = "spacy.Tok2Vec.v2"
    
    [components.tok2vec.model.embed]
    @architectures = "spacy.MultiHashEmbed.v2"
    width = ${components.tok2vec.model.encode.width}
    attrs = ["ORTH", "SHAPE"]
    rows = [5000, 2500]
    include_static_vectors = false
    
    [components.tok2vec.model.encode]
    @architectures = "spacy.MaxoutWindowEncoder.v2"
    width = 96
    depth = 4
    window_size = 1
    maxout_pieces = 3
    
    [components.ner]
    factory = "ner"
    
    [components.ner.model]
    @architectures = "spacy.TransitionBasedParser.v2"
    state_type = "ner"
    extra_state_tokens = false
    hidden_width = 64
    maxout_pieces = 2
    use_upper = true
    nO = null
    
    [components.ner.model.tok2vec]
    @architectures = "spacy.Tok2VecListener.v1"
    width = ${components.tok2vec.model.encode.width}
    
    [corpora]
    
    [corpora.train]
    @readers = "spacy.Corpus.v1"
    path = ${paths.train}
    max_length = 0
    
    [corpora.dev]
    @readers = "spacy.Corpus.v1"
    path = ${paths.dev}
    max_length = 0
    
    [training]
    dev_corpus = "corpora.dev"
    train_corpus = "corpora.train"
    
    [training.optimizer]
    @optimizers = "Adam.v1"
    
    [training.batcher]
    @batchers = "spacy.batch_by_words.v1"
    discard_oversize = false
    tolerance = 0.2
    
    [training.batcher.size]
    @schedules = "compounding.v1"
    start = 100
    stop = 1000
    compound = 1.001
    
    [initialize]
    vectors = ${paths.vectors}

## CONFIG INSTALLATION

    python -m spacy init fill-config ./base_config.cfg ./config.cfg