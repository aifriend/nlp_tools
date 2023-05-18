# ### Analyze performance


import statistics

from nltk.translate.bleu_score import sentence_bleu

from gpt2_pre_data import load_data

MAX_DATA_LOADED = 2

print(f"Using BLEU score to compare the real sentences with the generated ones")

scores = []
df, test_set = load_data(max_data_loaded=MAX_DATA_LOADED)
for i in enumerate(test_set):
    reference = test_set['True_end_lyrics'][i]
    candidate = test_set['Generated_lyrics'][i]
    scores.append(sentence_bleu(reference, candidate))

mean = statistics.mean(scores)
print(f"Scores: {mean}")

from rouge import Rouge

rouge = Rouge()
score = rouge.get_scores(test_set['Generated_lyrics'], test_set['True_end_lyrics'], avg=True)
print(f"Scores: {score}")

print(f"GPT2 without any fine-Tuning")

import transformers

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')


def gen_text(prompt_text, tokenizer, model, n_seqs=1, max_length=374):
    print("Making a function that will generate text for us")
    # n_seqs is the number of sequences to generate
    # max_length is the maximum length of the sequence
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    # We are encoding the text using the gpt tokenizer. The return tensors are of type "pt"
    # since we are using PyTorch, not tensorflow
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length + len(encoded_prompt),  # The model has to generate something,
        # so we add the length of the original sequence to max_length
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.2,  # To ensure that we dont get repeated phrases
        do_sample=True,
        num_return_sequences=n_seqs
    )  # We feed the encoded input into the model.
    ## Getting the output ##
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()  # the _ indicates that the operation will be done in-place
    generated_sequences = []
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence)
        total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True, )):]
        )
        generated_sequences.append(total_sequence)

    return generated_sequences


# Generate sequences
gen_text(df['Lyric'][0], tokenizer, model)


def text_generation(test_data):
    print("Function to generate multiple sentences. Test data should be a dataframe")
    generated_lyrics = []
    for i in range(len(test_data)):
        x = gen_text(test_data['Lyric'][i], tokenizer, model)
        generated_lyrics.append(x)
    return generated_lyrics


generated_lyrics = text_generation(test_set)

print("Loop to keep only generated text and add it as a new column in the dataframe")
my_generations = []
for i in range(len(generated_lyrics)):
    a = test_set['Lyric'][i].split()[-30:]  # Get the matching string we want (30 words)
    b = ' '.join(a)
    c = ' '.join(generated_lyrics[i])  # Get all that comes after the matching string
    my_generations.append(c.split(b)[-1])

test_set['Generated_lyrics'] = my_generations

print("Finish the sentences when there is a point, remove after that")
final = []
for i in range(len(test_set)):
    to_remove = test_set['Generated_lyrics'][i].split('.')[-1]
    final.append(test_set['Generated_lyrics'][i].replace(to_remove, ''))

test_set['Generated_lyrics'] = final
test_set.head()

print("Using BLEU score to compare the real sentences with the generated ones")
import statistics
from nltk.translate.bleu_score import sentence_bleu

scores = []
for i in range(len(test_set)):
    reference = test_set['True_end_lyrics'][i]
    candidate = test_set['Generated_lyrics'][i]
    scores.append(sentence_bleu(reference, candidate))

mean = statistics.mean(scores)
print(f"Scores: {mean}")

from rouge import Rouge

rouge = Rouge()
score = rouge.get_scores(test_set['Generated_lyrics'], test_set['True_end_lyrics'], avg=True, ignore_empty=True)
print(f"Scores: {score}")
