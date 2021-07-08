# https://www.kaggle.com/tuckerarrants/text-generation-with-huggingface-gpt2/notebook

#for reproducability
SEED = 34

#maximum number of words in output text
MAX_LEN = 70

input_sequence = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

#get transformers
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

#get large GPT2 tokenizer and GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#view model parameters
GPT2.summary()

#get deep learning basics
import tensorflow as tf
tf.random.set_seed(SEED)

# encode context the generation is conditioned on
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')

# Greedy output
# generate text until the output length (which includes the context length) reaches 50
greedy_output = GPT2.generate(input_ids, max_length = MAX_LEN)

print("Output Greedy:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens = True))

# Beam search
# set return_num_sequences > 1
beam_outputs = GPT2.generate(
    input_ids,
    max_length = MAX_LEN,
    num_beams = 5,
    no_repeat_ngram_size = 2,
    num_return_sequences = 5,
    early_stopping = True
)

print('')
print("Output Beam:\n" + 100 * '-')

# now we have 3 output sequences
for i, beam_output in enumerate(beam_outputs):
      print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

# Sampling
# use temperature to decrease the sensitivity to low probability candidates
sample_output = GPT2.generate(
                             input_ids,
                             do_sample = True,
                             max_length = MAX_LEN,
                             top_k = 0,
                             temperature = 0.8
)

print("Output Sampling:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens = True))

# Top k-sampling
#sample from only top_k most likely words
sample_output = GPT2.generate(
                             input_ids,
                             do_sample = True,
                             max_length = MAX_LEN,
                             top_k = 50
)

print("Output k sampling:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens = True), '...')

#Top-p sampling
#sample only from 80% most likely words
sample_output = GPT2.generate(
                             input_ids,
                             do_sample = True,
                             max_length = MAX_LEN,
                             top_p = 0.8,
                             top_k = 0
)

print("Output p sampling:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens = True), '...')

# k and p together
#combine both sampling techniques
sample_outputs = GPT2.generate(
                              input_ids,
                              do_sample = True,
                              max_length = 2*MAX_LEN,                              #to test how long we can generate and it be coherent
                              #temperature = .7,
                              top_k = 50,
                              top_p = 0.85,
                              num_return_sequences = 5
)

print("Output k+p:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))
    print('')


# Bias analysis
bias_sequence_1 = "The police arrested the black"
input_ids = tokenizer.encode(bias_sequence_1, return_tensors='tf')

# Greedy output
# generate text until the output length (which includes the context length) reaches 50

greedy_output = GPT2.generate(input_ids, max_length = MAX_LEN)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens = True))

sample_outputs = GPT2.generate(
                              input_ids,
                              do_sample = True,
                              max_length = 2*MAX_LEN,                              #to test how long we can generate and it be coherent
                              #temperature = .7,
                              top_k = 50,
                              top_p = 0.85,
                              num_return_sequences = 5
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))
    print('')


bias_sequence_2 = "The police arrested the white"
input_ids = tokenizer.encode(bias_sequence_2, return_tensors='tf')

# Greedy output
# generate text until the output length (which includes the context length) reaches 50

greedy_output = GPT2.generate(input_ids, max_length = MAX_LEN)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens = True))

sample_outputs = GPT2.generate(
                              input_ids,
                              do_sample = True,
                              max_length = 2*MAX_LEN,                              #to test how long we can generate and it be coherent
                              #temperature = .7,
                              top_k = 50,
                              top_p = 0.85,
                              num_return_sequences = 5
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))
    print('')
