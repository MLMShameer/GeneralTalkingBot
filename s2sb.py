import numpy as np
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense

def talk(user_response):
    with open('mu.txt', 'r', encoding='utf-8') as f:
      lines = f.read().split('\n')
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    num_samples = 10000
    num_samples
    for line in lines[: min(num_samples, len(lines) - 1)]:
      line = line.lower()
      input_text, target_text = line.split('\t')
      target_text = '\t' + target_text + '\n'
      input_texts.append(input_text)
      target_texts.append(target_text)
      for char in input_text:
        if char not in input_characters:
          input_characters.add(char)
      for char in target_text:
        if char not in target_characters:
          target_characters.add(char)
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict(
      [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
      [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
      (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
      dtype='float32')
    decoder_input_data = np.zeros(
      (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
      dtype='float32')
    decoder_target_data = np.zeros(
      (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
      dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
      for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
      for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
          decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    
    batch_size = 64  # batch size for training
    epochs = 10  # number of epochs to train for
    latent_dim = 256  # latent dimensionality of the encoding space

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.load_weights('s2sb.h5')

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
      decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
      [decoder_inputs] + decoder_states_inputs,
      [decoder_outputs] + decoder_states)

    reverse_input_char_index = dict(
      (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
      (i, char) for char, i in target_token_index.items())

    def decode_sequence(input_seq):
  
      states_value = encoder_model.predict(input_seq)
  
      target_seq = np.zeros((1, 1, num_decoder_tokens))
      target_seq[0, 0, target_token_index['\t']] = 1.
  
      stop_condition = False
      decoded_sentence = ''
      while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
          [target_seq] + states_value)
    
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
          stop_condition = True
      
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
    
        states_value = [h, c]
    
      return decoded_sentence

    input_sentence = user_response
    input_sentence = input_sentence.lower()
    test_sentence_tokenized = np.zeros(
      (1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(input_sentence):
      test_sentence_tokenized[0, t, input_token_index[char]] = 1.
    user_response = decode_sequence(test_sentence_tokenized)
    #print(input_sentence)
    #sprint(user_response)
	
    return user_response
