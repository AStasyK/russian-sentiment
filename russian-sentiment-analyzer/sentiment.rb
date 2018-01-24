require 'mysql2'
require 'yandex_mystem'
require 'nmatrix'
require 'pp'

require './neural_network'

# DB-connect
client = Mysql2::Client.new(
    host: '192.168.1.125',
    port: 3306,
    database: 'sentiment',
    username: 'sentiment',
    password: 'sentiment')

# form two arrays for pos and neg statements
pos = client.query("SELECT ttext FROM sortpos LIMIT 500").map { |row| row['ttext'].encode('utf-8')}
neg = client.query("SELECT ttext FROM sortneg LIMIT 500").map { |row| row['ttext'].encode('utf-8')}

#pp puts pos.size
#pp puts neg.size

vocabulary = Hash.new(0)
client.query("SELECT word, SUM(count) as cnt FROM vocabulary GROUP BY word HAVING cnt >= 10 ORDER BY cnt DESC").map { |row| vocabulary[row['word']] = row['cnt'] }


vocabulary = vocabulary.keys[0...vocabulary.keys.size]
token_to_idx = Hash.new(0)
tokenized_vocab_size = vocabulary.size
pp tokenized_vocab_size

(0...tokenized_vocab_size).each { |i| token_to_idx[vocabulary[i]] = i}

pp token_to_idx['и']

def sentence_to_vec sentence, vocab_size, token_to_idx
  vector = NMatrix.zeroes( [vocab_size, 1])
  stemmed = YandexMystem::Simple.stem sentence
  stemmed.each do |k, v|
    word = v.empty? ? k : v[0]
    index = token_to_idx[word] || nil
    vector[index] = 1 unless index.nil?
  end
  vector
end

puts pos.first
puts sentence_to_vec pos.first, tokenized_vocab_size, token_to_idx

all_sentences  = []
x_matrix = sentence_to_vec pos.first, tokenized_vocab_size, token_to_idx
all_sentences << pos.first

pos[1...pos.size].each do |sentence|
  vector = sentence_to_vec sentence,tokenized_vocab_size, token_to_idx
  x_matrix = x_matrix.hconcat(vector)
  all_sentences << sentence
end

neg.each do |sentence|
  vector = sentence_to_vec sentence,tokenized_vocab_size, token_to_idx
  x_matrix = x_matrix.hconcat(vector)
  all_sentences << sentence
end

pp x_matrix.shape

y_matrix = NMatrix.ones([1, pos.size]).hconcat(NMatrix.zeroes([1, neg.size]))

pp y_matrix.shape

X = x_matrix
Y = y_matrix
X_train = X.slice(0..tokenized_vocab_size - 1, 0..399).hconcat(X.slice(0..tokenized_vocab_size - 1 ,500..899))
X_test = X.slice(0..tokenized_vocab_size - 1, 400..499).hconcat(X.slice(0..tokenized_vocab_size - 1, 900..999))
Y_train = Y.slice(0,0..399).hconcat(Y.slice(0,500..899))
Y_test = Y.slice(0,400..499).hconcat(Y.slice(0,900..999))


parameters = nn_model(X_train, Y_train, 12, 1.3, num_iterations=10000, print_cost=true)
pp "W1 = #{parameters[:W1]}"
pp "b1 = #{parameters[:b1]}"
pp "W2 = #{parameters[:W2]}"
pp "b2 = #{parameters[:b2]}"

predictions = predict(parameters, X_test)

predictions1 = predictions.map { |e| pos = e ? 1 : 0 }
predictions2 = predictions.map do |e|
  pos = e ? 1 : 0
  1 - pos
end
y = Y_test.map { |e| (1 - e) }

accuracy = Y_test.dot(predictions1.transpose) + y.dot(predictions2.transpose)

puts "Accuracy: #{ accuracy.to_f / Y_test.shape[1].to_f * 100} %"