require 'mysql2'
require 'yandex_mystem'

# DB-connect
client = Mysql2::Client.new(
    host: '192.168.1.125',
    port: 3306,
    database: 'sentiment',
    username: 'sentiment',
    password: 'sentiment')

# form two arrays for pos and neg statements
pos = client.query("SELECT ttext FROM sortpos LIMIT 200, 5000").map { |row| row['ttext'].encode('utf-8')}
neg = client.query("SELECT ttext FROM sortneg LIMIT 200, 5000").map { |row| row['ttext'].encode('utf-8')}

puts pos.size
puts neg.size

vocabulary = Hash.new(0)

pos.each do |tweet|
  stemmed = YandexMystem::Simple.stem tweet
  stemmed.each do |k, v|
    word = v.empty? ? k : v[0]
    vocabulary[word] += 1
  end
end
neg.each do |tweet|
  stemmed = YandexMystem::Simple.stem tweet
  stemmed.each do |k, v|
    word = v.empty? ? k : v[0]
    vocabulary[word] += 1
  end
end

vocabulary.each do |k, v|
  query = 'INSERT INTO vocabulary(word, count) VALUES '
  query += "('#{k.to_s}', #{v.to_s})"
  client.query(query)
end