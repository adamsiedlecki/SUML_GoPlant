Zbudowanie obrazu

docker build -t goplant .

Uruchomienie obrazu:

docker run -p 5000:5000 my-ml-model

Testy:

curl -X POST -H "Content-Type: application/json" -d '{"input": [5.1, 3.5, 1.4, 0.2]}' http://localhost:5000/predict
