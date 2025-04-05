Jest założenie, że model znadjuje się w katalogu docker.

https://github.com/iamsonubisht/Hate-Speech-Detection/tree/main

Zbudowanie obrazu

```cmd
docker build -t goplant .
```

Uruchomienie obrazu:

```cmd
docker run -p 5000:5000 my-ml-model
```

Testy:

```cmd
curl -X POST -H "Content-Type: application/json" -d '{"input": [5.1, 3.5, 1.4, 0.2]}' http://localhost:5000/predict

```