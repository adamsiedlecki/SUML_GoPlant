FROM python:3.9

WORKDIR /app

COPY ["app.py", "trening/*", "requirements.txt", "./"]

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python trening.py


CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
