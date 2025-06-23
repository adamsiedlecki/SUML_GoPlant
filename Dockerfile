FROM python:3.9

WORKDIR /app

COPY ["app.py",  "trening",  "requirements.txt", "./"]

RUN  pip install --upgrade pip && pip install -r requirements.txt

RUN python trening.py  \
    && chmod +x /app/best_model.pkl

EXPOSE 8501

ENTRYPOINT ["sh", "-c", "streamlit run app.py"]