FROM python:3.13

WORKDIR /app

RUN pip install poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /app/

RUN poetry install --no-root

COPY . /app

CMD ["python", "/app/main.py"]

