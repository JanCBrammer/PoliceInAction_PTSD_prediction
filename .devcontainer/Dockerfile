FROM python:3.8.10

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.1.12

ENV PATH /root/.local/bin:$PATH

# Not have Poetry create a virtual environment since we're already isolated in a Docker container.
RUN poetry config virtualenvs.create false
