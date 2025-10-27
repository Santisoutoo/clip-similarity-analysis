# can be changed, however this offers a good compromise between recency and compatibility
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /opt/requirements.txt

RUN python -m venv /opt/.venv \
    && /opt/.venv/bin/pip install --upgrade pip \
    && /opt/.venv/bin/pip install --no-cache-dir -r /opt/requirements.txt

WORKDIR /opt

