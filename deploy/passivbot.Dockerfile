# ====== Builder ======
FROM python:3.10-slim AS builder
WORKDIR /app

# Системные зависимости (при необходимости дополняйте)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Скопируем только нужное
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip wheel -r requirements.txt -w /wheels

# ====== Runtime ======
FROM python:3.10-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Системные зависимости для рантайма
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini tzdata && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем собранные колёса
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*

# Копируем исходники форка (бота) в контейнер
# Если корень репо — это passivbot, достаточно:
COPY . /app

# Папки под внешние конфиги/логи
VOLUME ["/data"]

# Порт для GUI-API (опционально)
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "PYTHONPATH=src python3 -m passivbot --help"]
