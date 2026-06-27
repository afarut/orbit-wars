# Orbit Wars SFT

Бот на имитационном обучении (SFT / behavioral cloning) для Kaggle-соревнования
**Orbit Wars**. Set-transformer policy/value-сеть обучается на публичных реплеях
лидерборда, имитируя сильных игроков, и сабмитится как агент `act(obs) -> moves`.

- Архитектура модели и пайплайн обучения — `CLAUDE.md`.
- Правила игры — `orbit_wars_rules.md`, заметки по механикам — `insights/*.md`.

---

# Быстрый старт: dev-среда под CUDA (Docker)

Воспроизводимое окружение для обучения/эвала на GPU. Код монтируется внутрь живым
volume'ом — правишь на хосте, запускаешь внутри. **Не проверено на железе** (собиралось
без GPU под рукой); при первом запуске сверься с разделом «Проверка» ниже.

## Требования к хосту
- NVIDIA GPU + драйвер (`nvidia-smi` работает на хосте).
- [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (даёт `--gpus all`).
- Docker + Docker Compose v2.30+ (для синтаксиса `gpus: all`; для старых — см. коммент в `docker-compose.yml`).

## Запуск
```bash
docker compose build              # собрать образ (ставит CUDA-torch и зависимости)
docker compose up -d              # поднять контейнер в фоне
docker compose exec dev bash      # зайти внутрь
```
Внутри ты в `/workspace` (это корень репо). Остановить: `docker compose down`.

## Проверка, что CUDA видна
```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# ожидаем: ...  True  <число GPU>
```

## Запуск экспериментов внутри контейнера
Активный интерпретатор — `/opt/venv` (уже в `PATH`), с **CUDA-torch**. Поэтому команды
из `CLAUDE.md` запускай через `python`, **а не** `.venv/bin/python` (хостовый `.venv` —
CPU-сборка и внутри контейнера не используется):

```bash
python smoke_test.py
python -m sft.check --path data/sft.full_send.jsonl
python -m sft.train                                      # один GPU
torchrun --standalone --nproc_per_node=2 sft/train.py    # мульти-GPU DDP
python -m eval.run pool=baselines mode=1v1 episodes=25
tensorboard --logdir outputs --host 0.0.0.0              # порт пробрось при необходимости
```

`outputs/`, `data/`, `replays/` лежат на хосте (volume) — чекпойнты и логи переживают
пересоздание контейнера.

## Заметки / подводные камни
- **`ipc: host`** в compose обязателен: DataLoader кодирует ходы в воркерах, при дефолтных
  64 МБ `/dev/shm` обучение падает с `Bus error`. Альтернатива — `shm_size: '16gb'`.
- **torch ставится из PyPI** (`requirements.txt`): на linux это CUDA-сборка, отдельный
  `--extra-index-url` не нужен. Версия CUDA в базовом образе (`12.4.1`) на совместимость
  почти не влияет — torch несёт свои CUDA-либы; важен лишь драйвер хоста.
- **Kaggle-креды**: смонтированы из `~/.kaggle` (read-only) для `dataprep.download`. Нет
  файла — убери volume-строку, остальное работает.
- **Правки `requirements.txt`** требуют `docker compose build` заново; правки кода — нет.
- Удобный коннект из IDE: VS Code/Cursor «Dev Containers → Attach to running container»
  цепляется к `orbits-dev` без доп. конфига.
