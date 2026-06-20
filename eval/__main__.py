"""``python -m eval`` -> Hydra-вход :func:`eval.run.main` (см. eval/run.py).

Канонический запуск — ``python -m eval.run`` (зеркало ``python -m sft.train``); этот
делегат оставлен для удобства ``python -m eval``. Все параметры — оверрайды Hydra, напр.:
  python -m eval mode=4p episodes=25 roster=[best,sniper] ckpt_dir=outputs/<ts>/checkpoints
"""

from eval.run import main

if __name__ == "__main__":
    main()
