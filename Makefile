PY := python

.PHONY: venv install lint run docker-build docker-run clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

lint:
	. .venv/bin/activate && python -m pyflakes src || true

run:
	. .venv/bin/activate && $(PY) src/yolo_batch.py --input-dir data/images --project runs/detect --name batch

docker-build:
	docker build -t face-recognition:latest .

docker-run:
	docker compose up --build

clean:
	rm -rf runs __pycache__ .pytest_cache .coverage
