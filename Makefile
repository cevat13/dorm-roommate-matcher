.PHONY: help install seed test lint type check run api ui clean docker

help:  ## Bu yardımı göster
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Bağımlılıkları kur
	pip install -e ".[ui,dev]"
	pre-commit install

seed:  ## Veritabanını doldur
	roommate seed --count 400

test:  ## Testleri çalıştır
	pytest

lint:  ## Lint ve biçim kontrolü
	ruff check .
	ruff format --check .

type:  ## Tip kontrolü
	mypy

check: lint type test  ## CI'ın çalıştırdığı her şey

assign:  ## Optimal yerleşimi hesapla
	roommate assign

api:  ## API'yi başlat
	uvicorn roommate_matcher.api.main:app --reload

ui:  ## Streamlit arayüzünü başlat
	streamlit run app/streamlit_app.py

docker:  ## Docker yığınını başlat
	docker compose -f docker/docker-compose.yml up --build

clean:  ## Üretilen dosyaları temizle
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
