# Katkı Rehberi

## Geliştirme ortamı

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[ui,dev]"
pre-commit install
```

## Göndermeden önce

```bash
ruff check . && ruff format .
mypy
pytest
```

Üçü de CI'da zorunlu. `mypy` strict modda çalışır; `# type: ignore` eklemek yerine
tipi düzeltin.

## Kurallar

- **Kod ve tanımlayıcılar İngilizce, kullanıcıya görünen metinler Türkçe.**
- Her genel fonksiyon ve sınıf Google biçiminde docstring taşır (ruff `D` kuralları).
- Yeni bir uyum metriği `CompatibilityMetric`'ten türer ve `@register` ile kaydedilir.
- Yeni bir yerleşim stratejisi `AssignmentStrategy`'den türer ve `@register` ile
  kaydedilir. `optimal` sınıf değişkenini yalnızca sonucun optimal olduğu
  kanıtlanabiliyorsa `True` yapın.
- Yeni bir boyut `matching/dimensions.py` içine bir `Dimension` olarak eklenir ve
  `DEFAULT_WEIGHTS` içinde bir ağırlık alır. Boyut fonksiyonları **simetrik**
  olmalıdır; test bunu denetler.
- Kısıtlar `matching/constraints.py` içinde tanımlanır. Yerleşimde kesin filtre
  kullanmayın, çözümü imkânsız kılabilir; ceza mekanizmasını kullanın.
- **Düzeltilen her hata için bir regresyon testi** yazılır.
- Kapsam %80'in altına düşmemelidir.

## Commit mesajları

[Conventional Commits](https://www.conventionalcommits.org/tr/):
`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`, `ci:`
