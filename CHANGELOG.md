# Changelog

Bu dosya [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) biçimini,
sürümleme [Semantic Versioning](https://semver.org/lang/tr/) kurallarını izler.

## [1.0.0] - 2026-10-04

İlk yayın.

### Eklenenler

- **Maksimum ağırlıklı eşleştirme ile kesin optimal yerleşim.** Oda kapasitesi iki
  ve hedef toplam uyumu maksimize etmek olduğunda problem tam olarak *maximum
  weight perfect matching*; Edmonds' blossom algoritması (`networkx`) bunu kesin
  optimal çözüyor. `maxcardinality=True` ile önce herkesin eşlenmesi, sonra toplam
  uyum maksimize ediliyor.
- **Dört yerleşim stratejisi** ortak arayüz arkasında: `optimal` (varsayılan),
  `greedy`, `stable` (Irving), `random`. Aynı skor matrisi üzerinde karşılaştırılıyor.
- **Ceza tabanlı kısıt modeli.** Zorunlu yerleşimde kesin filtre çözümü imkânsız
  kılabildiği için ihlaller kenar ağırlığından 1000 düşürüyor: çözüm her zaman
  bulunuyor, kaçınılabilir ihlal kaçınılıyor, kaçınılamayan ihlaller raporlanıyor.
- **Ağırlıklı Gower uyum skoru**, 15 boyut üzerinde. Her boyut kendi `[0,1]`
  uyumluluk fonksiyonunu tanımlıyor; `study_location` ve `cleanliness` aynılığı
  kasten cezalandırıyor, çünkü iki oda-çalışanı aynı masayı ve aynı sessizliği
  aynı saatte istiyor.
- **Özellik bazlı açıklama.** Her oda ve her aday, hangi boyutun skora ne kadar
  katkı verdiğini gösteren bir dökümle geliyor.
- **İki eksende değerlendirme**: sıralama kalitesi (P@K, R@K, MRR, nDCG) ve yerleşim
  kalitesi (toplam uyum, en kötü oda, oracle onayı, optimale fark, blocking pair,
  kaçınılamayan ihlal). Ground truth, skorlayıcıdan bağımsız kural tabanlı bir
  oracle'dan geliyor.
- **Kararlılık teşhisi.** Blocking pair araması her planda çalışıyor; toplam uyumu
  maksimize etmek kararlılığı garanti etmediği için bu bir hedef değil, rapor.
- Altı uyum metriği ortak arayüz arkasında; geometrik olanlar karşılaştırma için
  kayıtlı tutuluyor.
- FastAPI ve JWT (yalnızca yurt personeli), Typer ve Rich komut satırı aracı,
  Streamlit arayüzü, SQLAlchemy depolama (SQLite / PostgreSQL).
- Korelasyonlu sentetik veri üretici: gece kuşları daha geç çalışıyor ve gürültüye
  daha dayanıklı, odada çalışanlar odada daha çok vakit geçiriyor.
- Plan kalıcılığı: hesaplanan yerleşim `plans` ve `rooms` tablolarına yazılıyor.
- 336 test, %94 kapsam, ruff + mypy `strict`, GitHub Actions CI, Docker.
