# Dorm Roommate Matcher

Yurtlarda iki kişilik odalara öğrenci yerleştiren analiz motoru: ağırlıklı Gower uyumluluğu, maksimum ağırlıklı eşleştirme ile kesin optimal yerleşim ve her odanın arkasındaki özellik bazlı gerekçe.

[![CI](https://github.com/ahmetcevatdeniz/dorm-roommate-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmetcevatdeniz/dorm-roommate-matcher/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Coverage 94%](https://img.shields.io/badge/coverage-94%25-0ca30c)](#test-ve-kalite)
[![mypy strict](https://img.shields.io/badge/mypy-strict-2a78d6)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-eb6834?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Problem

Yurt yerleştirmesi bir arama problemi değil, bir **bölüşüm** problemi. Öğrenci oda arkadaşını seçmiyor; kurum yerleştiriyor ve herkesin bir yatağı olmak zorunda. Dolayısıyla çıktı sıralı bir aday listesi değil, tüm popülasyonu kapsayan bir plan.

İki şey de bunun doğal sonucu:

**Benzerlik uyumluluk değildir.** İkisi de dağınık olan iki öğrenci birbirine çok benzer ama oda yaşanmaz olur. İkisi de odada ders çalışan iki öğrenci "aynı" olmalarına rağmen aynı masayı ve aynı sessizliği aynı saatte istiyor; bu yüzden skorlayıcı bu iki durumda aynılığı **cezalandırıyor**.

**Zorunlu yerleşimde kesin filtre tehlikelidir.** "Sigara içen ile içmeyen aynı odada olmaz" kuralını mutlak uygularsan ve sigara içenlerin sayısı tekse, bazı öğrenciler yerleştirilemez. Bu yüzden ihlaller büyük negatif ağırlık olarak modellendi: çözüm her zaman bulunur, ihlaller minimize edilir ve raporlanır.

## Ölçüm

### Metrik kalitesi

403 öğrenci, 150 sorgu, bağımsız kural tabanlı oracle'a karşı:

| Metrik | P@5 | P@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Ağırlıklı Gower | **0.745** | **0.697** | **0.899** | **0.727** |
| Hibrit (Gower + Öklid) | 0.719 | 0.677 | 0.871 | 0.699 |
| Manhattan (ölçeklenmiş) | 0.420 | 0.407 | 0.639 | 0.422 |
| Öklid (ölçeklenmiş) | 0.397 | 0.381 | 0.633 | 0.399 |
| Kosinüs | 0.363 | 0.347 | 0.575 | 0.359 |
| Jaccard (yalnız ilgi alanları) | 0.212 | 0.202 | 0.411 | 0.212 |

Ağırlıklı Gower, kosinüse göre nDCG@10'da **2.02 kat** daha iyi.

### Yerleşim kalitesi

200 öğrenci, 100 oda, aynı skor matrisi üzerinde dört strateji:

| Strateji | Toplam uyum | En kötü oda | Oracle onayı | Optimale fark | Blocking pair | İhlal |
|---|---:|---:|---:|---:|---:|---:|
| **Optimal** (Blossom) | **84.49** | **0.733** | **%78** | %0.00 | 35 | 0 |
| Açgözlü | 83.90 | 0.560 | %73 | %0.70 | 0 | 0 |
| Kararlı (Irving) | 83.90 | 0.560 | %73 | %0.70 | 0 | 0 |
| Rastgele | 63.35 | 0.351 | %15 | %25.03 | 7016 | 12 |

Buradaki asıl bulgu toplam sütununda değil: optimal çözüm toplamda açgözlüden yalnızca **%0.7** önde, ama **en kötü odada %31** daha iyi (0.733 vs 0.560). Açgözlü yaklaşım güçlü çiftleri önce kapıp artakalanları birbirine mahkûm ediyor; toplamı maksimize etmek en kötü durumdaki öğrenciyi de dolaylı olarak koruyor.

Bedeli de var: optimal çözüm 35 blocking pair içeriyor, açgözlü ve kararlı stratejiler ise sıfır. Toplam uyumu maksimize etmek kararlılığı garanti etmiyor; sistem bunu hedef almıyor ama her planda ölçüp raporluyor.

Rakamlar `roommate evaluate` ile yeniden üretilebilir.

### Ölçeklenebilirlik

Ölçülmüş değerler (tahmin değil):

| Öğrenci | Çift | Skor matrisi | Blossom | Toplam |
|---:|---:|---:|---:|---:|
| 100 | 4.950 | 0.27 s | 0.29 s | 0.56 s |
| 200 | 19.900 | 1.09 s | 2.32 s | 3.41 s |
| 400 | 79.800 | 4.49 s | 20.07 s | 24.56 s |
| 800 | 319.600 | 17.86 s | 168.42 s | 186.28 s |

Skor matrisi O(n²), Blossom O(n³). Tipik bir yurt bloğu 200-600 öğrenci olduğu için kesin optimal çözüm gerçekçi aralığı kapsıyor. 800 öğrencide 3 dakika toplu iş olarak kabul edilebilir; bunun üstünde kesin çözüm pratik değil.

## Kurulum

```bash
git clone https://github.com/ahmetcevatdeniz/dorm-roommate-matcher.git
cd dorm-roommate-matcher

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[ui,dev]"

roommate seed --count 400
roommate assign
roommate room 1
```

`seed` dolu bir veritabanının üzerine yazmaz; yeniden üretmek için `roommate seed --reset` kullanın.

Docker ile:

```bash
cp .env.example .env             # RM_JWT_SECRET değerini değiştirin
docker compose -f docker/docker-compose.yml up --build
# API  http://localhost:8000/docs
# UI   http://localhost:8501
```

## Kullanım

### Komut satırı

```bash
roommate seed --count 400           # öğrenci popülasyonu üret
roommate seed --reset               # mevcut veriyi silip yeniden üret
roommate assign                     # optimal yerleşimi hesapla ve kaydet
roommate assign --strategy greedy    # taban çizgisiyle karşılaştır
roommate rooms --worst              # en zayıf odalar
roommate room 12                    # bir odanın gerekçesi
roommate candidates 5 --detail      # bir öğrenci için uyumlu adaylar
roommate compare 5 12               # iki öğrencinin uyum dökümü
roommate evaluate                   # metrik ve strateji karşılaştırması
roommate stats                      # popülasyon dağılımları
roommate strategies                 # kayıtlı stratejiler
roommate create-admin               # API için yönetici hesabı
roommate serve                      # API
```

```
┌───────────────────────── Yerleşim planı ─────────────────────────┐
│ Strateji: optimal (kesin optimal)                                │
│ Öğrenci: 403 · Oda: 201 · Yerleşmeyen: 1                         │
│ Toplam uyum: 169.0124 · Ortalama: 0.8408 · En kötü oda: 0.7604   │
│ Süre: 26.50 sn                                                    │
│ Hiçbir kısıt ihlali yok.                                          │
│ Kararlılık: 81 blocking pair                                      │
│ Yerleşmeyen öğrenci: [384]                                        │
└───────────────────────────────────────────────────────────────────┘
                        Odalar (en uyumlu 5)
┌─────┬────────────────────────┬───────────────────────┬──────┬───────────────┐
│ Oda │ Öğrenci A              │ Öğrenci B             │ Uyum │ Dikkat        │
├─────┼────────────────────────┼───────────────────────┼──────┼───────────────┤
│   1 │ #287 Çağlasın Akdurmuş │ #377 Oliver İhsanoğlu │  %93 │ İlgi alanları │
│   2 │ #191 Sevginur Misra    │ #345 Atiyye Bilgin    │  %93 │ -             │
└─────┴────────────────────────┴───────────────────────┴──────┴───────────────┘
```

### Web arayüzü

```bash
streamlit run app/streamlit_app.py
```

Dört sekme: **Yerleşim** (plan, oda gerekçesi, oda skor dağılımı) · **Öğrenci analizi** (profil, uyumlu adaylar, açıklama) · **Popülasyon** (dağılımlar, uyum matrisi, strateji karşılaştırması) · **Öğrenci ekle**.

### HTTP API

```bash
roommate create-admin
roommate serve      # http://127.0.0.1:8000/docs
```

Okuma herkese açık; plan hesaplamak ve öğrenci eklemek yönetici girişi gerektirir. Öğrenciler API kullanıcısı değil, veridir; bu yüzden kayıt uç noktası yok.

| Yöntem | Uç nokta | Yetki | Açıklama |
|---|---|---|---|
| `GET` | `/health` | - | Servis durumu, oda kapasitesi, güvenlik uyarısı |
| `GET` | `/metrics` | - | Kayıtlı uyum metrikleri |
| `GET` | `/strategies` | - | Kayıtlı yerleşim stratejileri |
| `POST` | `/auth/login` | - | Yönetici girişi, JWT döner |
| `GET` | `/auth/me` | yönetici | Oturum bilgisi |
| `GET` | `/students` | - | Listeleme, sınıf filtresi, sayfalama |
| `GET` | `/students/{id}` | - | Tek öğrenci |
| `POST` | `/students` | yönetici | Öğrenci kaydı ekle |
| `GET` | `/students/{id}/candidates` | - | Uyumlu adaylar ve açıklamaları |
| `POST` | `/students/{id}/candidates` | - | Özel ağırlık ve kısıtlarla |
| `GET` | `/students/{id}/compare/{other}` | - | İki öğrencinin ayrıntılı dökümü |
| `POST` | `/assignments/run` | yönetici | Yerleşimi hesapla |
| `GET` | `/assignments/latest` | - | Son kaydedilen plan |
| `GET` | `/assignments/latest/rooms/{no}` | - | Bir odanın gerekçesi |

```bash
curl "http://127.0.0.1:8000/assignments/latest/rooms/1" \
  | jq '{first_name, second_name, percentage, summary}'
```

```json
{
  "first_name": "Çağlasın Akdurmuş Akça",
  "second_name": "Oliver İhsanoğlu",
  "percentage": 93,
  "summary": "Uyum skoru %93. Güçlü yanlar: Sigara (İçmiyor / İçmiyor); Uyku düzeni (Esnek / Esnek); Temizlik (4/5 / 4/5). Dikkat edilmesi gerekenler: İlgi alanları (Sanat, Teknoloji / Sanat, Yemek)."
}
```

## Nasıl çalışıyor

### 1. Uyum skoru: ağırlıklı Gower

Her boyut kendi `[0, 1]` uyumluluk fonksiyonunu tanımlar, sonuç bunların ağırlıklı ortalaması:

$$\text{score}(a,b) = \frac{\sum_{d \in D} w_d \cdot s_d(a,b)}{\sum_{d \in D} w_d}$$

Bu formülasyon sayısal, sıralı, kategorik ve küme değerli özellikleri tek modelde ele alır.

| Boyut | Ağırlık | Uyumluluk mantığı |
|---|---:|---|
| Sigara | 3.0 | Matris; içmeyen ile odada içen eşleşmesi 0.0 |
| Uyku düzeni | 2.5 | Erken ile gece kuşu 0.1; esnek herkesle 0.8 |
| Temizlik | 2.5 | Fark bazlı; **ikisi de dağınıksa 0.7 ile çarpılır** |
| Ders çalışma yeri | 2.0 | **Aynılık cezalı**: ikisi de odada 0.45, oda + kütüphane 1.0 |
| Gürültü toleransı | 2.0 | Fark bazlı |
| Çalışma saatleri | 1.5 | Aynı saatler yüksek (oda aynı anda sessiz olur) |
| Odada geçirilen zaman | 1.2 | Fark bazlı |
| Misafir sıklığı | 1.2 | Sıralı |
| İlgi alanları | 1.2 | Jaccard, tam token eşleşmesiyle |
| Sosyallik, alkol | 1.0 | Sıralı |
| Bölüm | 0.8 | Yapılandırılabilir: aynı bölüm artı, nötr veya eksi |
| Sınıf | 0.8 | `exp(-fark / 2.5)` |
| Ortak dil, beslenme | 0.5 | Jaccard / matris |

15 boyutun tamamı `matching/dimensions.py` içinde.

### 2. Kısıtlar: filtre değil, ceza

Aynı kısıt tanımı iki yerde kullanılır:

- **Aday analizinde** filtre gibi davranır, uymayan adayı listeden çıkarır.
- **Yerleşimde** kenar ağırlığından `1000` düşürür. Skorlar `[0, 1]` aralığında olduğu için ceza her zaman baskın; çözüm kaçınılabilir ihlalden kaçınır, kaçınılamayanı kabul eder ve raporlar.

Doğrulanmış davranış: 2 içen + 2 içmeyen verildiğinde içenler birlikte yerleşir ve **sıfır** ihlal olur; 1 içen + 3 içmeyen verildiğinde herkes yerleşir ve **tam 1** (minimum) ihlal raporlanır.

### 3. Yerleşim: maksimum ağırlıklı eşleştirme

Oda kapasitesi iki ve hedef toplam uyumu maksimize etmek olduğunda problem tam olarak *maximum weight perfect matching*. [Edmonds' blossom algoritması](https://en.wikipedia.org/wiki/Blossom_algorithm) bunu **kesin optimal** olarak, polinom zamanda çözer.

`maxcardinality=True` seçimi kritik: önce kardinaliteyi (yani herkesin eşlenmesini), sonra toplam ağırlığı maksimize eder. Negatif cezalı kenarlar çözümü engellemez, yalnızca kaçınılır.

Dört strateji ortak bir arayüz arkasında kayıtlı:

| Strateji | Yöntem | Rol |
|---|---|---|
`optimal` | Blossom | Varsayılan; kesin optimal
`greedy` | En iyi çifti sırayla seç | Taban çizgisi
`stable` | Irving kararlı oda arkadaşı algoritması | Farklı hedef: kararlılık
`random` | Tohumlanmış rastgele | Zemin

### 4. Açıklama

Her boyut kendi skorunu ayrı raporladığı için döküm doğrudan elde edilir:

```
┌──────────────────┬──────────┬──────────┬───────────┬─────────┬─────────────┬─────────┐
│ Özellik          │ Öğrenci A│ Öğrenci B│ Benzerlik │ Ağırlık │ Skora katkı │ Durum   │
├──────────────────┼──────────┼──────────┼───────────┼─────────┼─────────────┼─────────┤
│ Ders çalışma yeri│ Odada    │ Odada    │      0.45 │     2.0 │        4.1% │ Kısmen  │
│ Temizlik         │ 1/5      │ 1/5      │      0.70 │     2.5 │        8.1% │ Kısmen  │
│ Sigara           │ İçmiyor  │ İçmiyor  │      1.00 │     3.0 │       13.8% │ Uyumlu  │
└──────────────────┴──────────┴──────────┴───────────┴─────────┴─────────────┴─────────┘
```

İki satır da aynılığın ödüllendirilmediği yerler: aynı temizlik seviyesi 1/5 olduğu için 0.70, ikisi de odada çalıştığı için 0.45.

### 5. Kararlılık teşhisi

Blocking pair, farklı odalardaki iki öğrencinin karşılıklı olarak birbirini tercih etmesi. Toplam uyumu maksimize etmek bunu dışlamaz, bu yüzden her plan ayrıca denetlenir ve sonuç raporlanır. `find_blocking_pairs` dedektörünün kendisi, kasten bozuk bir eşleştirmeyle test edilir; aksi halde "hata bulamadı" ile "hata arayamıyor" ayırt edilemez.

## Mimari

```
src/roommate_matcher/
├── domain/              # StudentProfile, Room, AssignmentPlan, ağırlıklar
│   ├── enums.py             Türkçe etiketli kontrollü sözlükler
│   ├── models.py            Pydantic modelleri, hesaplanan özellikler
│   └── preferences.py       Ağırlıklar ve kısıtlar
├── features/            # Öğrenci → sayısal uzay
│   ├── interests.py         İlgi alanı taksonomisi ve normalizasyonu
│   └── vectorizer.py        [0,1] aralığına ölçeklenmiş kodlama
├── matching/
│   ├── dimensions.py        15 uyumluluk fonksiyonu ve açıklamaları
│   ├── metrics.py           Strategy: gower/cosine/euclidean/manhattan/jaccard/hybrid
│   ├── constraints.py       Kısıtlar; filtre ve ceza üretici
│   ├── engine.py            Skor matrisi ve aday analizi
│   ├── assignment.py        4 yerleşim stratejisi + kararlılık teşhisi
│   ├── explain.py           Skordan açıklamaya
│   └── evaluation.py        Sıralama ve yerleşim kalitesi + oracle
├── data/
│   ├── repository.py        Protocol, bellek içi ve CSV uygulamaları
│   ├── database.py          SQLAlchemy; öğrenci, plan, oda, yönetici tabloları
│   └── generator.py         Korelasyonlu sentetik popülasyon
├── api/                     FastAPI ve JWT (yalnızca yönetici)
├── cli/                     Typer ve Rich
└── viz/charts.py            Plotly grafikleri
```

Katman kuralı: `domain` iç modül import etmez, `matching` yalnızca `domain` ve `features`'ı bilir, `api`/`cli`/`app` sunum katmanıdır. Yerleşim mantığı tek yerde olduğu için CLI, API ve arayüz aynı sonucu üretir.

Yeni bir strateji eklemek tek dosya ve tek dekoratör:

```python
@register
class MyStrategy(AssignmentStrategy):
    key = "my_strategy"

    def pair_up(self, matrix) -> Pairing: ...
```

Tasarım kararlarının gerekçeleri ve değerlendirilen alternatifler: [docs/architecture.md](docs/architecture.md)

## Test ve kalite

```bash
pytest                    # 336 test, %94 kapsam
ruff check . && ruff format --check .
mypy                      # strict mod, 46 dosya
pre-commit install
```

Test paketinin kapsadıkları:

- **Yerleşim doğruluğu:** herkes yerleşir, tek sayıda öğrencide tam bir kişi kalır, hiçbir öğrenci iki odada olmaz, optimal hiçbir stratejiden geride kalmaz, dört kişilik elle doğrulanmış örnekte üç olası eşleştirme sayılarak optimalin bulunduğu kanıtlanır
- **Ceza mekanizması:** kaçınılabilir ihlalin kaçınıldığı, kaçınılamazda herkesin yerleştiği ve ihlalin minimum olduğu, cezanın her zaman skora baskın olduğu
- **Boyut sözleşmesi:** 15 boyutun tamamı sınırlı ve simetrik; aynılığı cezalandıran iki boyutun gerçekten cezalandırdığı
- **Özellik tabanlı testler** (Hypothesis): skor her zaman `[0,1]`'de, simetrik, bir klon her zaman yenilmez
- **Oracle:** simetrik, her kuralın ayrı ayrı çalıştığı, skorlayıcıdan bağımsız olduğu
- **Veri kalitesi:** tam token ilgi alanı eşleşmesi, satır sırasına güvenmeyen ID çözümleme, sütun ölçekleme, aralık doğrulama, üreticinin tekrarlanabilirliği
- **API entegrasyon:** yetkilendirme, 401/404/400/409/422 yolları, e-posta sızdırmayan giriş hatası
- **Streamlit:** `AppTest` ile script'in istisnasız render edildiği ve yerleşim ile strateji karşılaştırma düğmelerine basılan yolların çalıştığı
- **CLI:** on bir komutun tamamı izole bir veritabanına karşı; boş veritabanı ve eksik plan durumlarının anlaşılır hata vermesi

## Yapılandırma

Ayarlar `RM_` önekiyle ortam değişkeni veya `.env` ile geçersiz kılınabilir; tam liste `.env.example` içinde.

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `RM_JWT_SECRET` | geliştirme değeri | Üretimde değiştirilmeli; `/health` bunu bildirir |
| `RM_DATABASE_URL` | `sqlite:///data/dormitory.db` | PostgreSQL için `postgresql+psycopg://...` |
| `RM_DEFAULT_METRIC` | `weighted_gower` | Varsayılan uyum metriği |
| `RM_DEFAULT_STRATEGY` | `optimal` | Varsayılan yerleşim stratejisi |
| `RM_ROOM_CAPACITY` | `2` | Oda başına yatak; optimal çözüm bir eşleştirme olduğu için sabit |
| `RM_CHECK_STABILITY` | `true` | Planda blocking pair ara |

## Yol haritası

- [ ] 1000+ öğrenci için ANN ile ön eleme, ardından kalan adaylarda tam skorlama
- [ ] İkiden büyük oda kapasitesi (kapasiteli gruplama, sezgisel + yerel arama)
- [ ] Karşılıklı oda arkadaşı talebi desteği
- [ ] Geçmiş dönem sonuçlarından ağırlık öğrenme
- [ ] React ve TypeScript arayüz (şu an Streamlit)
- [ ] Alembic ile şema migrasyonları

## Lisans

[MIT](LICENSE)
