# Mimari ve Tasarım Kararları

Projede alınan kararlar ve değerlendirilip seçilmeyen alternatifler. Kodun ne yaptığı kodda; burada neden öyle olduğu var.

---

## ADR-001: Kosinüs benzerliği yerine ağırlıklı Gower

**Bağlam.** Profil benzerliği için akla ilk gelen yol, sütunları bir vektöre dizip `cosine_similarity` veya `euclidean_distances` uygulamaktır.

**Sorun.** Bu veride iki ayrı matematiksel kusur doğuruyor:

1. **Kosinüs yön ölçer.** İki öğrencinin de "hayır" dediği ikili bir özellik (ör. `smoker = 0`) iç çarpıma katkı vermez. Yani "ikimiz de sigara içmiyoruz" bilgisi benzerliğe sıfır katkı yapıyordu, oysa bir yurt odasında belirleyici bir uyum noktası.
2. **Ölçekleme yoktu.** Sütun aralıkları farklıydı; Öklid mesafesinde en geniş aralıklı sütun sonucu neredeyse tek başına belirliyordu.

**Karar.** Karışık tipli veri için tasarlanmış **Gower** formülasyonu: her boyut kendi `[0,1]` uyumluluk fonksiyonunu tanımlar, sonuç ağırlıklı ortalamadır.

**Sonuç.** nDCG@10 0.359 → 0.727 (2.02×). Açıklanabilirlik de aynı yapıdan geliyor: her boyut kendi skorunu ayrı raporladığı için skor dökümü ek bir iş gerektirmiyor.

**Alternatifler.**
- *Sütunları ölçekleyip kosinüsü korumak:* ölçekleme sorunu çözülür ama sıfır-bilgi sorunu kalır. Ölçüldü; `cosine` metriği kayıtlı kalmaya devam ediyor ve karşılaştırma tablosunda görünüyor.
- *Öğrenilmiş embedding:* etiketli veri olmadan eğitilemez ve açıklanabilirliği ortadan kaldırırdı.

---

## ADR-002: Kararlı eşleştirme değil, maksimum toplam ağırlık

**Bağlam.** "Toplam uyumu maksimize et" hedefi seçildi ve oda kapasitesi iki kişiye sabitlendi.

**Karar.** Bu iki koşul birlikte problemi tam olarak *maximum weight perfect matching* yapıyor. Edmonds' blossom algoritması (`networkx.max_weight_matching`) bunu kesin optimal olarak, polinom zamanda çözer. Varsayılan strateji bu.

**Neden Irving'in kararlı oda arkadaşı algoritması varsayılan değil.** Kararlılık farklı bir hedef: kimsenin eşini bırakıp başkasına kaçmak istememesi. Toplam uyumu maksimize etmekle aynı cevabı vermez ve ölçümde vermiyor da: optimal çözüm toplamda açgözlü/kararlı çözümlerden %0.7 önde ama 35 blocking pair içeriyor, onlarda ise sıfır.

Irving algoritması silinmedi, bir **karşılaştırma stratejisi** olarak kaldı. İki hedef arasındaki farkı göstermek analiz odaklı bir uygulama için değerli.

**`maxcardinality=True` seçimi.** Önce kardinaliteyi (herkesin eşlenmesini), sonra toplam ağırlığı maksimize eder. Yurtta öncelik sırası tam olarak bu: herkesin yatağı olacak, sonra en uyumlu bölüşüm seçilecek.

**Doğrulama.** Dört öğrencilik bir örnekte üç olası tam eşleştirmenin hepsi elle sayılıyor ve çözücünün en iyisini bulduğu kanıtlanıyor.

---

## ADR-003: Kesin kısıt yerine büyük ceza

**Sorun.** Sıralama dünyasında bir filtre listeyi daraltır. **Zorunlu yerleşimde** ise çözümü tamamen imkânsız kılabilir: "sigara içen ile içmeyen aynı odada olmaz" kuralını mutlak uygularsan ve sigara içenlerin sayısı tekse, tam eşleştirme matematiksel olarak var olmayabilir. Bir öğrenciyi yatakszı bırakmak kabul edilebilir bir çıktı değil.

**Karar.** İhlaller kenar ağırlığından `VIOLATION_PENALTY = 1000` düşürüyor. Skorlar `[0,1]` aralığında olduğu için ceza her zaman baskın.

**Sonuç.** Üç özellik birden sağlanıyor:
- Çözüm her zaman bulunur, kimse yerleşimsiz kalmaz.
- Çözücü kaçınılabilir ihlalden kendiliğinden kaçınır.
- Kaçınılamayan ihlaller planda türüne göre raporlanır.

**Doğrulanmış davranış.** 2 içen + 2 içmeyen → içenler birlikte, sıfır ihlal. 1 içen + 3 içmeyen → herkes yerleşir, tam 1 (minimum) ihlal. İkisi de test.

**Filtre tanımı tek yerde.** `violations_for()` hem aday analizinde filtre olarak hem yerleşimde ceza üretici olarak kullanılıyor, böylece iki görünüm neyin ihlal olduğu konusunda asla ayrışamaz.

---

## ADR-004: Benzerlik değil, uyumluluk

**Karar.** Boyut fonksiyonları benzerlik değil uyumluluk ölçer. Ayrım iki yerde somutlaşıyor:

- **Temizlik:** ikisi de 1/5 ise benzerlik tam, ama oda yaşanmaz olur. Skor `×0.7` ile cezalandırılır.
- **Ders çalışma yeri:** ikisi de odada çalışıyorsa "aynı" olmaları avantaj değil; aynı masayı ve aynı sessizliği aynı saatte istiyorlar. Bu eşleşme 0.45 alır, oda + kütüphane ise 1.0.

**Gerekçe.** Hedef birbirine benzeyen öğrencileri bulmak değil, birlikte yaşayabilecek öğrencileri bulmak. Bu iki fonksiyonun ayrı testleri var.

**Test edilebilir değişmez.** Aynılığı cezalandıran iki boyut yüzünden "özdeş profiller 1.0 alır" doğru bir beklenti değil. Testlerin dayandığı değişmez şu: *bir klon her zaman yenilmez bir oda arkadaşıdır*: skor 1.0 olmasa da hiçbir alternatif onu geçemez.

---

## ADR-005: Değerlendirme için kural tabanlı oracle

**Sorun.** Gerçek etiket yok. "Doğru eşleşme" verisi olmadan metrikler karşılaştırılamaz.

**Karar.** Bağımsız ve basit bir oracle: bir yurt görevlisinin yüksek sesle söyleyeceği kısıtlardan kuruluyor (evde sigara içilmesin, temizlik farkı en fazla 1, uyku düzenleri zıt olmasın, ikisi de odada çalışmasın, sınıf farkı en fazla 2, sigara alerjisine uyulsun).

**Bağımsızlık.** Oracle, Gower skorlayıcısının kendisi değil. Olsaydı model kendi çıktısına göre notlandırılmış olur ve karşılaştırma bir şey ölçmezdi.

**İki eksende kullanılıyor.** Sıralama tarafında precision/nDCG için ground truth; yerleşim tarafında ise `oracle_share` olarak, planın kaç odasının "yaşanabilir" sayıldığını ölçüyor. Aynı ölçütün iki tarafta da kullanılması karşılaştırmaları tutarlı kılıyor.

**Sınırı.** Oracle sentetik ve basit. Mutlak rakamlar değil, metrikler ve stratejiler arası sıralama anlam taşıyor.

---

## ADR-006: Ölçeklenebilirlik ve kesin çözümün sınırı

**Ölçüm.** Blossom pure-Python ve O(n³); skor matrisi O(n²).

| Öğrenci | Skor matrisi | Blossom | Toplam |
|---:|---:|---:|---:|
| 100 | 0.27 s | 0.29 s | 0.56 s |
| 200 | 1.09 s | 2.32 s | 3.41 s |
| 400 | 4.49 s | 20.07 s | 24.56 s |
| 800 | 17.86 s | 168.42 s | 186.28 s |

**Karar.** Grafik seyreltilmiyor; her öğrenci için en iyi K adayı tutup gerisini atmak akla gelen ilk hızlandırma ama tercih edilmiyor:

- Tipik bir yurt bloğu 200-600 öğrenci; kesin çözüm bu aralığı zaten kapsıyor.
- Seyreltme "optimal" iddiasını düşürür; ki projenin en güçlü teknik iddiası bu.
- Nadiren devreye girecek, test edilmesi zor ve ana garantiyi sessizce bozan bir kaçış yolu, sınırı belgelemekten kötü.

Sınır şu: 800 öğrencide 3 dakika toplu iş olarak kabul edilebilir, bunun üstünde kesin çözüm pratik değil. 1000+ ölçeği yol haritasında ANN ile ön eleme olarak duruyor.

**Skor matrisi paylaşımı.** Strateji karşılaştırması matrisi bir kez kurup dört stratejiye veriyor, böylece ölçüm stratejileri karşılaştırıyor, skorlamayı değil.

---

## ADR-007: Simetrik skorlama

**Karar.** Skor tek yönlü hesaplanıyor: `score(a, b)`. İki yönü ayrı hesaplayıp harmanlayan bir yapı (`(1-α)·score(a,b) + α·score(b,a)`) kurulmuyor.

**Gerekçe.** Oda paylaşımı tek taraflı bir düzen değil, ama 15 boyutun tamamı tanımı gereği simetrik olduğu için `score(a,b) == score(b,a)` her zaman doğru. Harman hiçbir şey değiştirmez, yalnızca her çift için iki kat hesap yapar.

Simetrinin boyut tasarımından gelmesi tesadüf değil: tek taraflı olabilecek tek şey "biri diğerine katlanabilir mi" sorusu, o da bir skor değil **kısıt**. Alerji bu yüzden boyut değil, `matching/constraints.py` içinde bir kısıt.

**Sonucu.** Skor matrisi yalnızca üst üçgeni hesaplayıp aynalıyor, yani popülasyon başına iş yarıya iniyor.

**Doğrulama.** Simetri hem 15 boyut için ayrı ayrı hem de altı metriğin tamamı için test ediliyor, ayrıca Hypothesis ile özellik tabanlı olarak.

---

## ADR-008: Oda, envanter değil çıktı

**Karar.** `Room` önceden tanımlı bir kaynak değil, eşleştirmenin ürettiği bir sonuç: oda numarası, iki öğrenci, skor ve gerekçe.

**Gerekçe.** Kapsam tek cinsiyet, blok/kat ayrımı yok ve kapasite sabit iki olarak belirlendi. Bu koşullarda bir oda envanteri hiçbir kısıt getirmiyor; kapasite kontrolü, cinsiyet tahsisi, erişilebilirlik kontrolü hepsi gereksizleşiyor. Odayı çıktı olarak modellemek domaini önemli ölçüde sadeleştiriyor.

**Sonucu.** Odalar skora göre sıralanıp 1'den numaralanıyor, yani "Oda 1" her zaman en uyumlu oda. Yurt yönetimi için en kötü odalar da `worst_rooms()` ile ayrıca listelenir; müdahale edilecek yer orası.

---

## ADR-009: Depolama arkasında repository, plan kalıcı

**Karar.** Her şey `StudentRepository` protokolüyle konuşur. Üç uygulama var: bellek içi (testler), CSV (deneyler), SQLAlchemy (SQLite/PostgreSQL).

**Gerekçe.** Yerleşim mantığının verinin nerede durduğunu bilmesi gerekmiyor. Testler dosya sistemine dokunmadan bellek içi depoyla çalışıyor, üretim `RM_DATABASE_URL` ile PostgreSQL'e geçiyor, arada kod değişmiyor.

**Plan da kalıcı.** Hesaplanan plan `plans` ve `rooms` tablolarına yazılıyor, böylece 25 saniyelik bir hesap bir kez yapılıp sonra incelenebiliyor, açıklanabiliyor ve API'den okunabiliyor.

**Üretilen veri ayrı tutuluyor.** Sentetik popülasyon `data/generated/` altına yazılır ve depo bunun dışına yazmaz; tohum ve üretici verildiğinde aynı popülasyon her zaman yeniden kurulabilir, dolayısıyla ölçüm sonuçları tekrarlanabilir kalır.

---

## ADR-010: Sentetik veri korelasyonlu üretilir

**Karar.** Üretici, bağımsız düzgün dağılımlar yerine ilişkili özellikler üretir: gece kuşları daha yüksek gürültü toleransına sahip ve daha geç çalışıyor, odada çalışanlar odada daha çok akşam geçiriyor ve daha çok sessizlik istiyor, sosyal öğrenciler daha sık misafir ağırlıyor, sınıf ile yaş birlikte artıyor.

**Gerekçe.** Bağımsız gürültüden oluşan bir popülasyonda keşfedilecek bir yapı yoktur, dolayısıyla hiçbir metrik diğerinden iyi çıkamaz. Değerlendirmenin anlam taşıması için popülasyonun içsel tutarlılığa sahip olması gerekiyor.

**Doğrulama.** Korelasyonlar teste bağlı, aksi halde bir değişiklikte sessizce kaybolabilirler.

---

## ADR-011: Yalnızca yönetici kimliği

**Karar.** Tek rol var: yurt personeli. Öğrenciler API kullanıcısı değil, veritabanı kaydı. Kayıt uç noktası yok.

**Gerekçe.** Kapsam gereği öğrenci etkileşimi ve talep mekanizması yok; sistem tamamen yönetici tarafında çalışıyor. Öğrenci kimlik doğrulaması eklemek kullanılmayacak bir saldırı yüzeyi ve bakım yükü olurdu.

**Sonucu.** Okuma uç noktaları açık, yazma (öğrenci ekleme, plan hesaplama) yönetici girişi gerektiriyor. İlk hesap `roommate create-admin` ile oluşturuluyor.

**Ağırlıklar sabit.** Öğrenci etkileşimi olmadığı için ağırlıkları geri bildirimden öğrenecek bir veri kaynağı yok; `DEFAULT_WEIGHTS` elle ayarlanmış değerler taşıyor. Yurt yönetiminin geçmiş dönem odalarını "sorunsuz / şikayet geldi" olarak işaretlemesi gerçekçi bir girdi olurdu ve yol haritasında duruyor.

---

## Katman kuralları

```
domain/     → hiçbir iç modülü import etmez
features/   → yalnızca domain
matching/   → domain + features
data/       → domain + features
api/ cli/ viz/ app/ → sunum; iş mantığı içermez
```

CLI, API ve Streamlit arayüzü aynı `CompatibilityEngine` ve `RoomAssigner`'ı çağırdığı için aralarında davranış farkı oluşamaz.
