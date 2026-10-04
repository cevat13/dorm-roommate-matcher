"""Keşifsel analiz, metrik ve strateji karşılaştırması.

`jupytext` uyumlu "percent" formatında: hem `python notebooks/01_analysis.py` ile
doğrudan çalışır, hem de Jupyter veya VS Code içinde hücre hücre çalıştırılabilir.
`.ipynb` yerine düz Python tutuluyor, çünkü notebook çıktıları diff'i okunmaz hale
getiriyor.

Notebook'a çevirmek için::

    pip install jupytext && jupytext --to notebook notebooks/01_analysis.py
"""

# %% [markdown]
# # Yurt Oda Arkadaşı Eşleştirme: Keşifsel Analiz
#
# Dört soru:
#
# 1. Sentetik popülasyonda keşfedilecek bir yapı var mı?
# 2. Kosinüs benzerliği bu veride neden zayıf kalıyor?
# 3. Ağırlıklı Gower ne kadar daha iyi?
# 4. Optimal yerleşim, açgözlü ve rastgele yaklaşımlardan ne kazandırıyor?

# %%
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from roommate_matcher.data.generator import generate_students
from roommate_matcher.data.serialization import to_record
from roommate_matcher.domain.enums import StudyLocation
from roommate_matcher.features.vectorizer import vectorize
from roommate_matcher.matching.assignment import OptimalAssignment, RoomAssigner
from roommate_matcher.matching.engine import CompatibilityEngine
from roommate_matcher.matching.evaluation import compare_metrics, evaluate_plans, relevant_set
from roommate_matcher.matching.metrics import get_metric

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs"
SAMPLE_SIZE = 400

students = generate_students(SAMPLE_SIZE, seed=2026)
frame = pd.DataFrame([to_record(student) for student in students])
print(f"{len(frame)} öğrenci, {frame.shape[1]} alan")

# %% [markdown]
# ## 1. Popülasyonda yapı var mı?
#
# Bağımsız düzgün dağılımlardan üretilen bir veri setinde keşfedilecek bir yapı
# olmaz ve hiçbir metrik diğerinden iyi çıkmaz. Üretici bu yüzden ilişkili
# özellikler kuruyor; aşağıdaki tablolar bu ilişkileri doğruluyor.

# %%
print("Sınıf dağılımı:")
print(frame["study_year"].value_counts().sort_index().to_string(), "\n")

print("Uyku düzenine göre ortalama gürültü toleransı:")
print(frame.groupby("sleep_schedule")["noise_tolerance"].agg(["mean", "count"]).round(2), "\n")

print("Ders çalışma yerine göre odada geçen akşam sayısı:")
print(frame.groupby("study_location")["room_time_weekdays"].agg(["mean", "count"]).round(2), "\n")

print("Uyku düzeni ve çalışma saati çapraz tablosu:")
print(pd.crosstab(frame["sleep_schedule"], frame["study_time"]))

# %% [markdown]
# Gece kuşlarının gürültü toleransı daha yüksek ve geç saatte çalışıyorlar; odada
# çalışanlar odada daha çok akşam geçiriyor.

# %%
numeric_columns = [
    "age",
    "study_year",
    "cleanliness",
    "noise_tolerance",
    "social_energy",
    "room_time_weekdays",
]
print("Sayısal özellikler arası korelasyon:")
print(frame[numeric_columns].corr().round(2))

# %% [markdown]
# ## 2. Kosinüs bu veride neden zayıf?
#
# İki sorun var. Birincisi: kosinüs **yön** ölçer, bu yüzden iki öğrencinin de
# "hayır" dediği ikili bir özellik benzerliğe katkı vermez. Oysa "ikimiz de sigara
# içmiyoruz" bir yurt odasında güçlü bir uyum sinyali.

# %%
left = np.array([0.0, 0.0, 1.0])  # sigara yok, alerji yok, erken yatıyor
right = np.array([0.0, 0.0, 1.0])  # birebir aynı
different = np.array([1.0, 1.0, 1.0])  # tek ortak nokta son alan


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Ham kosinüs benzerliği."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"Birebir aynı iki profil        : {cosine(left, right):.3f}")
print(f"Yalnız bir alanı ortak olanlar : {cosine(left, different):.3f}")
print("\nİkisi de 1.0 çıkıyor: kosinüs, paylaşılan sıfırları hiç görmüyor.")

# %% [markdown]
# İkincisi ölçekleme. Kodlayıcı her sütunu `[0,1]` aralığına çekiyor, böylece geniş
# aralıklı tek bir sütun mesafeyi domine edemiyor.

# %%
space = vectorize(students)
ranges = pd.DataFrame(
    {
        "feature": space.feature_names,
        "min": space.matrix.min(axis=0).round(3),
        "max": space.matrix.max(axis=0).round(3),
        "std": space.matrix.std(axis=0).round(3),
    }
)
print(f"Tüm sütunlar [{space.matrix.min():.2f}, {space.matrix.max():.2f}] aralığında.")
print(ranges.nlargest(8, "std").to_string(index=False))

# %% [markdown]
# ## 3. Metrikler ne kadar farklı?
#
# Değerlendirme kural tabanlı bir oracle'a karşı yapılıyor: evde sigara içilmesin,
# temizlik farkı en fazla 1, uyku düzenleri zıt olmasın, ikisi de odada çalışmasın,
# sınıf farkı en fazla 2. Oracle, Gower skorlayıcısının kendisi değil; olsaydı model
# kendini notlandırırdı.

# %%
report = compare_metrics(students, sample=120)
results = pd.DataFrame(report.rows())
print(results.to_string(index=False))

best = report.winner
baseline = next(s for s in report.scores if s.metric == "cosine")
if best is not None:
    print(f"\nEn iyi: {best.metric} (nDCG@10 = {best.ndcg_at_10:.4f})")
    print(f"Kosinüse göre kazanç: {best.ndcg_at_10 / baseline.ndcg_at_10:.2f}x")

results.to_csv(OUTPUT_DIR / "metric_comparison.csv", index=False)

# %% [markdown]
# Tek bir öğrenci için iki metriğin ürettiği ilk beş adayı yan yana koyalım.

# %%
target = students[0]
print(f"Hedef: {target.summary()}\n")
relevant = relevant_set(target, students)

for key in ("weighted_gower", "cosine"):
    engine = CompatibilityEngine(get_metric(key))
    candidates = engine.candidates_for(
        target, students[1:], top_n=5, explain=False, apply_constraint_filter=False
    )
    rows = [
        {
            "sıra": rank,
            "id": pair.second_id,
            "skor": round(pair.score, 3),
            "oracle": "✓" if pair.second_id in relevant else "✗",
        }
        for rank, pair in enumerate(candidates.results, start=1)
    ]
    print(f"--- {key} ---")
    print(pd.DataFrame(rows).to_string(index=False), "\n")

# %% [markdown]
# ## 4. Yerleşim: optimal ne kazandırıyor?
#
# Oda kapasitesi iki kişi ve hedef toplam uyumu maksimize etmek olduğunda problem
# tam olarak *maximum weight perfect matching*; Blossom algoritması kesin optimal
# çözümü veriyor. Aşağıda aynı skor matrisi üzerinde dört strateji karşılaştırılıyor.

# %%
subset = students[:200]
plan_scores = evaluate_plans(subset)
plans = pd.DataFrame([s.as_row() for s in plan_scores])
print(plans.to_string(index=False))
plans.to_csv(OUTPUT_DIR / "strategy_comparison.csv", index=False)

# %% [markdown]
# Toplam skorda fark küçük görünüyor, ama **en kötü oda** ve **oracle oranı**
# sütunlarında belirgin: açgözlü yaklaşım güçlü çiftleri önce kapıp artakalanları
# birbirine mahkûm ediyor. Optimal çözüm toplamı maksimize ederken en kötü durumdaki
# öğrenciyi de dolaylı olarak koruyor.

# %%
optimal_row = next(row for row in plan_scores if row.strategy == "optimal")
greedy_row = next(row for row in plan_scores if row.strategy == "greedy")
print(f"Toplam uyum farkı      : {greedy_row.gap_percent:.2f}%")
print(f"En kötü oda iyileşmesi : {(optimal_row.min_score / greedy_row.min_score - 1) * 100:.1f}%")
print(
    "Oracle oranı farkı     : "
    f"{(optimal_row.oracle_share - greedy_row.oracle_share) * 100:.1f} puan"
)
print(
    f"Kararlılık bedeli      : {optimal_row.blocking_pairs} blocking pair "
    f"(açgözlüde {greedy_row.blocking_pairs})"
)

# %% [markdown]
# ## 5. Ölçeklenebilirlik
#
# İki maliyet var: skor matrisi O(n²) çift, Blossom ise O(n³). Aşağıdaki rakamlar
# ölçülmüş değerler.

# %%
print(f"{'n':>6} {'çift':>9} {'matris s':>10} {'blossom s':>11} {'toplam s':>10}")
for size in (100, 200, 400):
    sample = generate_students(size, seed=7)
    engine = CompatibilityEngine()
    start = time.perf_counter()
    matrix = engine.score_matrix(sample)
    middle = time.perf_counter()
    OptimalAssignment().pair_up(matrix)
    end = time.perf_counter()
    print(
        f"{size:6} {size * (size - 1) // 2:9} "
        f"{middle - start:10.2f} {end - middle:11.2f} {end - start:10.2f}"
    )

# %% [markdown]
# ## 6. Planın kendisi
#
# Son olarak bir planın dağılımına ve en zayıf odalarına bakalım; yurt yönetiminin
# gerçekten müdahale edeceği yer burası.

# %%
plan = RoomAssigner("optimal").assign(subset)
print(f"Oda: {plan.room_count} | toplam: {plan.total_score:.4f} | ortalama: {plan.mean_score:.4f}")
print(f"En kötü oda: {plan.min_score:.4f} | yerleşmeyen: {len(plan.unplaced)}")
print(f"Kısıt ihlali: {plan.violation_count} | blocking pair: {len(plan.blocking_pairs)}\n")

print("En zayıf üç oda:")
for room in plan.worst_rooms(3):
    print(f"  {room.label()}")
    concerns = ", ".join(c.label for c in room.pair.concerns(3))
    print(f"    Dikkat: {concerns or '-'}")

print(
    "\nOdada çalışan öğrenci sayısı:",
    sum(1 for s in subset if s.study_location is StudyLocation.ROOM),
)
print(f"Kaydedildi: {OUTPUT_DIR / 'strategy_comparison.csv'}")
