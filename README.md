# Akıllı Rehabilitasyon Eldiveni — AI Öneri Sistemi

Felç ve nörolojik rehabilitasyon için taşınabilir akıllı eldiven projesinin bilgisayar mühendisliği bileşeni. Eldiven sensör verilerini makine öğrenmesi ile analiz eder ve kişiselleştirilmiş terapi önerileri üretir.

## Özellikler

- **Canlı simülasyon**: Gerçek eldiven olmadan sensör verisini replay modunda oynatır
- **Sentetik veri üreticisi**: Çoklu hasta profili ve seanslar arası iyileşme verisi üretir (demo/sunum için)
- **ML sınıflandırma**: Hareket fazı ve klinik yorum tahmini (RandomForest)
- **Kural tabanlı öneri motoru**: Yorgunluk, kavrama gücü ve ilerlemeye göre terapi önerileri
- **Profesyonel Türkçe arayüz**: Sekmeli yapı (Canlı İzleme / Seans Geçmişi / Klinik Rapor), KPI kartları, hasta başlık şeridi
- **Klinik rapor dışa aktarımı**: İndirilebilir HTML (Ctrl+P ile PDF), PNG grafikler ve CSV

## Kurulum

Windows'ta `pip` komutu PATH'te olmayabilir. Bu durumda Python launcher kullanın:

```powershell
py -m pip install -r requirements.txt
```

Alternatif:

```powershell
python -m pip install -r requirements.txt
```

## Model Eğitimi

```powershell
py -m src.train
```

Modeller `models/` klasörüne kaydedilir. Uygulama ilk açılışta modeller yoksa otomatik eğitir.

## Uygulamayı Çalıştırma

```powershell
py -m streamlit run app.py
```

`streamlit` komutu da tanınmıyorsa yukarıdaki `py -m streamlit` formunu kullanın.

Tarayıcıda `http://localhost:8501` adresinde açılır.

## Sentetik Demo Verisi Üretme (opsiyonel)

Arayüzdeki **"Demo verisi üret"** butonu verileri anlık üretir. İsteğe bağlı olarak diske de kaydedebilirsiniz:

```powershell
py -m src.data_generator
```

Bu komut `data/generated/` altına her hasta için seans CSV'leri ve `*_history.csv` ilerleme dosyaları yazar.

## Arayüz Sekmeleri

- **Canlı İzleme**: Seans replay'i, canlı sensör grafikleri, anlık AI analizi ve öneri kartı, seans özeti
- **Seans Geçmişi**: Seanslar arası ilerleme grafiği (puan/kuvvet/yorgunluk), KPI kartları, seans tablosu
- **Klinik Rapor**: İndirilebilir HTML rapor (Ctrl+P → PDF), PNG grafikler ve CSV dışa aktarımı

## Hasta Profilleri (sentetik)

| Hasta | Durum | Özellik |
|-------|-------|---------|
| Ahmet Y. | İnme sonrası | Düşük başlangıç, yavaş iyileşme, yüksek yorgunluk |
| Elif K. | Ortopedik | Orta-yüksek başlangıç, hızlı iyileşme |
| Mehmet D. | Nörolojik | Orta başlangıç, kademeli iyileşme |

## Demo Senaryoları (state filtresi)

| Senaryo | Beklenen AI Davranışı |
|---------|----------------------|
| Normal Seans | Tüm fazlar + seans sonu özeti |
| Başarılı Kavrama | "Güçlü kavrama, tekrar artır" önerisi |
| Yorgunluk | "Hedefi düşür, dinlen" uyarısı |
| Dinlenme | Dinlenme fazı bilgisi |

## Proje Yapısı

```
kmtproje/
├── app.py                  # Streamlit arayüzü (sekmeli)
├── data/
│   ├── rehab_glove_sample_300.csv
│   └── generated/          # Üretilen sentetik seanslar (opsiyonel)
├── models/                 # Eğitilmiş modeller (joblib)
├── src/
│   ├── features.py         # Özellik mühendisliği
│   ├── train.py            # Model eğitimi
│   ├── predict.py          # Anlık tahmin
│   ├── recommendation.py   # Öneri motoru
│   ├── simulator.py        # Replay (CSV + bellek içi DataFrame)
│   ├── data_generator.py   # Sentetik çoklu seans üreticisi
│   ├── charts.py           # Plotly grafik yardımcıları
│   └── report.py           # Klinik rapor (HTML/PNG)
├── requirements.txt
└── README.md
```

## Mimari

```
Sentetik Üretici / CSV → Simülatör → Özellik Çıkarımı → ML Modelleri
                                                          ↓
                                              Öneri Motoru (Kurallar)
                                                          ↓
                            Streamlit Arayüzü → Klinik Rapor (HTML/PDF/PNG)
```

## Sınırlamalar

- Eğitim verisi tek gerçek seans (300 satır); model augmentasyonu ve sentetik hasta verisi demo amaçlıdır, klinik genelleme yapmaz
- Gerçek eldiven entegrasyonu için `simulator.py` BLE/serial stream ile değiştirilebilir; arayüz ve öneri motoru aynı kalır
- Çoklu gerçek hasta verisi geldiğinde modeller yeniden eğitilmelidir
- PNG dışa aktarım `kaleido` paketini gerektirir; kurulu değilse rapor yalnızca HTML olarak üretilir

## Geliştirici Notları

Hasta profili (sidebar):
- Hedef günlük tekrar (varsayılan 15)
- Seans numarası
- Zorluk seviyesi (kolay / orta / zor)

Öneri örnekleri:
- *"Bugün 15 tekrar yerine 18 tekrar yapın; ilerleme %12 arttı."*
- *"Yorgunluk tespit edildi. Hedefi %20 düşürün, 2 dk dinlenme."*
