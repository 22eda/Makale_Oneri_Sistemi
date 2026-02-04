# ScholarMind: Akademik Makale Öneri Sistemi

ScholarMind, akademik literatürdeki "bilgi aşırı yüklemesi" (information overload) problemini hafifletmek amacıyla geliştirilmiş, derin öğrenme tabanlı bir hibrit öneri sistemi prototipidir. Proje, makalelerin anlamsal derinliğini analiz ederek araştırmacılara en alakalı yayınları kişiselleştirilmiş bir deneyimle sunar.

Streamlit arayüz: https://makaleonerisistemi-d3v4abrbbtnmzjurcherxz.streamlit.app

##  Proje Genel Bakış
Bilimsel yayınların üstel artışı, araştırmacıların kendi alanlarındaki en güncel ve alakalı çalışmaları bulmasını zorlaştırmaktadır. ScholarMind, geleneksel anahtar kelime tabanlı yöntemlerin (TF-IDF) sınırlılıklarını aşmak için Sentence-BERT (SBERT) mimarisini kullanarak metinleri anlamsal bir vektör uzayında modeller.

### Temel Özellikler
* **Semantik Arama:** Makale başlık ve özetlerini 384 boyutlu vektörlere dönüştürerek anlamsal benzerlik yakalama.
* **Çok Kriterli Hibrit Filtreleme:** İçerik alakasını (%40), kullanıcı profilini (%40) ve makale popülaritesini (%20) harmanlayan özgün bir skorlama algoritması.
* **Kişiselleştirilmiş Kütüphane:** Kullanıcının ilgi duyduğu makalelere göre dinamik profil oluşturma ve öneri listesini buna göre manipüle etme.
* **İnteraktif Web Arayüzü:** Streamlit ile geliştirilmiş, kullanıcıların anlık sorgular yapabildiği ve popüler makaleleri inceleyebildiği kullanıcı dostu panel.

## 🛠️ Teknik Altyapı
* **Dil:** Python 3.12
* **Geliştirme Ortamı:** Kaggle Notebook (Jupyter tabanlı)
* **Ana Kütüphaneler:**
    * `sentence-transformers`: SBERT "all-MiniLM-L6-v2" modeli için.
    * `streamlit`: Web arayüzü tasarımı için.
    * `scikit-learn`: Kosinüs benzerliği ve TF-IDF hesaplamaları için.
    * `pandas` & `numpy`: Veri manipülasyonu ve matris işlemleri için.



##  Hibrit Skorlama Modeli
Sistem, nihai öneri sıralamasını aşağıdaki matematiksel formül üzerinden hesaplar:

$$\text{Final Score} = (0.4 \times \text{Content Sim}) + (0.4 \times \text{User Profile Sim}) + (0.2 \times \text{Popularity})$$

##  Veri Seti Yapısı
Çalışmada Cornell Üniversitesi tarafından sağlanan **arXiv Dataset** kullanılmıştır.
* **Örneklem:** 6 ana disiplinden (cs, math, astro-ph, cond-mat, physics, eess) 2.000'er adet olmak üzere toplam **12.000 makale**.
* **Ön İşleme:** Başlık ve özet alanları birleştirilmiş, stop-word temizliği ve normalizasyon uygulanmıştır.

##  Performans ve Karşılaştırma
Yapılan nicel testlerde (Top-10 öneri üzerinden) SBERT tabanlı modelin üstünlüğü kanıtlanmıştır:

| Model | Ortalama Cosine Similarity | Kategori Hit Rate |
| :--- | :---: | :---: |
| **TF-IDF (Baseline)** | 0.45 | %72 |
| **SBERT** | **0.57** | **%83** |
| **Hibrit Model** | 0.44 | %50 |

##  Nitel Değerlendirme (Senaryolar)
"Deep Learning for Image Recognition" sorgusu için model çıktıları:

1. **TF-IDF:** Kelime eşleşmesine odaklanarak sorguyla ilgisiz "Speech Recognition" makalelerini listeleyebilmektedir.
2. **SBERT:** Başlıkta geçmese dahi anlamsal olarak ilgili "Neural Networks" ve "ResNet" çalışmalarını başarıyla bulmaktadır.
3. **Hibrit:** Kullanıcının geçmiş ilgi alanı olan "Astrofizik" (astro-ph) makalelerini öneri listesinde önceliklendirmektedir.

##  SWOT Analizi
* **Güçlü Yönler:** Anlamsal derinlik, kişiselleştirme, soğuk başlangıç sorununu minimize etme.
* **Zayıf Yönler:** SBERT için yüksek GPU maliyeti, gerçek etkileşim verisi eksikliği.

##  Referans
Bu proje, **Öz vd. (2021)** tarafından geliştirilen içerik tabanlı model prototipi temel alınarak, derin öğrenme ve popülarite katmanlarıyla zenginleştirilmiştir.

---
**Hazırlayan:** Edanur DEMİREL  
**Bilecik Şeyh Edebali Üniversitesi - İstatistik ve Bilgisayar Bilimleri**
