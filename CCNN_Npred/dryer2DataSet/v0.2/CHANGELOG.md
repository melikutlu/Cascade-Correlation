# Değişiklik Kaydı

## v0.2 — 2 Mart 2026

### Temel Değişiklik: Önceki Çıkış Ağırlıklarının Dondurulması

**Kaynak:** `dryer2DataSet/v0.1/Npred_MiniBatch_Adam_maxCandidate.m`  
**Hedef:**  `dryer2DataSet/v0.2/Npred_MiniBatch_Adam_maxCandidate.m`

#### Ne değişti?

| Konu | v0.1 | v0.2 |
|------|------|------|
| Hidden eklenince çıkış eğitimi | Tüm `w_o` vektörü yeniden eğitilir | `w_o_frozen` dondurulur, sadece yeni scalar `w_new` eğitilir |
| Stage-1 (gizli birim yok) | Tüm çıkış ağırlıkları serbest eğitilir | Aynı (değişiklik yok) |
| Adam state sıfırlanması | Her hidden sonrası tamamen sıfırlanır | Her hidden sonrası tamamen sıfırlanır (aynı) |

#### Yeni eklenen fonksiyonlar

- **`trainOutputLayer_FrozenPrev`**  
  `w_o_frozen` normal dizi olarak alır (gradyan akmaz), sadece yeni `w_new` (1×1 scalar) dlarray olarak eğitilir.

- **`loss_output_frozen`**  
  `dlgradient` yalnızca `w_new`'e göre hesaplanır. `w_o_frozen` MATLAB'ın autograd mekanizması dışında tutulmuş olur.

#### Motivasyon

Orijinal Cascade-Correlation algoritmasında (Fahlman & Lebiere, 1990) yeni bir hidden unit eklendiğinde mevcut ağırlıklar korunur; sadece yeni bağlantılar eğitilir. Bu versiyon bu davranışı çıkış katmanına uygulamaktadır:

```
w_o = [w_o_frozen ; w_new]
         ↑               ↑
    (dondurulmuş)   (eğitiliyor)
```

`w_o_frozen`, bir sonraki hidden unit eklendiğinde tekrar yeni `w_o_frozen` haline gelir.

#### Config değişiklikleri (v0.1'e göre)

| Parametre | v0.1 | v0.2 |
|-----------|------|------|
| `regressors.u` | `[0]` | `[1,2,3,4]` (dead time düzeltmesi) |
| `regressors.y` | `[1]` | `[1,2,3,4]` |
| `max_epochs_output` | 10 | 500 |
| `eta_output` | 0.008 | 0.005 |
| `max_epochs_candidate` | 10 | 300 |
| `eta_candidate` | 0.005 | 0.003 |
| `max_hidden_units` | 15 | 20 |
| `target_mse` | 1e-3 | 5e-4 |
| `min_mse_improvement` | 1e-5 | 1e-6 |

---

## v0.1 — Şubat 2026

- İlk versiyon: dryer2 dataseti için N-step trajectory eğitimi
- Mini-batch Adam optimizer
- Candidate pool ile en iyi aday seçimi (`candidateCorrelationMetric`)
- Parametre log dosyası ve figure kaydetme altyapısı
