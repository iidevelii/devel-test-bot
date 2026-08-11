# 🔑 بيانات الربط وإعدادات مشروع devel-test-bot

هذا الملف يحتوي على كافة مفاتيح الربط، التوكنات، وروابط الخدمات الخاصة بـ **devel-test-bot** للربط المباشر مع GitHub و Railway وتيليجرام.

---

## 1. 🐙 بيانات GitHub
* **الرابط الرئيسي (Repo URL)**: `https://github.com/iidevelii/devel-test-bot.git`
* **الفرع (Branch)**: `main`
* **رمز الوصول الشخصي (Personal Access Token)**:
  * مفتاح PAT المخصص لـ GitHub مرفق بحسابك `iidevelii`
* **أوامر الربط المباشرة مع GitHub**:
  ```bash
  git remote set-url origin https://<YOUR_GITHUB_TOKEN>@github.com/iidevelii/devel-test-bot.git
  git push origin main
  ```


---

## 2. 🤖 بيانات بوت وتيليبجرام (Telegram Credentials)
* **توكن البوت (Bot Token)**:
  ```
  8860361053:AAFdb2zkeYIB-0W2h38ex4iarfud72MH1Fg
  ```
* **معرف القناة الاختيارية (Test Channel ID)**:
  ```
  -1004310116394
  ```
* **اسم القناة**: `DevelTest1`

---

## 3. 🚆 إعدادات السيرفر والاستضافة (Railway Environment Variables)
عند إضافة المشروع على Railway، تأكد من ضبط المتغيرات التالية في قسم **Variables**:

```env
TEST_BOT_TOKEN=8860361053:AAFdb2zkeYIB-0W2h38ex4iarfud72MH1Fg
TEST_CHANNEL_ID=-1004310116394
DM_MIN_SCORE=5.0
DM_RR_FUTURES=1.5
DM_RR_SPOT=1.3
ENABLE_FUTURES=true
ENABLE_SPOT=true
SCAN_INTERVAL_H=4
```

---

## 📁 4. هيكل الملفات والمكونات الرئيسية
* `main.py`: البوت الرئيسي والماكينة التي تفحص الصفقات وتتتبع الأهداف والستوب وتنبيهات القرب.
* `devel_master_strategy.py`: المحرك الأساسي للاستراتيجية (Classic, ICT, Wyckoff, Candles, Momentum).
* `mtf_futures_engine.py`: محرك الفيوتشر الدقيق متعدد الأطر الزمنية (1H Bias + 5M Entry).
* `chart_generator.py`: محرك رسم الشارتات الفنية والتنبيهات وشارتات النتائج وقرب الهدف/الستوب.
* `requirements.txt`: المكتبات اللازمة للتشغيل (python-telegram-bot, matplotlib, requests, numpy).
* `Procfile`: ملف تشغيل الخدمة على Railway (`worker: python main.py`).
