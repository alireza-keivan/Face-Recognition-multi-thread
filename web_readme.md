#پروژه سیستم تشخیص چهره با استفاده از InsightFace و MQTT

این داکیومنت راهنمای جامع نصب، راه‌اندازی و پیکربندی سیستم تشخیص چهره با استفاده از کتابخانه‌های InsightFace و OpenCV است. این سیستم قابلیت تشخیص چهره از یک منبع ویدیویی (وب‌کم یا استریم RTSP) را دارد و اطلاعات چهره‌های شناسایی شده را از طریق پروتکل MQTT منتشر می‌کند.

-----

## ۱. پیش‌نیازها و آماده‌سازی محیط

قبل از شروع، باید مطمئن شوید که پایتون و ابزارهای مورد نیاز برای مدیریت پکیج‌ها روی سیستم شما نصب هستند.

### ۱.۱. نصب پایتون

این پروژه با پایتون نسخه ۳.۷ به بالا سازگار است. اگر پایتون را نصب ندارید، آن را از وب‌سایت رسمی [python.org](https://www.python.org/downloads/) دانلود و نصب کنید.

-----

### ۱.۲. آماده‌سازی محیط مجازی (توصیه می‌شود)

استفاده از محیط مجازی به شما کمک می‌کند تا کتابخانه‌های پروژه از سایر پروژه‌های پایتون جدا بمانند و تداخلی ایجاد نشود.


۱.  باز کردن ترمینال یا Command Prompt:


 * در ویندوز، کلید `Windows` را فشار داده و عبارت `cmd` یا `powershell` را جستجو کنید.
 * در لینوکس یا مک، برنامه `Terminal` را باز کنید.

۲.  ساخت محیط مجازی:
    در ترمینال، دستور زیر را اجرا کنید:


``` Bash
    python -m venv my_face_recognition_env
```



 این دستور یک پوشه به نام `my_face_recognition_env` ایجاد می‌کند.

۳.  **فعال‌سازی محیط مجازی:**

 * ویندوز:

``` bash
        my_face_recognition_env\Scripts\activate
```
 * لینوکس و macOS:
``` bash
        source my_face_recognition_env/bin/activate
```

* پس از فعال‌سازی، نام محیط مجازی در ابتدای خط فرمان نمایش داده می‌شود (مثلاً `(my_face_recognition_env) C:\...`).

-----
برای نصب کتابخانه‌های مورد نیاز پروژه، ابتدا باید محیط مجازی خود را فعال کنید. پس از آن، می‌توانید هر کتابخانه را با دستور pip install نصب کنید. در ادامه، جزئیات بیشتری در مورد هر کتابخانه ارائه شده است.

. دریافت پروژه از گیت‌هاب

برای اینکه بتوانید کد پروژه را روی کامپیوتر خود داشته باشید، باید آن را از گیت‌هاب (GitHub) دانلود کنید. این کار به سادگی با استفاده از دستور git clone انجام می‌شود.

۱.۰.۱. نصب Git

گیت یک ابزار برای مدیریت نسخه‌ها و دریافت کد از مخازن آنلاین مانند گیت‌هاب است.

دانلود Git: به وب‌سایت رسمی git-scm.com بروید و نسخه مناسب با سیستم عامل خود (ویندوز، macOS یا لینوکس) را دانلود و نصب کنید.

بررسی نصب: پس از نصب، ترمینال یا Command Prompt را باز کرده و دستور زیر را اجرا کنید:
   ``` Bash

    git --version
```
اگر نصب با موفقیت انجام شده باشد، نسخه Git نصب شده نمایش داده می‌شود.

۱.۰.۲. کلون کردن پروژه

حالا که Git روی سیستم شما نصب است، می‌توانید پروژه را کلون کنید.

آدرس مخزن (Repository) پروژه را پیدا کنید: اگر پروژه در گیت‌هاب عمومی است، آدرس URL آن را پیدا کنید. معمولاً در صفحه اصلی مخزن، یک دکمه سبز رنگ با عنوان Code وجود دارد که با کلیک روی آن، آدرس URL نمایش داده می‌شود.

باز کردن ترمینال: یک ترمینال یا Command Prompt را باز کنید و به مسیری که می‌خواهید پروژه در آنجا ذخیره شود بروید.

اجرای دستور کلون: دستور زیر را با آدرس مخزن خود جایگزین کرده و اجرا کنید:
   ``` Bash

git clone https://github.com/alireza-keivan/Face-Recognition-multi-thread/tree/alireza-keivan
```

پس از اجرای این دستور، تمامی فایل‌های پروژه در یک پوشه جدید با نام پروژه در مسیر فعلی شما دانلود خواهند شد.

۱.۰.۳. ورود به پوشه پروژه

پس از کلون کردن، باید وارد پوشه جدیدی شوید که حاوی فایل‌های پروژه است.
``` Bash

cd <نام پوشه پروژه>
```
حالا شما در پوشه اصلی پروژه قرار دارید و می‌توانید مراحل بعدی نصب (مانند راه‌اندازی محیط مجازی و نصب کتابخانه‌ها) را مطابق با داکیومنت اصلی دنبال کنید.



۱. نصب onnxruntime (پیش‌نیاز InsightFace)

چرا به آن نیاز داریم؟
کتابخانه insightface برای اجرای مدل‌های هوش مصنوعی خود از فرمت ONNX استفاده می‌کند. onnxruntime یک ابزار ضروری برای اجرای این مدل‌ها است و باید قبل از insightface نصب شود.

دستور نصب:
``` Bash

pip install onnxruntime
```
اگر از کارت گرافیک NVIDIA استفاده می‌کنید و می‌خواهید از پردازش GPU بهره ببرید، به جای دستور بالا، از دستور زیر استفاده کنید:
``` Bash

pip install onnxruntime-gpu
```
۲. نصب insightface

چرا به آن نیاز داریم؟
این کتابخانه هسته اصلی پروژه است که وظیفه تشخیص چهره، استخراج ویژگی‌های آن (embedding) و مقایسه چهره‌ها را بر عهده دارد.


دستور نصب:
``` Bash

pip install insightface
```
۳. نصب OpenCV (opencv-python)

چرا به آن نیاز داریم؟
OpenCV یک کتابخانه قدرتمند برای پردازش تصویر است. در این پروژه از آن برای موارد زیر استفاده می‌شود:

خواندن فریم‌ها: برای دریافت تصویر از وب‌کم یا استریم دوربین.

تبدیل رنگ: برای تبدیل فرمت رنگی تصاویر از BGR (فرمت پیش‌فرض OpenCV) به RGB (فرمت مورد انتظار InsightFace).

برش تصویر: برای ایجاد تصویر چهره برش‌خورده جهت ارسال از طریق MQTT.

ذخیره‌سازی تصویر: برای تبدیل تصاویر NumPy به فرمت Base64 برای ارسال.

مدیریت فریم‌ها: برای تغییر اندازه فریم‌ها جهت افزایش سرعت پردازش.

دستور نصب:
``` Bash

pip install opencv-python
```
۴. نصب paho-mqtt

چرا به آن نیاز داریم؟
این کتابخانه کلاینت پروتکل MQTT را فراهم می‌کند و به پروژه اجازه می‌دهد تا به یک کارگزار (Broker) MQTT متصل شود و داده‌های مربوط به چهره‌های شناسایی شده را منتشر کند.

دستور نصب:

``` Bash
pip install paho-mqtt
```
۵. نصب NumPy

چرا به آن نیاز داریم؟
NumPy برای انجام محاسبات عددی پیچیده و کار با آرایه‌ها در پایتون استفاده می‌شود. این کتابخانه در این پروژه برای موارد زیر ضروری است:

ذخیره ویژگی‌های چهره: ویژگی‌های عددی چهره (embedding) در قالب آرایه‌های NumPy ذخیره و مدیریت می‌شوند.

محاسبه شباهت: برای محاسبه شباهت بین چهره‌های جدید و چهره‌های شناخته‌شده، از ضرب داخلی (Dot Product) آرایه‌های NumPy استفاده می‌شود.

دستور نصب:
``` Bash

pip install numpy
```
## ۲. پیکربندی پروژه (فایل `config.json`)

قبل از اجرای برنامه، باید فایل `config.json` را بر اساس نیاز خود ویرایش کنید. این فایل شامل تمام تنظیمات کلیدی پروژه است.

### ۲.۱. تنظیمات دوربین (`camera_settings`)

| کلید | توضیحات |
| :--- | :--- |
| **`url`** | آدرس استریم دوربین (مثلاً وب‌کم یا دوربین IP). برای استفاده از وب‌کم داخلی، این مقدار را خالی بگذارید یا روی `0` تنظیم کنید.  |
| **`stream_id`** | اگر از وب‌کم استفاده می‌کنید، شناسه وب‌کم را اینجا وارد کنید (مثلاً `0` برای وب‌کم پیش‌فرض). |
| **`fps_limit`** | حداکثر فریم در ثانیه برای پردازش. هرچه این عدد کمتر باشد، پردازش سبک‌تر است. |
| **`OUTPUT_WINDOW_NAME`** | نام پنجره‌ای که خروجی ویدیو را نمایش می‌دهد. |
| **`resize_scale`** | برای کاهش ابعاد فریم و افزایش سرعت پردازش، این مقدار را به عددی بین `۰` و `۱` تنظیم کنید (مثلاً `0.5`). |

-----

### ۲.۲. تنظیمات تشخیص چهره (`face_recognition_settings`)

| کلید | توضیحات |
| :--- | :--- |
| **`PROCESS_FRAME_SCALE`** | مقیاس‌بندی فریم‌ها قبل از تشخیص چهره. مقدار `۰.۲` به معنی پردازش ۲۰ درصد از اندازه اصلی فریم است که سرعت را به شدت افزایش می‌دهد. |
| **`COSINE_SIMILARITY_THRESHOLD`** | **مهم:** آستانه شباهت برای تشخیص یک چهره به‌عنوان یک چهره **شناخته‌شده**. هرچه این عدد به `۱.۰` نزدیک‌تر باشد، دقت بیشتر و سخت‌گیری بالاتر است. مقدار پیشنهادی `۰.۳۵` است. |
| **`IMAGE_SEND_COOLDOWN`** | حداقل زمان بین ارسال دو تصویر متوالی یک چهره به سرور MQTT (بر حسب ثانیه). |

-----

### ۲.۳. تنظیمات دایرکتوری‌ها (`directories`)

| کلید | توضیحات |
| :--- | :--- |
| **`KNOWN_FACES_DIR`** | **مهم:** مسیر پوشه‌ای که تصاویر چهره‌های شناخته‌شده را در آن قرار می‌دهید. نام هر فایل تصویر (بدون پسوند) به‌عنوان نام آن شخص در نظر گرفته می‌شود (مثلاً `alireza.jpg`). |
| **`ENCODINGS_FILE`** | مسیر فایل `.pkl` که اطلاعات عددی چهره‌ها در آن ذخیره می‌شود تا در دفعات بعدی برنامه سریع‌تر اجرا شود. |

-----

### ۲.۴. تنظیمات MQTT (`mqtt_settings`)

| کلید | توضیحات |
| :--- | :--- |
| **`MQTT_BROKER_ADDRESS`** | آدرس IP یا نام هاست کارگزار MQTT. |
| **`MQTT_BROKER_PORT`** | پورت کارگزار MQTT (معمولاً `۱۸۸۳`). |
| **`MQTT_CLIENT_ID`** | شناسه منحصربه‌فرد برای این کلاینت. |
| **`MQTT_USERNAME`** | نام کاربری برای اتصال به کارگزار. |
| **`MQTT_PASSWORD`** | رمز عبور برای اتصال به کارگزار. |
| **`MQTT_FACE_TOPIC`** | موضوعی که اطلاعات چهره‌ها روی آن منتشر می‌شود. |

-----

### ۲.۵. تنظیمات منطقه عملیات (`operation_region`)

| کلید | توضیحات |
| :--- | :--- |
| **`x1`, `y1`, `x2`, `y2`** | مختصات (از بالا به چپ) یک مستطیل که فقط چهره‌های داخل آن پردازش می‌شوند. این کار به افزایش سرعت و کاهش پردازش‌های غیرضروری کمک می‌کند. |
| **`door_num`** | شماره درب یا مکان مورد نظر که اطلاعات به آن مرتبط می‌شود. |

-----

### ۲.۶. تنظیمات بهینه‌سازی (`optimization_settings`)

| کلید | توضیحات |
| :--- | :--- |
| **`detection_interval_seconds`** | حداقل زمان بین هر بار تشخیص کامل چهره. در فواصل بین این زمان‌ها، برنامه از ردیاب‌های سبک‌تری برای دنبال کردن چهره‌ها استفاده می‌کند که سرعت را بسیار بالا می‌برد. |
| **`tracker_type`** | نوع ردیاب مورد استفاده. `KCF` یک گزینه سریع و خوب است. |

-----

## ۳. افزودن چهره‌های شناخته‌شده

1.  یک پوشه برای تصاویر چهره‌های شناخته‌شده ایجاد کنید (بر اساس مسیری که در `config.json` برای `KNOWN_FACES_DIR` تنظیم کرده‌اید).
2.  تصاویر چهره‌ها را با فرمت‌های `png` یا `jpg` در این پوشه قرار دهید.
3.  **نام هر فایل را به نام شخص تغییر دهید.** به عنوان مثال، برای چهره علیرضا، فایل را `alireza.jpg` نامگذاری کنید.

در اولین اجرای برنامه، سیستم به طور خودکار این تصاویر را پردازش کرده و اطلاعات آن‌ها را در یک فایل `.pkl` ذخیره می‌کند تا دفعات بعدی سریع‌تر اجرا شود.

-----

## ۴. اجرای برنامه

پس از انجام تمام مراحل بالا، برنامه آماده اجرا است.

1.  **مطمئن شوید که در محیط مجازی قرار دارید.**
2.  **فایل `recognizer.py` را اجرا کنید:**
    ``` bash
    python recognizer.py
    ```

برنامه شروع به کار کرده و سپس اطلاعات پردازش و تشخیص چهره در ترمینال و همچنین در فایل `face_recognition.log` نمایش داده می‌شود.


## ۵. راهنمای عیب‌یابی

  * **اگر برنامه اجرا نمی‌شود:**
      * مطمئن شوید که تمامی کتابخانه‌ها به درستی نصب شده‌اند.
      * مسیر فایل‌ها و پوشه‌ها در `config.json` را دوباره بررسی کنید.
  * **اگر چهره‌ها شناسایی نمی‌شوند:**
      * مقدار `COSINE_SIMILARITY_THRESHOLD` در `config.json` را به یک عدد پایین‌تر کاهش دهید (مثلاً `۰.۳`).
      * مطمئن شوید تصاویر در پوشه `KNOWN_FACES_DIR` واضح و با کیفیت مناسب هستند.
  * **اگر به MQTT متصل نمی‌شود:**
      * آدرس و پورت کارگزار MQTT را بررسی کنید.
      * اطلاعات کاربری (`username` و `password`) را چک کنید.
      * اطمینان حاصل کنید که کارگزار MQTT فعال است.





****# راهنمای جامع راه‌اندازی رابط وب و PostgreSQL
## سیستم تشخیص چهره - از صفر تا صد

---

## فهرست مطالب
1. [معرفی و پیش‌نیازها](#۱-معرفی-و-پیش‌نیازها)
2. [نصب و راه‌اندازی PostgreSQL](#۲-نصب-و-راه‌اندازی-postgresql)
3. [پیکربندی پایگاه داده](#۳-پیکربندی-پایگاه-داده)
4. [نصب Python و محیط مجازی](#۴-نصب-python-و-محیط-مجازی)
5. [نصب کتابخانه‌های مورد نیاز](#۵-نصب-کتابخانه‌های-مورد-نیاز)
6. [ساختار پروژه و فایل‌های کلیدی](#۶-ساختار-پروژه-و-فایل‌های-کلیدی)
7. [پیکربندی رابط وب](#۷-پیکربندی-رابط-وب)
8. [راه‌اندازی سرور وب](#۸-راه‌اندازی-سرور-وب)
9. [مدیریت کاربران](#۹-مدیریت-کاربران)
10. [راه‌اندازی سرویس Systemd](#۱۰-راه‌اندازی-سرویس-systemd)
11. [استفاده از رابط وب](#۱۱-استفاده-از-رابط-وب)
12. [عیب‌یابی و رفع مشکلات](#۱۲-عیب‌یابی-و-رفع-مشکلات)

---

## ۱. معرفی و پیش‌نیازها

### ۱.۱ معرفی پروژه
این پروژه یک سیستم تشخیص چهره است که شامل دو بخش اصلی می‌باشد:
- **بخش تشخیص چهره**: که در سند دیگری توضیح داده شده است
- **رابط وب و پایگاه داده**: که در این سند به تفصیل شرح داده می‌شود

رابط وب این پروژه با استفاده از Flask ساخته شده و از PostgreSQL برای ذخیره‌سازی داده‌ها استفاده می‌کند.

### ۱.۲ قابلیت‌های رابط وب
- **داشبورد**: نمایش آمار و وضعیت سیستم
- **مدیریت افراد**: افزودن، حذف و مشاهده افراد ثبت شده
- **مدیریت کاربران**: سیستم احراز هویت کامل با نقش‌های مختلف
- **تنظیمات**: پیکربندی سیستم تشخیص چهره
- **لاگ‌ها**: مشاهده تاریخچه تشخیص چهره‌ها
- **رابط کاربری فارسی**: کاملاً به زبان فارسی و RTL

### ۱.۳ پیش‌نیازها
- سیستم عامل: Linux (Ubuntu 20.04 یا بالاتر توصیه می‌شود)
- Python 3.8 یا بالاتر
- PostgreSQL 12 یا بالاتر
- دسترسی sudo برای نصب بسته‌ها
- حداقل 2GB RAM
- اتصال به اینترنت برای نصب بسته‌ها

---

## ۲. نصب و راه‌اندازی PostgreSQL

### ۲.۱ نصب PostgreSQL

#### گام ۱: به‌روزرسانی لیست بسته‌ها
```bash
sudo apt update
```

#### گام ۲: نصب PostgreSQL
```bash
sudo apt install postgresql postgresql-contrib -y
```

#### گام ۳: بررسی نصب موفق
```bash
sudo systemctl status postgresql
```

خروجی باید نشان دهد که سرویس PostgreSQL در حال اجرا است (active).

#### گام ۴: فعال‌سازی شروع خودکار هنگام بوت
```bash
sudo systemctl enable postgresql
```

### ۲.۲ پیکربندی اولیه PostgreSQL

#### گام ۱: ورود به حساب کاربری postgres
```bash
sudo -u postgres psql
```

#### گام ۲: تغییر رمز عبور کاربر postgres (اختیاری اما توصیه می‌شود)
در محیط psql:
```sql
ALTER USER postgres WITH PASSWORD 'رمز_عبور_قوی_خود';
\q
```

### ۲.۳ راه‌اندازی دسترسی از راه دور (در صورت نیاز)

اگر می‌خواهید از راه دور به PostgreSQL متصل شوید:

#### گام ۱: ویرایش فایل پیکربندی
```bash
sudo nano /etc/postgresql/[VERSION]/main/postgresql.conf
```

خط زیر را پیدا کرده و از حالت comment خارج کنید:
```
listen_addresses = '*'
```

#### گام ۲: ویرایش فایل احراز هویت
```bash
sudo nano /etc/postgresql/[VERSION]/main/pg_hba.conf
```

در انتهای فایل اضافه کنید:
```
host    all             all             0.0.0.0/0               md5
host    all             all             ::/0                    md5
```

#### گام ۳: راه‌اندازی مجدد PostgreSQL
```bash
sudo systemctl restart postgresql
```

---

## ۳. پیکربندی پایگاه داده

### ۳.۱ ایجاد پایگاه داده و کاربر

#### گام ۱: ورود به PostgreSQL
```bash
sudo -u postgres psql
```

#### گام ۲: ایجاد کاربر برای برنامه
**نکته امنیتی مهم**: در محیط production حتماً از رمز عبور قوی استفاده کنید
```sql
CREATE USER face_user WITH PASSWORD '123456';
```

.

#### گام ۳: ایجاد پایگاه داده
```sql
CREATE DATABASE face_recognition_db OWNER face_user;
```

#### گام ۴: اعطای دسترسی‌های لازم
```sql
GRANT ALL PRIVILEGES ON DATABASE face_recognition_db TO face_user;
```

#### گام ۵: اتصال به پایگاه داده جدید
```sql
\c face_recognition_db
```

#### گام ۶: اعطای دسترسی به schema
```sql
GRANT ALL ON SCHEMA public TO face_user;
```

#### گام ۷: خروج از psql
```sql
\q
```

### ۳.۲ ساختار جداول

جداول به صورت خودکار توسط برنامه ساخته می‌شوند، اما در اینجا ساختار آن‌ها توضیح داده می‌شود:

#### جدول users (کاربران)
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_username ON users(username);
```

#### جدول recognition_logs (لاگ تشخیص چهره)
```sql
CREATE TABLE recognition_logs (
    id SERIAL PRIMARY KEY,
    recognized_person VARCHAR(255) NOT NULL,
    identity_verified BOOLEAN NOT NULL DEFAULT FALSE,
    cosine_similarity FLOAT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(512),
    image_filename VARCHAR(255),
    face_x1 INTEGER,
    face_y1 INTEGER,
    face_x2 INTEGER,
    face_y2 INTEGER,
    door_num INTEGER,
    region_x1 INTEGER,
    region_y1 INTEGER,
    region_x2 INTEGER,
    region_y2 INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_recognized_person ON recognition_logs(recognized_person);
CREATE INDEX idx_timestamp ON recognition_logs(timestamp);
CREATE INDEX idx_person_timestamp ON recognition_logs(recognized_person, timestamp);
CREATE INDEX idx_verified_timestamp ON recognition_logs(identity_verified, timestamp);
```

#### جدول config_snapshots (تنظیمات)
```sql
CREATE TABLE config_snapshots (
    id SERIAL PRIMARY KEY,
    config_json TEXT NOT NULL,
    service_start_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### ۳.۳ تست اتصال به پایگاه داده

```bash
psql -h localhost -U face_user -d face_recognition_db
```

رمز عبور را وارد کنید (123456 یا رمز عبوری که تنظیم کرده‌اید).

اگر با موفقیت وارد شدید، پیکربندی صحیح است. با `\q` خارج شوید.

---Python 3


#### گام 1: رفتن به پوشه پروژه
```bash
cd /home/face/Face-Recognition-multi-thread
```

#### گام 2: فعال‌سازی محیط مجازی
```bash
source bin/activate
```

پس از اجرا، باید `(Face-Recognition-multi-thread)` در ابتدای خط فرمان نمایش داده شود.

**نکته**: هر بار که می‌خواهید با پروژه کار کنید، باید محیط مجازی را فعال کنید.

#### گام ۴: به‌روزرسانی pip
```bash
pip install --upgrade pip
```

---

## ۵. نصب کتابخانه‌های مورد نیاز
curl http://localhost
**توضیح کتابخانه‌ها**:
- **Flask**: فریمورک اصلی وب
- **Flask-SQLAlchemy**: ORM برای کار با پایگاه داده
- **Flask-Login**: مدیریت احراز هویت کاربران
- **psycopg2-binary**: درایور PostgreSQL
- **gunicorn**: سرور WSGI برای production
- **Pillow**: پردازش تصویر
- **numpy**: محاسبات عددی
- **jdatetime**: تاریخ شمسی (جلالی)

### ۵.۲ نصب کتابخانه‌ها

```bash
cd /home/face/Face-Recognition-multi-thread
source bin/activate
pip install -r webapp/requirements.txt
pip install -r requirements.txt
```

این فرآیند ممکن است چند دقیقه طول بکشد.

### ۵.۳ تایید نصب

```bash
pip list
```

باید تمام کتابخانه‌های نصب شده را مشاهده کنید.

---

## ۶. ساختار پروژه و فایل‌های کلیدی

### ۶.۱ ساختار کلی پروژه

```
/home/face/Face-Recognition-multi-thread/
├── bin/                    # اجرایی‌های محیط مجازی
├── lib/                    # کتابخانه‌های Python
├── include/                # فایل‌های هدر
├── webapp/                 # اپلیکیشن وب (اصلی‌ترین پوشه)
│   ├── app.py             # فایل اصلی برنامه
│   ├── config.py          # تنظیمات
│   ├── database.py        # مدیریت پایگاه داده
│   ├── models.py          # مدل‌های SQLAlchemy
│   ├── wsgi.py            # نقطه ورود WSGI
│   ├── init_auth.py       # اسکریپت ایجاد کاربر
│   ├── requirements.txt   # وابستگی‌ها
│   ├── templates/         # قالب‌های HTML
│   ├── static/            # فایل‌های استاتیک (CSS, JS, تصاویر)
│   ├── routes/            # مسیرهای اضافی
│   └── services/          # سرویس‌های کمکی
├── captured_known_faces/   # تصاویر افراد شناخته شده
├── saved_faces/           # تصاویر ذخیره شده از تشخیص
├── logs/                  # فایل‌های لاگ
└── config.json            # تنظیمات سیستم تشخیص چهره
```

### ۷.۲ تست اتصال به پایگاه داده

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/face/Face-Recognition-multi-thread/webapp')
from database import db_manager

# Initialize database
if db_manager.initialize():
    print("✓ Database connection successful!")
    
    # Test query
    session = db_manager.get_session()
    from database import User
    users = session.query(User).all()
    print(f"✓ Found {len(users)} user(s) in database")
    session.close()
else:
    print("✗ Database connection failed!")
EOF
```

---

## ۸. راه‌اندازی سرور وب

### ۸.۱ اجرای سرور در حالت Development


## ۹. مدیریت کاربران

### ۹.۱ ورود با حساب Admin

#### گام ۱: باز کردن صفحه ورود
```
http://[IP_ADDRESS]:5000/login
```

#### گام ۲: وارد کردن اطلاعات
- **نام کاربری**: admin
- **رمز عبور**:******

#### گام ۳: تغییر رمز عبور (بسیار مهم!)
بعد از ورود، حتماً رمز عبور را تغییر دهید.

### ۹.۲ ایجاد کاربر جدید از رابط وب

#### از صفحه ثبت‌نام:
```
http://[IP_ADDRESS]:5000/register
```

فیلدهای مورد نیاز:
- نام کاربری (حداقل 3 کاراکتر)
- ایمیل (اختیاری)
- رمز عبور (حداقل 6 کاراکتر)
- تایید رمز عبور

### ۹.۳ ایجاد کاربر از خط فرمان

```bash
cd /home/face/Face-Recognition-multi-thread
source bin/activate

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/face/Face-Recognition-multi-thread/webapp')
from database import db_manager, User

db_manager.initialize()
session = db_manager.get_session()

# Create new user
new_user = User(
    username='username_here', #تغییر دهید
    email='email@example.com', # تغییر دهید
    is_admin=False,  # True for admin, False for regular user
    is_active=True
)
new_user.set_password('password_here') # تغییر دهید

session.add(new_user)
session.commit()

print(f"✓ User '{new_user.username}' created successfully!")
session.close()
EOF
```

### ۹.۴ نقش‌های کاربری

سیستم دارای دو نقش است:

#### کاربر عادی (User):
- مشاهده داشبورد
- مشاهده لیست افراد و لاگ‌ها
- افزودن فرد جدید
- مشاهده تنظیمات

#### مدیر (Admin):
- تمام دسترسی‌های کاربر عادی
- مدیریت کاربران (افزودن، حذف، فعال/غیرفعال کردن)
- تغییر تنظیمات سیستم
- حذف افراد
- بازسازی encodings

---

## ۱۰. راه‌اندازی سرویس Systemd

برای اجرای خودکار وب سرور هنگام بوت سیستم:

### ۱۰.۱ ایجاد فایل سرویس

```bash
sudo nano /etc/systemd/system/face-webapp.service
```

محتوا:
```ini
[Unit]
Description=Face Recognition Web Admin
After=network.target postgresql.service

[Service]
User=face
Group=face
WorkingDirectory=/home/face/Face-Recognition-multi-thread/webapp
Environment="PATH=/home/face/Face-Recognition-multi-thread/bin:/usr/bin"
Environment="VIRTUAL_ENV=/home/face/Face-Recognition-multi-thread"
ExecStart=/home/face/Face-Recognition-multi-thread/bin/python /home/face/Face-Recognition-multi-thread/bin/gunicorn --workers 3 --bind 127.0.0.1:5001 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```


```bash
sudo nano /etc/systemd/system/face-recognition.service
```

محتوا:
```ini
[Unit]
Description=Face Recognition Application Service
After=network.target

[Service]
# User and Group to run the service as
Type=simple
User=face
Group=face

# The directory where your script is located
WorkingDirectory=/home/face/Face-Recognition-multi-thread # این آدرس منطبق با آدرس خودتان اصلاح شود

# The command to start your application
# Use the FULL PATH to your virtual environment's python and your script
ExecStart=/home/face/Face-Recognition-multi-thread/exit.bash
# Restart the service if it fails
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```
دستور زیر را اجرا کنید:
```bash
sudo apt update
sudo apt install -y nginx
sudo nano /usr/lib/systemd/system/nginx.service
```
نصب nginx را تایید کنید:
```bash
systemctl list-unit-files | grep nginx
```
خروجی مورد نظر:
```bash
nginx.service    enabled
```
دستور زیر را اجرا و محتوای سرور زیر را به فایل اضافه کنید:
```bash
sudo nano /etc/nginx/sites-available/face-webapp
```
```bash
server {
    listen 80;
    server_name 192.168.45.235;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
   
    }
    location /static {
        alias /home/face/Face-Recognition-multi-thread/webapp/static;
        expires 1d;
    }

    location /saved_faces {
        alias /home/face/Face-Recognition-multi-thread/saved_faces;
    }
}
```
لینک سیمبولیک را با اجرای دستور زیر بسازید:
```bash
sudo ln -s /etc/nginx/sites-available/face-webapp \
           /etc/nginx/sites-enabled/face-webapp
```
تایید ساخت لینک سیمبولیک و خروجی مورد نظر:

```bash
sudo nginx -t
```

```bash
syntax is ok
test is successful
```

راه اندازی سرویس nginx:
```bash
sudo systemctl restart nginx
```

```bash
systemctl status nginx
```
خروجی مورد نظر:
```bash
Active: active (running)
```
شروع nginx از بوت:
```bash
sudo systemctl enable nginx
```




**نکته مهم**: اگر کاربر شما `face` نیست، `User` و `Group` را با نام کاربری خود جایگزین کنید.

### ۱۰.۲ فعال‌سازی و شروع سرویس

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable face-webapp.service

# Start service
sudo systemctl start face-webapp.service

# Check status
sudo systemctl status face-webapp.service
```

### ۱۰.۳ دستورات مفید سرویس

```bash
# مشاهده وضعیت
sudo systemctl status face-webapp.service

# توقف سرویس
sudo systemctl stop face-webapp.service

# راه‌اندازی مجدد
sudo systemctl restart face-webapp.service

# مشاهده لاگ‌ها
sudo journalctl -u face-webapp.service -f

# مشاهده 100 خط آخر لاگ
sudo journalctl -u face-webapp.service -n 100

# غیرفعال کردن شروع خودکار
sudo systemctl disable face-webapp.service
```

---

## ۱۱. استفاده از رابط وب

### ۱۱.۱ صفحه اصلی - داشبورد

آدرس: `http://[IP]:5000/` یا `http://[IP]:5000/dashboard`

**قابلیت‌ها**:
- نمایش تعداد کل افراد ثبت شده
- نمایش تعداد تشخیص‌های امروز
- نمایش تشخیص‌های اخیر
- نمودارها و آمار

### ۱۱.۲ مدیریت افراد

آدرس: `http://[IP]:5000/persons`

#### افزودن فرد جدید:
1. کلیک روی "افزودن فرد جدید"
2. نام فرد را وارد کنید
3. یک یا چند تصویر از چهره انتخاب کنید
4. کلیک روی "ذخیره"

**نکات مهم**:
- تصویر باید واضح و روشن باشد
- چهره باید رو به دوربین باشد
- حداقل 1 تصویر لازم است
- توصیه می‌شود 3-5 تصویر از زوایای مختلف

#### مشاهده جزئیات فرد:
- کلیک روی نام فرد
- مشاهده تمام تصاویر ثبت شده
- مشاهده تاریخچه تشخیص‌ها

#### حذف فرد:
- فقط مدیر می‌تواند افراد را حذف کند
- کلیک روی دکمه حذف
- تایید حذف

### ۱۱.۳ مشاهده لاگ‌ها

آدرس: `http://[IP]:5000/logs`

**قابلیت‌ها**:
- مشاهده تمام تشخیص‌های ثبت شده
- فیلتر بر اساس:
  - نام فرد
  - بازه زمانی
  - وضعیت تایید
- صفحه‌بندی
- نمایش تصویر تشخیص داده شده
- نمایش درصد شباهت

### ۱۱.۴ تنظیمات

آدرس: `http://[IP]:5000/settings`

**تنظیمات قابل تغییر**:
- آدرس دوربین (RTSP URL)
- آستانه شباهت (Cosine Similarity Threshold)
- تنظیمات MQTT
- منطقه عملیاتی (Operation Region)
- و تنظیمات دیگر...

**نحوه تغییر تنظیمات**:
1. مقادیر مورد نظر را تغییر دهید
2. کلیک روی "ذخیره تنظیمات"
3. برای اعمال تغییرات، سرویس تشخیص چهره باید restart شود

### ۱۱.۵ مدیریت کاربران (فقط Admin)

آدرس: `http://[IP]:5000/users`

**قابلیت‌ها برای Admin**:
- مشاهده لیست تمام کاربران
- فعال/غیرفعال کردن کاربران
- حذف کاربران
- مشاهده تاریخ ایجاد و آخرین ورود

**محدودیت‌ها**:
- نمی‌توان حساب خود را حذف کرد
- نمی‌توان آخرین admin را حذف کرد
- نمی‌توان خود را غیرفعال کرد

---

## ۱۲. عیب‌یابی و رفع مشکلات

### ۱۲.۱ مشکلات اتصال به پایگاه داده

#### خطا: "could not connect to server"

**علت**: PostgreSQL در حال اجرا نیست

**راه حل**:
```bash
sudo systemctl start postgresql
sudo systemctl status postgresql
```

#### خطا: "password authentication failed"

**علت**: نام کاربری یا رمز عبور اشتباه است

**راه حل**:
1. بررسی تنظیمات در `webapp/config.py`
2. مطابقت با اطلاعات ایجاد شده در PostgreSQL
3. تست اتصال:
```bash
psql -h localhost -U face_user -d face_recognition_db
```

#### خطا: "database does not exist"

**علت**: پایگاه داده ایجاد نشده است

**راه حل**:
```bash
sudo -u postgres psql
CREATE DATABASE face_recognition_db OWNER face_user;
\q
```

### ۱۲.۲ مشکلات نصب کتابخانه‌ها

#### خطا: "psycopg2" نصب نمی‌شود

**راه حل**:
```bash
# نصب وابستگی‌های سیستمی
sudo apt install python3-dev libpq-dev build-essential -y

# نصب مجدد
pip install psycopg2-binary
```

#### خطا: "Permission denied"

**راه حل**:
```bash
# اطمینان از فعال بودن محیط مجازی
source /home/face/Face-Recognition-multi-thread/bin/activate

# نصب بدون sudo
pip install -r requirements.txt
```

### ۱۲.۳ مشکلات راه‌اندازی سرور

#### خطا: "Address already in use"

**علت**: پورت 5000 در حال استفاده است

**راه حل**:
```bash
# پیدا کردن پروسه
sudo lsof -i :5000

# کشتن پروسه
sudo kill -9 [PID]

# یا استفاده از پورت دیگر
gunicorn --bind 0.0.0.0:5001 wsgi:app
```

#### خطا: "ModuleNotFoundError"

**علت**: کتابخانه‌ها نصب نشده‌اند یا محیط مجازی فعال نیست

**راه حل**:
```bash
cd /home/face/Face-Recognition-multi-thread
source bin/activate
pip install -r webapp/requirements.txt
```

### ۱۲.۴ مشکلات احراز هویت

#### نمی‌توانم وارد شوم

**بررسی**:
1. آیا جدول users ایجاد شده است؟
```bash
psql -h localhost -U face_user -d face_recognition_db -c "\dt"
```

2. آیا کاربر admin وجود دارد؟
```bash
python3 webapp/init_auth.py
```

3. آیا رمز عبور صحیح است؟
- رمز پیش‌فرض: `admin123`

#### صفحه login نمایش داده نمی‌شود

**بررسی**:
1. لاگ‌های Flask را مشاهده کنید:
```bash
# اگر از systemd استفاده می‌کنید
sudo journalctl -u face-webapp.service -f

# اگر دستی اجرا کرده‌اید
# خطاها در ترمینال نمایش داده می‌شوند
```

### ۱۲.۵ بررسی لاگ‌ها

#### لاگ‌های Flask (Development):
وقتی Flask را دستی اجرا می‌کنید، لاگ‌ها در ترمینال نمایش داده می‌شوند.

#### لاگ‌های Gunicorn:
```bash
# Access logs
tail -f /home/face/Face-Recognition-multi-thread/logs/webapp-access.log

# Error logs
tail -f /home/face/Face-Recognition-multi-thread/logs/webapp-error.log
```

#### لاگ‌های Systemd:
```bash
# Live monitoring
sudo journalctl -u face-webapp.service -f

# Last 100 lines
sudo journalctl -u face-webapp.service -n 100

# Since boot
sudo journalctl -u face-webapp.service -b
```

### ۱۲.۶ مشکلات عملکرد

#### سرور خیلی کند است

**راه حل**:
1. افزایش تعداد workers در Gunicorn:
```bash
gunicorn --workers 8 --bind 0.0.0.0:5000 wsgi:app
```

2. بررسی منابع سیستم:
```bash
htop
# یا
top
```

3. بهینه‌سازی پایگاه داده:
```bash
sudo -u postgres psql face_recognition_db
VACUUM ANALYZE;
REINDEX DATABASE face_recognition_db;
\q
```

#### خطای "Out of Memory"

**راه حل**:
1. کاهش تعداد workers
2. افزودن swap space:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### ۱۲.۷ تست سلامت سیستم

#### اسکریپت تست کامل:

```bash
nano test_system.sh
```

محتوا:
```bash
#!/bin/bash

echo "=== Testing Face Recognition Web System ==="
echo

# 1. Check PostgreSQL
echo "1. Checking PostgreSQL..."
if sudo systemctl is-active --quiet postgresql; then
    echo "   ✓ PostgreSQL is running"
else
    echo "   ✗ PostgreSQL is not running"
fi

# 2. Check database connection
echo "2. Testing database connection..."
if psql -h localhost -U face_user -d face_recognition_db -c "SELECT 1" > /dev/null 2>&1; then
    echo "   ✓ Database connection successful"
else
    echo "   ✗ Database connection failed"
fi

# 3. Check webapp service
echo "3. Checking webapp service..."
if sudo systemctl is-active --quiet face-webapp.service; then
    echo "   ✓ Webapp service is running"
else
    echo "   ✗ Webapp service is not running"
fi

# 4. Check web server response
echo "4. Testing web server..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 | grep -q "200\|302"; then
    echo "   ✓ Web server is responding"
else
    echo "   ✗ Web server is not responding"
fi

# 5. Check directories
echo "5. Checking directories..."
if [ -d "/home/face/Face-Recognition-multi-thread/captured_known_faces" ]; then
    echo "   ✓ Known faces directory exists"
else
    echo "   ✗ Known faces directory missing"
fi

echo
echo "=== Test Complete ==="
```

اجرا:
```bash
chmod +x test_system.sh
./test_system.sh
```

---

## نکات امنیتی مهم

### ۱. تغییر رمزهای پیش‌فرض
```bash
# تغییر رمز عبور admin
# از رابط وب: Settings > Change Password

# تغییر SECRET_KEY در config.py
nano webapp/config.py
# SECRET_KEY را به یک رشته تصادفی طولانی تغییر دهید
```

### ۲. محدود کردن دسترسی PostgreSQL
```bash
sudo nano /etc/postgresql/[VERSION]/main/pg_hba.conf
```

فقط از localhost اجازه دسترسی دهید:
```
# IPv4 local connections:
host    face_recognition_db    face_user    127.0.0.1/32    md5
```

### ۳. استفاده از HTTPS (توصیه می‌شود)
برای production، از Nginx با SSL استفاده کنید:

```bash
sudo apt install nginx certbot python3-certbot-nginx
```

پیکربندی Nginx:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### ۴. فایروال
```bash
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 5432/tcp  # Block PostgreSQL from outside
```

---

## پشتیبان‌گیری

### پشتیبان‌گیری از پایگاه داده

```bash
# Full backup
pg_dump -h localhost -U face_user face_recognition_db > backup_$(date +%Y%m%d).sql

# Compressed backup
pg_dump -h localhost -U face_user face_recognition_db | gzip > backup_$(date +%Y%m%d).sql.gz
```

### بازگردانی پشتیبان

```bash
# Restore
psql -h localhost -U face_user face_recognition_db < backup_20260104.sql

# Restore compressed
gunzip -c backup_20260104.sql.gz | psql -h localhost -U face_user face_recognition_db
```

### اسکریپت خودکار پشتیبان‌گیری

```bash
nano backup_db.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/face/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

pg_dump -h localhost -U face_user face_recognition_db | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "db_*.sql.gz" -mtime +7 -delete

echo "Backup completed: db_$DATE.sql.gz"
```

افزودن به crontab (روزانه ساعت 2 صبح):
```bash
chmod +x backup_db.sh
crontab -e
```

اضافه کنید:
```
0 2 * * * /home/face/backup_db.sh >> /home/face/backups/backup.log 2>&1
```

---

## خلاصه دستورات سریع

### شروع سیستم از صفر
```bash
# 1. PostgreSQL
sudo systemctl start postgresql

# 2. فعال‌سازی محیط مجازی
cd /home/face/Face-Recognition-multi-thread
source bin/activate

# 3. اجرای webapp (Development)
cd webapp
python3 app.py

# یا با Gunicorn (Production)
gunicorn --bind 0.0.0.0:5000 --workers 4 wsgi:app
```

### توقف سیستم
```bash
# توقف webapp
sudo systemctl stop face-webapp.service

# یا اگر دستی اجرا کرده‌اید
Ctrl+C
```

### راه‌اندازی مجدد
```bash
sudo systemctl restart face-webapp.service
```

### مشاهده وضعیت
```bash
sudo systemctl status face-webapp.service
sudo systemctl status postgresql
```

---

## منابع و مستندات بیشتر

- **Flask**: https://flask.palletsprojects.com/
- **PostgreSQL**: https://www.postgresql.org/docs/
- **SQLAlchemy**: https://docs.sqlalchemy.org/
- **Gunicorn**: https://docs.gunicorn.org/
- **Flask-Login**: https://flask-login.readthedocs.io/

---

## پشتیبانی و ارتباط

اگر با مشکلی مواجه شدید که در این سند پوشش داده نشده، لطفاً:
1. ابتدا لاگ‌ها را بررسی کنید
2. اسکریپت تست سیستم را اجرا کنید
3. خطا را به طور دقیق یادداشت کنید
4. مستندات مربوطه را مطالعه کنید

---

## تاریخچه تغییرات

- **نسخه 1.0** (4 ژانویه 2026): انتشار اولیه مستندات
  - راه‌اندازی PostgreSQL
  - پیکربندی رابط وب Flask
  - سیستم احراز هویت
  - مدیریت کاربران و افراد

---

**پایان مستندات**

این سند تمام مراحل لازم برای راه‌اندازی رابط وب و پایگاه داده سیستم تشخیص چهره را پوشش می‌دهد. با دنبال کردن این مراحل، شما باید بتوانید سیستم را از صفر راه‌اندازی کنید.

موفق باشید! 🚀
