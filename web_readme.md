# راهنمای جامع راه‌اندازی رابط وب و PostgreSQL
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
Description=Face Recognition Web Application
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=notify
User=face
Group=face
WorkingDirectory=/home/face/Face-Recognition-multi-thread/webapp
Environment="PATH=/home/face/Face-Recognition-multi-thread/bin"
ExecStart=/home/face/Face-Recognition-multi-thread/bin/gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 4 \
    --threads 2 \
    --timeout 120 \
    --access-logfile /home/face/Face-Recognition-multi-thread/logs/webapp-access.log \
    --error-logfile /home/face/Face-Recognition-multi-thread/logs/webapp-error.log \
    wsgi:app

# Restart policy
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
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
