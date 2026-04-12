# Teknik Güncelleme ve Kurulum Notları

Bu branch üzerinde Backend entegrasyonu ve Veritabanı bağlantıları tamamlanmıştır. Çalıştırmak için aşağıdaki adımları izlemeniz yeterlidir.

### ✅ Yapılan Güncellemeler
* **Backend:** Node.js & Express altyapısı kuruldu.
* **Database:** MongoDB bağlantısı sağlandı (Doctors ve Hospitals koleksiyonları).
* **Frontend:** Kayıt (Register) ekranı backend ile bağlandı. iOS ve Android için UI optimizasyonları yapıldı.
* **Network:** Cihazlar arası bağlantı için IP tabanlı API erişimi yapılandırıldı.

---

### 🛠️ Kurulum ve Çalıştırma

**1. Kütüphaneleri Yükleyin:**
Terminalde ana dizindeyken:

``bash
npm install
Ardından backend klasörüne girip oradaki kütüphaneleri de yükleyin:

Bash
cd backend
npm install
cd ..
2. Veritabanı Bağlantısı (.env):
backend klasörü içinde bir .env dosyası oluşturun ve içine size ilettiğim MONGO_URI bilgisini yapıştırın:

Plaintext
PORT=5000
MONGO_URI=mongodb+srv://... (Buraya bağlantı linki gelecek)
3. API IP Yapılandırması:
Mobil cihazdan (Expo Go) bağlanabilmeniz için src/app/register.tsx (ve login ekranı) içindeki API_URL değişkenine kendi bilgisayarınızın yerel IP adresini yazmalısınız:

JavaScript
const API_URL = "[http://192.168.](http://192.168.)x.x:5000/api";
4. Projeyi Başlatın:
İki ayrı terminal açın:

1. Terminal (Backend): cd backend -> npm start

2. Terminal (Frontend): npx expo start -c

⚠️ Not: Telefon ve bilgisayarın aynı Wi-Fi ağına bağlı olduğundan emin olun.
