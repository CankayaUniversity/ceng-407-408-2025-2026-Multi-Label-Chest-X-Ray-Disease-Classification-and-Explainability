const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

const Hospital = require("./models/Hospital");
const Doctor = require("./models/Doctor");

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB Bağlantısı
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("✅ Tebrikler! Kod şu an MongoDB'ye bağlı."))
  .catch((err) => console.log("❌ Bağlantı hatası:", err));

// 1. Hastaneleri Listeleme API'si (Frontend'deki Picker için)
app.get("/api/hospitals", async (req, res) => {
  try {
    const hospitals = await Hospital.find();
    res.json(hospitals);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// 2. Doktor Kayıt API'si
app.post("/api/register", async (req, res) => {
  try {
    const { name, email, password, hospitalId } = req.body;

    // Yeni doktoru 'pending' (beklemede) durumuyla oluştur
    const newDoctor = new Doctor({
      fullName: name,
      email,
      password, // İleride buraya bcrypt şifrelemesi ekleyeceğiz
      hospital: hospitalId,
      status: "pending",
    });

    await newDoctor.save();
    res
      .status(201)
      .json({ message: "Kayıt alındı. Hastane onayı bekleniyor." });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(`🚀 Sunucu ${PORT} portunda hazır.`));
