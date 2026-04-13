import axios from "axios";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useEffect, useState } from "react";
import {
  Alert,
  Dimensions,
  ImageBackground,
  KeyboardAvoidingView,
  Modal,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import DoctorImg from "../../assets/doctor.png";

const { height } = Dimensions.get("window");

// BİLGİSAYARININ IP ADRESİ
const API_URL = "http://192.168.1.90:8000/auth";

export default function RegisterScreen() {
  const router = useRouter();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [hospitals, setHospitals] = useState([]);
  const [selectedHospital, setSelectedHospital] = useState("");

  // AÇILIR PENCERE (MODAL) İÇİN STATE
  const [isModalVisible, setModalVisible] = useState(false);

  useEffect(() => {
    const fetchHospitals = async () => {
      try {
        const response = await axios.get(`${API_URL}/hospitals`);
        setHospitals(response.data);
      } catch (error) {
        console.error("Hastaneler yüklenemedi:", error);
      }
    };
    fetchHospitals();
  }, []);

  const handleRegister = async () => {
    // 1. KONTROL: Tüm alanlar dolu mu? (confirmPassword'ü de ekledik)
    if (!name || !email || !password || !confirmPassword || !selectedHospital) {
      Alert.alert(
        "Eksik Bilgi",
        "Lütfen tüm alanları doldurun ve hastane seçin.",
      );
      return;
    }

    // 2. YENİ KONTROL: Şifreler birbiriyle aynı mı?
    if (password !== confirmPassword) {
      Alert.alert(
        "Hata",
        "Girdiğiniz şifreler birbiriyle uyuşmuyor! Lütfen kontrol edin."
      );
      return; // Şifreler uyuşmuyorsa işlemi burada kes, backend'e gitme!
    }

    try {
      // DİKKAT: Backend'e sadece 'password' gönderiyoruz, confirmPassword'e gerek yok!
      const response = await axios.post(`${API_URL}/register`, {
        name_surname: name,                  
        email: email,
        password: password, 
        hospital: getSelectedHospitalName(), 
      });

      Alert.alert("Başarılı!", response.data.message);
      router.replace("/");
    } catch (error: any) {
      Alert.alert("Hata", "Kayıt işlemi başarısız oldu.");
      console.log("🚨 422 HATA DETAYI:", error.response?.data?.detail);
    }
  };

  // Seçilen hastanenin adını bulmak için yardımcı fonksiyon
  const getSelectedHospitalName = () => {
    if (!selectedHospital) return "Hastane Seçiniz...";
    const hospital: any = hospitals.find(
      (h: any) => h._id === selectedHospital,
    );
    return hospital
      ? `${hospital.name} (${hospital.city})`
      : "Hastane Seçiniz...";
  };

  return (
    <View style={styles.mainContainer}>
      <ImageBackground
        source={DoctorImg}
        style={StyleSheet.absoluteFill}
        resizeMode="cover"
      />

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled" // Bu çok önemli, dokunmaları kaybetmemizi engeller
          bounces={false}
        >
          <View style={{ height: height * 0.3 }} />

          <View style={styles.sheetContainer}>
            <LinearGradient
              colors={[
                "transparent",
                "rgb(8, 39, 97)",
                "rgba(64, 98, 154, 0.98)",
                "#071836",
              ]}
              style={styles.sheetBackground}
            >
              <View style={styles.handleBar} />
              <View style={styles.headerArea}>
                <Text style={styles.title}>KAYIT OL</Text>
              </View>

              <View style={styles.glassCard}>
                <TextInput
                  style={styles.input}
                  placeholder="Ad Soyad"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={name}
                  onChangeText={setName}
                />
                <TextInput
                  style={styles.input}
                  placeholder="E-posta Adresi"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                />
                <TextInput
                  style={styles.input}
                  placeholder="Şifre"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  secureTextEntry
                  value={password}
                  onChangeText={setPassword}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Şifre Tekrar"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  secureTextEntry
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                />

                {/* YENİ HASTANE SEÇİM BUTONU (Tıklanınca Modal Açılır) */}
                <Text
                  style={{
                    color: "rgba(255,255,255,0.6)",
                    marginTop: 10,
                    marginLeft: 5,
                    fontSize: 13,
                    marginBottom: 5,
                  }}
                >
                  Görev Yaptığınız Hastane:
                </Text>
                <TouchableOpacity
                  style={styles.customPickerButton}
                  onPress={() => setModalVisible(true)}
                  activeOpacity={0.7}
                >
                  <Text
                    style={{
                      color: selectedHospital
                        ? "#fff"
                        : "rgba(255,255,255,0.4)",
                      fontSize: 15,
                    }}
                  >
                    {getSelectedHospitalName()}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.primaryBtn}
                  onPress={handleRegister}
                >
                  <Text style={styles.btnText}>Hesabı Oluştur</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={() => router.replace("/")}
                  style={styles.footerLink}
                >
                  <Text style={styles.linkText}>
                    Zaten hesabın var mı?{" "}
                    <Text style={{ fontWeight: "bold" }}>Giriş Yap</Text>
                  </Text>
                </TouchableOpacity>
              </View>
              <View style={{ height: 150 }} />
            </LinearGradient>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* YENİ: AŞAĞIDAN KAYARAK AÇILAN HASTANE SEÇİM PENCERESİ */}
      <Modal
        visible={isModalVisible}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHandleBar} />
            <Text style={styles.modalTitle}>Hastane Seçin</Text>

            <ScrollView style={{ width: "100%", maxHeight: height * 0.4 }}>
              {hospitals.map((hosp: any) => (
                <TouchableOpacity
                  key={hosp._id}
                  style={styles.modalItem}
                  onPress={() => {
                    setSelectedHospital(hosp._id);
                    setModalVisible(false); // Seçim yapınca pencereyi kapat
                  }}
                >
                  <Text style={styles.modalItemText}>
                    {hosp.name} ({hosp.city})
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            <TouchableOpacity
              style={styles.modalCancelBtn}
              onPress={() => setModalVisible(false)}
            >
              <Text style={styles.modalCancelText}>İptal</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  mainContainer: { flex: 1, backgroundColor: "#000" },
  scrollContent: { flexGrow: 1 },
  sheetContainer: { flex: 1, backgroundColor: "transparent" },
  sheetBackground: {
    flex: 1,
    paddingHorizontal: 25,
    paddingTop: 15,
    minHeight: height * 0.7,
  },
  handleBar: {
    width: 40,
    height: 4,
    backgroundColor: "rgba(255,255,255,0.15)",
    borderRadius: 10,
    alignSelf: "center",
    marginBottom: 20,
  },
  headerArea: { alignItems: "center", marginBottom: 20 },
  title: { fontSize: 28, fontWeight: "700", color: "#fff", letterSpacing: 2 },
  glassCard: {
    backgroundColor: "rgba(255, 255, 255, 0.08)",
    borderRadius: 30,
    padding: 25,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  input: {
    height: 48,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.15)",
    color: "#fff",
    fontSize: 15,
    marginBottom: 10,
    paddingHorizontal: 5,
  },
  primaryBtn: {
    height: 52,
    backgroundColor: "#fff",
    borderRadius: 15,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 25,
  },
  btnText: { color: "#071836", fontWeight: "bold", fontSize: 16 },
  footerLink: { alignItems: "center", marginTop: 20 },
  linkText: { color: "rgba(255,255,255,0.8)", fontSize: 14 },

  // YENİ EKLENEN STİLLER (Modal ve Buton için)
  customPickerButton: {
    height: 48,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.15)",
    justifyContent: "center",
    paddingHorizontal: 5,
    marginBottom: 10,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.6)", // Arka planı hafif karart
    justifyContent: "flex-end", // Pencereyi en alta yasla
  },
  modalContent: {
    backgroundColor: "#0a2044", // Uygulamanın temasına uygun koyu mavi
    borderTopLeftRadius: 25,
    borderTopRightRadius: 25,
    padding: 25,
    alignItems: "center",
  },
  modalHandleBar: {
    width: 50,
    height: 5,
    backgroundColor: "rgba(255,255,255,0.3)",
    borderRadius: 10,
    marginBottom: 20,
  },
  modalTitle: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 15,
  },
  modalItem: {
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.1)",
    width: "100%",
  },
  modalItemText: {
    color: "#fff",
    fontSize: 16,
    textAlign: "center",
  },
  modalCancelBtn: {
    marginTop: 20,
    width: "100%",
    padding: 15,
    backgroundColor: "rgba(255,255,255,0.1)",
    borderRadius: 15,
    alignItems: "center",
  },
  modalCancelText: {
    color: "#FF6B6B",
    fontSize: 16,
    fontWeight: "bold",
  },
});
