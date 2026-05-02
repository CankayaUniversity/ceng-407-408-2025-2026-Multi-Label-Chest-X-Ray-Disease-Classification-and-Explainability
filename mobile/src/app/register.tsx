import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useState, useEffect } from "react";
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
  ActivityIndicator
} from "react-native";
import DoctorImg from "../../assets/doctor.png";
import { registerDoctor, getHospitalsFromDB } from "../services/authService";

const { height } = Dimensions.get("window");

export default function RegisterScreen() {
  const router = useRouter();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [hospitals, setHospitals] = useState<any[]>([]); 
  const [selectedHospitalName, setSelectedHospitalName] = useState(""); 
  const [isModalVisible, setModalVisible] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingHospitals, setIsLoadingHospitals] = useState(true);

  useEffect(() => {
    const fetchHospitals = async () => {
      try {
        const data = await getHospitalsFromDB();
        setHospitals(data);
      } catch (error) {
        console.error("Hastane çekme hatası:", error);
        Alert.alert("Hata", "Hastane listesi yüklenemedi.");
      } finally {
        setIsLoadingHospitals(false);
      }
    };
    fetchHospitals();
  }, []);

  const getSelectedHospitalLabel = () => {
    if (isLoadingHospitals) return "Hastaneler yükleniyor...";
    if (!selectedHospitalName) return "Hastane Seçiniz...";
    return selectedHospitalName;
  };

  const validateForm = () => {
    if (
      !fullName.trim() ||
      !email.trim() ||
      !password.trim() ||
      !selectedHospitalName
    ) {
      Alert.alert("Eksik Bilgi", "Lütfen tüm alanları doldurun ve hastane seçin.");
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      Alert.alert("Geçersiz E-posta", "Lütfen geçerli bir e-posta adresi girin.");
      return false;
    }

    if (password.trim().length < 6) {
      Alert.alert("Zayıf Şifre", "Şifre en az 6 karakter olmalıdır.");
      return false;
    }

    return true;
  };

  const handleRegister = async () => {
    if (!validateForm()) return;

    try {
      setIsSubmitting(true);

      await registerDoctor({
        full_name: fullName.trim(), 
        email: email.trim().toLowerCase(),
        password: password.trim(),
        hospital: selectedHospitalName,
      });

      router.replace("/pending-approval");
    } catch (error: any) {
      console.error("Register error:", error);
      Alert.alert("Kayıt Başarısız", error.message || "Bir hata oluştu.");
    } finally {
      setIsSubmitting(false);
    }
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
          keyboardShouldPersistTaps="handled"
          bounces={false}
        >
          <View style={{ height: height * 0.28 }} />

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
                <Text style={styles.title}>DOKTOR KAYIT</Text>
                <Text style={styles.subtitle}>
                  Başvurunuz hastane onayından sonra aktif olacaktır.
                </Text>
              </View>

              <View style={styles.glassCard}>
                <TextInput
                  style={styles.input}
                  placeholder="Ad Soyad"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={fullName}
                  onChangeText={setFullName}
                />

                <TextInput
                  style={styles.input}
                  placeholder="E-posta Adresi"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                  keyboardType="email-address"
                />

                <TextInput
                  style={styles.input}
                  placeholder="Şifre"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  secureTextEntry
                  value={password}
                  onChangeText={setPassword}
                  autoCapitalize="none"
                />

                <Text style={styles.fieldLabel}>Görev Yaptığınız Hastane</Text>

                <TouchableOpacity
                  style={styles.customPickerButton}
                  onPress={() => !isLoadingHospitals && setModalVisible(true)}
                  activeOpacity={0.75}
                >
                  <Text
                    style={[
                      styles.customPickerText,
                      !selectedHospitalName && styles.placeholderText,
                    ]}
                  >
                    {getSelectedHospitalLabel()}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.primaryBtn, isSubmitting && { opacity: 0.7 }]}
                  onPress={handleRegister}
                  disabled={isSubmitting || isLoadingHospitals}
                >
                  <Text style={styles.btnText}>
                    {isSubmitting ? "Oluşturuluyor..." : "Hesabı Oluştur"}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={() => router.replace("/doctor-login")}
                  style={styles.footerLink}
                >
                  <Text style={styles.linkText}>
                    Zaten hesabın var mı?{" "}
                    <Text style={{ fontWeight: "700" }}>Giriş Yap</Text>
                  </Text>
                </TouchableOpacity>
              </View>

              <View style={{ height: 140 }} />
            </LinearGradient>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Dinamik Hastane Listesi Modalı */}
      <Modal
        visible={isModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHandleBar} />
            <Text style={styles.modalTitle}>Hastane Seçin</Text>

            <ScrollView
              style={styles.modalScroll}
              showsVerticalScrollIndicator={false}
            >
              {hospitals.map((hospital) => (
                <TouchableOpacity
                  key={hospital._id}
                  style={styles.modalItem}
                  onPress={() => {
                    setSelectedHospitalName(hospital.name);
                    setModalVisible(false);
                  }}
                >
                  <Text style={styles.modalItemText}>
                    {hospital.name} {hospital.city ? `(${hospital.city})` : ""}
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
    paddingHorizontal: 24,
    paddingTop: 14,
    minHeight: height * 0.72,
  },
  handleBar: {
    width: 42,
    height: 4,
    backgroundColor: "rgba(255,255,255,0.18)",
    borderRadius: 999,
    alignSelf: "center",
    marginBottom: 18,
  },
  headerArea: { alignItems: "center", marginBottom: 18 },
  title: { fontSize: 28, fontWeight: "700", color: "#fff", letterSpacing: 2 },
  subtitle: {
    color: "rgba(255,255,255,0.58)",
    fontSize: 13,
    marginTop: 8,
    textAlign: "center",
    lineHeight: 20,
    paddingHorizontal: 10,
  },
  glassCard: {
    backgroundColor: "rgba(255, 255, 255, 0.08)",
    borderRadius: 30,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  input: {
    height: 50,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.15)",
    color: "#fff",
    fontSize: 15,
    marginBottom: 12,
    paddingHorizontal: 4,
  },
  fieldLabel: {
    color: "rgba(255,255,255,0.62)",
    marginTop: 10,
    marginLeft: 4,
    fontSize: 13,
    marginBottom: 6,
  },
  customPickerButton: {
    minHeight: 50,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.15)",
    justifyContent: "center",
    paddingHorizontal: 4,
    marginBottom: 12,
  },
  customPickerText: { color: "#fff", fontSize: 15 },
  placeholderText: { color: "rgba(255,255,255,0.4)" },
  primaryBtn: {
    height: 54,
    backgroundColor: "#fff",
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 22,
  },
  btnText: { color: "#071836", fontWeight: "700", fontSize: 16 },
  footerLink: { alignItems: "center", marginTop: 20 },
  linkText: { color: "rgba(255,255,255,0.82)", fontSize: 14 },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.6)",
    justifyContent: "flex-end",
  },
  modalContent: {
    backgroundColor: "#0A2044",
    borderTopLeftRadius: 26,
    borderTopRightRadius: 26,
    padding: 24,
    alignItems: "center",
  },
  modalHandleBar: {
    width: 52,
    height: 5,
    backgroundColor: "rgba(255,255,255,0.28)",
    borderRadius: 999,
    marginBottom: 18,
  },
  modalTitle: { color: "#fff", fontSize: 20, fontWeight: "700", marginBottom: 16 },
  modalScroll: { width: "100%", maxHeight: height * 0.4 },
  modalItem: {
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.09)",
    width: "100%",
  },
  modalItemText: { color: "#fff", fontSize: 16, textAlign: "center" },
  modalCancelBtn: {
    marginTop: 18,
    width: "100%",
    padding: 15,
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 15,
    alignItems: "center",
  },
  modalCancelText: { color: "#FF8C8C", fontSize: 16, fontWeight: "700" },
});