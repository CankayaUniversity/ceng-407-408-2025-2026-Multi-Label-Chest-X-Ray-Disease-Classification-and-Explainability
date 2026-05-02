import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useState, useEffect } from "react";
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  ActivityIndicator
} from "react-native";
import { Picker } from '@react-native-picker/picker';
import { loginHospital } from "../services/authService";
import { API_BASE_URL } from "../services/api"; 

export default function HospitalLoginScreen() {
  const router = useRouter();

  const [hospitals, setHospitals] = useState<any[]>([]);
  const [selectedHospital, setSelectedHospital] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingHospitals, setIsLoadingHospitals] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE_URL}/auth/hospitals`)
      .then(res => res.json())
      .then(data => {
        setHospitals(data);
        if (data.length > 0) setSelectedHospital(data[0].name);
      })
      .catch(err => {
        console.error("Hastane çekme hatası:", err);
        Alert.alert("Hata", "Hastane listesi yüklenemedi. Sunucu bağlantınızı kontrol edin.");
      })
      .finally(() => setIsLoadingHospitals(false));
  }, []);

  const handleHospitalLogin = async () => {
    if (!selectedHospital.trim() || !password.trim()) {
      Alert.alert("Eksik Bilgi", "Lütfen hastane seçin ve şifre girin.");
      return;
    }

    try {
      setIsSubmitting(true);
      const result = await loginHospital(selectedHospital, password);

      if (result && result.hospital_name) {
        router.replace({
          pathname: "/hospital-dashboard",
          params: { hospitalName: result.hospital_name },
        });
      } else {
        Alert.alert("Hata", "Beklenmedik bir yanıt alındı.");
      }
    } catch (error: any) {
      Alert.alert("Giriş Başarısız", error.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={["#071836", "#0D47A1"]}
        style={StyleSheet.absoluteFill}
      />

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.card}>
            <View style={styles.topBadge}>
              <Text style={styles.topBadgeText}>HOSPITAL PANEL</Text>
            </View>

            <Text style={styles.title}>Hastane Girişi</Text>
            <Text style={styles.subtitle}>
              Hastane paneline giriş yaptıktan sonra bekleyen doktor
              başvurularını görüntüleyip onaylayabilirsiniz.
            </Text>

            {/* Picker Açılır Liste Alanı */}
            <View style={styles.pickerContainer}>
              {isLoadingHospitals ? (
                <ActivityIndicator color="#fff" style={{ padding: 10 }} />
              ) : (
                <Picker
                  selectedValue={selectedHospital}
                  onValueChange={(itemValue) => setSelectedHospital(itemValue)}
                  dropdownIconColor="#fff"
                  style={styles.picker}
                >
                  {hospitals.map((h, index) => (
                    <Picker.Item 
                      key={index} 
                      label={h.name} 
                      value={h.name} 
                      color={Platform.OS === 'ios' ? '#fff' : '#000'} 
                    />
                  ))}
                </Picker>
              )}
            </View>

            <TextInput
              style={styles.input}
              placeholder="Şifre"
              placeholderTextColor="rgba(255,255,255,0.4)"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />

            <TouchableOpacity
              style={[styles.primaryBtn, isSubmitting && { opacity: 0.7 }]}
              onPress={handleHospitalLogin}
              disabled={isSubmitting || isLoadingHospitals}
            >
              <Text style={styles.primaryBtnText}>
                {isSubmitting ? "Panele Giriliyor..." : "Panele Giriş Yap"}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.backLink}
              onPress={() => router.replace("/")}
            >
              <Text style={styles.backLinkText}>Başlangıç ekranına dön</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.helperCard}>
            <Text style={styles.helperTitle}>Test Hesabı</Text>
            <Text style={styles.helperText}>
              Backend'deki "hospitals" koleksiyonunda olan bir hastaneyi seçip
              ilgili şifreyi giriniz. (Örn: 123 veya 123456)
            </Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    padding: 24,
  },
  card: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 30,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  topBadge: {
    alignSelf: "center",
    backgroundColor: "rgba(255,255,255,0.08)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    marginBottom: 18,
  },
  topBadgeText: {
    color: "#D6E4FF",
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 1,
  },
  title: {
    color: "#fff",
    fontSize: 28,
    fontWeight: "700",
    textAlign: "center",
  },
  subtitle: {
    color: "rgba(255,255,255,0.62)",
    fontSize: 14,
    lineHeight: 22,
    textAlign: "center",
    marginTop: 10,
    marginBottom: 24,
  },
  pickerContainer: {
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.15)",
    marginBottom: 16,
    justifyContent: "center",
    height: 52, 
  },
  picker: {
    color: "#fff",
    width: "100%",
    marginLeft: -8, 
  },
  input: {
    height: 52,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.15)",
    color: "#fff",
    fontSize: 15,
    marginBottom: 16,
    paddingHorizontal: 4,
  },
  primaryBtn: {
    backgroundColor: "#fff",
    height: 52,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 12,
  },
  primaryBtnText: {
    color: "#071836",
    fontSize: 16,
    fontWeight: "700",
  },
  backLink: {
    marginTop: 18,
    alignItems: "center",
  },
  backLinkText: {
    color: "rgba(255,255,255,0.55)",
    fontSize: 13,
  },
  helperCard: {
    marginTop: 16,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  helperTitle: {
    color: "#fff",
    fontWeight: "700",
    marginBottom: 10,
    fontSize: 15,
  },
  helperText: {
    color: "rgba(255,255,255,0.68)",
    fontSize: 13,
    marginBottom: 4,
    lineHeight: 20,
  },
});