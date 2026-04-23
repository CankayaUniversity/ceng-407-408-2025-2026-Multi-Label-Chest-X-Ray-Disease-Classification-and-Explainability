import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useState } from "react";
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
} from "react-native";
import { loginHospital } from "../services/authService";

export default function HospitalLoginScreen() {
  const router = useRouter();

  const [hospitalName, setHospitalName] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleHospitalLogin = async () => {
    if (!hospitalName.trim() || !password.trim()) {
      Alert.alert("Eksik Bilgi", "Lütfen hastane adı ve şifre girin.");
      return;
    }

    try {
      setIsSubmitting(true);

      const result = await loginHospital(hospitalName, password);

      if (!result.success) {
        Alert.alert("Giriş Başarısız", result.message || "Bir hata oluştu.");
        return;
      }

      router.replace({
        pathname: "/hospital-dashboard",
        params: { hospitalName: result.hospitalName || hospitalName.trim() },
      });
    } catch (error) {
      console.error("Hospital login error:", error);
      Alert.alert("Hata", "Giriş işlemi sırasında bir sorun oluştu.");
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

            <TextInput
              style={styles.input}
              placeholder="Hastane Adı"
              placeholderTextColor="rgba(255,255,255,0.4)"
              value={hospitalName}
              onChangeText={setHospitalName}
            />

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
              disabled={isSubmitting}
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
              Hastane Adı: Ankara Şehir Hastanesi
            </Text>
            <Text style={styles.helperText}>Şifre: 123456</Text>
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
  },
});