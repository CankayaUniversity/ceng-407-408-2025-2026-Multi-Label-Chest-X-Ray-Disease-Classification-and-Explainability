import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useState } from "react";
import AsyncStorage from '@react-native-async-storage/async-storage';
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
import { loginDoctor } from "../services/authService";

export default function DoctorLoginScreen() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleDoctorLogin = async () => {
    if (!email.trim() || !password.trim()) {
      Alert.alert("Eksik Bilgi", "Lütfen e-posta ve şifre girin.");
      return;
    }

    try {
      setIsSubmitting(true);

      const result = await loginDoctor({
        email: email.trim().toLowerCase(),
        password: password.trim(),
      });

      if (result.role === "doctor") {
        

        const doctorName = result.full_name || "Doktor";
        const hospitalName = result.hospital || "";

        await AsyncStorage.setItem('userEmail', email.trim().toLowerCase());
        await AsyncStorage.setItem('full_name', doctorName);
        await AsyncStorage.setItem('hospital_name', hospitalName);
        
        console.log(`Giriş Başarılı: ${doctorName} (@${hospitalName})`);

        router.replace("/home");
      } else {
        Alert.alert("Yetki Hatası", "Bu panel sadece doktor erişimine açıktır.");
      }
    } catch (error: any) {
      console.error("Doctor login error:", error);
      Alert.alert("Giriş Başarısız", error.message || "Bir hata oluştu.");
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
              <Text style={styles.topBadgeText}>DOCTOR ACCESS</Text>
            </View>

            <Text style={styles.title}>Doktor Girişi</Text>
            <Text style={styles.subtitle}>
              Onaylı hesabınızla sisteme giriş yapabilirsiniz.
            </Text>

            <TextInput
              style={styles.input}
              placeholder="E-posta Adresi"
              placeholderTextColor="rgba(255,255,255,0.4)"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
            />

            <TextInput
              style={styles.input}
              placeholder="Şifre"
              placeholderTextColor="rgba(255,255,255,0.4)"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
              autoCapitalize="none"
            />

            <TouchableOpacity
              style={[styles.primaryBtn, isSubmitting && { opacity: 0.7 }]}
              onPress={handleDoctorLogin}
              disabled={isSubmitting}
            >
              <Text style={styles.primaryBtnText}>
                {isSubmitting ? "Giriş Yapılıyor..." : "Giriş Yap"}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.secondaryLink}
              onPress={() => router.push("/register")}
            >
              <Text style={styles.secondaryLinkText}>
                Hesabın yok mu? <Text style={styles.boldText}>Kaydol</Text>
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.backLink}
              onPress={() => router.replace("/")}
            >
              <Text style={styles.backLinkText}>Başlangıç ekranına dön</Text>
            </TouchableOpacity>
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
  secondaryLink: {
    marginTop: 22,
    alignItems: "center",
  },
  secondaryLinkText: {
    color: "rgba(255,255,255,0.86)",
    fontSize: 14,
  },
  boldText: {
    fontWeight: "700",
  },
  backLink: {
    marginTop: 14,
    alignItems: "center",
  },
  backLinkText: {
    color: "rgba(255,255,255,0.55)",
    fontSize: 13,
  },
});