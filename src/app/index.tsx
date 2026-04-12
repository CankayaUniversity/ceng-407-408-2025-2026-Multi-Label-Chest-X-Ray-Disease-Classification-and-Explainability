import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useState } from "react";
import {
  Dimensions,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import XrayImg from "../../assets/xray_img.png";

const { height } = Dimensions.get("window");

export default function LoginScreen() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // Giriş kontrol fonksiyonu (admin/admin için)
  const handleLogin = () => {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    if (cleanEmail === "admin" && cleanPassword === "admin") {
      router.replace("/home");
    } else {
      alert("Hatalı giriş! (E-posta: admin, Şifre: admin)");
    }
  };

  return (
    <View style={styles.mainContainer}>
      <ImageBackground
        source={XrayImg}
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
          <View style={{ height: height * 0.45 }} />

          <View style={styles.sheetContainer}>
            <LinearGradient
              colors={[
                "transparent",
                "rgb(8, 37, 86)",
                "rgba(95, 120, 161, 0.78)",
                "#071836",
              ]}
              style={styles.sheetBackground}
            >
              <View style={styles.handleBar} />

              <View style={styles.headerArea}>
                <Text style={styles.brandName}>CHESTXPLAIN</Text>
                <Text style={styles.tagline}>
                  AI-Powered Radiology Assistant
                </Text>
              </View>

              <View style={styles.glassCard}>
                <TextInput
                  style={styles.input}
                  placeholder="E-posta Adresi"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none" // admin yazarken büyük harf hatasını önler
                  autoCorrect={false}
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

                <TouchableOpacity
                  style={styles.primaryBtn}
                  onPress={handleLogin} // Yeni kontrol fonksiyonu
                >
                  <Text style={styles.btnText}>Giriş Yap</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={() => router.push("/register")}
                  style={styles.footerLink}
                >
                  <Text style={styles.linkText}>
                    Hesabın yok mu?{" "}
                    <Text style={{ fontWeight: "bold" }}>Kaydol</Text>
                  </Text>
                </TouchableOpacity>
              </View>
              <View style={{ height: 150 }} />
            </LinearGradient>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  mainContainer: { flex: 1, backgroundColor: "#000" },
  scrollContent: { flexGrow: 1 },
  sheetContainer: {
    flex: 1,
    backgroundColor: "transparent",
  },
  sheetBackground: {
    flex: 1,
    paddingHorizontal: 25,
    paddingTop: 15,
    minHeight: height * 0.6,
  },
  handleBar: {
    width: 40,
    height: 4,
    backgroundColor: "rgba(255,255,255,0.15)",
    borderRadius: 10,
    alignSelf: "center",
    marginBottom: 20,
  },
  headerArea: { alignItems: "center", marginBottom: 25 },
  brandName: {
    fontSize: 32,
    fontWeight: "200",
    color: "#fff",
    letterSpacing: 6,
  },
  tagline: {
    color: "rgba(255,255,255,0.4)",
    fontSize: 11,
    letterSpacing: 1.5,
    marginTop: 5,
  },
  glassCard: {
    backgroundColor: "rgba(255, 255, 255, 0.08)",
    borderRadius: 30,
    padding: 25,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  input: {
    height: 50,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.15)",
    color: "#fff",
    fontSize: 15,
    marginBottom: 15,
    paddingHorizontal: 5,
  },
  primaryBtn: {
    height: 52,
    backgroundColor: "#fff",
    borderRadius: 15,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 15,
  },
  btnText: { color: "#071836", fontWeight: "bold", fontSize: 16 },
  footerLink: { alignItems: "center", marginTop: 25 },
  linkText: { color: "rgba(255,255,255,0.8)", fontSize: 14 },
});
