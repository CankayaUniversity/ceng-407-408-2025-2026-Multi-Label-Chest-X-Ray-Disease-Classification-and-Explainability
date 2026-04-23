import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React from "react";
import {
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

export default function EntrySelectionScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={["#071836", "#0D47A1"]}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={styles.safeArea}>
        <View style={styles.content}>
          <View style={styles.heroBox}>
            <Text style={styles.brand}>CHESTXPLAIN</Text>
            <Text style={styles.subtitle}>
              Multi-Label Chest X-Ray Disease Classification and Explainability
            </Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.title}>Giriş Türünü Seçin</Text>
            <Text style={styles.description}>
              Sisteme doktor olarak giriş yapabilir veya hastane panelinden
              bekleyen doktor başvurularını yönetebilirsiniz.
            </Text>

            <TouchableOpacity
              style={styles.primaryButton}
              onPress={() => router.push("/doctor-login")}
            >
              <Ionicons name="medkit-outline" size={22} color="#071836" />
              <Text style={styles.primaryButtonText}>Doktor Girişi</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={() => router.push("/hospital-login")}
            >
              <Ionicons name="business-outline" size={22} color="#fff" />
              <Text style={styles.secondaryButtonText}>Hastane Girişi</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  content: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
  },
  heroBox: {
    marginBottom: 30,
    alignItems: "center",
  },
  brand: {
    color: "#fff",
    fontSize: 32,
    fontWeight: "200",
    letterSpacing: 6,
    textAlign: "center",
  },
  subtitle: {
    color: "rgba(255,255,255,0.58)",
    textAlign: "center",
    marginTop: 14,
    lineHeight: 22,
    fontSize: 14,
  },
  card: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 30,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  title: {
    color: "#fff",
    fontSize: 26,
    fontWeight: "700",
    textAlign: "center",
  },
  description: {
    color: "rgba(255,255,255,0.65)",
    textAlign: "center",
    fontSize: 14,
    lineHeight: 22,
    marginTop: 12,
    marginBottom: 26,
  },
  primaryButton: {
    height: 56,
    borderRadius: 16,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    marginBottom: 14,
  },
  primaryButtonText: {
    color: "#071836",
    fontWeight: "700",
    fontSize: 16,
    marginLeft: 8,
  },
  secondaryButton: {
    height: 56,
    borderRadius: 16,
    backgroundColor: "rgba(255,255,255,0.08)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.15)",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
  },
  secondaryButtonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 16,
    marginLeft: 8,
  },
});