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

export default function RejectedAccountScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={["#2A0E12", "#7A1F2B"]}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={styles.safeArea}>
        <View style={styles.content}>
          <View style={styles.iconWrapper}>
            <Ionicons name="close-circle-outline" size={64} color="#fff" />
          </View>

          <Text style={styles.title}>Başvurunuz Reddedildi</Text>

          <Text style={styles.description}>
            Hesabınızın aktivasyonu onaylanmadı.{"\n\n"}
            Daha fazla bilgi almak için sistem yöneticiniz veya hastane yetkiliniz ile iletişime geçebilirsiniz.
          </Text>

          <TouchableOpacity
            style={styles.button}
            onPress={() => router.replace("/")}
          >
            <Text style={styles.buttonText}>Giriş Ekranına Dön</Text>
          </TouchableOpacity>
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
    alignItems: "center",
    paddingHorizontal: 28,
  },
  iconWrapper: {
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: "rgba(255,255,255,0.12)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 28,
  },
  title: {
    color: "#fff",
    fontSize: 28,
    fontWeight: "700",
    marginBottom: 18,
    textAlign: "center",
  },
  description: {
    color: "rgba(255,255,255,0.82)",
    fontSize: 16,
    lineHeight: 25,
    textAlign: "center",
    marginBottom: 34,
  },
  button: {
    backgroundColor: "#fff",
    paddingVertical: 15,
    paddingHorizontal: 28,
    borderRadius: 16,
    minWidth: 220,
    alignItems: "center",
  },
  buttonText: {
    color: "#7A1F2B",
    fontSize: 16,
    fontWeight: "700",
  },
});