import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React from "react";
import { SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from "react-native";

export default function PendingApprovalScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={["#071836", "#0D47A1"]}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={styles.safeArea}>
        <View style={styles.content}>
          <View style={styles.iconWrapper}>
            <Ionicons name="mail-unread-outline" size={64} color="#fff" />
          </View>

          <Text style={styles.title}>Başvurunuz Alındı</Text>

          <Text style={styles.description}>
            Hesabınız başarıyla oluşturuldu.{"\n\n"}
            Şu anda hastane / yönetici onayı bekleniyor. Onay tamamlandıktan sonra sisteme giriş yapabilirsiniz.
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
    color: "rgba(255,255,255,0.78)",
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
    color: "#071836",
    fontSize: 16,
    fontWeight: "700",
  },
});