import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useState } from "react";
import {
  Alert,
  Dimensions,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

const { width } = Dimensions.get("window");

export default function HomeScreen() {
  const router = useRouter();

  // TypeScript hatasını çözen state tanımı
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (status !== "granted") {
      Alert.alert("İzin Gerekli", "Galeriye erişim izni vermeniz gerekiyor.");
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={["#071836", "#0D47A1"]}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={styles.safeArea}>
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <View>
              <Text style={styles.welcomeText}>Hoş Geldiniz,</Text>
              <Text style={styles.doctorName}>Dr. Umut Demir</Text>
            </View>
            <TouchableOpacity style={styles.profileBadge}>
              <Ionicons name="person-circle-outline" size={40} color="#fff" />
            </TouchableOpacity>
          </View>

          {/* ANA KART */}
          <TouchableOpacity
            style={styles.mainActionCard}
            activeOpacity={0.8}
            onPress={pickImage}
          >
            <LinearGradient
              colors={["rgba(255,255,255,0.2)", "rgba(255,255,255,0.05)"]}
              style={styles.cardGradient}
            >
              {selectedImage ? (
                <Image
                  source={{ uri: selectedImage }}
                  style={styles.previewImage}
                />
              ) : (
                <Ionicons name="scan-outline" size={50} color="#fff" />
              )}
              <Text style={styles.cardTitle}>
                {selectedImage ? "Görüntü Değiştir" : "Yeni Analiz Başlat"}
              </Text>
              <Text style={styles.cardDesc}>
                Röntgen dosyasını yükleyerek AI analizini başlatın.
              </Text>
            </LinearGradient>
          </TouchableOpacity>

          {/* ANALİZ BUTONU - Daha az transparan, uzun ve ince */}
          {selectedImage && (
            <TouchableOpacity
              style={styles.analyzeBtn}
              onPress={() => Alert.alert("Sistem", "AI taraması başlatıldı.")}
            >
              <LinearGradient
                colors={["rgba(255,255,255,0.3)", "rgba(255,255,255,0.1)"]} // Opaklığı artırdık
                style={styles.analyzeGradient}
              >
                <Text style={styles.analyzeBtnText}>ANALİZİ ÇALIŞTIR</Text>
                <Ionicons
                  name="chevron-forward"
                  size={16}
                  color="#fff"
                  style={{ marginLeft: 8 }}
                />
              </LinearGradient>
            </TouchableOpacity>
          )}

          <View style={styles.statsRow}>
            <View style={styles.statBox}>
              <Text style={styles.statNumber}>24</Text>
              <Text style={styles.statLabel}>Bugünkü Analiz</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statNumber}>%98</Text>
              <Text style={styles.statLabel}>Doğruluk Oranı</Text>
            </View>
          </View>

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Son İşlemler</Text>
          </View>

          {[1, 2, 3].map((item) => (
            <View key={item} style={styles.historyItem}>
              <View style={styles.historyIcon}>
                <Ionicons
                  name="document-text-outline"
                  size={24}
                  color="#A7C2F0"
                />
              </View>
              <View style={styles.historyInfo}>
                <Text style={styles.historyName}>Hasta #{1200 + item}</Text>
                <Text style={styles.historyDate}>28 Mart 2026</Text>
              </View>
            </View>
          ))}

          <TouchableOpacity
            style={styles.logoutBtn}
            onPress={() => router.replace("/")}
          >
            <Ionicons name="log-out-outline" size={20} color="#FF6B6B" />
            <Text style={styles.logoutText}>Oturumu Kapat</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  scrollContent: { padding: 25 },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 30,
    marginTop: 20,
  },
  welcomeText: { color: "rgba(255,255,255,0.6)", fontSize: 16 },
  doctorName: { color: "#fff", fontSize: 24, fontWeight: "bold" },
  profileBadge: { opacity: 0.8 },
  mainActionCard: {
    borderRadius: 30,
    overflow: "hidden",
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.15)",
  },
  cardGradient: { padding: 40, alignItems: "center" },
  previewImage: { width: 140, height: 140, borderRadius: 15, marginBottom: 10 },
  cardTitle: { color: "#fff", fontSize: 22, fontWeight: "bold", marginTop: 15 },
  cardDesc: {
    color: "rgba(255,255,255,0.5)",
    textAlign: "center",
    marginTop: 10,
  },

  // İsteklere göre güncellenen buton stili
  analyzeBtn: {
    borderRadius: 15,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.3)", // Kenarlık daha belirgin
    marginBottom: 30,
    width: "100%", // Tam genişlik (Uzun)
  },
  analyzeGradient: {
    paddingVertical: 10, // Yüksekliği iyice azalttık (İnce)
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
  },
  analyzeBtnText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 14, // Yazıyı biraz küçülttük
    letterSpacing: 2, // Harf arasını açtık
  },

  statsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 30,
  },
  statBox: {
    backgroundColor: "rgba(255,255,255,0.05)",
    width: (width - 70) / 2,
    padding: 20,
    borderRadius: 20,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.05)",
  },
  statNumber: { color: "#fff", fontSize: 24, fontWeight: "bold" },
  statLabel: { color: "rgba(255,255,255,0.4)", fontSize: 12, marginTop: 5 },
  sectionHeader: { marginBottom: 15 },
  sectionTitle: { color: "#fff", fontSize: 18, fontWeight: "bold" },
  historyItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.03)",
    padding: 15,
    borderRadius: 15,
    marginBottom: 10,
  },
  historyIcon: {
    width: 45,
    height: 45,
    backgroundColor: "rgba(167, 194, 240, 0.1)",
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 15,
  },
  historyInfo: { flex: 1 },
  historyName: { color: "#fff", fontSize: 16, fontWeight: "600" },
  historyDate: { color: "rgba(255,255,255,0.3)", fontSize: 12 },
  logoutBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 30,
    padding: 15,
  },
  logoutText: { color: "#FF6B6B", marginLeft: 8, fontWeight: "600" },
});
