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
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const doctorName = "Dr. Umut Demir";

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (status !== "granted") {
      Alert.alert("İzin Gerekli", "Galeriye erişim izni vermeniz gerekiyor.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) {
      Alert.alert("Görüntü Seçilmedi", "Lütfen önce bir akciğer röntgeni seçin.");
      return;
    }

    try {
      setIsAnalyzing(true);

      // Backend hazır olduğunda burada analiz endpoint'ine istek atılacak.
      // Örnek:
      // const result = await analyzeXray(selectedImage);
      // router.push({ pathname: "/analysis-result", params: { ... } });

      await new Promise((resolve) => setTimeout(resolve, 1200));

      Alert.alert(
        "Analiz Başlatıldı",
        "Demo akışında analiz başarıyla tetiklendi. Backend bağlandığında sonuç ekranına yönlendirme yapılacak."
      );
    } catch (error) {
      console.error("Analyze error:", error);
      Alert.alert("Hata", "Analiz sırasında bir sorun oluştu.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const recentAnalyses = [
    { id: 1, patient: "Hasta #1201", date: "13 Nisan 2026", status: "Tamamlandı" },
    { id: 2, patient: "Hasta #1202", date: "13 Nisan 2026", status: "Tamamlandı" },
    { id: 3, patient: "Hasta #1203", date: "12 Nisan 2026", status: "Tamamlandı" },
  ];

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
              <Text style={styles.doctorName}>{doctorName}</Text>
            </View>

            <TouchableOpacity style={styles.profileBadge}>
              <Ionicons name="person-circle-outline" size={40} color="#fff" />
            </TouchableOpacity>
          </View>

          <View style={styles.heroInfoCard}>
            <Text style={styles.heroTitle}>Chest X-Ray Analysis Workspace</Text>
            <Text style={styles.heroDesc}>
              Göğüs röntgeni görüntüsünü yükleyerek çoklu hastalık sınıflandırma
              analizini başlatın ve explainability çıktıları için altyapıyı kullanın.
            </Text>
          </View>

          <TouchableOpacity
            style={styles.mainActionCard}
            activeOpacity={0.85}
            onPress={pickImage}
          >
            <LinearGradient
              colors={["rgba(255,255,255,0.18)", "rgba(255,255,255,0.05)"]}
              style={styles.cardGradient}
            >
              {selectedImage ? (
                <Image source={{ uri: selectedImage }} style={styles.previewImage} />
              ) : (
                <View style={styles.uploadPlaceholder}>
                  <Ionicons name="scan-outline" size={52} color="#fff" />
                </View>
              )}

              <Text style={styles.cardTitle}>
                {selectedImage ? "Görüntüyü Değiştir" : "Yeni Analiz Başlat"}
              </Text>

              <Text style={styles.cardDesc}>
                {selectedImage
                  ? "Seçili görüntü hazır. Analizi çalıştırabilirsiniz."
                  : "Akciğer röntgeni yükleyerek AI destekli değerlendirmeyi başlatın."}
              </Text>
            </LinearGradient>
          </TouchableOpacity>

          {selectedImage && (
            <TouchableOpacity
              style={[styles.analyzeBtn, isAnalyzing && { opacity: 0.7 }]}
              onPress={handleAnalyze}
              disabled={isAnalyzing}
            >
              <LinearGradient
                colors={["rgba(255,255,255,0.28)", "rgba(255,255,255,0.10)"]}
                style={styles.analyzeGradient}
              >
                <Text style={styles.analyzeBtnText}>
                  {isAnalyzing ? "ANALİZ EDİLİYOR..." : "ANALİZİ ÇALIŞTIR"}
                </Text>
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
              <Text style={styles.statLabel}>Model Güven Skoru</Text>
            </View>
          </View>

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Son İşlemler</Text>
          </View>

          {recentAnalyses.map((item) => (
            <View key={item.id} style={styles.historyItem}>
              <View style={styles.historyIcon}>
                <Ionicons
                  name="document-text-outline"
                  size={24}
                  color="#A7C2F0"
                />
              </View>

              <View style={styles.historyInfo}>
                <Text style={styles.historyName}>{item.patient}</Text>
                <Text style={styles.historyDate}>{item.date}</Text>
              </View>

              <View style={styles.historyStatusBadge}>
                <Text style={styles.historyStatusText}>{item.status}</Text>
              </View>
            </View>
          ))}

          <TouchableOpacity
            style={styles.logoutBtn}
            onPress={() => router.replace("/")}
          >
            <Ionicons name="log-out-outline" size={20} color="#FF9A9A" />
            <Text style={styles.logoutText}>Oturumu Kapat</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 10,
    marginBottom: 24,
  },
  welcomeText: {
    color: "rgba(255,255,255,0.6)",
    fontSize: 16,
  },
  doctorName: {
    color: "#fff",
    fontSize: 25,
    fontWeight: "700",
    marginTop: 4,
  },
  profileBadge: {
    opacity: 0.9,
  },
  heroInfoCard: {
    backgroundColor: "rgba(255,255,255,0.06)",
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    marginBottom: 18,
  },
  heroTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
  heroDesc: {
    color: "rgba(255,255,255,0.62)",
    lineHeight: 21,
    marginTop: 8,
    fontSize: 14,
  },
  mainActionCard: {
    borderRadius: 28,
    overflow: "hidden",
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.14)",
  },
  cardGradient: {
    padding: 34,
    alignItems: "center",
  },
  uploadPlaceholder: {
    width: 132,
    height: 132,
    borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.08)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 10,
  },
  previewImage: {
    width: 140,
    height: 140,
    borderRadius: 18,
    marginBottom: 10,
  },
  cardTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "700",
    marginTop: 12,
  },
  cardDesc: {
    color: "rgba(255,255,255,0.52)",
    textAlign: "center",
    marginTop: 10,
    lineHeight: 20,
    fontSize: 14,
  },
  analyzeBtn: {
    borderRadius: 16,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.26)",
    marginBottom: 28,
    width: "100%",
  },
  analyzeGradient: {
    paddingVertical: 13,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
  },
  analyzeBtnText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 14,
    letterSpacing: 1.5,
  },
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 28,
  },
  statBox: {
    backgroundColor: "rgba(255,255,255,0.05)",
    width: (width - 64) / 2,
    paddingVertical: 20,
    paddingHorizontal: 12,
    borderRadius: 20,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.07)",
  },
  statNumber: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "700",
  },
  statLabel: {
    color: "rgba(255,255,255,0.42)",
    fontSize: 12,
    marginTop: 6,
    textAlign: "center",
  },
  sectionHeader: {
    marginBottom: 14,
  },
  sectionTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
  historyItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.04)",
    padding: 15,
    borderRadius: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.05)",
  },
  historyIcon: {
    width: 46,
    height: 46,
    backgroundColor: "rgba(167, 194, 240, 0.10)",
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 14,
  },
  historyInfo: {
    flex: 1,
  },
  historyName: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  historyDate: {
    color: "rgba(255,255,255,0.34)",
    fontSize: 12,
    marginTop: 3,
  },
  historyStatusBadge: {
    backgroundColor: "rgba(46, 204, 113, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(46, 204, 113, 0.22)",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  historyStatusText: {
    color: "#D9FFE6",
    fontSize: 11,
    fontWeight: "700",
  },
  logoutBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 26,
    paddingVertical: 14,
  },
  logoutText: {
    color: "#FF9A9A",
    marginLeft: 8,
    fontWeight: "700",
  },
});