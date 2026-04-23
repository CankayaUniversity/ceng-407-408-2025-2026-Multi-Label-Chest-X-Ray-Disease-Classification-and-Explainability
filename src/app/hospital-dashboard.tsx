import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, useRouter } from "expo-router";
import React, { useEffect, useState } from "react";
import {
  Alert,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import PendingDoctorCard from "../components/PendingDoctorCard";
import {
  approveDoctor,
  getPendingDoctorsByHospital,
  rejectDoctor,
} from "../services/storageService";
import { Doctor } from "../types";

export default function HospitalDashboardScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const hospitalName =
    typeof params.hospitalName === "string"
      ? params.hospitalName
      : "Ankara Şehir Hastanesi";

  const [pendingDoctors, setPendingDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPendingDoctors();
  }, [hospitalName]);

  const loadPendingDoctors = async () => {
    try {
      setLoading(true);
      const doctors = await getPendingDoctorsByHospital(hospitalName);
      setPendingDoctors(doctors);
    } catch (error) {
      console.error("Load pending doctors error:", error);
      Alert.alert("Hata", "Bekleyen doktor başvuruları yüklenemedi.");
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (doctorId: string) => {
    try {
      const success = await approveDoctor(doctorId);

      if (!success) {
        Alert.alert("Hata", "Onay işlemi başarısız oldu.");
        return;
      }

      await loadPendingDoctors();
      Alert.alert("Başarılı", "Doktor hesabı onaylandı.");
    } catch (error) {
      console.error("Approve error:", error);
      Alert.alert("Hata", "Onay işlemi sırasında bir sorun oluştu.");
    }
  };

  const handleReject = async (doctorId: string) => {
    try {
      const success = await rejectDoctor(doctorId);

      if (!success) {
        Alert.alert("Hata", "Red işlemi başarısız oldu.");
        return;
      }

      await loadPendingDoctors();
      Alert.alert("Bilgi", "Doktor başvurusu reddedildi.");
    } catch (error) {
      console.error("Reject error:", error);
      Alert.alert("Hata", "Red işlemi sırasında bir sorun oluştu.");
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
              <Text style={styles.welcomeText}>Hastane Paneli</Text>
              <Text style={styles.hospitalName}>{hospitalName}</Text>
            </View>

            <TouchableOpacity
              style={styles.logoutBtn}
              onPress={() => router.replace("/")}
            >
              <Ionicons name="log-out-outline" size={20} color="#FFB3B3" />
            </TouchableOpacity>
          </View>

          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>Pending Doktor Başvuruları</Text>
            <Text style={styles.summaryDesc}>
              Bu hastaneye ait bekleyen doktor hesaplarını inceleyin ve uygun
              olan başvurular için onay veya red işlemi yapın.
            </Text>

            <View style={styles.summaryStats}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{pendingDoctors.length}</Text>
                <Text style={styles.statLabel}>Bekleyen Başvuru</Text>
              </View>

              <View style={styles.statDivider} />

              <View style={styles.statItem}>
                <Text style={styles.statNumber}>Live</Text>
                <Text style={styles.statLabel}>Approval Queue</Text>
              </View>
            </View>
          </View>

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Başvuru Listesi</Text>
          </View>

          {loading ? (
            <View style={styles.emptyBox}>
              <Ionicons name="time-outline" size={32} color="#D6E4FF" />
              <Text style={styles.emptyTitle}>Yükleniyor...</Text>
              <Text style={styles.emptyText}>
                Pending doktor başvuruları getiriliyor.
              </Text>
            </View>
          ) : pendingDoctors.length > 0 ? (
            pendingDoctors.map((doctor) => (
              <PendingDoctorCard
                key={doctor._id}
                doctor={doctor}
                onApprove={() => handleApprove(doctor._id)}
                onReject={() => handleReject(doctor._id)}
              />
            ))
          ) : (
            <View style={styles.emptyBox}>
              <Ionicons
                name="checkmark-done-outline"
                size={34}
                color="#D6E4FF"
              />
              <Text style={styles.emptyTitle}>Bekleyen başvuru kalmadı</Text>
              <Text style={styles.emptyText}>
                Şu anda bu hastane için onay bekleyen doktor kaydı bulunmuyor.
              </Text>
            </View>
          )}

          <TouchableOpacity
            style={styles.refreshBtn}
            onPress={loadPendingDoctors}
          >
            <Ionicons name="refresh-outline" size={18} color="#D6E4FF" />
            <Text style={styles.refreshBtnText}>Listeyi Yenile</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  scrollContent: {
    padding: 20,
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
    color: "rgba(255,255,255,0.56)",
    fontSize: 15,
  },
  hospitalName: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "700",
    marginTop: 4,
  },
  logoutBtn: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.08)",
    justifyContent: "center",
    alignItems: "center",
  },
  summaryCard: {
    backgroundColor: "rgba(255,255,255,0.07)",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    marginBottom: 22,
  },
  summaryTitle: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "700",
  },
  summaryDesc: {
    color: "rgba(255,255,255,0.62)",
    fontSize: 14,
    lineHeight: 22,
    marginTop: 10,
  },
  summaryStats: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 18,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 18,
    paddingVertical: 14,
    paddingHorizontal: 10,
  },
  statItem: {
    flex: 1,
    alignItems: "center",
  },
  statNumber: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "700",
  },
  statLabel: {
    color: "rgba(255,255,255,0.5)",
    fontSize: 12,
    marginTop: 5,
    textAlign: "center",
  },
  statDivider: {
    width: 1,
    height: 34,
    backgroundColor: "rgba(255,255,255,0.08)",
  },
  sectionHeader: {
    marginBottom: 12,
  },
  sectionTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
  emptyBox: {
    marginTop: 20,
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.06)",
    borderRadius: 22,
    padding: 28,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  emptyTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
    marginTop: 12,
  },
  emptyText: {
    color: "rgba(255,255,255,0.6)",
    textAlign: "center",
    lineHeight: 22,
    marginTop: 8,
  },
  refreshBtn: {
    marginTop: 14,
    alignSelf: "center",
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.06)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  refreshBtnText: {
    color: "#D6E4FF",
    fontWeight: "700",
    marginLeft: 8,
  },
});