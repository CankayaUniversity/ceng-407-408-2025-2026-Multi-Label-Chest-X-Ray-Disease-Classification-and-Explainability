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
  ActivityIndicator
} from "react-native";
import PendingDoctorCard from "../components/PendingDoctorCard";
import { API_BASE_URL } from "../services/api"; 

export default function HospitalDashboardScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const hospitalName = typeof params.hospitalName === "string" ? params.hospitalName : "Hastane";

  const [activeTab, setActiveTab] = useState<'bekleyen' | 'onayli'>('bekleyen');
  const [doctors, setDoctors] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDoctors();
  }, [hospitalName, activeTab]);

  const loadDoctors = async () => {
    try {
      setLoading(true);
      const endpoint = activeTab === 'bekleyen' ? 'pending-doctors' : 'approved-doctors';
      const response = await fetch(`${API_BASE_URL}/auth/${endpoint}/${hospitalName}`);
      if (!response.ok) throw new Error("Ağ hatası");
      const data = await response.json();
      setDoctors(data);
    } catch (error) {
      console.error("Doktorlar yüklenirken hata:", error);
      Alert.alert("Hata", "Doktor listesi yüklenemedi.");
    } finally {
      setLoading(false);
    }
  };

  const handleAction = async (email: string, action: 'onayla' | 'reddet' | 'sil') => {
    try {
      const apiAction = action === 'onayla' ? 'approve' : action === 'reddet' ? 'reject' : 'delete';
      const response = await fetch(`${API_BASE_URL}/auth/manage-doctor`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, action: apiAction })
      });

      if (response.ok) {
        const actionLabel = action === 'onayla' ? 'Onaylama' : action === 'reddet' ? 'Reddetme' : 'Silme';
        Alert.alert("Başarılı", `${actionLabel} işlemi tamamlandı.`);
        loadDoctors();
      } else {
        const errorData = await response.json();
        Alert.alert("Hata", errorData.detail || "İşlem yapılamadı.");
      }
    } catch (error) {
      Alert.alert("Hata", "Sunucu ile iletişim kurulamadı.");
    }
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={["#071836", "#0D47A1"]} style={StyleSheet.absoluteFill} />
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View>
              <Text style={styles.welcomeText}>Hastane Paneli</Text>
              <Text style={styles.hospitalName}>{hospitalName}</Text>
            </View>
            <TouchableOpacity style={styles.logoutBtn} onPress={() => router.replace("/")}>
              <Ionicons name="log-out-outline" size={20} color="#FFB3B3" />
            </TouchableOpacity>
          </View>

          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>
              {activeTab === 'bekleyen' ? 'Bekleyen Başvurular' : 'Mevcut Doktorlar'}
            </Text>
            <Text style={styles.summaryDesc}>
              {activeTab === 'bekleyen' 
                ? 'Onay bekleyen doktor başvurularını inceleyin.'
                : 'Aktif olarak çalışan doktorların listesi.'}
            </Text>
            <View style={styles.summaryStats}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{doctors.length}</Text>
                <Text style={styles.statLabel}>{activeTab === 'bekleyen' ? 'Bekleyen' : 'Aktif'}</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>Aktif</Text>
                <Text style={styles.statLabel}>Sistem</Text>
              </View>
            </View>
          </View>

          <View style={styles.tabContainer}>
            <TouchableOpacity 
              style={[styles.tab, activeTab === 'bekleyen' && styles.activeTab]} 
              onPress={() => setActiveTab('bekleyen')}
            >
              <Text style={[styles.tabText, activeTab === 'bekleyen' && styles.activeTabText]}>Bekleyenler</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.tab, activeTab === 'onayli' && styles.activeTab]} 
              onPress={() => setActiveTab('onayli')}
            >
              <Text style={[styles.tabText, activeTab === 'onayli' && styles.activeTabText]}>Onaylılar</Text>
            </TouchableOpacity>
          </View>

          {loading ? (
            <View style={styles.emptyBox}><ActivityIndicator size="large" color="#D6E4FF" /></View>
          ) : doctors.length > 0 ? (
            doctors.map((doctor) => (
              activeTab === 'bekleyen' ? (
                <PendingDoctorCard
                  key={doctor._id}
                  doctor={doctor}
                  onApprove={() => handleAction(doctor.email, 'onayla')}
                  onReject={() => handleAction(doctor.email, 'reddet')}
                />
              ) : (
                <View key={doctor._id} style={styles.approvedDocCard}>
                  <View style={styles.approvedDocInfo}>
                    <Text style={styles.docNameText}>{doctor.full_name}</Text>
                    <Text style={styles.docEmailText}>{doctor.email}</Text>
                  </View>
                  <TouchableOpacity 
                    style={styles.deleteBtn}
                    onPress={() => {
                      Alert.alert(
                        "Doktoru Sil",
                        `${doctor.full_name} isimli doktoru silmek istiyor musunuz?`,
                        [
                          { text: "İptal", style: "cancel" },
                          { text: "Sil", style: "destructive", onPress: () => handleAction(doctor.email, 'sil') }
                        ]
                      );
                    }}
                  >
                    <Text style={styles.btnTextSmall}>Sil</Text>
                  </TouchableOpacity>
                </View>
              )
            ))
          ) : (
            <View style={styles.emptyBox}>
              <Text style={styles.emptyTitle}>Liste boş</Text>
            </View>
          )}

          <TouchableOpacity style={styles.refreshBtn} onPress={loadDoctors}>
            <Ionicons name="refresh-outline" size={18} color="#D6E4FF" />
            <Text style={styles.refreshBtnText}>Yenile</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 40 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 10, marginBottom: 24 },
  welcomeText: { color: "rgba(255,255,255,0.56)", fontSize: 15 },
  hospitalName: { color: "#fff", fontSize: 24, fontWeight: "700", marginTop: 4 },
  logoutBtn: { width: 44, height: 44, borderRadius: 14, backgroundColor: "rgba(255,255,255,0.08)", justifyContent: "center", alignItems: "center" },
  summaryCard: { backgroundColor: "rgba(255,255,255,0.07)", borderRadius: 24, padding: 18, borderWidth: 1, borderColor: "rgba(255,255,255,0.10)", marginBottom: 22 },
  summaryTitle: { color: "#fff", fontSize: 20, fontWeight: "700" },
  summaryDesc: { color: "rgba(255,255,255,0.62)", fontSize: 14, lineHeight: 22, marginTop: 10 },
  summaryStats: { flexDirection: "row", alignItems: "center", marginTop: 18, backgroundColor: "rgba(255,255,255,0.04)", borderRadius: 18, paddingVertical: 14, paddingHorizontal: 10 },
  statItem: { flex: 1, alignItems: "center" },
  statNumber: { color: "#fff", fontSize: 20, fontWeight: "700" },
  statLabel: { color: "rgba(255,255,255,0.5)", fontSize: 12, marginTop: 5, textAlign: "center" },
  statDivider: { width: 1, height: 34, backgroundColor: "rgba(255,255,255,0.08)" },
  tabContainer: { flexDirection: 'row', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: 16, padding: 6, marginBottom: 24, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  tab: { flex: 1, paddingVertical: 12, alignItems: 'center', borderRadius: 12 },
  activeTab: { backgroundColor: '#fff' },
  tabText: { color: 'rgba(255,255,255,0.6)', fontWeight: '600', fontSize: 15 },
  activeTabText: { color: '#071836' },
  approvedDocCard: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.05)', padding: 16, borderRadius: 18, marginBottom: 12, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  approvedDocInfo: { flex: 1 },
  docNameText: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginBottom: 4 },
  docEmailText: { color: 'rgba(255,255,255,0.5)', fontSize: 13 },
  deleteBtn: { paddingHorizontal: 15, paddingVertical: 8, backgroundColor: 'rgba(255,50,50,0.2)', borderRadius: 10, marginLeft: 10 },
  btnTextSmall: { color: '#FFB3B3', fontWeight: 'bold', fontSize: 13 },
  emptyBox: { marginTop: 20, alignItems: "center", backgroundColor: "rgba(255,255,255,0.06)", borderRadius: 22, padding: 28, borderWidth: 1, borderColor: "rgba(255,255,255,0.10)" },
  emptyTitle: { color: "#fff", fontSize: 18, fontWeight: "700" },
  refreshBtn: { marginTop: 14, alignSelf: "center", flexDirection: "row", alignItems: "center", paddingHorizontal: 16, paddingVertical: 12, borderRadius: 14, backgroundColor: "rgba(255,255,255,0.06)", borderWidth: 1, borderColor: "rgba(255,255,255,0.10)" },
  refreshBtnText: { color: "#D6E4FF", fontWeight: "700", marginLeft: 8 }
});