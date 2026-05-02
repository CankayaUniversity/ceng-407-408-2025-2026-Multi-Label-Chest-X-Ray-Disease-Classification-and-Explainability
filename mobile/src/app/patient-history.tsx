import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, useRouter } from "expo-router";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { 
  SafeAreaView, 
  StyleSheet, 
  Text, 
  TouchableOpacity, 
  View, 
  FlatList, 
  ActivityIndicator,
  Dimensions
} from "react-native";

const { width } = Dimensions.get("window");
const BASE_URL = "http://10.125.73.179:8000";

export default function PatientHistoryScreen() {
  const router = useRouter();
  const { protocol_id } = useLocalSearchParams();
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'xrays' | 'reports'>('xrays');

  useEffect(() => {
    if (protocol_id) {
      fetchHistory();
    }
  }, [protocol_id]);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${BASE_URL}/analyze/history/${protocol_id}`, {
        headers: { "Bypass-Tunnel-Reminder": "true" }
      });
      setHistory(response.data);
    } catch (error) {
      console.error("Geçmiş çekilemedi:", error);
    } finally {
      setLoading(false);
    }
  };

  const displayedData = activeTab === 'xrays' 
    ? history 
    : history.filter(item => item.report !== undefined);

  const renderItem = ({ item }: { item: any }) => {
    if (activeTab === 'xrays') {
      return (
        <TouchableOpacity 
          style={styles.historyCard}
          onPress={() => router.push({
            pathname: "/analysis-result" as any,
            params: { analysisId: item._id }
          })}
        >
          <View style={styles.cardLeft}>
            <View style={[styles.statusIndicator, { backgroundColor: item.has_disease ? '#FF3B30' : '#34C759' }]} />
            <View>
              <Text style={styles.dateText}>{item.upload_date}</Text>
              <Text style={styles.statusText}>
                {item.has_disease ? "Bulgu Tespit Edildi" : "Temiz / Normal"}
              </Text>
            </View>
          </View>
          <Ionicons name="image-outline" size={24} color="rgba(255,255,255,0.3)" />
        </TouchableOpacity>
      );
    } else {
      const report = item.report;
      return (
        <TouchableOpacity 
          style={styles.reportCard}
          onPress={() => router.push({
            pathname: "/report-view" as any,
            params: { analysisId: item._id }
          })}
        >
          <View style={styles.reportHeader}>
            <View style={styles.reportIconContainer}>
              <Ionicons name="document-text" size={22} color="#34C759" />
            </View>
            <View style={styles.reportInfo}>
              <Text style={styles.reportHospital}>{report.hospital_name}</Text>
              <Text style={styles.reportDoctor}>Dr. {report.doctor_name}</Text>
            </View>
          </View>
          
          <View style={styles.reportFooter}>
            <Text style={styles.reportDate}>{report.finalized_at}</Text>
            <View style={styles.pdfBtn}>
              <Text style={styles.pdfBtnText}>Raporu Gör</Text>
              <Ionicons name="arrow-forward" size={14} color="#fff" style={{marginLeft: 4}} />
            </View>
          </View>
        </TouchableOpacity>
      );
    }
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={["#071836", "#0D47A1"]} style={StyleSheet.absoluteFill} />
      
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
            <Ionicons name="arrow-back" size={24} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Hasta Arşivi</Text>
          <TouchableOpacity onPress={fetchHistory}>
            <Ionicons name="refresh" size={22} color="#fff" />
          </TouchableOpacity>
        </View>

        <View style={styles.patientInfo}>
          <Ionicons name="person-circle-outline" size={40} color="#A7C2F0" />
          <View style={{ marginLeft: 12 }}>
            <Text style={styles.label}>Protokol Numarası</Text>
            <Text style={styles.protocolValue}>{protocol_id}</Text>
          </View>
        </View>

        {/* Sekme Butonları */}
        <View style={styles.tabContainer}>
          <TouchableOpacity 
            style={[styles.tab, activeTab === 'xrays' && styles.activeTab]} 
            onPress={() => setActiveTab('xrays')}
          >
            <Text style={[styles.tabText, activeTab === 'xrays' && styles.activeTabText]}>X-Rayler</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.tab, activeTab === 'reports' && styles.activeTab]} 
            onPress={() => setActiveTab('reports')}
          >
            <Text style={[styles.tabText, activeTab === 'reports' && styles.activeTabText]}>Raporlar</Text>
          </TouchableOpacity>
        </View>

        {loading ? (
          <View style={styles.center}>
            <ActivityIndicator size="large" color="#A7C2F0" />
          </View>
        ) : (
          <FlatList
            data={displayedData}
            keyExtractor={(item) => item._id}
            renderItem={renderItem}
            contentContainerStyle={styles.listContent}
            ListEmptyComponent={
              <View style={styles.center}>
                <Ionicons 
                  name={activeTab === 'xrays' ? "images-outline" : "folder-open-outline"} 
                  size={60} 
                  color="rgba(255,255,255,0.1)" 
                />
                <Text style={styles.emptyText}>
                  {activeTab === 'xrays' 
                    ? "Henüz bir X-Ray analizi bulunmuyor." 
                    : "Onaylanmış resmi rapor bulunmuyor."}
                </Text>
              </View>
            }
          />
        )}
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  header: { 
    flexDirection: "row", 
    alignItems: "center", 
    justifyContent: "space-between", 
    paddingHorizontal: 20, 
    paddingVertical: 15
  },
  backBtn: { padding: 5 },
  headerTitle: { color: "#fff", fontSize: 18, fontWeight: "700" },
  patientInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.06)',
    marginHorizontal: 20,
    marginTop: 10,
    marginBottom: 20,
    padding: 15,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)'
  },
  label: { color: 'rgba(255,255,255,0.5)', fontSize: 12 },
  protocolValue: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.05)',
    marginHorizontal: 20,
    borderRadius: 12,
    padding: 5,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)'
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8
  },
  activeTab: { backgroundColor: '#fff' },
  tabText: { color: 'rgba(255,255,255,0.6)', fontWeight: '600', fontSize: 14 },
  activeTabText: { color: '#071836', fontWeight: 'bold' },

  listContent: { paddingHorizontal: 20, paddingBottom: 30 },
  
  historyCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255,255,255,0.04)',
    padding: 16,
    borderRadius: 15,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.05)'
  },
  cardLeft: { flexDirection: 'row', alignItems: 'center' },
  statusIndicator: { width: 4, height: 35, borderRadius: 2, marginRight: 15 },
  dateText: { color: '#fff', fontSize: 15, fontWeight: '600' },
  statusText: { color: 'rgba(255,255,255,0.4)', fontSize: 13, marginTop: 2 },
  
  reportCard: {
    backgroundColor: 'rgba(52, 199, 89, 0.05)',
    padding: 16,
    borderRadius: 15,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(52, 199, 89, 0.2)'
  },
  reportHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  reportIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: 'rgba(52, 199, 89, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12
  },
  reportInfo: { flex: 1 },
  reportHospital: { color: '#fff', fontSize: 15, fontWeight: 'bold' },
  reportDoctor: { color: 'rgba(255,255,255,0.7)', fontSize: 13, marginTop: 2 },
  reportFooter: { 
    flexDirection: 'row', 
    justifyContent: 'space-between', 
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.05)',
    paddingTop: 12
  },
  reportDate: { color: 'rgba(255,255,255,0.4)', fontSize: 12 },
  pdfBtn: { 
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#34C759', 
    paddingHorizontal: 12, 
    paddingVertical: 6, 
    borderRadius: 8 
  },
  pdfBtnText: { color: '#fff', fontSize: 12, fontWeight: 'bold' },

  center: { flex: 1, justifyContent: "center", alignItems: "center", marginTop: 50 },
  emptyText: { color: "rgba(255,255,255,0.3)", marginTop: 15, fontSize: 14 }
});