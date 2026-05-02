import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useState, useEffect } from "react";
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from "axios";
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
  Modal,
  TextInput,
  ActivityIndicator
} from "react-native";

const BASE_URL = "http://10.125.73.179:8000"; 
const { width, height } = Dimensions.get("window");

export default function HomeScreen() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [doctorName, setDoctorName] = useState("Doktor");
  const [hospitalName, setHospitalName] = useState<string | null>(null); 
  const [doctorEmail, setDoctorEmail] = useState<string | null>(null); 

  const [patients, setPatients] = useState<any[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<any>(null);
  const [isPatientModalVisible, setPatientModalVisible] = useState(false);
  const [isAddFormVisible, setAddFormVisible] = useState(false);
  const [isAddingPatient, setIsAddingPatient] = useState(false);
  const [newPatient, setNewPatient] = useState({ full_name: "", age: "", gender: "Erkek", tc_no: "" });
  const [recentAnalyses, setRecentAnalyses] = useState<any[]>([]);

  useEffect(() => {
    const initPage = async () => {
      try {
        const name = await AsyncStorage.getItem("full_name") || await AsyncStorage.getItem("userName");
        const hospital = await AsyncStorage.getItem("hospital_name");
        const email = await AsyncStorage.getItem("userEmail"); 
        
        if (name) {
          setDoctorName(`Dr. ${name}`);
        }
        
        if (hospital) {
          setHospitalName(hospital);
        }

        if (email) {
          setDoctorEmail(email);
          fetchPatients(email); 
          fetchRecentAnalyses(email);
        } else {
          fetchPatients(""); 
        }
      } catch (error) {
        console.error("Başlatma hatası:", error);
      }
    };
    initPage();
  }, []);

  const fetchPatients = async (email: string) => {
    try {
      const url = email ? `${BASE_URL}/patients?email=${email}` : `${BASE_URL}/patients`;
      const response = await axios.get(url, {
          headers: {
              "Bypass-Tunnel-Reminder": "true" 
          }
      });
      
      if (response.data) {
        setPatients(response.data);
      }
    } catch (error) {
      console.log("Hastalar çekilemedi:", error);
    }
  };

  const fetchRecentAnalyses = async (email: string) => {
    try {
      const url = `${BASE_URL}/analyze/recent/doctor?email=${email}`; 
      const response = await axios.get(url, { headers: { "Bypass-Tunnel-Reminder": "true" } });
      if (response.data) {
        setRecentAnalyses(response.data);
      }
    } catch (error) {
      console.log("Son analizler çekilemedi:", error);
    }
  };

  const getPatientName = (protocolId: string) => {
    const patient = patients.find((p) => p.protocol_id === protocolId);
    return patient ? patient.full_name : `Protokol: ${protocolId}`;
  };

  const handleAddNewPatient = async () => {
    if (!newPatient.full_name || !newPatient.age) {
      Alert.alert("Eksik Bilgi", "Lütfen hastanın adını ve yaşını girin.");
      return;
    }

    if (!doctorEmail) {
      Alert.alert("Oturum Hatası", "Doktor bilgisi bulunamadı.");
      return;
    }

    try {
      setIsAddingPatient(true);
      const payload = {
        full_name: newPatient.full_name.trim(),
        age: parseInt(newPatient.age),
        gender: newPatient.gender,
        hospital: hospitalName || "Bilinmiyor", 
        doctor_email: doctorEmail, 
        is_foreign: false,
        tc_no: newPatient.tc_no || null
      };

      const response = await axios.post(`${BASE_URL}/patients/add`, payload, {
          headers: { "Bypass-Tunnel-Reminder": "true" }
      });
      
      Alert.alert("Başarılı", "Hasta sisteme eklendi.");
      
      setNewPatient({ full_name: "", age: "", gender: "Erkek", tc_no: "" });
      setAddFormVisible(false);
      await fetchPatients(doctorEmail);
      
      if (response.data && response.data.protocol_id) {
          setSelectedPatient({ protocol_id: response.data.protocol_id, full_name: payload.full_name });
          setPatientModalVisible(false);
      }

    } catch (error: any) {
      Alert.alert("Hata", error.response?.data?.detail || "Hasta kaydedilemedi.");
    } finally {
      setIsAddingPatient(false);
    }
  };

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (status !== "granted") {
      Alert.alert("İzin Gerekli", "Galeriye erişim izni vermeniz gerekiyor.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'] as any, 
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedPatient || !selectedImage) {
      Alert.alert("Eksik Bilgi", "Lütfen hasta ve görüntü seçin.");
      return;
    }

    try {
      setIsAnalyzing(true);
      const formData = new FormData();
      const filename = selectedImage.split('/').pop() || 'image.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : `image/jpeg`;

      // @ts-ignore
      formData.append('file', { uri: selectedImage, name: filename, type });
      formData.append('doctor_email', doctorEmail || "musa@mail.com");
      formData.append('protocol_id', selectedPatient.protocol_id); 

      const uploadResponse = await axios.post(`${BASE_URL}/analyze/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data', "Bypass-Tunnel-Reminder": "true" },
        timeout: 60000, 
      });

      const analysisId = uploadResponse.data.id;
      if(doctorEmail) fetchRecentAnalyses(doctorEmail);

      router.push({
        pathname: "/analysis-result",
        params: { analysisId: analysisId }
      });

    } catch (error: any) {
      Alert.alert("Hata", error.response?.data?.detail || "Analiz yapılamadı.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleLogout = async () => {
      try {
          await AsyncStorage.multiRemove(['userEmail', 'full_name', 'hospital_name']);
          router.replace("/");
      } catch (error) {
          console.error("Çıkış yaparken hata:", error);
      }
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={["#071836", "#0D47A1"]} style={StyleSheet.absoluteFill} />

      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View>
              <Text style={styles.welcomeText}>Hoş Geldiniz,</Text>
              <Text style={styles.doctorName}>{doctorName}</Text>
              {hospitalName && <Text style={{color: 'rgba(255,255,255,0.5)', fontSize: 12, marginTop: 2}}>{hospitalName}</Text>}
            </View>
            <TouchableOpacity style={styles.profileBadge}>
              <Ionicons name="person-circle-outline" size={40} color="#fff" />
            </TouchableOpacity>
          </View>

          <View style={styles.patientSelectionCard}>
            <Text style={styles.sectionLabel}>İşlem Yapılacak Hasta</Text>
            <TouchableOpacity 
              style={styles.patientSelectBtn}
              onPress={() => setPatientModalVisible(true)}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="person-outline" size={20} color={selectedPatient ? "#A7C2F0" : "rgba(255,255,255,0.4)"} />
                <Text style={[styles.patientSelectText, !selectedPatient && { color: "rgba(255,255,255,0.4)" }]}>
                  {selectedPatient ? `${selectedPatient.full_name} (${selectedPatient.protocol_id})` : "Hasta Seçiniz veya Ekleyiniz..."}
                </Text>
              </View>
              <Ionicons name="chevron-down" size={20} color="rgba(255,255,255,0.4)" />
            </TouchableOpacity>
          </View>

          {selectedPatient && (
            <TouchableOpacity 
              style={styles.historyBtn} 
              onPress={() => router.push({
                pathname: "/patient-history" as any,
                params: { protocol_id: selectedPatient.protocol_id }
              })}
            >
              <Ionicons name="time-outline" size={20} color="#A7C2F0" />
              <Text style={styles.historyBtnText}>Seçili Hastanın Eski Röntgenlerini Gör</Text>
            </TouchableOpacity>
          )}

          <View style={styles.heroInfoCard}>
            <Text style={styles.heroTitle}>Chest X-Ray Analysis</Text>
            <Text style={styles.heroDesc}>
              Göğüs röntgeni görüntüsünü yükleyerek çoklu hastalık sınıflandırma
              analizini başlatın ve explainability çıktıları için altyapıyı kullanın.
            </Text>
          </View>

          <TouchableOpacity style={styles.mainActionCard} activeOpacity={0.85} onPress={pickImage}>
            <LinearGradient colors={["rgba(255,255,255,0.18)", "rgba(255,255,255,0.05)"]} style={styles.cardGradient}>
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
              <LinearGradient colors={["rgba(255,255,255,0.28)", "rgba(255,255,255,0.10)"]} style={styles.analyzeGradient}>
                <Text style={styles.analyzeBtnText}>
                  {isAnalyzing ? "ANALİZ EDİLİYOR..." : "ANALİZİ ÇALIŞTIR"}
                </Text>
                <Ionicons name="chevron-forward" size={16} color="#fff" style={{ marginLeft: 8 }} />
              </LinearGradient>
            </TouchableOpacity>
          )}

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Son Yaptığınız Analizler</Text>
          </View>

          {recentAnalyses.length > 0 ? (
            recentAnalyses.map((item, index) => (
              <TouchableOpacity 
                key={index} 
                style={styles.historyItem}
                onPress={() => router.push({
                  pathname: "/analysis-result" as any,
                  params: { analysisId: item._id }
                })}
              >
                <View style={styles.historyIcon}>
                  <Ionicons name="document-text-outline" size={24} color="#A7C2F0" />
                </View>

                <View style={styles.historyInfo}>
                  {/* YENİ: getPatientName fonksiyonu ile ismi yazdırıyoruz */}
                  <Text style={styles.historyName}>{getPatientName(item.protocol_id)}</Text>
                  <Text style={styles.historyDate}>{item.upload_date}</Text>
                </View>

                <View style={[styles.historyStatusBadge, { backgroundColor: item.has_disease ? 'rgba(255, 59, 48, 0.12)' : 'rgba(52, 199, 89, 0.12)' }]}>
                  <Text style={[styles.historyStatusText, { color: item.has_disease ? '#FF3B30' : '#34C759' }]}>
                    {item.has_disease ? "Bulgu Var" : "Temiz"}
                  </Text>
                </View>
              </TouchableOpacity>
            ))
          ) : (
            <Text style={{ color: 'rgba(255,255,255,0.5)', textAlign: 'center', marginTop: 10, marginBottom: 20 }}>
              Henüz geçmiş bir analiz kaydınız bulunmuyor.
            </Text>
          )}

          <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
            <Ionicons name="log-out-outline" size={20} color="#FF9A9A" />
            <Text style={styles.logoutText}>Oturumu Kapat</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>

      <Modal visible={isPatientModalVisible} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{isAddFormVisible ? "Yeni Hasta Kaydı" : "Kayıtlı Hastalar"}</Text>
              <TouchableOpacity onPress={() => { setPatientModalVisible(false); setAddFormVisible(false); }}>
                <Ionicons name="close-circle" size={28} color="rgba(255,255,255,0.5)" />
              </TouchableOpacity>
            </View>

            {isAddFormVisible ? (
              <ScrollView style={{ width: '100%' }}>
                <TextInput
                  style={styles.input}
                  placeholder="Ad Soyad"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={newPatient.full_name}
                  onChangeText={(t) => setNewPatient({...newPatient, full_name: t})}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Yaş"
                  keyboardType="numeric"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={newPatient.age}
                  onChangeText={(t) => setNewPatient({...newPatient, age: t})}
                />
                <TextInput
                  style={styles.input}
                  placeholder="TC Kimlik No (Opsiyonel)"
                  keyboardType="numeric"
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  value={newPatient.tc_no}
                  onChangeText={(t) => setNewPatient({...newPatient, tc_no: t})}
                />
                
                <TouchableOpacity style={styles.saveBtn} onPress={handleAddNewPatient} disabled={isAddingPatient}>
                  {isAddingPatient ? <ActivityIndicator color="#071836" /> : <Text style={styles.saveBtnText}>Kaydet ve Seç</Text>}
                </TouchableOpacity>
                
                <TouchableOpacity onPress={() => setAddFormVisible(false)} style={{ marginTop: 15, alignItems: 'center' }}>
                  <Text style={{ color: '#FF8C8C' }}>İptal Et ve Listeye Dön</Text>
                </TouchableOpacity>
              </ScrollView>
            ) : (
              <>
                <TouchableOpacity style={styles.addNewPatientBtn} onPress={() => setAddFormVisible(true)}>
                  <Ionicons name="add-circle-outline" size={20} color="#071836" />
                  <Text style={styles.addNewPatientText}>Sisteme Yeni Hasta Ekle</Text>
                </TouchableOpacity>

                <ScrollView style={{ width: '100%', maxHeight: height * 0.4 }}>
                  {patients.length > 0 ? patients.map((patient: any, idx: number) => (
                    <TouchableOpacity 
                      key={idx} 
                      style={styles.patientListItem}
                      onPress={() => {
                        setSelectedPatient(patient);
                        setPatientModalVisible(false);
                      }}
                    >
                      <Text style={styles.patientListName}>{patient.full_name}</Text>
                      <Text style={styles.patientListDetails}>Protokol: {patient.protocol_id} | Yaş: {patient.age}</Text>
                    </TouchableOpacity>
                  )) : (
                    <Text style={{ color: 'rgba(255,255,255,0.5)', textAlign: 'center', marginTop: 20 }}>Kayıtlı hasta bulunamadı.</Text>
                  )}
                </ScrollView>
              </>
            )}

          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  scrollContent: { padding: 24, paddingBottom: 40 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 10, marginBottom: 24 },
  welcomeText: { color: "rgba(255,255,255,0.6)", fontSize: 16 },
  doctorName: { color: "#fff", fontSize: 25, fontWeight: "700", marginTop: 4 },
  profileBadge: { opacity: 0.9 },
  
  patientSelectionCard: { marginBottom: 14 },
  sectionLabel: { color: "rgba(255,255,255,0.7)", fontSize: 13, marginBottom: 8, marginLeft: 4, fontWeight: '600' },
  patientSelectBtn: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.08)', padding: 16, borderRadius: 16, borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)' },
  patientSelectText: { color: "#fff", fontSize: 15, marginLeft: 10, fontWeight: '500' },
  
  historyBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(167, 194, 240, 0.15)',
    padding: 14,
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(167, 194, 240, 0.3)',
  },
  historyBtnText: {
    color: '#A7C2F0',
    fontWeight: '600',
    marginLeft: 8,
    fontSize: 14,
  },

  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: '#0A2044', borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: 24, minHeight: height * 0.5, alignItems: 'center' },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', width: '100%', marginBottom: 20 },
  modalTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  addNewPatientBtn: { flexDirection: 'row', backgroundColor: '#A7C2F0', padding: 14, borderRadius: 12, width: '100%', justifyContent: 'center', alignItems: 'center', marginBottom: 20 },
  addNewPatientText: { color: '#071836', fontWeight: 'bold', marginLeft: 8, fontSize: 15 },
  patientListItem: { borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.1)', paddingVertical: 14, width: '100%' },
  patientListName: { color: '#fff', fontSize: 16, fontWeight: '600' },
  patientListDetails: { color: 'rgba(255,255,255,0.5)', fontSize: 13, marginTop: 4 },
  input: { backgroundColor: 'rgba(255,255,255,0.08)', color: '#fff', padding: 15, borderRadius: 12, marginBottom: 15, fontSize: 15, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  saveBtn: { backgroundColor: '#fff', padding: 15, borderRadius: 12, alignItems: 'center', marginTop: 10 },
  saveBtnText: { color: '#071836', fontWeight: 'bold', fontSize: 16 },

  heroInfoCard: { backgroundColor: "rgba(255,255,255,0.06)", borderRadius: 22, padding: 18, borderWidth: 1, borderColor: "rgba(255,255,255,0.10)", marginBottom: 18 },
  heroTitle: { color: "#fff", fontSize: 18, fontWeight: "700" },
  heroDesc: { color: "rgba(255,255,255,0.62)", lineHeight: 21, marginTop: 8, fontSize: 14 },
  mainActionCard: { borderRadius: 28, overflow: "hidden", marginBottom: 18, borderWidth: 1, borderColor: "rgba(255,255,255,0.14)" },
  cardGradient: { padding: 34, alignItems: "center" },
  uploadPlaceholder: { width: 132, height: 132, borderRadius: 20, backgroundColor: "rgba(255,255,255,0.08)", justifyContent: "center", alignItems: "center", marginBottom: 10 },
  previewImage: { width: 140, height: 140, borderRadius: 18, marginBottom: 10 },
  cardTitle: { color: "#fff", fontSize: 22, fontWeight: "700", marginTop: 12 },
  cardDesc: { color: "rgba(255,255,255,0.52)", textAlign: "center", marginTop: 10, lineHeight: 20, fontSize: 14 },
  analyzeBtn: { borderRadius: 16, overflow: "hidden", borderWidth: 1, borderColor: "rgba(255,255,255,0.26)", marginBottom: 28, width: "100%" },
  analyzeGradient: { paddingVertical: 13, flexDirection: "row", justifyContent: "center", alignItems: "center" },
  analyzeBtnText: { color: "#fff", fontWeight: "700", fontSize: 14, letterSpacing: 1.5 },
  
  sectionHeader: { marginBottom: 14, marginTop: 10 },
  sectionTitle: { color: "#fff", fontSize: 18, fontWeight: "700" },
  historyItem: { flexDirection: "row", alignItems: "center", backgroundColor: "rgba(255,255,255,0.04)", padding: 15, borderRadius: 16, marginBottom: 10, borderWidth: 1, borderColor: "rgba(255,255,255,0.05)" },
  historyIcon: { width: 46, height: 46, backgroundColor: "rgba(167, 194, 240, 0.10)", borderRadius: 12, justifyContent: "center", alignItems: "center", marginRight: 14 },
  historyInfo: { flex: 1 },
  historyName: { color: "#fff", fontSize: 16, fontWeight: "600" },
  historyDate: { color: "rgba(255,255,255,0.34)", fontSize: 12, marginTop: 3 },
  historyStatusBadge: { borderWidth: 1, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 999 },
  historyStatusText: { fontSize: 11, fontWeight: "700" },
  logoutBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", marginTop: 26, paddingVertical: 14 },
  logoutText: { color: "#FF9A9A", marginLeft: 8, fontWeight: "700" },
});