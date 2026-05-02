import React, { useEffect, useState } from 'react';
import { 
  View, Text, StyleSheet, Image, ActivityIndicator, Alert, 
  ScrollView, Dimensions, TouchableOpacity, TextInput,
  KeyboardAvoidingView, Platform
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import axios from 'axios';
import Svg, { Polyline } from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get("window");
const BASE_URL = "http://10.125.73.179:8000"; 

const DISEASE_THRESHOLDS: { [key: string]: number } = {
  "Atelectasis": 0.50, "Cardiomegaly": 0.45, "Consolidation": 0.50,
  "Edema": 0.55, "Effusion": 0.55, "Emphysema": 0.65,
  "Fibrosis": 0.50, "Hernia": 0.75, "Infiltration": 0.50,
  "Mass": 0.65, "Nodule": 0.55, "Pleural_Thickening": 0.55,
  "Pneumonia": 0.35, "Pneumothorax": 0.55
};

const COLORS = [
  '#FF3B30', '#34C759', '#007AFF', '#FF9500', '#AF52DE', 
  '#FF2D55', '#5856D6', '#5AC8FA', '#FFCC00', '#FF7F50',
  '#00FA9A', '#00CED1', '#BA55D3', '#FF69B4'
];

export default function AnalysisResultScreen() {
  const { analysisId } = useLocalSearchParams();
  const router = useRouter();
  
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>(null);
  const [allResults, setAllResults] = useState<any[]>([]);
  const [activeDiseases, setActiveDiseases] = useState<string[]>([]);
  
  const [doctorName, setDoctorName] = useState("Yükleniyor...");
  const [hospitalName, setHospitalName] = useState("Yükleniyor...");
  const [comment, setComment] = useState("");
  const [isFinalizing, setIsFinalizing] = useState(false);

  useEffect(() => {
    const initPage = async () => {
      try {
        const name = await AsyncStorage.getItem("full_name") || await AsyncStorage.getItem("userName");
        const hospital = await AsyncStorage.getItem("hospital_name");
        
        if (name) {
          setDoctorName(`Dr. ${name}`);
        } else {
          setDoctorName("Bilinmeyen Doktor");
        }
        
        if (hospital) {
          setHospitalName(hospital);
        } else {
          setHospitalName("Bilinmeyen Hastane");
        }
      } catch (error) {
        console.error("Doktor bilgileri çekilemedi:", error);
        setDoctorName("Bilinmeyen Doktor");
        setHospitalName("Bilinmeyen Hastane");
      }

      if (analysisId && analysisId !== 'undefined') {
        fetchResult();
      }
    };

    initPage();
  }, [analysisId]);

  const fetchResult = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${BASE_URL}/analyze/result/${analysisId}`, {
          headers: { "Bypass-Tunnel-Reminder": "true" }
      });
      const resultData = response.data;
      setData(resultData);

      if (resultData.ai_result && Array.isArray(resultData.ai_result)) {
        const processed = resultData.ai_result.map((item: any, index: number) => {
          const name = item.name || `Bulgu ${index + 1}`;
          const scoreValue = parseFloat(item.prob || 0);

          return {
            ...item,
            displayName: name,
            displayScore: isNaN(scoreValue) ? 0 : scoreValue
          };
        });

        const sortedAll = processed.sort((a: any, b: any) => b.displayScore - a.displayScore);
        setAllResults(sortedAll);
        
        const defaultActive = processed
          .filter((item: any) => {
            const threshold = DISEASE_THRESHOLDS[item.displayName] || 0.50;
            return item.displayScore >= threshold;
          })
          .map((item: any) => item.displayName);
        
        setActiveDiseases(defaultActive);
      }
    } catch (error) {
      console.error("Sonuç çekme hatası:", error);
      Alert.alert("Hata", "Analiz verileri yüklenemedi.");
    } finally {
      setLoading(false);
    }
  };

  const toggleDisease = (name: string) => {
    setActiveDiseases(prev => 
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    );
  };

  const handleFinalize = async () => {
    if (!comment.trim()) {
      Alert.alert("Eksik Bilgi", "Lütfen rapor için klinik notunuzu ekleyin.");
      return;
    }

    try {
      setIsFinalizing(true);
      const formData = new FormData();
      formData.append("analysis_id", analysisId as string);
      formData.append("doctor_comment", comment);
      
      formData.append("doctor_name", doctorName); 
      formData.append("hospital_name", hospitalName);

      formData.append("detected_findings", JSON.stringify(activeDiseases));

      const response = await axios.post(`${BASE_URL}/analyze/finalize-report`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      if (response.status === 200) {
        Alert.alert("Başarılı", "Rapor resmileştirildi ve hasta arşivine eklendi.");
        router.push({
          pathname: "/patient-history",
          params: { protocol_id: data.protocol_id }
        });
      }
    } catch (error) {
      console.error("Rapor hatası:", error);
      Alert.alert("Hata", "Rapor kaydedilemedi. Backend bağlantısını kontrol edin.");
    } finally {
      setIsFinalizing(false);
    }
  };

  if (loading || !data) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#fff" />
        <Text style={{ color: '#fff', marginTop: 10 }}>Analiz Sonuçları Hazırlanıyor...</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView 
      behavior={Platform.OS === 'ios' ? 'padding' : undefined} 
      style={{ flex: 1, backgroundColor: '#071836' }}
    >
      <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 40 }} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Analiz Detay Sonucu</Text>
        
        <View style={styles.imageWrapper}>
          <Image 
            source={{ uri: data.original_image }} 
            style={styles.image} 
            resizeMode="contain"
          />
          <Svg height="100%" width="100%" viewBox="0 0 1024 1024" style={StyleSheet.absoluteFill}>
            {allResults.map((disease: any, i: number) => {
              const name = disease.displayName;
              if (!activeDiseases.includes(name)) return null;

              return disease.contours?.map((contour: any, j: number) => (
                <Polyline
                  key={`poly-${i}-${j}`}
                  points={contour.map((p: any) => `${p[1]},${p[0]}`).join(' ')}
                  fill="none" 
                  stroke={COLORS[i % COLORS.length]} 
                  strokeWidth="4" 
                />
              ));
            })}
          </Svg>
        </View>

        <View style={styles.infoArea}>
          <Text style={styles.infoTitle}>Tüm Tespit Edilen Bulgular</Text>
          <Text style={styles.subLabel}>Tıbbi eşiği geçenler otomatik seçilmiştir. Manuel seçim yapabilirsiniz.</Text>
          
          {allResults.map((d: any, idx: number) => {
            const isSelected = activeDiseases.includes(d.displayName);
            const color = COLORS[idx % COLORS.length];
            const threshold = DISEASE_THRESHOLDS[d.displayName] || 0.50;

            if (d.displayScore < 0.05) return null;

            return (
              <TouchableOpacity 
                key={`row-${idx}`} 
                style={[styles.diseaseRow, isSelected && { backgroundColor: 'rgba(255,255,255,0.06)' }]} 
                onPress={() => toggleDisease(d.displayName)}
                activeOpacity={0.7}
              >
                <View style={styles.rowLeft}>
                  <Ionicons 
                    name={isSelected ? "checkbox" : "square-outline"} 
                    size={24} 
                    color={isSelected ? color : "rgba(255,255,255,0.3)"} 
                  />
                  <View style={{ marginLeft: 12 }}>
                    <Text style={[styles.diseaseName, { color: isSelected ? '#fff' : 'rgba(255,255,255,0.4)' }]}>
                      {d.displayName}
                    </Text>
                    <Text style={{fontSize: 10, color: 'rgba(255,255,255,0.3)'}}>Eşik: {threshold}</Text>
                  </View>
                </View>
                <Text style={[styles.scoreText, { color: isSelected ? color : 'rgba(255,255,255,0.3)' }]}>
                  %{(d.displayScore * 100).toFixed(1)}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* RAPOR BÖLÜMÜ */}
        <View style={styles.reportSection}>
          <View style={styles.reportHeader}>
            <Ionicons name="document-text" size={20} color="#34C759" />
            <Text style={styles.reportTitle}>Resmi Rapor Kaydı</Text>
          </View>
          
          {/* OTOMATİK ÇEKİLEN SADE BİLGİ KARTLARI */}
          <View style={styles.autoFetchedBox}>
            <Text style={styles.autoFetchedLabel}>Kurum:</Text>
            <Text style={styles.autoFetchedText}>{hospitalName}</Text>
          </View>
          
          <View style={styles.autoFetchedBox}>
            <Text style={styles.autoFetchedLabel}>Hekim:</Text>
            <Text style={styles.autoFetchedText}>{doctorName}</Text>
          </View>

          {/* SADECE YORUM GİRİŞİ */}
          <TextInput
            style={styles.commentInput}
            placeholder="Klinik değerlendirmenizi buraya yazın..."
            placeholderTextColor="rgba(255,255,255,0.3)"
            multiline
            numberOfLines={4}
            value={comment}
            onChangeText={setComment}
          />
          
          <TouchableOpacity 
            style={[styles.finalizeBtn, isFinalizing && { opacity: 0.7 }]}
            onPress={handleFinalize}
            disabled={isFinalizing}
          >
            <Text style={styles.finalizeBtnText}>
              {isFinalizing ? "Kaydediliyor..." : "Raporu Onayla ve Arşivle"}
            </Text>
            {!isFinalizing && <Ionicons name="checkmark-done" size={20} color="#fff" style={{marginLeft: 8}} />}
          </TouchableOpacity>
        </View>

        <View style={styles.patientBrief}>
          <Text style={styles.briefText}>Hasta Protokol: {data.protocol_id}</Text>
          <Text style={styles.briefText}>Analiz Tarihi: {data.upload_date || "Belirtilmemiş"}</Text>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 15 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#071836' },
  title: { color: '#fff', fontSize: 22, fontWeight: 'bold', marginBottom: 20, textAlign: 'center', marginTop: 40 },
  imageWrapper: { 
    width: width - 30, 
    height: width - 30, 
    backgroundColor: '#000', 
    borderRadius: 20, 
    overflow: 'hidden', 
    position: 'relative',
    alignSelf: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)'
  },
  image: { width: '100%', height: '100%' },
  infoArea: { marginTop: 20, padding: 18, backgroundColor: '#0A2044', borderRadius: 20, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  infoTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold', marginBottom: 4 },
  subLabel: { color: 'rgba(255,255,255,0.4)', fontSize: 12, marginBottom: 16 },
  diseaseRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, paddingHorizontal: 12, borderRadius: 12, marginBottom: 6 },
  rowLeft: { flexDirection: 'row', alignItems: 'center' },
  diseaseName: { fontSize: 16, fontWeight: '600' },
  scoreText: { fontSize: 16, fontWeight: 'bold' },
  
  reportSection: {
    marginTop: 20,
    padding: 18,
    backgroundColor: 'rgba(52, 199, 89, 0.05)',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(52, 199, 89, 0.3)',
  },
  reportHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 15 },
  reportTitle: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginLeft: 8 },
  
  autoFetchedBox: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(0,0,0,0.15)',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.05)'
  },
  autoFetchedLabel: { color: 'rgba(255,255,255,0.5)', fontSize: 14, fontWeight: 'bold' },
  autoFetchedText: { color: '#fff', fontSize: 14, fontWeight: '500' },

  commentInput: {
    backgroundColor: 'rgba(0,0,0,0.3)',
    borderRadius: 12,
    padding: 15,
    color: '#fff',
    fontSize: 14,
    textAlignVertical: 'top',
    minHeight: 100,
    marginTop: 10,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)'
  },
  finalizeBtn: { flexDirection: 'row', backgroundColor: '#34C759', paddingVertical: 15, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  finalizeBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
  patientBrief: { marginTop: 20, paddingHorizontal: 10 },
  briefText: { color: 'rgba(255,255,255,0.4)', fontSize: 13, marginBottom: 4 }
});