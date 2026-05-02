import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Image, ActivityIndicator, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';

const BASE_URL = "http://10.125.73.179:8000";

const DISEASE_THRESHOLDS: { [key: string]: number } = {
  "Atelectasis": 0.50, "Cardiomegaly": 0.45, "Consolidation": 0.50,
  "Edema": 0.55, "Effusion": 0.55, "Emphysema": 0.65,
  "Fibrosis": 0.50, "Hernia": 0.75, "Infiltration": 0.50,
  "Mass": 0.65, "Nodule": 0.55, "Pleural_Thickening": 0.55,
  "Pneumonia": 0.35, "Pneumothorax": 0.55
};

export default function ReportViewScreen() {
  const { analysisId } = useLocalSearchParams();
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchReport();
  }, [analysisId]);

  const fetchReport = async () => {
    try {
      const response = await axios.get(`${BASE_URL}/analyze/result/${analysisId}`);
      setData(response.data);
    } catch (error) {
      Alert.alert("Hata", "Rapor yüklenemedi.");
    } finally {
      setLoading(false);
    }
  };

  const getParsedFindings = () => {
    if (!data) return [];
    
    let findings: string[] = [];
    const report = data.report;

    if (report && report.detected_findings) {
      try {
        let parsed = JSON.parse(report.detected_findings);
        if (typeof parsed === 'string') parsed = JSON.parse(parsed); 
        if (Array.isArray(parsed) && parsed.length > 0) {
          findings = parsed;
        }
      } catch (e) {
        console.error("UI Parse hatası", e);
      }
    }

    if (findings.length === 0 && data.ai_result && Array.isArray(data.ai_result)) {
      findings = data.ai_result
        .filter((item: any) => {
          const score = parseFloat(item.prob || 0);
          const threshold = DISEASE_THRESHOLDS[item.name] || 0.50;
          return score >= threshold;
        })
        .map((item: any) => item.name);
    }

    return findings;
  };

  const generateAndSharePDF = async () => {
    if (!data || !data.report) return;

    const parsedFindings = getParsedFindings();

    let findingsHtml = "<li>Belirgin bulgu saptanmadı.</li>";
    if (parsedFindings.length > 0) {
      findingsHtml = parsedFindings.map((f: string) => `<li>${f}</li>`).join('');
    }

    const htmlContent = `
      <html>
        <head>
          <style>
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 40px; color: #333; }
            .header { text-align: center; border-bottom: 2px solid #34C759; padding-bottom: 20px; margin-bottom: 30px; }
            .hospital-name { font-size: 24px; font-weight: bold; color: #071836; }
            .report-title { font-size: 18px; color: #666; margin-top: 5px; }
            .info-table { width: 100%; margin-bottom: 30px; border-collapse: collapse; }
            .info-table td { padding: 8px; border: 1px solid #ddd; }
            .info-table strong { color: #071836; }
            .image-container { text-align: center; margin-bottom: 30px; }
            .image-container img { max-width: 400px; max-height: 400px; border-radius: 10px; border: 1px solid #ccc; }
            .ai-section { background-color: #e8f5e9; padding: 20px; border: 1px solid #c8e6c9; border-radius: 8px; margin-bottom: 20px; }
            .ai-section h3 { color: #2e7d32; margin-top: 0; margin-bottom: 10px; font-size: 16px; }
            .ai-section ul { margin: 0; padding-left: 20px; color: #d32f2f; font-weight: bold; }
            .comment-section { background-color: #f9f9f9; padding: 20px; border-left: 4px solid #34C759; margin-bottom: 40px; }
            .footer { margin-top: 50px; text-align: right; }
            .signature { font-size: 18px; font-weight: bold; color: #071836; }
            .stamp { color: #34C759; font-size: 12px; margin-top: 5px; }
          </style>
        </head>
        <body>
          <div class="header">
            <div class="hospital-name">${data.report.hospital_name}</div>
            <div class="report-title">Radyolojik Analiz ve AI Destekli Değerlendirme Raporu</div>
          </div>

          <table class="info-table">
            <tr>
              <td><strong>Hasta Protokol NO:</strong> ${data.protocol_id}</td>
              <td><strong>Rapor Tarihi:</strong> ${data.report.finalized_at}</td>
            </tr>
            <tr>
              <td><strong>İnceleyen Hekim:</strong> ${data.report.doctor_name}</td>
              <td><strong>Sistem Durumu:</strong> Yapay Zeka Destekli - ONAYLI</td>
            </tr>
          </table>

          <div class="image-container">
            <img src="${data.original_image}" />
          </div>

          <div class="ai-section">
            <h3>Yapay Zeka Destekli Analiz Bulguları</h3>
            <ul>
              ${findingsHtml}
            </ul>
          </div>

          <div class="comment-section">
            <h3>Klinik Değerlendirme ve Bulgular</h3>
            <p>${data.report.doctor_comment}</p>
          </div>

          <div class="footer">
            <div class="signature">${data.report.doctor_name}</div>
            <div class="stamp">e-İmzalıdır / QR Doğrulamalıdır</div>
          </div>
        </body>
      </html>
    `;

    try {
      const { uri } = await Print.printToFileAsync({ html: htmlContent });
      
      const isAvailable = await Sharing.isAvailableAsync();
      if (isAvailable) {
        await Sharing.shareAsync(uri);
      } else {
        Alert.alert("Bilgi", "Cihazınızda paylaşım desteklenmiyor, PDF konumu: " + uri);
      }
    } catch (error) {
      console.error(error);
      Alert.alert("Hata", "PDF oluşturulurken bir sorun oluştu.");
    }
  };

  if (loading || !data) {
    return <ActivityIndicator size="large" color="#34C759" style={{ flex: 1, backgroundColor: '#071836' }} />;
  }

  const report = data.report;
  const parsedFindings = getParsedFindings();

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={{ padding: 5 }}>
          <Ionicons name="close" size={28} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Resmi Rapor</Text>
        <View style={{ width: 28 }} />
      </View>

      <ScrollView style={styles.paperContainer}>
        <View style={styles.paper}>
          
          <View style={styles.paperHeader}>
            <Text style={styles.hospitalName}>{report.hospital_name}</Text>
            <Text style={styles.paperTitle}>Radyoloji Değerlendirme Raporu</Text>
          </View>

          <View style={styles.infoBox}>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Protokol:</Text>
              <Text style={styles.infoValue}>{data.protocol_id}</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Tarih:</Text>
              <Text style={styles.infoValue}>{report.finalized_at}</Text>
            </View>
          </View>

          <View style={styles.imageSection}>
            <Image source={{ uri: data.original_image }} style={styles.xrayImage} resizeMode="contain" />
          </View>

          {/* EKRANDA GÖSTERİLECEK AI BULGULARI KUTUSU */}
          <View style={styles.aiBox}>
            <Text style={styles.aiBoxTitle}>Yapay Zeka Destekli Analiz Bulguları:</Text>
            {parsedFindings.length > 0 ? (
              parsedFindings.map((finding, index) => (
                <Text key={index} style={styles.findingItem}>• {finding}</Text>
              ))
            ) : (
              <Text style={styles.findingItem}>• Belirgin bulgu saptanmadı.</Text>
            )}
          </View>

          <View style={styles.commentBox}>
            <Text style={styles.commentLabel}>Klinik Notlar ve Bulgular:</Text>
            <Text style={styles.commentText}>{report.doctor_comment}</Text>
          </View>

          <View style={styles.signatureBox}>
            <Text style={styles.doctorName}>{report.doctor_name}</Text>
            <Text style={styles.stampText}>Dijital Olarak Onaylanmıştır</Text>
            <Ionicons name="checkmark-done-circle" size={24} color="#34C759" style={{ marginTop: 5 }} />
          </View>

        </View>
      </ScrollView>

      <View style={styles.bottomBar}>
        <TouchableOpacity style={styles.pdfButton} onPress={generateAndSharePDF}>
          <Ionicons name="share-outline" size={22} color="#fff" />
          <Text style={styles.pdfButtonText}>PDF Olarak Paylaş / İndir</Text>
        </TouchableOpacity>
      </View>

    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#071836' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 15, paddingTop: 50, paddingBottom: 15, backgroundColor: '#0A2044' },
  headerTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  paperContainer: { flex: 1, padding: 15 },
  
  paper: { backgroundColor: '#fff', borderRadius: 8, padding: 20, minHeight: 600, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 4, elevation: 5, marginBottom: 30 },
  paperHeader: { alignItems: 'center', borderBottomWidth: 2, borderBottomColor: '#34C759', paddingBottom: 15, marginBottom: 20 },
  hospitalName: { fontSize: 20, fontWeight: 'bold', color: '#071836', textAlign: 'center' },
  paperTitle: { fontSize: 14, color: '#666', marginTop: 5 },
  
  infoBox: { backgroundColor: '#f5f5f5', padding: 15, borderRadius: 8, marginBottom: 20 },
  infoRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 },
  infoLabel: { fontSize: 14, color: '#555', fontWeight: 'bold' },
  infoValue: { fontSize: 14, color: '#071836', fontWeight: '600' },
  
  imageSection: { alignItems: 'center', marginBottom: 20 },
  xrayImage: { width: '100%', height: 250, borderRadius: 8, borderWidth: 1, borderColor: '#eee' },
  
  aiBox: { backgroundColor: '#e8f5e9', padding: 15, borderRadius: 8, marginBottom: 20, borderWidth: 1, borderColor: '#c8e6c9' },
  aiBoxTitle: { fontSize: 16, fontWeight: 'bold', color: '#2e7d32', marginBottom: 8 },
  findingItem: { fontSize: 14, color: '#d32f2f', marginBottom: 4, marginLeft: 5, fontWeight: 'bold' },

  commentBox: { borderLeftWidth: 4, borderLeftColor: '#34C759', paddingLeft: 15, marginBottom: 40 },
  commentLabel: { fontSize: 16, fontWeight: 'bold', color: '#071836', marginBottom: 8 },
  commentText: { fontSize: 14, color: '#333', lineHeight: 22 },
  
  signatureBox: { alignItems: 'flex-end', marginTop: 20 },
  doctorName: { fontSize: 16, fontWeight: 'bold', color: '#071836' },
  stampText: { fontSize: 12, color: '#34C759', marginTop: 2 },
  
  bottomBar: { padding: 20, backgroundColor: '#0A2044', borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.1)' },
  pdfButton: { flexDirection: 'row', backgroundColor: '#34C759', padding: 15, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  pdfButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginLeft: 10 }
});