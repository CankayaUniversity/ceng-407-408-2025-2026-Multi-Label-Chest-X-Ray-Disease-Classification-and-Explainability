import { Doctor } from "@/types";
import { Ionicons } from "@expo/vector-icons";
import React from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";

interface PendingDoctorCardProps {
  doctor: Doctor;
  onApprove: () => void;
  onReject: () => void;
}

export default function PendingDoctorCard({
  doctor,
  onApprove,
  onReject,
}: PendingDoctorCardProps) {
  return (
    <View style={styles.card}>
      <View style={styles.topRow}>
        <View style={styles.iconBox}>
          <Ionicons name="person-outline" size={24} color="#D6E4FF" />
        </View>

        <View style={styles.infoArea}>
          <Text style={styles.name}>{doctor.fullName}</Text>
          <Text style={styles.email}>{doctor.email}</Text>
        </View>
      </View>

      <View style={styles.metaBox}>
        <View style={styles.metaRow}>
          <Text style={styles.metaLabel}>Hastane</Text>
          <Text style={styles.metaValue}>{doctor.hospitalName}</Text>
        </View>

        <View style={styles.metaRow}>
          <Text style={styles.metaLabel}>Durum</Text>
          <View style={styles.statusBadge}>
            <Text style={styles.statusText}>{doctor.status.toUpperCase()}</Text>
          </View>
        </View>
      </View>

      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.rejectButton} onPress={onReject}>
          <Ionicons name="close-outline" size={18} color="#FFD6D6" />
          <Text style={styles.rejectText}>Reject</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.approveButton} onPress={onApprove}>
          <Ionicons name="checkmark-outline" size={18} color="#D9FFE6" />
          <Text style={styles.approveText}>Approve</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "rgba(255,255,255,0.06)",
    borderRadius: 22,
    padding: 18,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  topRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  iconBox: {
    width: 52,
    height: 52,
    borderRadius: 16,
    backgroundColor: "rgba(167, 194, 240, 0.12)",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 14,
  },
  infoArea: {
    flex: 1,
  },
  name: {
    color: "#fff",
    fontSize: 17,
    fontWeight: "700",
  },
  email: {
    color: "rgba(255,255,255,0.62)",
    marginTop: 4,
    fontSize: 13,
  },
  metaBox: {
    marginBottom: 16,
    gap: 10,
  },
  metaRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  metaLabel: {
    color: "rgba(255,255,255,0.55)",
    fontSize: 13,
  },
  metaValue: {
    color: "#fff",
    fontSize: 13,
    fontWeight: "600",
    flexShrink: 1,
    textAlign: "right",
    marginLeft: 10,
  },
  statusBadge: {
    backgroundColor: "rgba(255, 193, 7, 0.14)",
    borderColor: "rgba(255, 193, 7, 0.30)",
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
  },
  statusText: {
    color: "#FFD76A",
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 10,
  },
  rejectButton: {
    flex: 1,
    minHeight: 46,
    borderRadius: 14,
    backgroundColor: "rgba(255, 107, 107, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(255, 107, 107, 0.22)",
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
  },
  approveButton: {
    flex: 1,
    minHeight: 46,
    borderRadius: 14,
    backgroundColor: "rgba(46, 204, 113, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(46, 204, 113, 0.22)",
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
  },
  rejectText: {
    color: "#FFD6D6",
    marginLeft: 6,
    fontWeight: "700",
  },
  approveText: {
    color: "#D9FFE6",
    marginLeft: 6,
    fontWeight: "700",
  },
});