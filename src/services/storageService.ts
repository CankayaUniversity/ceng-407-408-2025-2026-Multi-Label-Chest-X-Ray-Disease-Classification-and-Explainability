import AsyncStorage from "@react-native-async-storage/async-storage";
import { Doctor, DoctorLoginResponse } from "../types";

const DOCTORS_STORAGE_KEY = "doctors_storage_key";

export async function getDoctors(): Promise<Doctor[]> {
  try {
    const data = await AsyncStorage.getItem(DOCTORS_STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error("getDoctors error:", error);
    return [];
  }
}

export async function saveDoctors(doctors: Doctor[]): Promise<void> {
  try {
    await AsyncStorage.setItem(DOCTORS_STORAGE_KEY, JSON.stringify(doctors));
  } catch (error) {
    console.error("saveDoctors error:", error);
    throw error;
  }
}

export async function registerDoctor(data: {
  fullName: string;
  email: string;
  password: string;
  hospitalId: string;
  hospitalName: string;
}): Promise<{ success: boolean; message?: string }> {
  try {
    const doctors = await getDoctors();

    const existingDoctor = doctors.find(
      (doctor) => doctor.email.trim().toLowerCase() === data.email.trim().toLowerCase()
    );

    if (existingDoctor) {
      return {
        success: false,
        message: "Bu e-posta adresi ile kayıtlı bir doktor zaten var.",
      };
    }

    const newDoctor: Doctor = {
      _id: Date.now().toString(),
      fullName: data.fullName.trim(),
      email: data.email.trim().toLowerCase(),
      password: data.password.trim(),
      hospitalId: data.hospitalId,
      hospitalName: data.hospitalName,
      status: "pending",
      createdAt: new Date().toISOString(),
    };

    const updatedDoctors = [...doctors, newDoctor];
    await saveDoctors(updatedDoctors);

    return { success: true };
  } catch (error) {
    console.error("registerDoctor error:", error);
    return {
      success: false,
      message: "Kayıt işlemi sırasında bir hata oluştu.",
    };
  }
}

export async function loginDoctorLocal(
  email: string,
  password: string
): Promise<DoctorLoginResponse> {
  try {
    const doctors = await getDoctors();

    const doctor = doctors.find(
      (item) =>
        item.email.trim().toLowerCase() === email.trim().toLowerCase() &&
        item.password === password.trim()
    );

    if (!doctor) {
      return {
        success: false,
        message: "E-posta veya şifre hatalı.",
      };
    }

    return {
      success: true,
      status: doctor.status,
      doctorName: doctor.fullName,
    };
  } catch (error) {
    console.error("loginDoctorLocal error:", error);
    return {
      success: false,
      message: "Giriş yapılırken bir hata oluştu.",
    };
  }
}

export async function getPendingDoctorsByHospital(
  hospitalName: string
): Promise<Doctor[]> {
  try {
    const doctors = await getDoctors();

    return doctors.filter(
      (doctor) =>
        doctor.hospitalName.trim().toLowerCase() === hospitalName.trim().toLowerCase() &&
        doctor.status === "pending"
    );
  } catch (error) {
    console.error("getPendingDoctorsByHospital error:", error);
    return [];
  }
}

export async function approveDoctor(doctorId: string): Promise<boolean> {
  try {
    const doctors = await getDoctors();

    const updatedDoctors = doctors.map((doctor) =>
      doctor._id === doctorId ? { ...doctor, status: "approved" as const } : doctor
    );

    await saveDoctors(updatedDoctors);
    return true;
  } catch (error) {
    console.error("approveDoctor error:", error);
    return false;
  }
}

export async function rejectDoctor(doctorId: string): Promise<boolean> {
  try {
    const doctors = await getDoctors();

    const updatedDoctors = doctors.map((doctor) =>
      doctor._id === doctorId ? { ...doctor, status: "rejected" as const } : doctor
    );

    await saveDoctors(updatedDoctors);
    return true;
  } catch (error) {
    console.error("rejectDoctor error:", error);
    return false;
  }
}

/* İstersen test için bunu geçici kullanabilirsin
export async function clearAllDoctors(): Promise<void> {
  await AsyncStorage.removeItem(DOCTORS_STORAGE_KEY);
}
*/