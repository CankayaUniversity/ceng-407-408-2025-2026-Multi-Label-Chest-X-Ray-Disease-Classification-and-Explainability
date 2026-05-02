import { API_BASE_URL } from "./api";

// --- Arayüz Tanımlamaları ---
export interface RegisterDoctorPayload {
  full_name: string;
  email: string;
  password: string;
  hospital: string;
}

export interface LoginDoctorPayload {
  email: string;
  password: string;
}

export interface LoginDoctorResponse {
  message: string;
  full_name: string;
  role: string;
  hospital: string;
}

/**
 * Yardımcı Fonksiyon: JSON Yanıt Kontrolü
 * Localtunnel'ın HTML hata sayfalarını yakalar ve backend hatalarını düzgün metne çevirir.
 */
async function handleResponse(response: Response) {
  const textData = await response.text();
  let data;
  try {
    data = JSON.parse(textData);
  } catch (e) {
    // Eğer buraya düşüyorsa sunucu JSON dışında bir şey (muhtemelen HTML) döndürmüştür.
    console.error("Ham veri:", textData);
    throw new Error("Sunucu geçersiz bir format döndü. Localtunnel linkini tarayıcıdan bir kez onaylamanız gerekebilir.");
  }

  if (!response.ok) {
    const errorDetail = data.detail;
    // Pydantic hataları (liste/obje) gelirse stringe çeviriyoruz
    const finalMessage = typeof errorDetail === 'object' 
      ? JSON.stringify(errorDetail) 
      : errorDetail;
      
    throw new Error(finalMessage || "İşlem başarısız oldu.");
  }
  return data;
}

/**
 * Yeni Doktor Kaydı
 */
export async function registerDoctor(payload: RegisterDoctorPayload) {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
      "Bypass-Tunnel-Reminder": "true" 
    },
    body: JSON.stringify({
      full_name: payload.full_name, 
      email: payload.email,
      password: payload.password,
      hospital: payload.hospital,
    }),
  });

  return await handleResponse(response);
}

/**
 * Doktor Girişi
 */
export async function loginDoctor(payload: LoginDoctorPayload): Promise<LoginDoctorResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
      "Bypass-Tunnel-Reminder": "true" 
    },
    body: JSON.stringify(payload),
  });

  return await handleResponse(response);
}

/**
 * Hastane Paneli Girişi
 */
export const loginHospital = async (hospitalName: string, password: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/hospital-login`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Bypass-Tunnel-Reminder": "true" 
      },
      body: JSON.stringify({
        hospital_name: hospitalName,
        password: password,
      }),
    });

    return await handleResponse(response);
  } catch (error: any) {
    console.error("Hospital login error:", error);
    return { success: false, message: error.message || "Sunucuya bağlanılamadı." };
  }
};

/**
 * Veritabanındaki Hastane Listesini Getirir
 */
export const getHospitalsFromDB = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/auth/hospitals`, {
          headers: { 
            "Bypass-Tunnel-Reminder": "true" 
          }
        });
        if (!response.ok) throw new Error("Hastaneler çekilemedi");
        return await response.json();
    } catch (error) {
        console.error("getHospitals error:", error);
        throw error;
    }
};