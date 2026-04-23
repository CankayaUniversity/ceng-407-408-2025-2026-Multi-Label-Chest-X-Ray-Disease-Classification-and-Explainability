import { API_BASE_URL } from "./api";

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
}

export async function registerDoctor(payload: RegisterDoctorPayload) {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Kayıt işlemi başarısız oldu.");
  }

  return data;
}

export async function loginDoctor(
  payload: LoginDoctorPayload
): Promise<LoginDoctorResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Giriş işlemi başarısız oldu.");
  }

  return data;
}