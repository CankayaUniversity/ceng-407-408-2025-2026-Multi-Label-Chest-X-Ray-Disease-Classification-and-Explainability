export type DoctorStatus = "pending" | "approved" | "rejected";

export interface Hospital {
  _id: string;
  name: string;
  city: string;
}

export interface Doctor {
  _id: string;
  fullName: string;
  email: string;
  password: string;
  hospitalId: string;
  hospitalName: string;
  status: DoctorStatus;
  createdAt: string;
}

export interface DoctorLoginResponse {
  success: boolean;
  status?: DoctorStatus;
  doctorName?: string;
  message?: string;
}

export interface HospitalLoginResponse {
  success: boolean;
  hospitalName?: string;
  message?: string;
}