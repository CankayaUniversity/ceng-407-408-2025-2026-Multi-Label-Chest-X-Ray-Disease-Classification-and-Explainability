import { mockPendingDoctors } from "@/constants/mockData";
import { Doctor } from "@/types";

export async function getPendingDoctors(): Promise<Doctor[]> {
  await new Promise((resolve) => setTimeout(resolve, 400));
  return mockPendingDoctors;
}

export async function approveDoctor(doctorId: string): Promise<boolean> {
  await new Promise((resolve) => setTimeout(resolve, 300));
  console.log("Approved doctor:", doctorId);
  return true;
}

export async function rejectDoctor(doctorId: string): Promise<boolean> {
  await new Promise((resolve) => setTimeout(resolve, 300));
  console.log("Rejected doctor:", doctorId);
  return true;
}