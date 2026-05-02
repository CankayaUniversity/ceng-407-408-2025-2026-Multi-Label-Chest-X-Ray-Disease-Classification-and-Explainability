import { Stack } from "expo-router";

export default function Layout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" />
      <Stack.Screen name="doctor-login" />
      <Stack.Screen name="hospital-login" />
      <Stack.Screen name="register" />
      <Stack.Screen name="home" />
      <Stack.Screen name="pending-approval" />
      <Stack.Screen name="rejected-account" />
      <Stack.Screen name="hospital-dashboard" />
    </Stack>
  );
}