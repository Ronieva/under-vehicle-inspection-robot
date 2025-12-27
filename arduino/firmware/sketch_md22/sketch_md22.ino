#include <Wire.h>

// Dirección del MD22
#define MD22_ADDR 0x58
#define REG_MODE   0x00
#define REG_M1     0x01   // Motor izquierdo
#define REG_M2     0x02   // Motor derecho
#define REG_ACCEL  0x03

String buffer = "";
bool comandoCompleto = false;

// Velocidades básicas
const int V_BASE = 80;         // avance y retroceso
const int V_GIRO_MOV = 60;     // giro mientras avanza
const int V_GIRO_PURO = 60;    // giro sobre si mismo

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 2000));

  Wire.begin();

  // Modo 0 (control independiente motor 1 y motor 2)
  Wire.beginTransmission(MD22_ADDR);
  Wire.write(REG_MODE);
  Wire.write(0);
  Wire.endTransmission();

  // Aceleración suave
  Wire.beginTransmission(MD22_ADDR);
  Wire.write(REG_ACCEL);
  Wire.write(5);
  Wire.endTransmission();

  parar();

  Serial.println("MD22 listo en MODO 0");
}

void loop() {

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      comandoCompleto = true;
      break;
    } else buffer += c;
  }

  if (comandoCompleto) {
    buffer.trim();
    procesar(buffer);
    buffer = "";
    comandoCompleto = false;
  }
}

// ===============================
// CONTROL DE MOTORES (modo 0)
// ===============================
void setMotores(int velL, int velR) {
  uint8_t mL = constrain(128 + velL, 0, 255);
  uint8_t mR = constrain(128 + velR, 0, 255);

  Wire.beginTransmission(MD22_ADDR);
  Wire.write(REG_M1);
  Wire.write(mL);
  Wire.endTransmission();

  Wire.beginTransmission(MD22_ADDR);
  Wire.write(REG_M2);
  Wire.write(mR);
  Wire.endTransmission();

  Serial.print("Motores -> L:");
  Serial.print(mL);
  Serial.print(" R:");
  Serial.println(mR);
}

void parar() {
  setMotores(0, 0);
}

// ===============================
// PROCESAR COMANDOS
// ===============================
void procesar(String c) {
  Serial.print("CMD=");
  Serial.println(c);

  if (c == "adelante")
    setMotores(V_BASE, V_BASE);

  else if (c == "atras")
    setMotores(-V_BASE, -V_BASE);

  // GIRO mientras avanza
  else if (c == "der_adelante")
    setMotores(V_BASE + V_GIRO_MOV, V_BASE - V_GIRO_MOV);

  else if (c == "izq_adelante")
    setMotores(V_BASE - V_GIRO_MOV, V_BASE + V_GIRO_MOV);

  // GIRO mientras retrocede
  else if (c == "der_atras")
    setMotores(-V_BASE +V_GIRO_MOV, -V_BASE -V_GIRO_MOV);

  else if (c == "izq_atras")
    setMotores(-V_BASE -V_GIRO_MOV, -V_BASE +V_GIRO_MOV);

  // ⭐ GIRO SOBRE EL SITIO (ROTACIÓN PURA)
  else if (c == "giro_der")
    setMotores(-V_GIRO_PURO, +V_GIRO_PURO);

  else if (c == "giro_izq")
    setMotores(+V_GIRO_PURO, -V_GIRO_PURO);

  else if (c == "parar")
    parar();

  else
    parar();
}
