#include <Servo.h>

// Definición de los 10 servos con sus alias
Servo manzana;   // 0 a 30
Servo sandia;    // 0 a 30
Servo estrella;  // 0 a 30
Servo s77;       // 0 a 30 (alias '77')
Servo bar;       // 0 a 30
Servo campana;   // 180 a 150
Servo kiwi;      // 180 a 150
Servo naranja;   // 180 a 150
Servo cereza;    // 180 a 150
Servo start;     // 180 a 150 (Servo adicional)

void setup() {
  // Asignación de pines (Ajusta los pines según tu conexión física)
  manzana.attach(2);
  sandia.attach(3);
  estrella.attach(4);
  s77.attach(5);
  bar.attach(6);
  
  campana.attach(7);
  kiwi.attach(8);
  naranja(9);
  cereza.attach(10);
  start.attach(11);
}

void loop() {
  // --- POSICIÓN A (Estado Inicial) ---
  // Grupo 0 a 30 inicia en 0
  manzana.write(0);
  sandia.write(0);
  estrella.write(0);
  s77.write(0);
  bar.write(0);

  // Grupo 180 a 150 inicia en 180
  campana.write(180);
  kiwi.write(180);
  naranja.write(180);
  cereza.write(180);
  start.write(180);
  
  delay(2000); // Espera 2 segundos en Posición A

  // --- POSICIÓN B (Movimiento) ---
  // Grupo 0 a 30 se mueve a 30
  manzana.write(30);
  sandia.write(30);
  estrella.write(30);
  s77.write(30);
  bar.write(30);

  // Grupo 180 a 150 se mueve a 150
  campana.write(150);
  kiwi.write(150);
  naranja.write(150);
  cereza.write(150);
  start.write(150);

  delay(2000); // Espera 2 segundos en Posición B
}