'''
camera.py

Sample client for the Pioneer P3DX mobile robot that receives and
displays images from the camera.

Copyright (C) 2023 Javier de Lope

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import cv2
import numpy as np
import robotica

def procesar_mascara(img):
    """Convierte a HSV y aplica el doble umbral para el color rojo."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Rangos del rojo
    rango_bajo1 = np.array([0, 100, 20])
    rango_alto1 = np.array([10, 255, 255])
    rango_bajo2 = np.array([160, 100, 20])
    rango_alto2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, rango_bajo1, rango_alto1)
    mask2 = cv2.inRange(hsv, rango_bajo2, rango_alto2)
    
    return cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0)

def obtener_centro(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contornos:
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy, c
    return None, None, None

def dibujar_info(mask, cx, cy):
    cv2.circle(mask, (cx, cy), 10, (150), 2)
    cv2.line(mask, (cx - 15, cy), (cx + 15, cy), (150), 2)
    cv2.line(mask, (cx, cy - 15), (cx, cy + 15), (150), 2)
    
    cv2.putText(mask, f"Centro: {cx}, {cy}", (cx + 20, cy - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)
    return mask

def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()
    
    print("Sistema listo. Presiona 'q' en la ventana de imagen para salir.")

    try:
        while coppelia.is_running():
            img = robot.get_image()
            if img is None:
                continue

            mascara = procesar_mascara(img)
            
            cx, cy, contorno_valido = obtener_centro(mascara)
            
            if cx is not None:
                mascara = dibujar_info(mascara, cx, cy)
                print(f"Bola detectada -> X: {cx}, Y: {cy}")
            
            cv2.imshow('Seguimiento de Bola (Mascara)', mascara)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
