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


VEL_AVANCE = 1.4
VEL_GIRO_MAX = 1.2
K_GIRO = 1.5
AREA_MINIMA = 500


def procesar_mascara(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rojo_bajo1 = np.array([0, 100, 20])
    rojo_alto1 = np.array([10, 255, 255])

    rojo_bajo2 = np.array([160, 100, 20])
    rojo_alto2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mask2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)

    return mask1 + mask2


def obtener_centro(mask):
    contornos, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) == 0:
        return None, 0

    contorno = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(contorno)

    if area < AREA_MINIMA:
        return None, area

    M = cv2.moments(contorno)

    if M["m00"] == 0:
        return None, area

    cx = int(M["m10"] / M["m00"])
    return cx, area


def limitar(valor, minimo, maximo):
    return max(min(valor, maximo), minimo)


def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, "PioneerP3DX", True)

    coppelia.start_simulation()

    print("Programa iniciado. Pulsa q para salir.")

    try:
        while coppelia.is_running():
            img = robot.get_image()

            if img is None:
                continue

            alto, ancho, _ = img.shape
            centro_img = ancho / 2

            mask = procesar_mascara(img)
            cx, area = obtener_centro(mask)

            if cx is None:
                # Si no ve la bola, gira despacio buscándola
                v_izq = 0.4
                v_der = -0.4
                estado = "BUSCANDO BOLA"

            else:
                error = (cx - centro_img) / centro_img

                giro = K_GIRO * error
                giro = limitar(giro, -VEL_GIRO_MAX, VEL_GIRO_MAX)

                # Si la bola está muy descentrada, primero solo gira
                if abs(error) > 0.35:
                    avance = 0.0
                else:
                    avance = VEL_AVANCE

                # Si la bola está muy cerca, parar
                if area > 30000:
                    avance = 0.0

                v_izq = avance + giro
                v_der = avance - giro

                estado = f"SIGUIENDO BOLA | error={error:.2f}, area={area:.0f}"

                cv2.circle(mask, (cx, alto // 2), 10, 255, 2)
                cv2.line(mask, (int(centro_img), 0), (int(centro_img), alto), 255, 1)

            robot.set_speed(v_izq, v_der)

            cv2.putText(
                mask,
                estado,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                255,
                2
            )

            cv2.imshow("Mascara bola roja", mask)

            print(f"{estado} | v_izq={v_izq:.2f}, v_der={v_der:.2f}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        robot.set_speed(0, 0)
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()