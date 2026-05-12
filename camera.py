import cv2
import numpy as np
import robotica
import time


VEL_AVANCE_MAX = 2.0
VEL_GIRO_MAX = 1.2

K_P = 2.2
K_D = 0.35

AREA_MINIMA = 500

ALPHA_ERROR = 0.25
ALPHA_DERIVADA = 0.2


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

    print("Pulsa q para salir")

    error_anterior = 0
    error_filtrado = 0

    derivada_filtrada = 0

    tiempo_anterior = time.time()

    try:

        while coppelia.is_running():

            img = robot.get_image()

            if img is None:
                continue

            alto, ancho, _ = img.shape

            centro_img = ancho / 2

            mask = procesar_mascara(img)

            cx, area = obtener_centro(mask)

            tiempo_actual = time.time()

            dt = tiempo_actual - tiempo_anterior

            if dt <= 0:
                dt = 1e-6

            if cx is None:

                v_izq = 0.3
                v_der = -0.3

                estado = "BUSCANDO"

            else:

                error = (cx - centro_img) / centro_img

                # FILTRO DEL ERROR
                error_filtrado = (
                    ALPHA_ERROR * error
                    + (1 - ALPHA_ERROR) * error_filtrado
                )

                # DERIVADA
                derivada = (
                    error_filtrado - error_anterior
                ) / dt

                # FILTRO DERIVATIVO
                derivada_filtrada = (
                    ALPHA_DERIVADA * derivada
                    + (1 - ALPHA_DERIVADA) * derivada_filtrada
                )

                # CONTROL PD
                giro = (
                    K_P * error_filtrado
                    + K_D * derivada_filtrada
                )

                giro = limitar(
                    giro,
                    -VEL_GIRO_MAX,
                    VEL_GIRO_MAX
                )

                # avance progresivo
                avance = (
                    VEL_AVANCE_MAX
                    * (1 - min(abs(error_filtrado), 1))
                )

                # parar si está muy cerca
                if area > 30000:
                    avance = 0

                v_izq = avance + giro
                v_der = avance - giro

                estado = (
                    f"error={error_filtrado:.2f} "
                    f"der={derivada_filtrada:.2f}"
                )

                cv2.circle(
                    mask,
                    (cx, alto // 2),
                    10,
                    255,
                    2
                )

                error_anterior = error_filtrado

            tiempo_anterior = tiempo_actual

            robot.set_speed(v_izq, v_der)

            cv2.putText(
                mask,
                estado,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                255,
                2
            )

            cv2.imshow("Mascara", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        robot.set_speed(0, 0)

        coppelia.stop_simulation()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()