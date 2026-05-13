import cv2
import numpy as np
import robotica
import time


VEL_AVANCE_MAX = 2.0
VEL_GIRO_MAX = 1.2

K_P = 2.2
K_D = 0.35

AREA_MINIMA = 500
AREA_BOLA_CERCA = 30000

ALPHA_ERROR = 0.25
ALPHA_DERIVADA = 0.2

TIEMPO_MEMORIA_CORTA = 2.0
TIEMPO_BUSQUEDA_LARGA = 5.0

VEL_BUSQUEDA_GIRO = 0.5
VEL_BUSQUEDA_AVANCE = 0.5

DIST_OBSTACULO_FRONTAL = 0.35
DIST_OBSTACULO_LATERAL = 0.25


def procesar_mascara(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rojo_bajo1 = np.array([0, 100, 20])
    rojo_alto1 = np.array([10, 255, 255])

    rojo_bajo2 = np.array([160, 100, 20])
    rojo_alto2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mask2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


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


def leer_zonas_sonar(robot):
    sonar = robot.get_sonar()

    izquierda = min(sonar[0], sonar[1], sonar[2])
    frente = min(sonar[3], sonar[4])
    derecha = min(sonar[5], sonar[6], sonar[7])

    return izquierda, frente, derecha


def comportamiento_evitar_obstaculos(robot):
    izquierda, frente, derecha = leer_zonas_sonar(robot)

    obstaculo_frente = frente < DIST_OBSTACULO_FRONTAL
    obstaculo_izquierda = izquierda < DIST_OBSTACULO_LATERAL
    obstaculo_derecha = derecha < DIST_OBSTACULO_LATERAL

    if not obstaculo_frente and not obstaculo_izquierda and not obstaculo_derecha:
        return None

    if obstaculo_frente:
        if izquierda > derecha:
            return -0.4, 0.8
        else:
            return 0.8, -0.4

    if obstaculo_izquierda:
        return 0.8, 0.2

    if obstaculo_derecha:
        return 0.2, 0.8

    return None


def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, "PioneerP3DX", True)

    coppelia.start_simulation()

    print("Pulsa q para salir")

    error_anterior = 0.0
    error_filtrado = 0.0
    derivada_filtrada = 0.0

    tiempo_anterior = time.time()

    ultima_direccion = 0
    ultimo_tiempo_vista = time.time()
    ultimo_error_visto = 0.0

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

            if cx is not None:
                error = (cx - centro_img) / centro_img

                error_filtrado = (
                    ALPHA_ERROR * error
                    + (1 - ALPHA_ERROR) * error_filtrado
                )

                derivada = (
                    error_filtrado - error_anterior
                ) / dt

                derivada_filtrada = (
                    ALPHA_DERIVADA * derivada
                    + (1 - ALPHA_DERIVADA) * derivada_filtrada
                )

                giro = (
                    K_P * error_filtrado
                    + K_D * derivada_filtrada
                )

                giro = limitar(
                    giro,
                    -VEL_GIRO_MAX,
                    VEL_GIRO_MAX
                )

                avance = (
                    VEL_AVANCE_MAX
                    * (1 - min(abs(error_filtrado), 1))
                )

                if area > AREA_BOLA_CERCA:
                    avance = 0.0

                v_izq = avance + giro
                v_der = avance - giro

                ultimo_tiempo_vista = tiempo_actual
                ultimo_error_visto = error_filtrado

                if error_filtrado > 0.15:
                    ultima_direccion = 1
                elif error_filtrado < -0.15:
                    ultima_direccion = -1
                else:
                    ultima_direccion = 0

                error_anterior = error_filtrado

                estado = (
                    f"SIGUIENDO | error={error_filtrado:.2f} "
                    f"area={area:.0f}"
                )

                cv2.circle(
                    mask,
                    (cx, alto // 2),
                    10,
                    255,
                    2
                )

                cv2.line(
                    mask,
                    (int(centro_img), 0),
                    (int(centro_img), alto),
                    255,
                    1
                )

            else:
                tiempo_sin_ver = tiempo_actual - ultimo_tiempo_vista

                velocidades_evitacion = comportamiento_evitar_obstaculos(robot)

                if velocidades_evitacion is not None:
                    v_izq, v_der = velocidades_evitacion
                    estado = "EVITANDO OBSTACULO"

                elif tiempo_sin_ver < TIEMPO_MEMORIA_CORTA:
                    if ultima_direccion > 0:
                        v_izq = VEL_BUSQUEDA_GIRO
                        v_der = -VEL_BUSQUEDA_GIRO
                        estado = "PERDIDA RECIENTE | buscando derecha"

                    elif ultima_direccion < 0:
                        v_izq = -VEL_BUSQUEDA_GIRO
                        v_der = VEL_BUSQUEDA_GIRO
                        estado = "PERDIDA RECIENTE | buscando izquierda"

                    else:
                        v_izq = VEL_BUSQUEDA_AVANCE
                        v_der = VEL_BUSQUEDA_AVANCE
                        estado = "PERDIDA RECIENTE | avanzando"

                elif tiempo_sin_ver < TIEMPO_BUSQUEDA_LARGA:
                    if ultima_direccion >= 0:
                        v_izq = 0.7
                        v_der = 0.2
                        estado = "BUSQUEDA MEDIA | derecha"
                    else:
                        v_izq = 0.2
                        v_der = 0.7
                        estado = "BUSQUEDA MEDIA | izquierda"

                else:
                    if ultima_direccion >= 0:
                        v_izq = 0.5
                        v_der = -0.5
                        estado = "BUSQUEDA AMPLIA | derecha"
                    else:
                        v_izq = -0.5
                        v_der = 0.5
                        estado = "BUSQUEDA AMPLIA | izquierda"

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

            print(
                f"{estado} | "
                f"v_izq={v_izq:.2f}, v_der={v_der:.2f}"
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        robot.set_speed(0, 0)
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()