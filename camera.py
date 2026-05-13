import cv2
import numpy as np
import robotica
import time


VEL_AVANCE_MAX = 2.15
VEL_AVANCE_BUSQUEDA = 0.75
VEL_GIRO_MAX = 1.45

K_P = 2.45
K_D = 0.28

AREA_MINIMA = 450
AREA_BOLA_CERCA = 36000

ALPHA_ERROR = 0.35
ALPHA_DERIVADA = 0.18

TIEMPO_MEMORIA_CORTA = 1.8
TIEMPO_BUSQUEDA_LARGA = 5.0

VEL_BUSQUEDA_GIRO = 0.55
VEL_BUSQUEDA_AVANCE = 0.65

DIST_PRECAUCION_FRONTAL = 0.70
DIST_PRECAUCION_LATERAL = 0.48
DIST_CRITICA_FRONTAL = 0.24
DIST_CRITICA_LATERAL = 0.18

K_EVITA_FRONTAL = 1.55
K_EVITA_LATERAL = 1.10

REDUCCION_FRONTAL = 0.88
REDUCCION_LATERAL = 0.38

ALPHA_SONAR = 0.45

MAX_CAMBIO_VEL = 0.18
MAX_CAMBIO_GIRO = 0.22


def procesar_mascara(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rojo_bajo1 = np.array([0, 95, 25])
    rojo_alto1 = np.array([12, 255, 255])
    rojo_bajo2 = np.array([158, 95, 25])
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
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) == 0:
        return None, 0

    mejor_contorno = None
    mejor_puntuacion = 0

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < AREA_MINIMA:
            continue

        perimetro = cv2.arcLength(contorno, True)

        if perimetro <= 0:
            continue

        circularidad = 4.0 * np.pi * area / (perimetro * perimetro)
        puntuacion = area * max(0.35, min(circularidad, 1.0))

        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_contorno = contorno

    if mejor_contorno is None:
        return None, 0

    area = cv2.contourArea(mejor_contorno)
    M = cv2.moments(mejor_contorno)

    if M["m00"] == 0:
        return None, area

    cx = int(M["m10"] / M["m00"])

    return cx, area


def limitar(valor, minimo, maximo):
    return max(min(valor, maximo), minimo)


def acercar_suave(actual, objetivo, max_cambio):
    if objetivo > actual + max_cambio:
        return actual + max_cambio

    if objetivo < actual - max_cambio:
        return actual - max_cambio

    return objetivo


def leer_zonas_sonar(robot):
    sonar = robot.get_sonar()
    izquierda = min(sonar[0], sonar[1], sonar[2])
    frente = min(sonar[3], sonar[4])
    derecha = min(sonar[5], sonar[6], sonar[7])

    return izquierda, frente, derecha


def filtrar_sonar(zonas_filtradas, zonas_nuevas):
    if zonas_filtradas is None:
        return zonas_nuevas

    izq_ant, frente_ant, der_ant = zonas_filtradas
    izq, frente, der = zonas_nuevas

    izquierda = ALPHA_SONAR * izq + (1 - ALPHA_SONAR) * izq_ant
    frente = ALPHA_SONAR * frente + (1 - ALPHA_SONAR) * frente_ant
    derecha = ALPHA_SONAR * der + (1 - ALPHA_SONAR) * der_ant

    return izquierda, frente, derecha


def riesgo_por_distancia(distancia, distancia_precaucion, distancia_critica):
    if distancia >= distancia_precaucion:
        return 0.0

    if distancia <= distancia_critica:
        return 1.0

    return (
        distancia_precaucion - distancia
    ) / (
        distancia_precaucion - distancia_critica
    )


def calcular_evitacion(zonas_sonar):
    izquierda, frente, derecha = zonas_sonar

    riesgo_izq = riesgo_por_distancia(
        izquierda, DIST_PRECAUCION_LATERAL, DIST_CRITICA_LATERAL
    )
    riesgo_frente = riesgo_por_distancia(
        frente, DIST_PRECAUCION_FRONTAL, DIST_CRITICA_FRONTAL
    )
    riesgo_der = riesgo_por_distancia(
        derecha, DIST_PRECAUCION_LATERAL, DIST_CRITICA_LATERAL
    )

    giro_lateral = K_EVITA_LATERAL * (riesgo_izq - riesgo_der)

    if riesgo_frente > 0:
        if izquierda > derecha:
            direccion_libre = -1.0
        else:
            direccion_libre = 1.0
    else:
        direccion_libre = 0.0

    giro_frontal = K_EVITA_FRONTAL * riesgo_frente * direccion_libre
    giro_evitacion = giro_lateral + giro_frontal
    giro_evitacion = limitar(giro_evitacion, -VEL_GIRO_MAX, VEL_GIRO_MAX)

    factor_avance = 1.0
    factor_avance -= REDUCCION_FRONTAL * riesgo_frente
    factor_avance -= REDUCCION_LATERAL * max(riesgo_izq, riesgo_der)
    factor_avance = limitar(factor_avance, 0.18, 1.0)

    obstaculo_critico = (
        frente <= DIST_CRITICA_FRONTAL
        or izquierda <= DIST_CRITICA_LATERAL
        or derecha <= DIST_CRITICA_LATERAL
    )

    if obstaculo_critico and frente <= DIST_CRITICA_FRONTAL:
        factor_avance = 0.0

        if izquierda > derecha:
            giro_evitacion = -VEL_GIRO_MAX
        else:
            giro_evitacion = VEL_GIRO_MAX

    return giro_evitacion, factor_avance, obstaculo_critico


def combinar_seguimiento_y_evitacion(avance_base, giro_base, zonas_sonar):
    giro_evitacion, factor_avance, obstaculo_critico = calcular_evitacion(
        zonas_sonar
    )

    avance = avance_base * factor_avance
    giro = giro_base + giro_evitacion
    giro = limitar(giro, -VEL_GIRO_MAX, VEL_GIRO_MAX)

    if obstaculo_critico:
        giro = giro_evitacion

    v_izq = avance + giro
    v_der = avance - giro

    v_izq = limitar(v_izq, -VEL_GIRO_MAX, VEL_AVANCE_MAX)
    v_der = limitar(v_der, -VEL_GIRO_MAX, VEL_AVANCE_MAX)

    return v_izq, v_der, giro_evitacion, factor_avance, obstaculo_critico


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
    zonas_sonar_filtradas = None

    v_izq_actual = 0.0
    v_der_actual = 0.0

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

            zonas_sonar = leer_zonas_sonar(robot)
            zonas_sonar_filtradas = filtrar_sonar(
                zonas_sonar_filtradas, zonas_sonar
            )
            izquierda, frente, derecha = zonas_sonar_filtradas

            if cx is not None:
                error = (cx - centro_img) / centro_img
                error_filtrado = (
                    ALPHA_ERROR * error
                    + (1 - ALPHA_ERROR) * error_filtrado
                )

                derivada = (error_filtrado - error_anterior) / dt
                derivada_filtrada = (
                    ALPHA_DERIVADA * derivada
                    + (1 - ALPHA_DERIVADA) * derivada_filtrada
                )

                giro_base = K_P * error_filtrado + K_D * derivada_filtrada
                giro_base = limitar(giro_base, -VEL_GIRO_MAX, VEL_GIRO_MAX)

                avance_base = VEL_AVANCE_MAX * (
                    1.0 - 0.58 * min(abs(error_filtrado), 1.0)
                )
                avance_base = limitar(avance_base, 0.42, VEL_AVANCE_MAX)

                if area > AREA_BOLA_CERCA:
                    avance_base = 0.0

                (
                    v_izq_objetivo,
                    v_der_objetivo,
                    giro_evitacion,
                    factor_avance,
                    obstaculo_critico,
                ) = combinar_seguimiento_y_evitacion(
                    avance_base, giro_base, zonas_sonar_filtradas
                )

                ultimo_tiempo_vista = tiempo_actual
                ultimo_error_visto = error_filtrado

                if error_filtrado > 0.12:
                    ultima_direccion = 1
                elif error_filtrado < -0.12:
                    ultima_direccion = -1
                else:
                    ultima_direccion = 0

                error_anterior = error_filtrado

                if obstaculo_critico:
                    estado = (
                        f"SIGUIENDO+EVITANDO CRITICO | "
                        f"err={error_filtrado:.2f} area={area:.0f}"
                    )
                elif abs(giro_evitacion) > 0.08 or factor_avance < 0.95:
                    estado = (
                        f"SIGUIENDO+EVITANDO | "
                        f"err={error_filtrado:.2f} area={area:.0f}"
                    )
                else:
                    estado = (
                        f"SIGUIENDO | "
                        f"err={error_filtrado:.2f} area={area:.0f}"
                    )

                cv2.circle(mask, (cx, alto // 2), 10, 255, 2)
                cv2.line(
                    mask, (int(centro_img), 0),
                    (int(centro_img), alto), 255, 1
                )

            else:
                tiempo_sin_ver = tiempo_actual - ultimo_tiempo_vista

                if tiempo_sin_ver < TIEMPO_MEMORIA_CORTA:
                    if ultima_direccion > 0:
                        avance_base = VEL_BUSQUEDA_AVANCE * 0.70
                        giro_base = VEL_BUSQUEDA_GIRO
                        estado = "PERDIDA RECIENTE | buscando derecha"

                    elif ultima_direccion < 0:
                        avance_base = VEL_BUSQUEDA_AVANCE * 0.70
                        giro_base = -VEL_BUSQUEDA_GIRO
                        estado = "PERDIDA RECIENTE | buscando izquierda"

                    else:
                        avance_base = VEL_BUSQUEDA_AVANCE
                        giro_base = limitar(
                            0.55 * ultimo_error_visto,
                            -VEL_BUSQUEDA_GIRO,
                            VEL_BUSQUEDA_GIRO
                        )
                        estado = "PERDIDA RECIENTE | avanzando"

                elif tiempo_sin_ver < TIEMPO_BUSQUEDA_LARGA:
                    avance_base = VEL_AVANCE_BUSQUEDA

                    if ultima_direccion >= 0:
                        giro_base = 0.45
                        estado = "BUSQUEDA MEDIA | derecha"
                    else:
                        giro_base = -0.45
                        estado = "BUSQUEDA MEDIA | izquierda"

                else:
                    avance_base = 0.35

                    if ultima_direccion >= 0:
                        giro_base = 0.95
                        estado = "BUSQUEDA AMPLIA | derecha"
                    else:
                        giro_base = -0.95
                        estado = "BUSQUEDA AMPLIA | izquierda"

                (
                    v_izq_objetivo,
                    v_der_objetivo,
                    giro_evitacion,
                    factor_avance,
                    obstaculo_critico,
                ) = combinar_seguimiento_y_evitacion(
                    avance_base, giro_base, zonas_sonar_filtradas
                )

                if obstaculo_critico:
                    estado = "EVITANDO OBSTACULO CRITICO"
                elif abs(giro_evitacion) > 0.08 or factor_avance < 0.95:
                    estado += " + evitando"

            v_izq_actual = acercar_suave(
                v_izq_actual,
                v_izq_objetivo,
                MAX_CAMBIO_VEL
                + MAX_CAMBIO_GIRO * abs(v_izq_objetivo - v_izq_actual)
            )

            v_der_actual = acercar_suave(
                v_der_actual,
                v_der_objetivo,
                MAX_CAMBIO_VEL
                + MAX_CAMBIO_GIRO * abs(v_der_objetivo - v_der_actual)
            )

            tiempo_anterior = tiempo_actual
            robot.set_speed(v_izq_actual, v_der_actual)

            cv2.putText(
                mask, estado, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 2
            )

            cv2.putText(
                mask,
                f"S izq={izquierda:.2f} fr={frente:.2f} der={derecha:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1
            )

            cv2.imshow("Mascara", mask)

            print(
                f"{estado} | "
                f"v_izq={v_izq_actual:.2f}, v_der={v_der_actual:.2f} | "
                f"sonar=({izquierda:.2f}, {frente:.2f}, {derecha:.2f})"
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        robot.set_speed(0, 0)
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()