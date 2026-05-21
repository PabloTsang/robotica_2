import cv2
import numpy as np
import robotica
import time


VEL_MAX = 2
VEL_MIN_BUSQUEDA = 0.3
GIRO_MAX = 1.6

AREA_MINIMA = 450
AREA_OBJETIVO = 43000

ALPHA_CX = 0.72
ALPHA_AREA = 0.65
ALPHA_GIRO = 0.18
ALPHA_AVANCE = 0.22

MEMORIA_FRAMES = 35

DIST_STOP = 0.18
DIST_FRONTAL = 0.55
DIST_FRONT_LAT = 0.48
DIST_LATERAL = 0.35

K_GIRO_BOLA = 0.85
K_GIRO_OBSTACULO = 1.15

GIRO_BUSQUEDA = 0.38
AVANCE_BUSQUEDA = 0.10

MOSTRAR_CAMARA = True


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def limitar(x, a, b):
    return max(min(x, b), a)


def suavizar(valor_anterior, valor_nuevo, alpha):
    return alpha * valor_nuevo + (1.0 - alpha) * valor_anterior


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def hombro_izq(x, a, b):
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def hombro_der(x, a, b):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def media_ponderada(pares):
    num = 0.0
    den = 0.0

    for peso, valor in pares:
        num += peso * valor
        den += peso

    if den == 0:
        return 0.0

    return num / den


# ============================================================
# VISION: DETECCION DE BOLA ROJA
# ============================================================

def procesar_mascara_roja(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0, 100, 25), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 25), (180, 255, 255))
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detectar_bola(mask):
    contornos, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return None, 0

    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < AREA_MINIMA:
        return None, area

    M = cv2.moments(c)

    if M["m00"] == 0:
        return None, area

    cx = int(M["m10"] / M["m00"])

    return cx, area


# ============================================================
# SONAR
# ============================================================

def leer_sonar(robot):
    s = robot.get_sonar()

    izquierda = min(s[0], s[1], s[2])
    frente_izq = min(s[2], s[3])
    frente = min(s[3], s[4])
    frente_der = min(s[4], s[5])
    derecha = min(s[5], s[6], s[7])

    return izquierda, frente_izq, frente, frente_der, derecha


# ============================================================
# CONTROL BORROSO PARA SEGUIR LA BOLA
# ============================================================

def controlador_bola(error, area):
    muy_izq = hombro_izq(error, -0.85, -0.45)
    izq = triangular(error, -0.70, -0.35, -0.05)
    centro = triangular(error, -0.20, 0.0, 0.20)
    der = triangular(error, 0.05, 0.35, 0.70)
    muy_der = hombro_der(error, 0.45, 0.85)

    lejos = hombro_izq(area, AREA_OBJETIVO * 0.20, AREA_OBJETIVO * 0.70)
    medio = triangular(area, AREA_OBJETIVO * 0.35, AREA_OBJETIVO * 0.75, AREA_OBJETIVO * 1.05)
    cerca = hombro_der(area, AREA_OBJETIVO * 0.85, AREA_OBJETIVO * 1.20)

    giro = media_ponderada([
        (muy_izq, -GIRO_MAX),
        (izq, -0.55 * GIRO_MAX),
        (centro, 0.0),
        (der, 0.55 * GIRO_MAX),
        (muy_der, GIRO_MAX),
    ])

    avance = media_ponderada([
        (lejos, VEL_MAX),
        (medio, VEL_MAX * 0.65),
        (cerca, 0.0),
    ])

    factor_centrado = limitar(1.0 - 0.85 * abs(error), 0.28, 1.0)

    avance *= factor_centrado
    giro *= K_GIRO_BOLA

    return avance, giro


# ============================================================
# CONTROL BORROSO DE OBSTACULOS
# ============================================================

def controlador_obstaculos(izq, fizq, frente, fder, der):
    obs_izq = max(
        hombro_izq(izq, 0.05, DIST_LATERAL),
        hombro_izq(fizq, 0.08, DIST_FRONT_LAT)
    )

    obs_frente = hombro_izq(frente, 0.08, DIST_FRONTAL)

    obs_der = max(
        hombro_izq(der, 0.05, DIST_LATERAL),
        hombro_izq(fder, 0.08, DIST_FRONT_LAT)
    )

    peligro = max(obs_izq, obs_frente, obs_der)

    giro_lateral = obs_izq - obs_der

    if izq > der:
        giro_frontal = obs_frente
    else:
        giro_frontal = -obs_frente

    giro = K_GIRO_OBSTACULO * GIRO_MAX * limitar(
        0.65 * giro_lateral + 0.95 * giro_frontal,
        -1.0,
        1.0
    )

    reduccion = limitar(
        0.95 * obs_frente + 0.40 * max(obs_izq, obs_der),
        0.0,
        0.92
    )

    factor_avance = 1.0 - reduccion

    return factor_avance, giro, peligro


# ============================================================
# CONVERSION A VELOCIDADES DE RUEDAS
# ============================================================

def diferencial(avance, giro):
    v_izq = avance + giro
    v_der = avance - giro

    v_izq = limitar(v_izq, -VEL_MAX, VEL_MAX)
    v_der = limitar(v_der, -VEL_MAX, VEL_MAX)

    return v_izq, v_der


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    sim = robotica.Coppelia()
    robot = robotica.P3DX(sim.sim, "PioneerP3DX", True)

    sim.start_simulation()

    print("Robot iniciado con control borroso mejorado. Pulsa q para salir.")

    cx_filtrado = None
    area_filtrada = 0.0

    frames_sin_bola = MEMORIA_FRAMES + 1

    ultimo_error = 0.0
    ultimo_lado = 1.0

    avance_suave = 0.0
    giro_suave = 0.0

    try:
        while sim.is_running():
            img = robot.get_image()

            if img is None:
                continue

            alto, ancho, _ = img.shape
            centro_img = ancho / 2.0

            mask = procesar_mascara_roja(img)
            cx, area = detectar_bola(mask)

            if cx is not None:
                if cx_filtrado is None:
                    cx_filtrado = float(cx)
                    area_filtrada = float(area)
                else:
                    cx_filtrado = suavizar(cx_filtrado, float(cx), ALPHA_CX)
                    area_filtrada = suavizar(area_filtrada, float(area), ALPHA_AREA)

                frames_sin_bola = 0

            else:
                frames_sin_bola += 1
                area_filtrada *= 0.92

            bola_en_memoria = cx_filtrado is not None and frames_sin_bola <= MEMORIA_FRAMES

            if bola_en_memoria:
                error = limitar((cx_filtrado - centro_img) / centro_img, -1.0, 1.0)
                ultimo_error = error

                if error > 0.12:
                    ultimo_lado = 1.0
                elif error < -0.12:
                    ultimo_lado = -1.0
            else:
                error = None
                cx_filtrado = None

            izq, fizq, frente, fder, der = leer_sonar(robot)

            factor_obs, giro_obs, peligro = controlador_obstaculos(
                izq,
                fizq,
                frente,
                fder,
                der
            )

            emergencia = (
                frente < DIST_STOP or
                fizq < DIST_STOP or
                fder < DIST_STOP
            )

            if emergencia:
                avance = -0.45

                if izq > der:
                    giro = 0.75
                else:
                    giro = -0.75

                estado = "EMERGENCIA"

            elif bola_en_memoria:
                avance_bola, giro_bola = controlador_bola(error, area_filtrada)

                peso_obs = limitar(peligro * 1.25, 0.0, 1.0)
                peso_bola = 1.0 - peso_obs

                avance = avance_bola * factor_obs
                giro = peso_bola * giro_bola + peso_obs * giro_obs

                estado = "SIGUIENDO"

            else:
                avance = AVANCE_BUSQUEDA * factor_obs
                giro = ultimo_lado * GIRO_BUSQUEDA + giro_obs * peligro

                estado = "BUSCANDO"

            avance = limitar(avance, -VEL_MAX, VEL_MAX)
            giro = limitar(giro, -GIRO_MAX, GIRO_MAX)

            avance_suave = suavizar(avance_suave, avance, ALPHA_AVANCE)
            giro_suave = suavizar(giro_suave, giro, ALPHA_GIRO)

            v_izq, v_der = diferencial(avance_suave, giro_suave)

            robot.set_speed(v_izq, v_der)

            print(
                f"{estado} | "
                f"VL={v_izq:.2f} VR={v_der:.2f} | "
                f"err={0.0 if error is None else error:.2f} | "
                f"area={area_filtrada:.0f} | "
                f"front={frente:.2f} L={izq:.2f} R={der:.2f} | "
                f"peligro={peligro:.2f}"
            )

            if MOSTRAR_CAMARA:
                debug = mask.copy()

                cv2.line(
                    debug,
                    (int(centro_img), 0),
                    (int(centro_img), alto),
                    255,
                    1
                )

                if cx_filtrado is not None:
                    cv2.circle(
                        debug,
                        (int(cx_filtrado), alto // 2),
                        10,
                        255,
                        2
                    )

                cv2.imshow("Mascara bola roja", debug)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.035)

    finally:
        robot.set_speed(0, 0)
        sim.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()