import cv2
import numpy as np
import robotica
import time


VEL_MAX = 1.35
GIRO_MAX = 0.85

AREA_MINIMA = 450
AREA_OBJETIVO = 43000

DIST_STOP = 0.17
DIST_FRONTAL = 0.48
DIST_LATERAL_DESEADA = 0.38
DIST_LATERAL_PERDIDA = 0.75

ALPHA_CX = 0.72
ALPHA_AREA = 0.62
ALPHA_AVANCE = 0.25
ALPHA_GIRO = 0.13

MEMORIA_BOLA = 35

K_BOLA = 0.75
K_PARED = 0.85
K_FRONTAL = 1.00

AVANCE_SEGUIR = 1.10
AVANCE_BORDEAR = 0.46
AVANCE_BUSCAR = 0.10

GIRO_BUSCAR = 0.30
GIRO_ESQUINA = 0.62
GIRO_ESCAPE = 0.65

FRAMES_GIRO_ESQUINA = 28
FRAMES_ESCAPE = 16

STATE_SEGUIR = 0
STATE_BORDEAR = 1
STATE_BUSCAR = 2
STATE_ESCAPE = 3
STATE_GIRAR_ESQUINA = 4

MOSTRAR_CAMARA = True


def limitar(x, a, b):
    return max(min(x, b), a)


def suavizar(prev, nuevo, alpha):
    return alpha * nuevo + (1.0 - alpha) * prev


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
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def leer_sonar(robot):
    s = robot.get_sonar()

    izquierda = min(s[0], s[1], s[2])
    frente_izq = min(s[2], s[3])
    frente = min(s[3], s[4])
    frente_der = min(s[4], s[5])
    derecha = min(s[5], s[6], s[7])

    return izquierda, frente_izq, frente, frente_der, derecha


def hay_obstaculo_frontal(frente, fizq, fder):
    return (
        frente < DIST_FRONTAL or
        fizq < DIST_FRONTAL * 0.85 or
        fder < DIST_FRONTAL * 0.85
    )


def emergencia(frente, fizq, fder):
    return frente < DIST_STOP or fizq < DIST_STOP or fder < DIST_STOP


def elegir_lado_pared(izq, fizq, fder, der):
    """
    Devuelve el lado de la pared que se va a seguir.
    1  = pared a la derecha
    -1 = pared a la izquierda
    """

    obst_izq = min(izq, fizq)
    obst_der = min(der, fder)

    if obst_der < obst_izq:
        return 1
    else:
        return -1


def control_bola(error, area):
    giro = K_BOLA * error

    proximidad = limitar(area / AREA_OBJETIVO, 0.0, 1.0)
    avance = AVANCE_SEGUIR * (1.0 - proximidad)

    avance *= limitar(1.0 - 0.65 * abs(error), 0.35, 1.0)

    if area >= AREA_OBJETIVO:
        avance = 0.0

    return avance, limitar(giro, -GIRO_MAX, GIRO_MAX)


def control_bordeo(lado_pared, izq, fizq, frente, fder, der):
    if lado_pared == 1:
        dist_lateral = der
        dist_diag = fder
    else:
        dist_lateral = izq
        dist_diag = fizq

    error_pared = dist_lateral - DIST_LATERAL_DESEADA

    giro_pared = lado_pared * K_PARED * error_pared

    peligro_frontal = limitar(
        (DIST_FRONTAL - min(frente, dist_diag)) / DIST_FRONTAL,
        0.0,
        1.0
    )

    giro_frontal = -lado_pared * K_FRONTAL * peligro_frontal

    giro = giro_pared + giro_frontal
    giro = limitar(giro, -GIRO_MAX, GIRO_MAX)

    avance = AVANCE_BORDEAR * (1.0 - 0.55 * peligro_frontal)
    avance = limitar(avance, 0.16, AVANCE_BORDEAR)

    return avance, giro


def final_de_pared(lado_pared, izq, fizq, frente, fder, der):
    if lado_pared == 1:
        lateral_libre = der > DIST_LATERAL_PERDIDA and fder > DIST_LATERAL_PERDIDA * 0.75
    else:
        lateral_libre = izq > DIST_LATERAL_PERDIDA and fizq > DIST_LATERAL_PERDIDA * 0.75

    frente_libre = frente > DIST_FRONTAL * 1.10

    return lateral_libre and frente_libre


def linea_visual_libre(error, frente, fizq, fder):
    if error is None:
        return False

    return (
        abs(error) < 0.42 and
        frente > DIST_FRONTAL * 1.10 and
        fizq > DIST_FRONTAL * 0.85 and
        fder > DIST_FRONTAL * 0.85
    )


def diferencial(avance, giro):
    v_izq = limitar(avance + giro, -VEL_MAX, VEL_MAX)
    v_der = limitar(avance - giro, -VEL_MAX, VEL_MAX)
    return v_izq, v_der


def main():
    sim = robotica.Coppelia()
    robot = robotica.P3DX(sim.sim, "PioneerP3DX", True)

    sim.start_simulation()

    print("Bug-2 corregido: mantiene lado de pared y rodea el final.")

    estado = STATE_BUSCAR

    cx_filtrado = None
    area_filtrada = 0.0
    frames_sin_bola = MEMORIA_BOLA + 1

    ultimo_lado_bola = 1
    lado_pared = 1

    avance_suave = 0.0
    giro_suave = 0.0

    timer = 0

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

            bola_memoria = cx_filtrado is not None and frames_sin_bola <= MEMORIA_BOLA

            if bola_memoria:
                error = limitar((cx_filtrado - centro_img) / centro_img, -1.0, 1.0)

                if error > 0.12:
                    ultimo_lado_bola = 1
                elif error < -0.12:
                    ultimo_lado_bola = -1
            else:
                error = None
                cx_filtrado = None

            izq, fizq, frente, fder, der = leer_sonar(robot)

            if emergencia(frente, fizq, fder) and estado != STATE_ESCAPE:
                estado = STATE_ESCAPE
                timer = FRAMES_ESCAPE

            if estado == STATE_ESCAPE:
                avance = -0.42

                if izq > der:
                    giro = GIRO_ESCAPE
                    lado_pared = -1
                else:
                    giro = -GIRO_ESCAPE
                    lado_pared = 1

                timer -= 1

                if timer <= 0:
                    estado = STATE_BORDEAR

            elif estado == STATE_BUSCAR:
                avance = AVANCE_BUSCAR
                giro = ultimo_lado_bola * GIRO_BUSCAR

                if bola_memoria:
                    estado = STATE_SEGUIR

            elif estado == STATE_SEGUIR:
                if not bola_memoria:
                    estado = STATE_BUSCAR
                    avance = AVANCE_BUSCAR
                    giro = ultimo_lado_bola * GIRO_BUSCAR

                elif hay_obstaculo_frontal(frente, fizq, fder):
                    lado_pared = elegir_lado_pared(izq, fizq, fder, der)
                    estado = STATE_BORDEAR

                    avance, giro = control_bordeo(
                        lado_pared,
                        izq,
                        fizq,
                        frente,
                        fder,
                        der
                    )

                else:
                    avance, giro = control_bola(error, area_filtrada)

            elif estado == STATE_BORDEAR:
                avance, giro = control_bordeo(
                    lado_pared,
                    izq,
                    fizq,
                    frente,
                    fder,
                    der
                )

                if final_de_pared(lado_pared, izq, fizq, frente, fder, der):
                    estado = STATE_GIRAR_ESQUINA
                    timer = FRAMES_GIRO_ESQUINA

                elif bola_memoria and linea_visual_libre(error, frente, fizq, fder):
                    estado = STATE_SEGUIR

            elif estado == STATE_GIRAR_ESQUINA:
                avance = 0.28
                giro = lado_pared * GIRO_ESQUINA

                timer -= 1

                if timer <= 0:
                    if bola_memoria and linea_visual_libre(error, frente, fizq, fder):
                        estado = STATE_SEGUIR
                    else:
                        estado = STATE_BORDEAR

            avance = limitar(avance, -VEL_MAX, VEL_MAX)
            giro = limitar(giro, -GIRO_MAX, GIRO_MAX)

            avance_suave = suavizar(avance_suave, avance, ALPHA_AVANCE)
            giro_suave = suavizar(giro_suave, giro, ALPHA_GIRO)

            v_izq, v_der = diferencial(avance_suave, giro_suave)

            robot.set_speed(v_izq, v_der)

            nombre_estado = {
                STATE_SEGUIR: "SEGUIR",
                STATE_BORDEAR: "BORDEAR_PARED",
                STATE_BUSCAR: "BUSCAR",
                STATE_ESCAPE: "ESCAPE",
                STATE_GIRAR_ESQUINA: "GIRAR_ESQUINA"
            }[estado]

            print(
                f"{nombre_estado} | "
                f"VL={v_izq:.2f} VR={v_der:.2f} | "
                f"F={frente:.2f} L={izq:.2f} R={der:.2f} | "
                f"lado_pared={lado_pared} | "
                f"err={0.0 if error is None else error:.2f}"
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