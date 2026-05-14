import cv2
import numpy as np
import robotica

# =========================
# VELOCIDADES
# =========================
VEL_AVANCE_MAX = 2.2
VEL_GIRO_MAX = 1.4

AREA_OBJETIVO = 55000

# =========================
# CONTROL PD (GIRO)
# =========================
KP = 1.1
KD = 0.6

# =========================
# SONAR (UMBRAL)
# =========================
U_FRONTAL = 0.35
U_FRONT_LAT = 0.45
U_LATERAL = 0.5

K_REP = 1.2

# =========================
# PESOS
# =========================
W_ATR = 0.75
W_REP = 0.25

# =========================
# MEMORIA
# =========================
MEMORY_FRAMES = 8
SEARCH_SPEED = 0.6


# =========================
def procesar_mascara(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 20), (10, 255, 255)) + \
           cv2.inRange(hsv, (160, 100, 20), (180, 255, 255))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def obtener_centro(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, 0

    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 400:
        return None, area

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, area

    cx = int(M["m10"] / M["m00"])
    return cx, area


def leer_sonar(robot):
    s = robot.get_sonar()

    left = min(s[0], s[1], s[2])
    front_left = s[2]
    front = min(s[3], s[4])
    front_right = s[4]
    right = min(s[5], s[6], s[7])

    return left, front_left, front, front_right, right


def repulsion(left, fl, front, fr, right):
    # fuerzas
    fL = max(0.0, U_LATERAL - left)
    fR = max(0.0, U_LATERAL - right)

    fFL = max(0.0, U_FRONT_LAT - fl)
    fFR = max(0.0, U_FRONT_LAT - fr)

    fF = max(0.0, U_FRONTAL - front)

    # prioridad:
    # frontal domina totalmente
    if fF > 0.0:
        giro = K_REP * (fR - fL)
        return giro, 0.0

    # frontolateral anticipa giro
    giro = K_REP * ((fFR + fR) - (fFL + fL))

    # reducción de avance si hay peligro cercano
    avance_penalty = max(0.0, 1.0 - (fFL + fFR + fL + fR) / 2.0)

    return giro, avance_penalty


def limitar(x, a, b):
    return max(min(x, b), a)


# =========================
def main():
    sim = robotica.Coppelia()
    robot = robotica.P3DX(sim.sim, "PioneerP3DX", True)
    sim.start_simulation()

    last_cx = None
    lost = 0

    prev_error = 0.0

    try:
        while sim.is_running():

            img = robot.get_image()
            if img is None:
                continue

            h, w, _ = img.shape
            center = w / 2

            mask = procesar_mascara(img)
            cx, area = obtener_centro(mask)

            left, fl, front, fr, right = leer_sonar(robot)

            giro_rep, avance_penalty = repulsion(left, fl, front, fr, right)

            # =========================
            # MEMORIA
            # =========================
            if cx is not None:
                last_cx = cx
                lost = 0
                cx_use = cx
                area_use = area
            else:
                if last_cx is not None and lost < MEMORY_FRAMES:
                    cx_use = last_cx
                    lost += 1
                    area_use = 0
                else:
                    cx_use = None
                    area_use = 0

            # =========================
            # PD VISIÓN
            # =========================
            if cx_use is not None:
                error = (cx_use - center) / center
            else:
                error = 0.0

            deriv = error - prev_error
            prev_error = error

            giro_atr = KP * error + KD * deriv

            if cx_use is None:
                giro_atr = SEARCH_SPEED

            # =========================
            # FUSIÓN GIRO
            # =========================
            giro = W_ATR * giro_atr + W_REP * giro_rep
            giro = limitar(giro, -VEL_GIRO_MAX, VEL_GIRO_MAX)

            # =========================
            # AVANCE
            # =========================
            if cx_use is not None:
                if area_use < AREA_OBJETIVO:
                    proximity = np.sqrt(area_use / AREA_OBJETIVO)
                    base = VEL_AVANCE_MAX * (1.0 - proximity)

                    alignment = max(0.2, 1.0 - abs(error))

                    avance = base * alignment * avance_penalty
                else:
                    avance = 0.0
            else:
                avance = 0.15 * VEL_AVANCE_MAX

            # seguridad frontal dura
            if front < U_FRONTAL:
                avance = 0.0

            avance = limitar(avance, 0.0, VEL_AVANCE_MAX)

            # =========================
            # CONTROL ROBOT
            # =========================
            v_left = avance + giro
            v_right = avance - giro

            robot.set_speed(v_left, v_right)

            # =========================
            # VISUAL
            # =========================
            cv2.circle(img, (int(center), 50), 5, (255, 255, 255), -1)

            if cx_use is not None:
                cv2.circle(img, (int(cx_use), 50), 8, (0, 0, 255), -1)

            cv2.imshow("cam", img)
            cv2.imshow("mask", mask)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        robot.set_speed(0, 0)
        sim.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()