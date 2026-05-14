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
KD = 0.6  # 🔥 clave para eliminar zigzag

# =========================
# REPULSIÓN
# =========================
UMBRAL_REP = 0.5
K_REP = 0.5

# mezcla estable (evita peleas)
W_ATRACCION = 0.8
W_REPULSION = 0.2

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
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


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
    front = min(s[3], s[4])
    right = min(s[5], s[6], s[7])
    return left, front, right


def repulsion(left, front, right):
    lf = max(0.0, UMBRAL_REP - left)
    rf = max(0.0, UMBRAL_REP - right)
    ff = max(0.0, UMBRAL_REP - front)

    giro_rep = K_REP * (rf - lf)
    return giro_rep


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

            left, front, right = leer_sonar(robot)
            giro_rep = repulsion(left, front, right)

            # =========================
            # DETECCIÓN / MEMORIA
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
            # CONTROL ANGULAR PD
            # =========================
            if cx_use is not None:
                error = (cx_use - center) / center
            else:
                error = 0.0

            deriv = error - prev_error
            prev_error = error

            giro_atraccion = KP * error + KD * deriv

            # búsqueda si no ve bola
            if cx_use is None:
                giro_atraccion = SEARCH_SPEED

            # =========================
            # COMBINACIÓN ESTABLE
            # =========================
            giro = (W_ATRACCION * giro_atraccion +
                    W_REPULSION * giro_rep)

            giro = limitar(giro, -VEL_GIRO_MAX, VEL_GIRO_MAX)

            # =========================
            # AVANCE
            # =========================
            if cx_use is not None:
                if area_use < AREA_OBJETIVO:
                    proximity = np.sqrt(area_use / AREA_OBJETIVO)
                    base = VEL_AVANCE_MAX * (1.0 - proximity)

                    alignment_penalty = max(0.2, 1.0 - abs(error))

                    avance = base * alignment_penalty
                else:
                    avance = 0.0
            else:
                avance = 0.15 * VEL_AVANCE_MAX

            avance = limitar(avance, 0.0, VEL_AVANCE_MAX)

            # =========================
            # ROBOT
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