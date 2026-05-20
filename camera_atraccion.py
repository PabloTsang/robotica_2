import cv2
import numpy as np
import robotica

VEL_AVANCE_MAX = 2.2
VEL_AVANCE_MIN_EVITANDO = 0.18
VEL_GIRO_MAX = 1.55

AREA_OBJETIVO = 55000
AREA_MINIMA = 350

KP = 1.05
KD = 0.42
KI_SUAVE = 0.015

U_EMERGENCIA = 0.22
U_FRONTAL = 0.48
U_FRONT_LAT = 0.58
U_LATERAL = 0.52

K_REP_LATERAL = 1.25
K_REP_FRONTAL = 1.65
K_REP_TANGENCIAL = 0.85

MEMORY_FRAMES = 28
SEARCH_SPEED_MIN = 0.22
SEARCH_SPEED_MAX = 0.55
AVOID_COMMIT_FRAMES = 12

ALPHA_CX = 0.62
ALPHA_AREA = 0.55

STALL_SONAR_THRESH   = 0.19
STALL_CMD_VEL_MIN    = 0.25
STALL_CONFIRM_FRAMES = 6    

STALL_RECOVER_BACK_FRAMES = 18
STALL_RECOVER_TURN_MAXFRAMES = 80  
STALL_COOLDOWN_FRAMES = 30

STATE_NORMAL  = 0
STATE_BACK    = 1
STATE_TURN    = 2  
STATE_RESUME  = 3


def procesar_mascara(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 100, 20), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 20), (180, 255, 255))
    mask = mask1 + mask2
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
    if area < AREA_MINIMA:
        return None, area
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, area
    cx = int(M["m10"] / M["m00"])
    return cx, area


def leer_sonar(robot):
    s = robot.get_sonar()
    left        = min(s[0], s[1], s[2])
    front_left  = min(s[2], s[3])
    front       = min(s[3], s[4])
    front_right = min(s[4], s[5])
    right       = min(s[5], s[6], s[7])
    front_raw   = [s[2], s[3], s[4], s[5]]
    return left, front_left, front, front_right, right, front_raw


def peligro(distancia, umbral):
    return limitar((umbral - distancia) / umbral, 0.0, 1.0)


def signo(x):
    if x > 0.05:
        return 1.0
    if x < -0.05:
        return -1.0
    return 0.0


def elegir_lado(left, fl, fr, right, error_objetivo, ultimo_error):
    despeje    = (right + fr) - (left + fl)
    referencia = error_objetivo if error_objetivo is not None else ultimo_error
    decision   = 1.15 * despeje + 0.85 * referencia
    if abs(decision) < 0.05:
        decision = referencia
    if abs(decision) < 0.05:
        decision = despeje
    lado = signo(decision)
    if lado == 0.0:
        lado = 1.0
    return lado


def repulsion(left, fl, front, fr, right, error_objetivo, ultimo_error, avoid_dir, avoid_timer):
    pL  = peligro(left,  U_LATERAL)
    pFL = peligro(fl,    U_FRONT_LAT)
    pF  = peligro(front, U_FRONTAL)
    pFR = peligro(fr,    U_FRONT_LAT)
    pR  = peligro(right, U_LATERAL)
    peligro_total = max(pF, pFL, pFR, 0.75 * pL, 0.75 * pR)
    if pF > 0.05 or front < U_FRONTAL:
        if avoid_timer <= 0:
            avoid_dir = elegir_lado(left, fl, fr, right, error_objetivo, ultimo_error)
            avoid_timer = AVOID_COMMIT_FRAMES
        else:
            avoid_timer -= 1
    else:
        avoid_timer = max(0, avoid_timer - 1)
        if avoid_timer == 0:
            avoid_dir = elegir_lado(left, fl, fr, right, error_objetivo, ultimo_error)
    giro_lateral    = K_REP_LATERAL    * ((pL + 0.75 * pFL) - (pR + 0.75 * pFR))
    giro_frontal    = K_REP_FRONTAL    * pF * avoid_dir
    giro_tangencial = K_REP_TANGENCIAL * (pFL - pFR)
    giro = limitar(giro_lateral + giro_frontal + giro_tangencial, -VEL_GIRO_MAX, VEL_GIRO_MAX)
    avance_factor = 1.0 - limitar(
        0.95 * pF + 0.45 * max(pFL, pFR) + 0.25 * max(pL, pR), 0.0, 0.92
    )
    return giro, avance_factor, peligro_total, avoid_dir, avoid_timer


def limitar(x, a, b):
    return max(min(x, b), a)


def main():
    sim   = robotica.Coppelia()
    robot = robotica.P3DX(sim.sim, "PioneerP3DX", True)
    sim.start_simulation()

    last_cx        = None
    last_error     = 0.0
    lost           = MEMORY_FRAMES + 1
    prev_error     = 0.0
    integral_error = 0.0
    area_filtrada  = 0.0
    avoid_dir      = 1.0
    avoid_timer    = 0

    recover_state  = STATE_NORMAL
    recover_timer  = 0
    recover_dir    = 1.0  
    stall_counter  = 0
    stall_cooldown = 0

    try:
        while sim.is_running():
            img = robot.get_image()
            if img is None:
                continue

            h, w, _ = img.shape
            center = w / 2.0

            mask     = procesar_mascara(img)
            cx, area = obtener_centro(mask)

            if recover_state == STATE_NORMAL:
                if cx is not None:
                    last_cx       = ALPHA_CX   * float(cx)   + (1.0 - ALPHA_CX)   * last_cx \
                                    if last_cx is not None else float(cx)
                    area_filtrada = ALPHA_AREA * float(area) + (1.0 - ALPHA_AREA)  * area_filtrada
                    lost = 0
                else:
                    lost          += 1
                    area_filtrada *= 0.88

            if last_cx is not None and lost <= MEMORY_FRAMES:
                cx_use           = last_cx
                area_use         = area_filtrada if cx is not None \
                                   else min(area_filtrada, AREA_OBJETIVO * 0.35)
                objetivo_memoria = True
            else:
                cx_use           = None
                area_use         = 0.0
                objetivo_memoria = False

            error = limitar((cx_use - center) / center, -1.0, 1.0) if cx_use is not None else None
            if error is not None and recover_state == STATE_NORMAL:
                last_error = error

            left, fl, front, fr, right, front_raw = leer_sonar(robot)
            giro_rep, avance_factor, peligro_total, avoid_dir, avoid_timer = repulsion(
                left, fl, front, fr, right, error, last_error, avoid_dir, avoid_timer
            )

            if cx_use is not None:
                if area_use < AREA_OBJETIVO:
                    proximity        = np.sqrt(limitar(area_use / AREA_OBJETIVO, 0.0, 1.0))
                    base             = VEL_AVANCE_MAX * (1.0 - proximity)
                    alignment        = limitar(1.0 - 0.75 * abs(error or 0), 0.25, 1.0)
                    avance_tentativo = base * alignment * avance_factor
                else:
                    avance_tentativo = 0.0
            else:
                avance_tentativo = VEL_AVANCE_MAX * 0.12 * avance_factor

            if stall_cooldown > 0:
                stall_cooldown -= 1
                stall_counter   = 0
            elif recover_state == STATE_NORMAL:
                sonar_en_contacto = any(s < STALL_SONAR_THRESH for s in front_raw)
                avanzando         = avance_tentativo > STALL_CMD_VEL_MIN
                if sonar_en_contacto and avanzando:
                    stall_counter += 1
                else:
                    stall_counter  = max(0, stall_counter - 1)

                if stall_counter >= STALL_CONFIRM_FRAMES:
                    despeje_lateral = (right + fr) - (left + fl)
                    recover_dir     = signo(despeje_lateral)
                    if recover_dir == 0.0:
                        recover_dir = 1.0

                    recover_state  = STATE_BACK
                    recover_timer  = STALL_RECOVER_BACK_FRAMES
                    stall_counter  = 0
                    stall_cooldown = STALL_COOLDOWN_FRAMES

                    last_cx        = None
                    lost           = MEMORY_FRAMES + 1
                    last_error     = 0.0
                    prev_error     = 0.0
                    integral_error = 0.0

            if recover_state == STATE_BACK:
                avance = -0.85
                giro   = recover_dir * 0.35
                recover_timer -= 1
                if recover_timer <= 0:
                    recover_state = STATE_TURN
                    recover_timer = STALL_RECOVER_TURN_MAXFRAMES

            elif recover_state == STATE_TURN:
                avance = 0.0
                giro   = recover_dir * VEL_GIRO_MAX * 0.75
                recover_timer -= 1

                bola_visible = cx is not None and area > AREA_MINIMA
                if bola_visible or recover_timer <= 0:
                    recover_state = STATE_RESUME

            elif recover_state == STATE_RESUME:
                if cx is not None:
                    last_cx       = float(cx)
                    area_filtrada = float(area)
                    lost          = 0
                    last_error    = limitar((last_cx - center) / center, -1.0, 1.0)
                recover_state = STATE_NORMAL
                avoid_dir     = 1.0
                avoid_timer   = 0

            else:
                if error is not None:
                    deriv          = error - prev_error
                    integral_error = limitar(integral_error + error, -8.0, 8.0)
                    giro_atr       = KP * error + KD * deriv + KI_SUAVE * integral_error
                    prev_error     = error
                else:
                    integral_error *= 0.82
                    prev_error     *= 0.65
                    busqueda        = signo(last_error) or signo(avoid_dir) or 1.0
                    giro_atr        = limitar(
                        busqueda * (SEARCH_SPEED_MIN + SEARCH_SPEED_MAX
                                    * min(lost, MEMORY_FRAMES) / MEMORY_FRAMES),
                        -SEARCH_SPEED_MAX, SEARCH_SPEED_MAX,
                    )

                peso_rep = limitar(0.25 + 0.75 * peligro_total, 0.25, 1.0) \
                           if peligro_total > 0.0 else 0.0
                peso_atr = (1.0 if objetivo_memoria else 0.55) \
                           if peligro_total > 0.0 else 1.0

                giro   = limitar(peso_atr * giro_atr + peso_rep * giro_rep,
                                 -VEL_GIRO_MAX, VEL_GIRO_MAX)
                avance = avance_tentativo

                if peligro_total > 0.35 and avance > 0.0:
                    avance = max(avance,
                                 VEL_AVANCE_MIN_EVITANDO * (1.0 - min(peligro_total, 0.8)))

                if front < U_EMERGENCIA:
                    avance = 0.0
                    giro   = limitar(giro + avoid_dir * 0.55, -VEL_GIRO_MAX, VEL_GIRO_MAX)

                avance = limitar(avance, 0.0, VEL_AVANCE_MAX)

            robot.set_speed(avance + giro, avance - giro)

            estado_txt = {
                STATE_NORMAL: "NORMAL",
                STATE_BACK:   "STALL>BACK",
                STATE_TURN:   "STALL>TURN",
                STATE_RESUME: "STALL>RESUME",
            }.get(recover_state, "?")
            color = (0, 220, 0) if recover_state == STATE_NORMAL else (0, 80, 255)
            cv2.putText(img, estado_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img, f"stall:{stall_counter}/{STALL_CONFIRM_FRAMES}  front:{front:.2f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 0), 1)
            cv2.circle(img, (int(center), 75), 5, (255, 255, 255), -1)
            if cx_use is not None:
                cv2.circle(img, (int(cx_use), 75), 8, (0, 0, 255), -1)
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