# -*- coding: utf-8 -*-
"""
Projet de robotique
Navigation par Champ de Potentiel et Remontée de Gradient

Auteurs : Étudiant / Ingénieur Robotique
(c) Efrei Paris - Majeure Systèmes Robotiques et Drones
"""

import math
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import Timer as tmr
import Potential

# 1. Initialisation

# Position initiale du robot
x0 = -20.0
y0 = -20.0
theta0 = np.pi / 4.0  # Orientation initiale=45 degrés
robot = rob.Robot(x0, y0, theta0)

# Choix de la difficulté et du scénario (1, 2 ou 3 sources, random=False ou True)
DIFFICULTY = 2
RANDOM_SCENARIO = False
pot = Potential.Potential(difficulty=DIFFICULTY, random=RANDOM_SCENARIO)


# 2. Paramètres de l'asservissement et de la navigation

# Boucle de position / calcul du gradient
kpPos = 0.5
positionCtrlPeriod = 0.1  # Période d'échantillonnage de la position (s)
timerPositionCtrl = tmr.Timer(positionCtrlPeriod)

# Boucle d'orientation (asservissement angulaire)
kpOrient = 3.0
orientationCtrlPeriod = 0.02  # Période d'échantillonnage de l'orientation (s)
timerOrientationCtrl = tmr.Timer(orientationCtrlPeriod)

# Paramètres de la méthode de remontée de gradient
delta_grad = 0.2  # Pas spatial pour l'estimation numérique du gradient
v_max = 1.0  # Vitesse linéaire maximale (m/s)
omega_max = 2.0  # Vitesse angulaire maximale (rad/s)

# Gestion des waypoints (optionnel pour la mission 2 ou approche multi-sources)
WPlist = [[x0, y0]]
epsilonWP = 0.5
WPManager = rob.WPManager(WPlist, epsilonWP)


# 3. Configuration de la simulation

t0 = 0.0
tf = 300.0  # Durée totale de la simulation (s)
dt = 0.01  # Pas d'intégration numérique d'Euler (s)
simu = rob.RobotSimulation(robot, t0, tf, dt)

# Initialisation des variables de commande
Vr = 0.0
thetar = theta0
omegar = 0.0

# Variables pour l'estimation numérique du gradient
grad_x = 0.0
grad_y = 0.0

print("Début de la simulation de recherche de source...")

# 4. Boucle principale de simulation
for t in simu.t:

    # --- Boucle de calcul de position et de gradient ---
    if timerPositionCtrl.isEllapsed(t):

        # Récupération de la position actuelle du robot
        curr_x = robot.x
        curr_y = robot.y

        # Mesure du potentiel de pollution au point courant
        current_potential = pot.value([curr_x, curr_y])

        # Estimation numérique du gradient par différences finies centrées :
        p_plus_x = pot.value([curr_x + delta_grad, curr_y])
        p_minus_x = pot.value([curr_x - delta_grad, curr_y])
        grad_x = (p_plus_x - p_minus_x) / (2.0 * delta_grad)

        p_plus_y = pot.value([curr_x, curr_y + delta_grad])
        p_minus_y = pot.value([curr_x, curr_y - delta_grad])
        grad_y = (p_plus_y - p_minus_y) / (2.0 * delta_grad)

        # Norme et direction du gradient (direction de la plus forte pente / source)
        grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)

        if grad_norm > 1e-5:
            # Orientation de référence vers le gradient max
            thetar = np.arctan2(grad_y, grad_x)

            # Adaptation de la vitesse linéaire : proportionnelle à la norme du gradient
            Vr = min(v_max, 0.1 + kpPos * grad_norm)
        else:
            # Si le gradient = nul (sommet atteint ou zone plate)
            Vr = 0.0
            thetar = robot.theta

    # --- Boucle d'asservissement en orientation ---
    if timerOrientationCtrl.isEllapsed(t):

        # Normalisation de l'erreur angulaire entre -pi et pi
        angle_error = thetar - robot.theta
        while angle_error > math.pi:
            angle_error -= 2.0 * math.pi

        while angle_error < -math.pi:
            angle_error += 2.0 * math.pi

        # Loi de commande proportionnelle en orientation
        omegar = kpOrient * angle_error
        # Saturation de la vitesse angulaire
        omegar = max(-omega_max, min(omega_max, omegar))

    # --- Application des commandes au robot ---
    robot.setV(Vr)
    robot.setOmega(omegar)

    # Intégration du mouvement
    robot.integrateMotion(dt)

    # Sauvegarde des données pour graphiques et post-traitement
    simu.addData(robot, WPManager, Vr, thetar, omegar, pot.value([robot.x, robot.y]))

print("Simulation terminée avec succès.")


# 5. Génération des graphiques et résultats

plt.close("all")

# 1. Trajectoire 2D du robot superposée au champ de potentiel
fig, ax = simu.plotXY(1, xmin=-25, xmax=25, ymin=-25, ymax=25)
pot.plot(noFigure=None, fig=fig, ax=ax)
ax.set_title("Trajectoire du robot")

# 2. Évolution temporelle/spatiale du potentiel de pollution mesuré
simu.plotPotential(4)

# Relief de pollution et trajectoire
simu.plotPotential3D(5)

plt.show()