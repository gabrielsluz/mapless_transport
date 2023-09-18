# Mapless Transport

Create python env.
````
conda create -n mapless_env python=3.9
conda activate mapless_env
pip install -r requirements_freeze.txt
````


Objetivo: agente controlável por teclado para navegar em um abiente com obstáculos,
usando laser. => visualizar tanto visão global do mundo, quanto a visão de laser.

A Fazer:
- Visualizar política final.
- Código de Eval.
- Reward Engeneering
- Testar PPO

Ideia inicial:
- Rodar um playground que permita controlar o agente.
- Montar o ambiente de treinamento para navegacao.

Sensing:
- Ver objeto (perfeito) e obstáculos (laser).

Navigation simulator:
- Agente, goal, obstáculos.