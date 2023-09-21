# Mapless Transport

Create python env.
````
conda create -n mapless_env python=3.9
conda activate mapless_env
pip install -r requirements_freeze.txt
````

Objetivo final: Mapless Transportation: Ter uma política de RL capaz de levar um objeto circular e um retangular para um goal em um ambiente simples com obstáculos. => Definir melhor o que é um ambiente simples.

Objetivo: agente controlável por teclado para navegar em um abiente com obstáculos,
usando laser. Tentar fazer o melhor dentro do possível => é um algoritmo local.

A Fazer:
Primeiro, vou deixar o mapless navigation rodando bem.
- Para quais cenários deveria funcionar? E quais não conseguiria? => dar uma olhada em artigos de mapless navigation.
- Atingir 100% de sucesso em cenários simples:
    - Definir os mapas:
        - 4 obstáculos circulares
        - Corredor
        - U
        - G
        - Aleatório 1
        - Aleatório 2
        - Aleatório 3
        - Complexo 1
        - COmplexo 2
    - Corrigir problema do loop local:
        - Modificar observacao para ter angulo e distancia do goal.
        - Frame stack + ação tomada. => implementar como um Wrapper?
    - Ajeitar: observacao e reward.
    - Ajeitar treinamento do PPO: Entender o algoritmo e tunar.
- Safe eval => Impedir que o agente tome ações que o levem a colidir com obstáculos.


Ideias para melhorar:
- Modificar observacao para ter angulo e distancia do goal.
- Aumentar o range do laser e número de raios => maior campo de visão para ver obstáculos e caminhos sem volta.
- Frame stack + ação tomada. => implementar como um Wrapper?
- Modificar a ação do robô => velocidade linear constante e altera apenas a angular. 
    - A ideia é criar um robô diferencial. => fazer que seja quadrado também.
    - Ajustar a observação para estar no sistema de coord do robô.
    - Problema: Para pushing vai funcionar? Talvez precise de vários ajustes de rota. Dá para fazer com um diferencial?


Ambientes:
- Range dos laser e número de raios é importante.
- Mas, e se o range for curto e o robô cair em loop?
    - Lembrar das últimas n ações e observações deve prevenir loops de tamanho 2*n.