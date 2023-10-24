# Mapless Transport

Create python env.
````
conda create -n mapless_env python=3.9
conda activate mapless_env
pip install -r requirements_freeze.txt
````

Objetivo final: Mapless Transportation: Ter uma política de RL capaz de levar um objeto circular e um retangular para um goal em um ambiente simples com obstáculos. => Definir melhor o que é um ambiente simples.

Passos:
- Voltar para o navigation => fazer funcionar nos mapas de pushing.





Pushing:
- Observacao: goal e centro do objeto (circular)

## Experimento no navigation
Ideias que melhoraram:
- Mudar para o Agent Forward
- Observação:
    - Mais lasers => não tenho certeza se ajudou.
    - Deixar entre 0 e 5 a distância normalizada.
- Recompensa:
    - Dar maior penalidade de morte.
    - DIminuir recompensa de progresso e penalidade de tempo.

Pontos melhorados:
- Agent stuck in a loop.
- Agente se matando ao passar muito perto dos obstáculos.

## Como resolver o problema de uma forma abstrata?
Em cada momento, a política deve:
- Qual direção o objeto deve ir?
- Como mover o robô?

Não é guloso:
- Tem que contornar obstáculos.
- Tem que mover o robô para checar na posição de empurrar.

São dois problemas separados.
Primeiro, define para onde o objeto deve ir e depois a ação do robô.
A movimentação do robô também depende dos lasers.
Mas, não são tão separados? => Pegar um caso.
A parte de definir onde o objeto vai, depende de onde o robô pode ir. São joint.

### Implicações: 
- Diminuir o force length
    - Entre 2, 1 e 0.5, 1 teve o melhor resultado.



# Checklist para melhorar.

Ambiente:

- Observação:
    - Laser:
        - N raios
        - Range
        - Not detected value
    - Goal:
        - Distância e ângulo
    - Memória:
        - N últimas ações
        - N últimas observações

- Recompensa:
    - Progresso
    - Colisão
    - Goal
    - Escala das recompensas.

- Agente:
    - MLP Policy
    - Ações:
        - Número e force length
    - PPO:


Ideias:
- Colocar penalidade de tempo
- Tirar as ações passadas.
- Tunar os hiperparâmetros do PPO:
- A solução fixa é:
    - Tome a ação que te leva mais perto do goal e que não colide.

Cirurgíca:
Pegar um ambiente simples => sem dead-ends e ambiguidade.
Ambiente => circle-line
E visualizar o aprendizado com o tempo:
- Comportamento do agente. => video ao vivo
- Parâmetros do agente. => plotar com o tempo.
Quero visualizar o dataset de aprendizado supervisionado => o que ele está tentando aprender?

Tentar entender: aprendizado supervisionado vai funcionar, mas está aprendendo na direção certa?
Comparar labels e predições.
Intuitivamente, como o algoritmo deve aprender ao longo das tentativas e recompensas?

Tentar visualizar:
- Pegar 5 cenários importantes e visualizar a função de valor e a política. e ver como ela se altera com o treinamento.
=> Talvez plotar no mapa setas mostrando a ação escolhida.

Qual relação com campos potenciais?
Ideia: Implementar um campos potenciais e visualizar o campo nos cenários.
=> Depois, tentar melhorar com ML:
- Tunar parâmetros com ML
- Dar o campo como entrada para o RL => pode funcionar para o transporte => damos o campo do objeto e do robô.


Descobri:
- Force length menor ajuda.
- Range do sensor menor => talvez nao ajude no RL..

Testar:
- Input = forças potenciais
- Comparar com taxa de sucesso do campos.
- Testar mais ações => 24