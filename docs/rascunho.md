Tem algumas decisões importantes:
- Mapas
- Qual o ápice do projeto?
- Usar ou não um pathfinder não neural?


Ápice do projeto
- Controlar a pose do objeto.
- 3 tipos de objetos: círculo, retângulo e triângulo.
- Obstáculos dinâmicos.


Pathfinder:
- Dá para fazer um teste sem o local path planner => use um global para validar a ideia.

Cada episódio de RL é: levar o objeto até onde o path planner mandar.
Vantagens:
- Facilita o aprendizado.
- Dá para fazer asserções como: sempre achamos o caminho, pois o path planner tal está sendo usado.

E se quisermos aprender o path planner?
=> podemos treiná-lo para gerar a saída do path planner.

Quero experimentar com isso.
A fazer:
- Implementar path planning
- Criar env.

Implementar path planning:
- Usar código da disciplina
- Usar shapely
    - Converter o mapa para shapely
    - Lidar com as coordenadas. => talvez nem precise, pois quem inverte é o cv2.
- Funções:
    - criar mapa shapely
    - Aplicar código da disciplina
    - Pegar o primeiro subgoal.

Como vai ser o env?:
- A cada novo episódio gerar o mapa inteiro? => sim, para simplificar.
- O episódio continuará sendo definido do jeito que é hoje, mas os gols estarão bem mais perto.