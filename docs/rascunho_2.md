# Mapless Transportation

## Problema
Mapless transportation via pushing consiste em um robô empurrar um objeto até um objetivo desviando de obstáculos com informação local.
Objeto: círculo, retângulo e triângulo.
Objetivo: posição 2D ou configuração: pos + orientação.

Obstáculos: fixos ou dinâmicos.

Ápice:
Obstáculos dinâmicos lentos e aleatórios, objetivo como a pose do objeto, todos os 3 formatos de objeto.


## Solução
Dois módulos: Mapless Navigation e Pushing

Mapless Navigation: Planeja o caminho do objeto até o objetivo.
Pushing: Empurra o objeto até o subgoal.

## Cenários

### Simples
- Sem obstáculos
- Objeto circular
=> já resolvemos.

### Retângulo sem obstáculos
Agora requer mais planejamento.

Ideia:
- Testar abordagens de RL
- Observation: object pose (não usar imagem por enquanto).

Problema:
- Parece que o RL não está conseguindo "pensar" a longo prazo.
- Exploração => fica preso em alguns cenários.