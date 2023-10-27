Ok.
Estamos com problemas.
O PPO não está conseguindo lidar com o vai e volta da política, fincando preso em alguns momentos.
Uma hipótese é: 
- Acertar na observação e na recompensa.
- Treinar por mais tempo com uma rede suficientemente capaz.

Observação:
- Últimas 5 ações => dá para a rede pegar os padrões de ficar preso.

Recompensas:
- Simplificar: não usar recomepnsa de progresso.
- Apenas:
    - Sucesso, morte e tempo

Testar:
- Box pushing simples

Também quero testar curriculum learning.

EM resumo, quero quebrar o problema em etapas.
Começar por mais fáceis e ir incrementando. Quando começar a dar errado, eu paro e tento corrigir.

Etapas
- Navigation simples => funcionou 99%
- Navigation medium => 93%
Problema:
    - Agente se mata as vezes
    - Agente fica preso.
Solução:
    - Colocar as 5 últimas ações na obs
    - Treinar por mais tempo com uma rede de 3x64. 

Teste atual:
- Com mais treinamento e 5 ações, consigo chegar em 99%?
=> Não.

Os artigos usam redes maiores: 3x512

Próximos testes:
- Aumentar a rede

Desafio atual: eliminar falha por stuck.
Ideias:
- Lembrar as ações passadas
- Lembrar as ações e observações passadas
- Usar uma rede recorrente.

Investir: Lembrar ações e observações.
Quantas lembrar e com qual granularidade?
Começar com simplicidade:
- Lembrar das 5 observacoes inteiras e acoes

Tentar usar o PPO-Recurrent => Muito lento

Opções para atingir o 99%:
- Curriculum learning
- Tunar o PPO
- Testar outro algoritmo



