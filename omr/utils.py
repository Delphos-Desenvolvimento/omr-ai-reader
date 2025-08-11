import pprint

def transformar_gabaritos(gabaritos_em_letras):
  """
  Converte uma lista de gabaritos do formato com chaves e valores em string
  para um formato com chaves e valores numéricos, com tratamento de erros.

  Mapeamento de respostas: a=0, b=1, c=2, d=3
  Mapeamento de questões: "1" -> 0, "2" -> 1, ..., "10" -> 9

  Args:
    gabaritos_em_letras: Uma lista de dicionários com respostas em letras.

  Returns:
    Uma lista de dicionários com os gabaritos convertidos para números.
  """
  # Dicionário que mapeia cada letra para seu número correspondente
  mapeamento_respostas = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
  
  gabaritos_numericos = []

  # 1. Tratamento de erro: Verifica se a entrada é uma lista
  if not isinstance(gabaritos_em_letras, list):
      print("Erro Crítico: O formato de entrada deve ser uma lista de gabaritos.")
      return [] # Retorna uma lista vazia em caso de erro crítico

  # Itera sobre cada gabarito e seu índice na lista de entrada
  for i, gabarito_original in enumerate(gabaritos_em_letras):
    
    # 2. Tratamento de erro: Verifica se cada item da lista é um dicionário
    if not isinstance(gabarito_original, dict):
        print(f"Aviso: O item de índice {i} não é um gabarito válido (dicionário) e foi ignorado.")
        continue # Pula para o próximo item da lista

    novo_gabarito = {}
    # Itera sobre cada item (ex: "1": "a") no dicionário do gabarito
    for questao_str, resposta_letra in gabarito_original.items():
      try:
        # Converte a chave da questão (string) para inteiro e ajusta o índice
        nova_chave = int(questao_str) - 1
        
        # Converte o valor da resposta (letra) para o número correspondente
        novo_valor = mapeamento_respostas[resposta_letra]
        
        # Adiciona a questão e resposta convertidas ao novo gabarito
        novo_gabarito[nova_chave] = novo_valor

      # 3. Tratamento de erro: Se a chave da questão não for um número
      except ValueError:
          return {
              "status": "bad_request",
              "message": f"Chave de questão inválida: '{questao_str}' no gabarito de índice {i}. Deve ser um número."
          }
      
      # 4. Tratamento de erro: Se a letra da resposta não for a, b, c ou d
      except KeyError:
            return {
                "status": "bad_request",
                "message": f"Resposta inválida: '{resposta_letra}' no gabarito de índice {i}. Deve ser 'a', 'b', 'c' ou 'd'."
            }
    
    # Adiciona o dicionário de gabarito (mesmo que parcial) à lista final
    if novo_gabarito:
        gabaritos_numericos.append(novo_gabarito)
    
  return gabaritos_numericos

if __name__ == "__main__":

    # --- Exemplo de Uso com Dados Corretos e Incorretos ---

    # Dados mistos para testar o tratamento de erros
    dados_com_erros = [
        # Gabarito válido
        {"1": "a", "2": "b", "3": "b", "4": "b", "5": "c", "6": "d", "7": "a", "8": "c", "9": "c", "10": "c"},
        
        # Gabarito com erros
        {"1": "c", "2": "c", "3": "z", "quatro": "b", "5": "c"}, # Resposta 'z' e chave 'quatro' inválidas
        
        # Item inválido na lista
        "não é um dicionário",

        # Gabarito parcialmente válido
        {"1": "a", "2": "d"}
    ]

    print("--- Processando dados com erros ---")
    GABARITOS = transformar_gabaritos(dados_com_erros)

    print("\n--- Resultado do Processamento ---")
    print("Gabaritos válidos que foram convertidos:")
    pprint.pprint(GABARITOS)


