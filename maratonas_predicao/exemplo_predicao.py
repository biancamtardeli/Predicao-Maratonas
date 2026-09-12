# -*- coding: utf-8 -*-
"""
exemplo_predicao.py
---------------------
Exemplo mínimo de uso do sistema já treinado para simular uma equipe
antes da inscrição em uma competição -- este é o uso "do dia a dia" que
o(a) orientador(a)/técnico(a) faria ao montar os times.

Uso:
    python exemplo_predicao.py
"""

from src.predict import carregar_preditor

if __name__ == "__main__":
    print("Treinando o modelo com todo o histórico disponível...\n")
    preditor = carregar_preditor(nome_modelo="RandomForest")

    # --------------------------------------------------------------
    # TROQUE os nomes abaixo pelos alunos que você quer testar juntos,
    # e a olimpíada-alvo. Alunos que nunca competiram entram como
    # "estreantes" (o sistema avisa quando isso acontece).
    # --------------------------------------------------------------
    equipes_para_testar = [
        (["Tiago Fernandes Moraes", "Miguel Custodio Pereira", "Lucio Garcia Leite Filho"], "OLIP"),
        (["Tiago Fernandes Moraes", "Yasmin Pereira Pasqua", "Jean Ferreira Dias"], "Regional"),
    ]

    for nomes, olimpiada in equipes_para_testar:
        prob, faltantes = preditor.prever(nomes, olimpiada)
        print(f"Equipe: {', '.join(nomes)}")
        print(f"Olimpíada-alvo: {olimpiada}")
        print(f"Probabilidade estimada de alto desempenho: {prob:.1%}")
        if faltantes:
            print(f"  (alunos sem histórico na base, tratados como "
                  f"estreantes: {', '.join(faltantes)})")
        print("-" * 60)
