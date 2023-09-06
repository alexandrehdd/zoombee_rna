# Zoombee - Identificação de espécie animal por meio de Redes Neurais Convolucionais

 
1 - Primeira Etapa: Obtenção de um modelo  a partir do dataset público Snapshot Serengethi.


2 - Segunda Etapa:
 https://drive.google.com/drive/folders/1_Q4s_TdpjXTcMhtWXDbkVtRllZQvl6ES?usp=sharing

O objetivo dessa etapa é a obtenção de classificador para as imagens do catálogo da Vale, com base na extração de características de um modelo pré-treinado obtido na primeira etapa. No entanto, um dos principais desafios enfrentados foi o desequilíbrio na quantidade de imagens disponíveis para cada espécie animal, com um total de apenas 122 imagens. Esse desequilíbrio poderia prejudicar a capacidade do modelo de realizar classificações precisas e robustas.

Para superar esse desafio, implementamos a técnica de aumento de dados (data augmentation). Essa técnica envolve a geração de amostras adicionais a partir das imagens originais, aplicando diversas operações de transformação, como variações de ângulo de visão, ajustes de brilho, contraste e outras modificações. Com esse procedimento, conseguimos artificialmente expandir o nosso conjunto de dados original de 122 imagens.

O aumento de dados permitiu que o modelo fosse exposto a uma variedade maior de variações nas imagens, tornando-o mais robusto e capaz de lidar melhor com imagens de entrada diferentes. Isso é especialmente importante quando se tem um conjunto de dados pequeno e desbalanceado, pois ajuda a evitar o sobreajuste (overfitting) e melhora o desempenho do modelo em dados de teste não vistos.

Espécies no catálogo:
   veado, sapo, cobra, rato, cuica, macaco, jabuti, jacare, gaviao, irara, porco, cachorro, preguica, gato, cutia, tamandua, capivara, lagarto, quati e gamba.
   
###**Imagem Original:**
![image](https://github.com/alexandrehdd/zoombee_rna/assets/78443037/a3ff485e-a992-4bf9-9ae5-ec1c44e4da68)


###**Imagem Aumentada:** 
![image](https://github.com/alexandrehdd/zoombee_rna/assets/78443037/85dd2331-945f-40b9-9ca7-53dd8f6d46cd)
