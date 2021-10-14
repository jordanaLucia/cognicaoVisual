# cognicaoVisual

Este trabalho utilizou uma rede neural pré treinada disponível no TensorFlow/Keras.
Além de imagens disponíveis no Flickr para treinamento e testes.

É necessário instalar os seguintes pacotes:</br>
1 - tensorFlow</br>
2 - keras</br>

É necessário descompactar as pastas </br>
1 - Flickr8k_text.zip</br>
2 - Flickr8k_Dataset.zip</br>
3 - descriptions.zip</br>

a)Para submeter uma imagem nova ao modelo definido como resultado deste trabalho:</br>
1 - coloque uma (apenas uma por vez) imagem na pasta ./Cognicao Visual/imagem_teste</br>
2 - no arquivo 'testeNovaFotografia.py',altere o endereço home_folder = 'C:/Users/jordanareis/Documents/Cognicao Visual/' para o correspondente na máquina que será utilizada.</br>
3 - é importante que os arquivos 'features.pkl', 'tokenizer.pkl', 'descriptions' e o modelo 'model-ep004-loss3.543-val_loss3.877.h5' estejam no diretório home_folder.</br>
4 - executar o arquivo 'testeNovaFotografia.py'</br>

b)Para executar todos os passos e verificar se alguma época traz um modelo mais performático</br>
1 - executar o arquivo 'cognicaoVisual.py'</br>
2 - identificar o modelo mais performático, a partir da nomenclatura, que contém a perda estimada</br>
3 - seguir o passo-a-passo do item a) alterando o valor da variável modelo para o identificado no passo 2 acima.</br>
