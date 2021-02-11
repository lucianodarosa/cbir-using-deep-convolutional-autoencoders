# cbir-using-deep-convolutional-autoencoders

Implementação de um pipeline de treinamento de redes neurais profundas baseado em python e framework tensorflow, contendo várias opções de parametrização e personalização. Após o treinamento das redes neurais, que podem ser acompanhados pelo painel tensorboard, segue a implementação de um mecanismo de extração de features das imagens, baseados em aprendizado não supervisionado (auto-encoders convolucionais profundos) que são utilizados para a extração das features baseados nos pesos da camada latente. Todas as features são então posteriormente armazenadas para uso em um mecanismo de busca por imagem.


