# 🐱🐶 Classificador de Gatos e Cachorros com TensorFlow

Este projeto utiliza uma rede neural convolucional (CNN) simples para classificar imagens entre **gatos** e **cachorros**, usando o dataset "Cats vs Dogs" disponível no `tensorflow_datasets`.

---

## 📂 Estrutura do Projeto

O arquivo de código `main.py` contém todos os passos para realizar a construção, treino e uso da rede. O arquivo contém:

- `criar_modelo()` — Função que define a arquitetura da rede neural.
- `carregar_modelo()` — Função que carrega a arquitetura da rede neural salva em arquivo.
- `treinar()` — Treina o modelo com o conjunto de dados e salva o modelo treinado.
- `salvar_history()` — Salvar dados do histórico.
- `carregar_history()` — Carregar dados do histórico.
- `visualizar_history()` — Plot dados do histórico.
- `test_image()` — Script para testar o modelo com uma imagem aleatória da base de validação.
- `visualizar_modelo()` — Demonstrar o conteúdo inserido na primeira iteração da rede.

---

## ✅ Funcionalidades

- Classificação binária de imagens (Gato ou Cachorro)
- Utilização de CNN com `Conv2D`, `MaxPooling2D` e `Dense`
- Salvamento automático do modelo treinado
- Reuso do modelo salvo sem necessidade de re-treinamento
- Visualização de predições com imagens aleatórias
- Visualização da primeira camada da rede

---

## 🧠 Pré-requisitos

> É recomendado o uso de um gerênciador de versão do python para este trabalho, como o [PyEnv](https://github.com/pyenv/pyenv).


> ⚙️ A aplicação faz uso de bibliotecas não-presentes neste repositório.
Para instalar estas, o projeto acompanha arquivo de requisitos `requirements.txt`.

> Em caso de incompatibilidade com hardware, é possível alterar os requisitos de execução e diminuir tempo de instalação, removendo `[and-cuda]` presente no arquivo `requirements.txt` como tensorflow[and-cuda].

> A aplicação possui alguns segmentos custosos de treino em algoritmos de aprendizado de máquina. [Neste repositório](https://1drv.ms/f/c/81eff739ca6e213e/ElzNFnvr2s5AnjhOt-kFQtEB_pqSiz2WTDTDxNtLHnkh8Q?e=nTF7D8) é possível encontrar os modelos pré-treinados.

Para realizar a instalação das dependências do projeto, execute: 

```bash
pip3 install -r requirements.txt
```



---

## ▶️ Como Executar

Executar o script em modo interativo apenas inicializa o TensorFlow.
```bash
python -i main.py

```

Executar o script direto produz um modelo, treina e exibe métricas, usando 5 Epochs.
```bash
python main.py

```